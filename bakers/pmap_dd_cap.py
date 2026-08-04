#!/usr/bin/env python3
"""pmap_dd_cap - CAP (lower) the per-model draw distance of OPAQUE, non-landmark
models IN PLACE, for BOTH v2 (raw) and v3 (lz4-streamed) .pmap tiles.

The mirror image of pmap_dd_bump.py (which RAISES alpha-model dd to a floor). This
tool LOWERS the draw distance of opaque, non-landmark models to a cap so fewer distant
instances survive the renderer's frustum+distance test (their LODxxx proxy is meant to
cover them past the cap) - aimed at the dense downtown belt where the frame is bound by
draw-call / texture-bind / CLUT-upload count, not triangles or streaming ().

Why this is safe on v3 (lz4) without decompressing anything
-----------------------------------------------------------
draw_dist lives in the RESIDENT model table - the engine culls by draw_dist BEFORE it
streams a model's geometry, so the whole model table sits in the always-resident prefix
[0, vertex_off). tools/pmap_lz4.py copies "header..grid+cell tables..instances" (the
resident prefix) VERBATIM into v3 and only replaces the vertex/index/texel/clut pools
with compressed blobs (which begin at vertex_off). header.model_off is an ABSOLUTE file
offset already shifted by the v3 writer, so it points straight at the model table in both
v2 and v3. draw_dist is a leaf field (no dependent offsets), so we patch its 4 bytes in
place - file length is unchanged and the compressed blobs / trailing comp u32s are never
touched.

Per model, cap iff ALL of:
 * OPAQUE - NO submesh references an alpha texture. Alpha is detected exactly as
 pmap_dd_bump does: a texture is alpha when (num_levels >> 8) & 3 != 0.
 (opaque = the inverse over every submesh of the model.)
 * NOT landmark - bound_radius < LANDMARK_R (default 60.0). Big skyline buildings keep
 their long draw distance so downtown still reads at range.
 * draw_dist > CAP (default 320.0).
then draw_dist := CAP. Alpha models (foliage - they rely on the pmap_dd_bump floor),
landmarks, and already-short models are left untouched.

Every file touched is BACKED UP first (to the scratchpad dir below) so the change is
fully reversible; the first backup of a file is preserved across re-runs.

Usage:
 pmap_dd_cap.py <region_dir-or-file> [cap=320] [landmark_r=60] [--dry-run]

 --dry-run : analyse + print what WOULD be capped, write nothing (no backup, no patch).
"""
import glob
import os
import shutil
import struct
import sys

# Reversibility: pristine copies of every touched file land here so the coordinator can
# revert if the skyline looks wrong on device.
BACKUP_DIR = (""
             r"\f1e716b9-ee1c-4107-956e-1c7477c0b3b6\scratchpad\dd_cap_backup")

PMAP_MAGIC = 0x50414D50          # 'PMAP' little-endian
MODEL_STRIDE   = 32              # first_submesh,submesh_count,scale,cx,cy,cz,bound_r,draw_dist
SUBMESH_STRIDE = 20              # texture(i32),vfirst,vcount,ifirst,icount
TEX_STRIDE     = 32              # w,h,format,texel_first,texel_bytes,bufw,clut_first,clut_entries,num_levels
BR_OFF  = 24                     # bound_radius, within a MODEL record
DD_OFF  = 28                     # draw_dist, within a MODEL record
NL_OFF  = 28                     # num_levels, within a TEXTURE record
SMTEX_OFF = 0                    # texture(i32), within a SUBMESH record


def _read_header(data):
    """Return the header fields we need. Field offsets are identical for v2/v3/v4 (v3/v4
 just carry extra u32s AFTER this fixed prefix, and shift the *_off values - which are
 absolute - to point past the grown header)."""
    magic, version, file_size = struct.unpack_from("<III", data, 0)
    model_count, model_off     = struct.unpack_from("<II", data, 0x0C)
    submesh_count, submesh_off = struct.unpack_from("<II", data, 0x14)
    texture_count, texture_off = struct.unpack_from("<II", data, 0x1C)
    vertex_off = struct.unpack_from("<I", data, 0x30)[0]   # start of the (v3: compressed) pools
    return dict(magic=magic, version=version, file_size=file_size,
                model_count=model_count, model_off=model_off,
                submesh_count=submesh_count, submesh_off=submesh_off,
                texture_count=texture_count, texture_off=texture_off,
                vertex_off=vertex_off)


def _alpha_flags(data, h):
    """Per-texture alpha flag: (num_levels >> 8) & 3 != 0 (same test as pmap_dd_bump)."""
    tc, toff = h["texture_count"], h["texture_off"]
    out = [False] * tc
    for ti in range(tc):
        nl = struct.unpack_from("<I", data, toff + ti * TEX_STRIDE + NL_OFF)[0]
        out[ti] = ((nl >> 8) & 3) != 0
    return out


def _submesh_tex(data, h):
    sc, soff = h["submesh_count"], h["submesh_off"]
    return [struct.unpack_from("<i", data, soff + si * SUBMESH_STRIDE + SMTEX_OFF)[0]
            for si in range(sc)]


def _model_is_alpha(data, h, mi, sm_tex, tex_alpha):
    moff = h["model_off"] + mi * MODEL_STRIDE
    first, count = struct.unpack_from("<II", data, moff)
    tc = h["texture_count"]
    for k in range(count):
        t = sm_tex[first + k]
        if 0 <= t < tc and tex_alpha[t]:
            return True
    return False


def _backup(path):
    """Copy path into BACKUP_DIR once (the first, pristine copy is kept on re-runs)."""
    if not os.path.isdir(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, os.path.basename(path))
    if os.path.exists(dst):
        return dst, False        # keep the existing pristine backup
    shutil.copy2(path, dst)
    return dst, True


def _plan(data, h, cap, lr):
    """Return (to_cap, opaque_total, landmarks_kept) where to_cap is a list of
 (model_index, old_dd, bound_radius) for opaque non-landmark models with dd>cap."""
    tex_alpha = _alpha_flags(data, h)
    sm_tex = _submesh_tex(data, h)
    to_cap = []
    opaque_total = 0
    landmarks_kept = 0
    for mi in range(h["model_count"]):
        moff = h["model_off"] + mi * MODEL_STRIDE
        br = struct.unpack_from("<f", data, moff + BR_OFF)[0]
        dd = struct.unpack_from("<f", data, moff + DD_OFF)[0]
        if _model_is_alpha(data, h, mi, sm_tex, tex_alpha):
            continue                       # foliage - leave the pmap_dd_bump floor alone
        opaque_total += 1
        if dd <= cap:
            continue
        if br >= lr:
            landmarks_kept += 1            # big skyline building - keep its long dd
            continue
        to_cap.append((mi, dd, br))
    return to_cap, opaque_total, landmarks_kept


def cap_file(path, cap, lr, dry):
    data = bytearray(open(path, "rb").read())
    if len(data) < 0x34:
        print("  %s: too small, skip" % os.path.basename(path)); return (0, 0, 0)
    h = _read_header(data)
    if h["magic"] != PMAP_MAGIC:
        print("  %s: bad magic, skip" % os.path.basename(path)); return (0, 0, 0)
    if h["version"] not in (2, 3, 4):
        print("  %s: unsupported version %d, skip" % (os.path.basename(path), h["version"]))
        return (0, 0, 0)

    # The model table MUST live entirely inside the resident prefix [0, vertex_off).
    # If not (only possible if the format changed), refuse - patching could hit a blob.
    model_end = h["model_off"] + h["model_count"] * MODEL_STRIDE
    if model_end > h["vertex_off"]:
        print("  %s: model table (end 0x%x) spills past vertex_off 0x%x - REFUSE"
              % (os.path.basename(path), model_end, h["vertex_off"]))
        return (0, 0, 0)

    to_cap, opaque_total, landmarks_kept = _plan(data, h, cap, lr)

    if dry:
        print("  %s v%d: WOULD cap %d/%d opaque models (kept %d landmarks) [dry-run]"
              % (os.path.basename(path), h["version"], len(to_cap), opaque_total, landmarks_kept))
        return (len(to_cap), opaque_total, landmarks_kept)

    if not to_cap:
        print("  %s v%d: capped 0/%d opaque models (kept %d landmarks) - nothing to do"
              % (os.path.basename(path), h["version"], opaque_total, landmarks_kept))
        return (0, opaque_total, landmarks_kept)

    bdst, fresh = _backup(path)

    orig_size = len(data)
    for mi, _old, _br in to_cap:
        struct.pack_into("<f", data, h["model_off"] + mi * MODEL_STRIDE + DD_OFF, float(cap))
    assert len(data) == orig_size, "in-place patch changed file length!"
    open(path, "wb").write(data)

    ok, why = _verify_file(path, cap, lr, h, to_cap)
    if not ok:
        shutil.copy2(bdst, path)           # restore pristine
        print("  %s: VERIFY FAILED (%s) - RESTORED from backup" % (os.path.basename(path), why))
        return (0, opaque_total, landmarks_kept)

    print("  %s v%d: capped %d/%d opaque models (kept %d landmarks)%s"
          % (os.path.basename(path), h["version"], len(to_cap), opaque_total, landmarks_kept,
             "" if fresh else "  [backup already existed - kept pristine]"))
    return (len(to_cap), opaque_total, landmarks_kept)


def _verify_file(path, cap, lr, h_before, to_cap):
    """Re-read the patched file and confirm: version unchanged, file size unchanged, every
 opaque non-landmark model now has dd<=cap, and NOTHING outside the cap set moved."""
    data = bytearray(open(path, "rb").read())
    if len(data) != h_before["file_size"] and len(data) != h_before["vertex_off"] + (
            h_before["file_size"] - h_before["vertex_off"]):
        # file_size field is authoritative; compare on-disk length to it
        pass
    h = _read_header(data)
    if h["version"] != h_before["version"]:
        return False, "version changed"
    if len(data) != os.path.getsize(path):
        return False, "size read mismatch"
    if h["file_size"] != h_before["file_size"]:
        return False, "header file_size changed"
    if os.path.getsize(path) != h_before["file_size"]:
        return False, "on-disk size != header file_size"
    # every opaque non-landmark must be <= cap now; capped set must equal cap
    to_cap_set = {mi for mi, _o, _b in to_cap}
    tex_alpha = _alpha_flags(data, h)
    sm_tex = _submesh_tex(data, h)
    for mi in range(h["model_count"]):
        moff = h["model_off"] + mi * MODEL_STRIDE
        br = struct.unpack_from("<f", data, moff + BR_OFF)[0]
        dd = struct.unpack_from("<f", data, moff + DD_OFF)[0]
        if _model_is_alpha(data, h, mi, sm_tex, tex_alpha):
            continue
        if br >= lr:
            continue
        if mi in to_cap_set and abs(dd - cap) > 1e-3:
            return False, "model %d not capped (dd=%g)" % (mi, dd)
        if dd > cap + 1e-3:
            return False, "opaque non-landmark model %d still dd=%g > cap" % (mi, dd)
    return True, "ok"


def main():
    argv = [a for a in sys.argv[1:]]
    dry = False
    if "--dry-run" in argv:
        dry = True
        argv.remove("--dry-run")
    if not argv:
        print(__doc__)
        return 1
    target = argv[0]
    cap = float(argv[1]) if len(argv) > 1 else 320.0
    lr = float(argv[2]) if len(argv) > 2 else 60.0

    files = (sorted(glob.glob(os.path.join(target, "region_*.pmap")))
             if os.path.isdir(target) else [target])
    if not files:
        print("no region_*.pmap under", target); return 1

    print("pmap_dd_cap: cap=%.1f landmark_r=%.1f  %s  (%d file%s)%s"
          % (cap, lr, target, len(files), "" if len(files) == 1 else "s",
             "  [DRY-RUN]" if dry else ""))
    tot_cap = tot_opaque = tot_land = 0
    for f in files:
        n, op, land = cap_file(f, cap, lr, dry)
        tot_cap += n; tot_opaque += op; tot_land += land
    print("%s: %d opaque models %s across %d files (of %d opaque total, kept %d landmarks). backups -> %s"
          % ("DRY-RUN" if dry else "DONE",
             tot_cap, "would be capped" if dry else "capped",
             len(files), tot_opaque, tot_land,
             "(none written)" if dry else BACKUP_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
