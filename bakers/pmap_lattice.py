#!/usr/bin/env python3
"""Put every model's vertices on ONE world lattice of 1/128 unit.

Today each model carries its own int16 scale - 7203 distinct values across the world, a
2816x spread, and 69% of neighbours quantise on different grids. Sony's fill rule (GE Users
Manual p21, top-left) is correct but needs coordinates that match EXACTLY, so different
grids leave the seams you can see down the middle of a road in daylight.

1/128 = 7.8125 mm fits every model in the world: the largest spans 506 units, which is
64780 of the 65534 steps available. A handful of models world-wide would end up with fewer
than 64 steps across them; they are left alone and reported (too_small), same as any model
whose OWN worst-case extent could not fit the new grid at all (refused).

This pass touches x, y and z of vertices, plus each converted model's scale/cx/cy/cz.
Vertex count does not change, so .night (one entry per vertex), .nightd (runs addressed by
vertex index) and the header's vertex_bytes stay correct by construction - the thing an
earlier tessellation pass got wrong. UV, colour, indices and the submesh table are copied
through unchanged, and pmap_lattice_verify.py proves that byte for byte.

Wired into a chain, this runs unattended and more than once (re-converts, re-bakes), so
every file is self-checked before it is allowed to touch disk: verify_bytes(expect_lattice=
True, ref=original) runs on this pass's OWN output, and on any problem - lattice_bytes
raising, or the self-check finding one - that ONE tile is left exactly as it was (old
per-model scale, its seams unfixed) rather than written unverified. The write itself is a
temp file next to the target then an atomic replace, so a kill mid-write cannot leave a
truncated .pmap either. See main()'s docstring-adjacent comment for why this is the
finest-grained rollback available.

Usage:
 python tools/pmap_lattice.py <dir> # rewrite every region_*.pmap in place
 python tools/pmap_lattice.py <dir> --dry # report only, do not write
"""
import os
import struct
import sys
import time

import lz4.block

from pmap_lattice_verify import verify_bytes, verify_world_positions, PMAP_VERSION_STRIPPED

POS_STEP = 1.0 / 128.0
MIN_STEPS_ACROSS = 64          # below this a model is left on its own scale
MAX_STEPS = 65534

# v3 header is 23 u32 = 92 bytes (cross-checked against tools/pmap_lz4.py's HDR_V3=92
# and against pmap_lattice_verify.py's own HDR). A 24th field (uvrange_off) exists only
# in v4 - reading it here would silently swallow 4 bytes that belong to whatever table
# sits right after the header (normally the first model's own first_submesh field).
HDR = struct.Struct("<23I")
MODEL = struct.Struct("<2I6f")       # first_submesh,submesh_count,scale,cx,cy,cz,bound_r,draw_dist
SUBMESH = struct.Struct("<i4I")      # texture,vfirst,vcount,ifirst,icount
COMP = struct.Struct("<2I")          # off,csize
VERT_SZ = 12                         # s16 u,v; u16 colour; s16 x,y,z


def _spans(models, subs, i):
    """(vfirst, vcount, icount) for model i - a model's span runs from its first
 submesh's start to its last submesh's end. Matches pmap_lattice_verify.py's
 _model_spans exactly; the two must never disagree about what a model's geometry is."""
    first, count = models[i][0], models[i][1]
    if count == 0:
        return 0, 0, 0
    s0, sN = subs[first], subs[first + count - 1]
    vfirst = s0[1]
    return vfirst, sN[1] + sN[2] - vfirst, sN[3] + sN[4] - s0[3]


def lattice_bytes(buf):
    """Returns (new_bytes, stats). Models that cannot fit the lattice (refused) or are
 too small to bother (too_small) are left on their own scale, geometry untouched.
 Raises ValueError for anything that is not a well-formed v3-or-later LZ4 .pmap, or
 if a requantised coordinate would leave the int16 range - this pass never clamps."""
    if buf[:4] != b"PMAP":
        raise ValueError("not a .pmap (bad magic)")
    if len(buf) < HDR.size:
        raise ValueError("file is shorter than the header (%d < %d bytes)"
                         % (len(buf), HDR.size))
    h = HDR.unpack_from(buf, 0)
    if h[1] == PMAP_VERSION_STRIPPED:
        raise ValueError(
            "this is a STRIPPED tile (version=%d), not a malformed .pmap - it belongs "
            "to the world store (tools/world_store_build.py): its comp_model/comp_tex "
            "tables hold GLOBAL ids into a companion world.idx/world.dat, not local "
            "byte offsets, and this pass has no way to resolve one" % h[1])
    if h[1] not in (3, 4) or not h[20]:
        raise ValueError("not a v3-or-later LZ4 .pmap (version=%d comp_flag=%d) - "
                         "this pass only handles compressed pools" % (h[1], h[20]))

    stats = {"converted": 0, "refused": 0, "too_small": 0, "empty": 0}
    mc, moff, sc, soff = h[3], h[4], h[5], h[6]
    cmoff = h[21]
    try:
        models = [list(MODEL.unpack_from(buf, moff + i * 32)) for i in range(mc)]
        subs = [SUBMESH.unpack_from(buf, soff + i * 20) for i in range(sc)]
        comp = [list(COMP.unpack_from(buf, cmoff + i * 8)) for i in range(mc)]
    except struct.error as exc:
        raise ValueError("a table runs past the end of the file (%s)" % exc)

    prefix = bytearray(buf[:h[12]])          # header + tables + grid, rewritten in place below
    blobs = []
    for i, m in enumerate(models):
        _, vcount, icount = _spans(models, subs, i)
        if vcount == 0 or not comp[i][1]:
            stats["empty"] += 1
            off, csize = comp[i]
            blobs.append(buf[off:off + csize])
            continue

        vbytes, ibytes = vcount * VERT_SZ, icount * 2
        off, csize = comp[i]
        raw = bytearray(lz4.block.decompress(buf[off:off + csize],
                                             uncompressed_size=vbytes + ibytes))

        scale, cx, cy, cz = m[2], m[3], m[4], m[5]
        # Worst case: this model's OWN scale times the full int16 range is the most it
        # could possibly span. If that worst case already fits the new grid, the real
        # data (which can only be smaller) certainly does - same test
        # pmap_lattice_verify.py uses to judge whether a refusal was legitimate.
        extent = scale * MAX_STEPS
        steps_needed = extent / POS_STEP
        if steps_needed > MAX_STEPS:
            stats["refused"] += 1
            blobs.append(lz4.block.compress(bytes(raw), mode="high_compression",
                                            store_size=False))
            continue
        if steps_needed < MIN_STEPS_ACROSS:
            # region_4_7 model #2 (extent exactly 0.5m) computes to
            # 63.999999940395355 steps here - one ULP below 64, because
            # scale = extent/65534 loses its last bit going through float32. It
            # classifies correctly either way (still too_small), but the real-world
            # too_small count (3) is only as stable as this expression's exact
            # rounding: if a future rewrite of this comparison changes evaluation
            # order and the count reads 2 instead of 3, this is why - go looking
            # for a changed expression, not a regression.
            stats["too_small"] += 1
            blobs.append(lz4.block.compress(bytes(raw), mode="high_compression",
                                            store_size=False))
            continue

        # Snap the centre to the lattice, then re-express every vertex against it.
        ncx = round(cx / POS_STEP) * POS_STEP
        ncy = round(cy / POS_STEP) * POS_STEP
        ncz = round(cz / POS_STEP) * POS_STEP
        for k in range(vcount):
            o = k * VERT_SZ + 6                       # skip u, v, colour
            x, y, z = struct.unpack_from("<3h", raw, o)
            wx, wy, wz = cx + x * scale, cy + y * scale, cz + z * scale
            nx = round((wx - ncx) / POS_STEP)
            ny = round((wy - ncy) / POS_STEP)
            nz = round((wz - ncz) / POS_STEP)
            if not (-32768 <= nx <= 32767 and -32768 <= ny <= 32767
                    and -32768 <= nz <= 32767):
                raise ValueError("model %d vertex %d left the int16 range "
                                 "(%d,%d,%d)" % (i, k, nx, ny, nz))
            struct.pack_into("<3h", raw, o, nx, ny, nz)

        m[2], m[3], m[4], m[5] = POS_STEP, ncx, ncy, ncz
        stats["converted"] += 1
        blobs.append(lz4.block.compress(bytes(raw), mode="high_compression",
                                        store_size=False))

    # Rewrite the model table in place (still 92-byte-header-relative offsets, same
    # length as the source - only scale/cx/cy/cz differ for converted models).
    out = bytearray(prefix)
    for i, m in enumerate(models):
        MODEL.pack_into(out, moff + i * 32, m[0], m[1],
                        m[2], m[3], m[4], m[5], m[6], m[7])

    # Rebuild the blob region: model blobs first (recompressed sizes may differ from
    # the source), then texture blobs copied through byte for byte at their new offsets.
    blob_base = len(out)
    for i, b in enumerate(blobs):
        COMP.pack_into(out, cmoff + i * 8, len(out) if b else 0, len(b))
        out += b

    ctoff, tc = h[22], h[7]
    for i in range(tc):
        toff, tcs = COMP.unpack_from(buf, ctoff + i * 8)
        tb = buf[toff:toff + tcs] if tcs else b""
        COMP.pack_into(out, ctoff + i * 8, len(out) if tb else 0, len(tb))
        out += tb

    struct.pack_into("<I", out, 8, len(out))          # file_size
    struct.pack_into("<I", out, 48, blob_base)         # vertex_off = start of blob region
    return bytes(out), stats


def main():
    # Parsed the same defensive way as pmap_lattice_verify.py's --ref: a typo like
    # --dryx must not be silently ignored while the directory argument is still
    # accepted, because the cost here is not a check that quietly didn't run - it is
    # 184 files overwritten when the caller asked for a report and nothing else.
    #
    # Report the WHOLE remaining args, not args[1:]: slicing off "the first token"
    # and calling the rest unrecognised assumes the first token is the legitimate
    # directory, which is exactly backwards for "--dryx <dir>" (blames <dir>).
    # And a flag-shaped token must be rejected even when it is the ONLY thing left
    # (len(args) == 1) - otherwise "--dryx" alone falls through to os.listdir on
    # a path that does not exist and dies with an unhandled FileNotFoundError
    # instead of the usage message.
    args = sys.argv[1:]
    dry = "--dry" in args
    args = [a for a in args if a != "--dry"]
    if len(args) != 1 or args[0].startswith("-"):
        if args:
            print("unrecognised argument(s): %s\n" % " ".join(args))
        print(__doc__)
        return 2
    d = args[0]
    total = {"converted": 0, "refused": 0, "too_small": 0, "empty": 0}
    failed = []             # [(name, reason)]; reason in parse/self-check/geometry/write
    t_struct = 0.0           # verify_bytes: structural + lattice checks
    t_geom = 0.0              # verify_world_positions: the actual position round-trip
    names = sorted(f for f in os.listdir(d) if f.startswith("region_") and f.endswith(".pmap"))
    for name in names:
        p = os.path.join(d, name)
        with open(p, "rb") as fh:
            buf = fh.read()

        # This pass is wired into a chain now, so it runs unattended and MORE THAN
        # ONCE (a re-bake reconverts an already-converted world). Nothing here may
        # reach the write below unverified: lattice_bytes can raise on a file it
        # cannot make sense of, and even a clean run can still produce output that
        # fails its own self-check (a bug, not a designed refusal - refused/
        # too_small models are ALREADY excluded from the failure this catches,
        # verify_bytes agrees they are legitimately left alone, see
        # pmap_lattice_verify.py's _lattice_problems). Either way this ONE tile is
        # left exactly as it was - old per-model scale, its seams unfixed --
        # instead of a bad file reaching disk and waiting for someone to notice the
        # separate verify_dir pass caught it later in the chain. That is the
        # finest-grained rollback available: one tile among 184, not one chain step
        # that already overwrote everything before anything noticed.
        try:
            out, stats = lattice_bytes(buf)
        except Exception as exc:
            # Broad, not `except ValueError`: this pass is wired into an unattended
            # chain whose entire purpose is "one tile failed, keep going", and
            # narrowing this catch to a specific type is what already broke that
            # promise twice (a bare struct.error once, an unguarded os.replace once)
            # - narrowing it further does not converge, there is always another type.
            # Confirmed live: lz4.block.decompress raises LZ4BlockError, which is NOT
            # a ValueError (LZ4BlockError -> Exception -> BaseException;
            # issubclass(LZ4BlockError, ValueError) is False), so a blob that will not
            # decompress - the single most likely real corruption - used to escape
            # this guard entirely and kill the whole directory run partway through
            # with a bare traceback, exactly the failure this per-tile loop exists to
            # prevent. type(exc).__name__ is printed alongside the message so a reader
            # can tell "genuinely malformed input" (LZ4BlockError, struct.error) from
            # "a bug in this pass itself" (anything else) without narrowing what gets
            # caught - see tools/pmap_tess_v3.py's own per-file loop for the same
            # broad-catch shape.
            print("%s: %s: %s - left on its old scale" % (name, type(exc).__name__, exc))
            failed.append((name, "parse"))
            continue

        t0 = time.perf_counter()
        ok, problems = verify_bytes(out, expect_lattice=True, ref=buf)
        t_struct += time.perf_counter() - t0
        if not ok:
            print("%s: self-check failed, left on its old scale:" % name)
            for prob in problems[:8]:
                print("    ", prob)
            if len(problems) > 8:
                print("     ... and %d more" % (len(problems) - 8))
            failed.append((name, "self-check"))
            continue

        # verify_bytes(ref=...) never opens bytes 6-12 of a vertex (x, y, z) - the
        # pass is allowed to move them, so it deliberately does not compare them.
        # That leaves a real gap: wrong-but-well-formed geometry (swapped axes, a
        # mis-indexed centre, wrong rounding) is byte-correct everywhere the check
        # above looks, and would sail through it. This is the check that actually
        # opens the geometry and asks whether each vertex ended up where it started
        # - see verify_world_positions's own docstring for the tolerance rule
        # (half a step for a converted model, exact for one the pass never touched).
        t0 = time.perf_counter()
        pos_ok, pos_problems = verify_world_positions(out, buf)
        t_geom += time.perf_counter() - t0
        if not pos_ok:
            print("%s: geometry self-check failed, left on its old scale:" % name)
            for prob in pos_problems[:8]:
                print("    ", prob)
            if len(pos_problems) > 8:
                print("     ... and %d more" % (len(pos_problems) - 8))
            failed.append((name, "geometry"))
            continue

        if not dry:
            # Atomic write: a temp file next to the target, then an OS-level
            # replace. os.replace is atomic on both POSIX and Windows, so a kill
            # mid-write - or the process losing power, or Quarry's own cancel
            # button - leaves either the complete old file or the complete new
            # one, never a truncated.pmap.
            #
            # That per-tile promise still needs an exception handler around it: a
            # read-only file, an AV scan or cloud-sync holding a handle, or any
            # other OSError here is a realistic Windows failure that has nothing to
            # do with this tile's geometry, and must not take the rest of the
            # directory down with it. Reproduced live before this guard existed: an
            # unguarded os.replace on a locked file raised PermissionError, which
            # propagated straight out of this whole for-loop - every tile sorted
            # AFTER the locked one was silently never even opened, and a complete,
            # valid alternate-scale.tmp was left sitting on disk. This tile is now
            # exactly as untouched as a self-check failure: log it, count it,
            # remove the stray temp file, move on to the next tile.
            tmp = p + ".tmp"
            try:
                with open(tmp, "wb") as fh:
                    fh.write(out)
                os.replace(tmp, p)
            except OSError as exc:
                print("%s: write failed, left on its old scale: %s" % (name, exc))
                failed.append((name, "write"))
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                continue

        for k in total:
            total[k] += stats[k]
        # refused is an ACCEPTED, by-design outcome (see the banner below), but a
        # per-tile line as it happens is what lets someone find WHICH tile without
        # re-running with more logging.
        if stats["refused"]:
            print("   ! %s: %d model(s) refused (own scale kept, seams remain there)"
                  % (name, stats["refused"]))

    print("tiles %d | converted %d refused %d too_small %d empty %d | failed %d"
          % (len(names), total["converted"], total["refused"],
             total["too_small"], total["empty"], len(failed)))
    print("self-check: %.2fs structural + %.2fs geometry = %.2fs across %d tile(s)"
          % (t_struct, t_geom, t_struct + t_geom, len(names)))
    # refused is ACCEPTABLE - the pass refuses per model, so one oversized model
    # (e.g. on a disc whose model sizes differ from the one this pass was measured
    # against - see ConvertPipeline.cs StepBakeWorld for the known v1.03/v2.01
    # gap) keeps its own scale instead of corrupting. But "acceptable" only holds if
    # it is actually SEEN: printed here as one number among four, on real 184-tile
    # world it would have read "tiles 184 | converted 14126 refused 3 too_small 3
    # empty 0" - a skim reads that as a clean run. This banner is deliberately
    # separate, loud and LAST, so it survives being read only from the tail of a
    # long chain log.
    if total["refused"]:
        print("!!! REFUSED: %d model(s) kept their own scale - seams remain at "
              "those models. See the '!' lines above for which tile(s)."
              % total["refused"])
    if failed:
        # Grouped by REASON: a parse failure (a genuinely malformed INPUT file) and
        # a self-check/geometry/write failure (this pass's own output, or this
        # run's own I/O, did not hold up) point a reader at completely different
        # places to look. Folding all of them under one "SELF-CHECK FAILED" label
        # sent a real reviewer looking at the verifier for a bug that was actually
        # 3 deliberately corrupted INPUT files - fixed by saying which is which.
        by_reason = {}
        for fname, reason in failed:
            by_reason.setdefault(reason, []).append(fname)
        print("!!! FAILED on %d tile(s), left unconverted:" % len(failed))
        for reason in ("parse", "self-check", "geometry", "write"):
            names_here = by_reason.get(reason)
            if names_here:
                print("     %s: %s" % (reason, ", ".join(names_here)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
