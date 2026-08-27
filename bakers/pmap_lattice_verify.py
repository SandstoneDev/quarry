#!/usr/bin/env python3
"""Whole-file check for a .pmap before and after the 1/128 lattice pass.

The lattice pass touches x, y, z and nothing else, so the strongest check available is
byte equality of everything else against the source. That is what catches the class of
bug that shipped twice: b949 rewrote vertex colour with the wrong mask, b952 rewrote UV
as if the GE read it signed. Neither was visible to a geometric invariant.

Also confirms, independent of what the pass claims: vertex/index COUNTS are unchanged
(the .night / .nightd sidecars ride on the vertex pool and desynchronise silently if a
count drifts - the exact way the b952 tessellation pass broke), the submesh table is
untouched byte for byte (it is the map from vertex-blob ranges to draw calls and
textures, and it sits directly against what the pass moves - span-based checks alone
never look at an INNER submesh's own fields), the instance table and grid (opaque
byte ranges in the resident prefix - the pass has no legitimate reason to touch either
one) and every texture blob (addressed via comp_tex, same as a model's own blob via
comp_model) are all byte-identical to the source, every model's compressed blob
inflates to exactly what its own tables promise, and - when expect_lattice is set - every model that could already sit on the lattice does: scale == 1/128, centre an
exact multiple of it, extent inside 65534 steps.

verify_bytes() never raises: anything it cannot make sense of (bad magic, wrong
version, a table that runs past the end of the file) comes back as
(False, ["one problem describing why"]) rather than an exception, so one corrupt
file cannot take the rest of a directory scan down with it.

verify_bytes() deliberately never looks at a vertex's own x/y/z bytes when ref is
given - the pass is allowed to move them, that is its entire job. verify_world_
positions(), alongside it, is the check that does: it recomputes each vertex's WORLD
position (centre + int16*scale) from both buf and ref and confirms it moved no more
than the pass is allowed to move it (half a lattice step for a converted model,
nothing at all for one the pass never touched). Wrong-but-well-formed geometry - swapped axes, a mis-indexed centre, wrong rounding - is invisible to verify_bytes
alone; this is the check that actually opens the geometry.

Usage:
 python tools/pmap_lattice_verify.py <dir> # structural + lattice checks
 python tools/pmap_lattice_verify.py <dir> --ref <origdir> # + byte-compare vs source
"""
import os
import struct
import sys

import lz4.block

POS_STEP = 1.0 / 128.0
MAX_STEPS = 65534        # ELIGIBILITY gate only. Inherited from the ORIGINAL per-model baker's
                          # own `scale = extent / 65534` convention (a symmetric +-32767
                          # assumption). Must stay in exact agreement with pmap_lattice.py's own
                          # MAX_STEPS - see test_gate_and_verifier_agree_at_the_real_worlds_ulp_edge.
                          # Do NOT reuse this for a REALIZED-extent check - see MAX_SPAN below.
MAX_SPAN = MAX_STEPS + 1  # 65535: the actual count of int16 values between -32768 and 32767.
                          # int16 is NOT symmetric (one more negative value than positive), so the
                          # true addressable span is ONE MORE than MAX_STEPS above - a DIFFERENT
                          # quantity, deliberately its own constant rather than MAX_STEPS reused.
                          # pmap_lattice.py's own writer allows the full -32768..32767 range (see
                          # its int16-range check in lattice_bytes), so it can legitimately produce
                          # a span of exactly 65535; using MAX_STEPS here rejected that as if the
                          # format could not hold it - see
                          # test_lattice_check_accepts_the_full_asymmetric_int16_span.
MIN_STEPS_ACROSS = 64    # below this the lattice pass leaves a model on its own scale too
# 23 u32: magic,version,file_size, model_count,model_off, submesh_count,submesh_off,
# texture_count,texture_off, instance_count,instance_off, grid_off, vertex_off,vertex_bytes,
# index_off,index_bytes, texel_off,texel_bytes, clut_off,clut_bytes, comp_flag,
# comp_model_off,comp_tex_off - see pmap_lz4.py HDR_V3 (92 bytes). Cross-checked against a
# real region_0_0.pmap: model_off there is exactly 92, not 96.
HDR = struct.Struct("<23I")
MODEL = struct.Struct("<2I6f")       # first_submesh,submesh_count,scale,cx,cy,cz,bound_r,draw_dist
SUBMESH = struct.Struct("<i4I")      # texture,vfirst,vcount,ifirst,icount
COMP = struct.Struct("<2I")          # off,csize
VERT_SZ = 12                         # s16 u,v; u16 colour; s16 x,y,z

# Matches src/platform_psp/pmap.h's own PMAP_VERSION_STRIPPED - a world-store stage 2a
# stripped tile (tools/world_store_build.py strip_tile): same v3 header shape and
# every table through comp_tex, but comp_model[i]/comp_tex[i] hold GLOBAL ids into a
# companion world.idx/world.dat, not local byte offsets, and the blob region is gone.
# This module has no way to resolve a global id (it has no world.idx open), so it must
# recognise version 5 BY NAME and refuse cleanly - see _parse's own check, added
# after a review found that comp_flag alone (tested for truthiness everywhere, never a
# specific value) was never a real gate against this file reaching a v3-or-later reader
# that cannot actually handle it. The two constants must never drift apart; see
# test_stripped_version_constant_matches_pmap_h in test_world_store.py.
PMAP_VERSION_STRIPPED = 5


def _parse(buf):
    """Header + model/submesh/comp tables. Raises ValueError for anything that is not
 a well-formed v3-or-later LZ4 .pmap - callers turn that into (False, [...]) rather
 than letting a corrupt or truncated file crash the run."""
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
            "byte offsets, and it has no blob region of its own for this tool to verify"
            % h[1])
    if h[1] < 3 or not h[20]:
        raise ValueError("not a v3-or-later LZ4 .pmap (version=%d comp_flag=%d) - "
                         "this pass only handles compressed pools" % (h[1], h[20]))
    mc, moff, sc, soff, cmoff = h[3], h[4], h[5], h[6], h[21]
    try:
        models = [MODEL.unpack_from(buf, moff + i * 32) for i in range(mc)]
        subs = [SUBMESH.unpack_from(buf, soff + i * 20) for i in range(sc)]
        comp = [COMP.unpack_from(buf, cmoff + i * 8) for i in range(mc)]
    except struct.error as exc:
        raise ValueError("a table runs past the end of the file (%s)" % exc)
    return h, models, subs, comp


def _model_spans(models, subs, i):
    """(vfirst, vcount, icount) for model i, exactly as pmap_lz4.py computes them: a
 model's span runs from its first submesh's start to its last submesh's end, so
 submeshes must be contiguous in vertex/index space within a model."""
    first, count = models[i][0], models[i][1]
    if count == 0:
        return 0, 0, 0
    s0, sN = subs[first], subs[first + count - 1]
    vfirst = s0[1]
    vcount = sN[1] + sN[2] - vfirst
    icount = sN[3] + sN[4] - s0[3]
    return vfirst, vcount, icount


def _owning_model(models, si):
    """Which model (if any) claims submesh index si - just for a clearer message,
 the submesh index alone is always enough to locate the problem."""
    for mi, m in enumerate(models):
        first, count = m[0], m[1]
        if first <= si < first + count:
            return mi
    return -1


def _decompress(buf, comp, need):
    off, csize = comp
    if not csize:
        return b""
    return lz4.block.decompress(buf[off:off + csize], uncompressed_size=need)


def _compare_range(buf, off, end, ref, roff, rend, label, problems):
    """Byte-range compare for a region the pass must never touch (instance table,
 grid, one texture blob). Reports a length mismatch distinctly from a content
 mismatch rather than treating differently-sized regions as trivially unequal - a length problem and a corruption problem read very differently to whoever has
 to act on this."""
    a = buf[off:end]
    b = ref[roff:rend]
    if len(a) != len(b):
        problems.append("%s: size differs from the source (%d -> %d bytes)"
                        % (label, len(b), len(a)))
        return
    if a != b:
        problems.append("%s: differs from the source - the pass must not touch it" % label)


def _lattice_problems(i, m, vcount, raw):
    """Checks for one model when expect_lattice=True. `raw` is its already-decompressed
 vertex+index blob, already confirmed to be the length the tables promise."""
    out = []
    scale, cx, cy, cz = m[2], m[3], m[4], m[5]
    if abs(scale - POS_STEP) > 1e-12:
        # Not on the lattice yet. Only complain if it COULD already be there: this is
        # exactly the pair of tests the (separately written) lattice pass itself uses
        # to decide "refused" (too coarse - current scale times the full int16 range
        # is the most it could possibly span, and that worst case already overflows
        # the new grid) and "too_small" (too fine to be worth it - that same worst
        # case needs fewer than MIN_STEPS_ACROSS steps on the new grid). Both leave a
        # model on its own scale, geometry untouched. Sharing both tests means this
        # verifier never disagrees with either decision the pass made correctly, and
        # never nags about a model that could never or need never convert.
        # Computed in the identical order to pmap_lattice.py's own gate on purpose --
        # a real model (region_4_7 #2) sits one ULP below MIN_STEPS_ACROSS here, so
        # this and the pass's expression must never drift apart or the two would
        # start disagreeing about it. See pmap_lattice.py's too_small branch for why.
        steps_needed = scale * MAX_STEPS / POS_STEP
        eligible = MIN_STEPS_ACROSS <= steps_needed <= MAX_STEPS
        if eligible and vcount >= 3:
            out.append("model %d: scale %.9g is not the lattice step" % (i, scale))
        return out
    for axis, c in zip("xyz", (cx, cy, cz)):
        k = c / POS_STEP
        if abs(k - round(k)) > 1e-4:
            out.append("model %d: centre %s=%.9g is off the lattice" % (i, axis, c))
    # Sanity-check the ACTUAL stored data, not just the scale/centre fields' say-so:
    # every vertex is already an int16, so this only fires if the real span exceeds
    # what the lattice can address.
    lo = [32767, 32767, 32767]
    hi = [-32768, -32768, -32768]
    for k in range(vcount):
        xyz = struct.unpack_from("<3h", raw, k * VERT_SZ + 6)
        for a in range(3):
            v = xyz[a]
            if v < lo[a]:
                lo[a] = v
            if v > hi[a]:
                hi[a] = v
    for axis, l, hgh in zip("xyz", lo, hi):
        if hgh - l > MAX_SPAN:
            out.append("model %d: %s extent needs %d steps, only %d exist"
                       % (i, axis, hgh - l, MAX_SPAN))
    return out


def verify_bytes(buf, expect_lattice, ref=None):
    """Returns (ok, [problems]). expect_lattice=True demands every model that could
 sit on the lattice does. ref, when given, is the pre-pass bytes: every model's
 vertex/index COUNTS must match it, the index bytes must be identical, and each
 vertex's first 6 bytes (u, v, colour) must be identical - the pass may only move
 x, y, z. Reports the first offending vertex per model, not all of them."""
    try:
        h, models, subs, comp = _parse(buf)
    except ValueError as exc:
        return False, [str(exc)]

    problems = []
    rmodels = rsubs = rcomp = None
    if ref is not None:
        try:
            rh, rmodels, rsubs, rcomp = _parse(ref)
        except ValueError as exc:
            return False, ["reference: %s" % exc]
        if rh[3] != h[3] or rh[5] != h[5]:
            problems.append("model/submesh counts differ from the source (%d/%d -> %d/%d)"
                            % (rh[3], rh[5], h[3], h[5]))
        else:
            # The whole submesh table, byte for byte - the pass has no legitimate
            # reason to touch a single byte of it, unlike file_size or the comp
            # tables. This matters because _model_spans above only ever reads a
            # model's FIRST and LAST submesh to derive its overall vertex/index
            # span: a model with 3+ submeshes has INNER submeshes - their texture
            # id, vfirst, vcount, ifirst, icount - that no span-based logic ever
            # looks at, in ref mode or otherwise. A raw table compare is the only
            # thing that sees them.
            soff, rsoff, sc = h[6], rh[6], h[5]
            a = buf[soff:soff + sc * SUBMESH.size]
            b = ref[rsoff:rsoff + sc * SUBMESH.size]
            if a != b:
                for si in range(sc):
                    o = si * SUBMESH.size
                    if a[o:o + SUBMESH.size] != b[o:o + SUBMESH.size]:
                        mi = _owning_model(models, si)
                        where = ("model %d " % mi) if mi >= 0 else ""
                        problems.append(
                            "%ssubmesh %d: table entry differs from the source - "
                            "the pass must not touch the submesh table" % (where, si))

        # Three regions the pass has no legitimate reason to touch at all, and that
        # nothing above ever looks at: the instance table and grid live in the
        # resident prefix (byte ranges, not parsed into records - their internal
        # layout is not needed to prove they are untouched), and texture blobs are
        # addressed by comp_tex exactly like model blobs are by comp_model. All three
        # were byte-identical on every real tile the pass has been run against; this
        # is what actually proves that, instead of asserting it from the pass's own
        # code never assigning to them.
        _compare_range(buf, h[10], h[11], ref, rh[10], rh[11], "instance table", problems)
        # The grid's own span ends at comp_model_off, NOT at vertex_off: v3 places the
        # comp_model/comp_tex tables AFTER the grid and BEFORE the blob region, so
        # vertex_off is the end of the whole resident prefix, not the end of the grid
        # specifically (cross-checked against pmap_lz4.py's compress: comp_model_off
        # = HDR_V3 + len(body), where body already includes the grid). Using vertex_off
        # here would fold the comp tables into the "grid" compare and false-positive on
        # every converted file, since comp_model entries legitimately change every run.
        _compare_range(buf, h[11], h[21], ref, rh[11], rh[21], "grid", problems)

        if rh[7] != h[7]:
            problems.append("texture count differs from the source (%d -> %d)"
                            % (rh[7], h[7]))
        else:
            ctoff, rctoff, tc = h[22], rh[22], h[7]
            for ti in range(tc):
                try:
                    toff, tcs = COMP.unpack_from(buf, ctoff + ti * COMP.size)
                    rtoff, rtcs = COMP.unpack_from(ref, rctoff + ti * COMP.size)
                except struct.error as exc:
                    problems.append("texture %d: comp table runs past the end of "
                                    "the file (%s)" % (ti, exc))
                    continue
                _compare_range(buf, toff, toff + tcs, ref, rtoff, rtoff + rtcs,
                               "texture %d blob" % ti, problems)

    for i, m in enumerate(models):
        try:
            _, vcount, icount = _model_spans(models, subs, i)
        except (IndexError, struct.error) as exc:
            problems.append("model %d: submesh range runs past the submesh table (%s)"
                            % (i, exc))
            continue

        if ref is not None and rmodels is not None and i < len(rmodels):
            try:
                _, rvcount, ricount = _model_spans(rmodels, rsubs, i)
            except (IndexError, struct.error) as exc:
                problems.append("model %d: reference submesh range runs past the "
                                "submesh table (%s)" % (i, exc))
                continue
            # Checked BEFORE the vcount==0 gate below: a model corrupted down to
            # nothing must still be compared against a reference that had geometry,
            # not silently skipped just because it is now empty.
            if rvcount != vcount or ricount != icount:
                problems.append("model %d: vertex/index count changed (%d/%d -> %d/%d)"
                                % (i, rvcount, ricount, vcount, icount))
                continue

        if vcount == 0:
            continue

        vbytes, ibytes = vcount * VERT_SZ, icount * 2
        need = vbytes + ibytes
        try:
            raw = _decompress(buf, comp[i], need)
        except Exception as exc:
            problems.append("model %d: blob does not decompress (%s)" % (i, exc))
            continue
        if len(raw) != need:
            problems.append("model %d: blob is %d bytes, tables promise %d"
                            % (i, len(raw), need))
            continue

        if expect_lattice:
            problems.extend(_lattice_problems(i, m, vcount, raw))

        if ref is not None and rmodels is not None and i < len(rmodels):
            try:
                rraw = _decompress(ref, rcomp[i], need)
            except Exception as exc:
                problems.append("model %d: reference blob does not decompress (%s)"
                                % (i, exc))
                continue
            if len(rraw) != need:
                problems.append("model %d: reference blob is %d bytes, tables promise %d"
                                % (i, len(rraw), need))
                continue
            if raw[vbytes:] != rraw[vbytes:]:
                problems.append("model %d: INDICES changed - the pass must not touch them" % i)
            for k in range(vcount):
                o = k * VERT_SZ
                if raw[o:o + 6] != rraw[o:o + 6]:
                    problems.append("model %d vertex %d: UV or COLOUR changed - "
                                    "the pass must only touch x,y,z" % (i, k))
                    break

    return (not problems), problems


def verify_world_positions(buf, ref):
    """Returns (ok, [problems]). Recomputes every vertex's WORLD position - centre +
 int16*scale - independently from `buf` and from `ref` (the pre-pass bytes), and
 confirms it did not move more than the pass is allowed to move it.

 This is the check verify_bytes() deliberately does NOT do: verify_bytes(ref=...)
 only ever compares bytes 0-6 of each vertex (UV, colour) against the reference,
 never bytes 6-12 (x, y, z) - by design, since a CONVERTED model's x/y/z are
 exactly what the pass is allowed to rewrite. That leaves a real gap: a pass that
 writes wrong-but-well-formed geometry - swapped axes, a mis-indexed centre, wrong
 rounding - produces a file that is byte-correct everywhere verify_bytes looks
 (right counts, right UV, right colour, right indices, right submesh/instance/grid/
 texture bytes) and still passes every check it runs. This is the one check that
 opens the compressed blob and asks whether each vertex ended up where it started,
 not just whether the file's bookkeeping is self-consistent.

 A model already at scale==POS_STEP in `buf` is one lattice_bytes CONVERTED (or
 re-processed, idempotently - see test_lattice_bytes_is_idempotent): round-to-
 nearest quantisation onto the 1/128 grid can move it by up to half a step, so
 that is the tolerance. Every other model (refused/too_small/empty) never has its
 vertices touched at all - decompressed and immediately recompressed byte for
 byte, no arithmetic on x/y/z - so ANY movement there, however small, is a real
 bug, and the tolerance is exact.

 Never raises, matching every other function in this module: an unparseable buf
 or ref comes back as a problem string, not an exception."""
    try:
        h, models, subs, comp = _parse(buf)
        rh, rmodels, rsubs, rcomp = _parse(ref)
    except ValueError as exc:
        return False, [str(exc)]

    problems = []
    mc = min(h[3], rh[3])          # a model-count mismatch is verify_bytes' job to report
    for i in range(mc):
        try:
            _, vcount, icount = _model_spans(models, subs, i)
            _, rvcount, ricount = _model_spans(rmodels, rsubs, i)
        except (IndexError, struct.error) as exc:
            problems.append("model %d: submesh range runs past the submesh table (%s)"
                            % (i, exc))
            continue
        if vcount == 0 or rvcount != vcount or ricount != icount:
            continue          # empty, or a count mismatch - verify_bytes' job to report

        need = vcount * VERT_SZ + icount * 2
        try:
            raw = _decompress(buf, comp[i], need)
            rraw = _decompress(ref, rcomp[i], need)
        except Exception as exc:
            problems.append("model %d: blob does not decompress (%s)" % (i, exc))
            continue
        if len(raw) != need or len(rraw) != need:
            continue          # a length mismatch - verify_bytes' job to report

        scale, cx, cy, cz = models[i][2], models[i][3], models[i][4], models[i][5]
        rscale, rcx, rcy, rcz = rmodels[i][2], rmodels[i][3], rmodels[i][4], rmodels[i][5]
        converted = abs(scale - POS_STEP) < 1e-12
        limit = (POS_STEP / 2 + 1e-9) if converted else 0.0
        for k in range(vcount):
            x, y, z = struct.unpack_from("<3h", raw, k * VERT_SZ + 6)
            rx, ry, rz = struct.unpack_from("<3h", rraw, k * VERT_SZ + 6)
            wx, wy, wz = cx + x * scale, cy + y * scale, cz + z * scale
            rwx, rwy, rwz = rcx + rx * rscale, rcy + ry * rscale, rcz + rz * rscale
            err = max(abs(wx - rwx), abs(wy - rwy), abs(wz - rwz))
            if err > limit:
                problems.append(
                    "model %d vertex %d: world position moved %.6g (%s)"
                    % (i, k, err, ("half-step limit %.6g" % limit) if converted
                                  else "must be exact - this model was never touched"))
                break          # one report per model is enough to point at the problem
    return (not problems), problems


def verify_dir(path, refdir=None):
    """Walk region_*.pmap in `path`, printing per-file problems and a summary.
 Returns a shell exit code: 0 if every tile was clean, 1 otherwise."""
    bad = 0
    names = sorted(f for f in os.listdir(path) if f.startswith("region_") and f.endswith(".pmap"))
    for name in names:
        with open(os.path.join(path, name), "rb") as fh:
            buf = fh.read()
        ref = None
        if refdir:
            rp = os.path.join(refdir, name)
            if os.path.exists(rp):
                with open(rp, "rb") as fh:
                    ref = fh.read()
        ok, problems = verify_bytes(buf, expect_lattice=True, ref=ref)
        if ref is not None:
            # verify_bytes(ref=...) never opens bytes 6-12 of a vertex (x, y, z) --
            # that is the pass's whole job, so it deliberately does not compare
            # them. This is the companion check that actually opens the geometry;
            # see verify_world_positions's own docstring for what it catches that
            # the check above cannot.
            pos_ok, pos_problems = verify_world_positions(buf, ref)
            if not pos_ok:
                ok = False
                problems = problems + pos_problems
        if not ok:
            bad += 1
            print("%s:" % name)
            for p in problems[:8]:
                print("   ", p)
            if len(problems) > 8:
                print("    ... and %d more" % (len(problems) - 8))
    print("checked %d tiles, %d with problems" % (len(names), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    # Parsed order-independently: "<dir> --ref <refdir>" and "--ref <refdir> <dir>"
    # must both work, and a missing/incomplete --ref must print usage, not raise
    # a bare FileNotFoundError out of os.listdir on whatever token got mistaken for
    # the directory.
    args = sys.argv[1:]
    refdir = None
    if "--ref" in args:
        idx = args.index("--ref")
        if idx + 1 >= len(args):
            print(__doc__)
            raise SystemExit(2)
        refdir = args[idx + 1]
        del args[idx:idx + 2]
    if len(args) != 1 or args[0].startswith("-"):
        # Refuse anything left over rather than silently using args[0] and dropping the
        # rest. A typo like `--refx <origdir>` would otherwise run WITHOUT a reference and
        # print a clean result, which reads as "the byte-identity check passed" when it
        # never ran - an instrument that silently does not run is indistinguishable from
        # one that passed, and that is the failure mode this whole tool exists to prevent.
        #
        # Report the WHOLE remaining args, not args[1:] (that slice assumes args[0] is the
        # legitimate directory, backwards for "--refx <origdir>" - it named <origdir> as
        # the problem). And reject a flag-shaped SOLE token instead of accepting it as the
        # directory: "--refx" alone has len(args)==1 and used to fall through to
        # verify_dir("--refx", None), dying with an unhandled FileNotFoundError.
        if args:
            print("unrecognised argument(s): %s\n" % " ".join(args))
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(verify_dir(args[0], refdir))
