#!/usr/bin/env python3
"""Verify a tessellated v3 .pmap against the invariants the engine depends on.

A tile that loads and renders garbage is worse than one that fails loudly, and the engine
trusts these tables without re-deriving them. Checked here, on the host, before anything
reaches a Memory Stick.

GEOMETRY (always)
 1. Every model blob decompresses, and its length equals exactly
 (vertex span * 12) + (index span * 2) from the submesh table. This is the same check
 the engine makes at draw time via mr->vbytes / mr->ibytes; failing it takes the
 g_oobSlice path, which drops geometry silently.
 2. Submesh vertex_first / index_first are contiguous and ascending within a model.
 3. Every index is inside its own submesh vertex count (indices are submesh-local u16).
 4. No blob exceeds LZ4_SCRATCH uncompressed.
 5. Blob offsets ascend, never overlap, never run past EOF - the streaming worker batches
 by file offset and relies on that ordering.
 6. Reports the longest surviving world edge.

COLOUR (with --ref <dir of untessellated originals>)
 7. Every vertex the original had survives BYTE FOR BYTE at the same position.
 8. Every vertex the split ADDED has, per channel, a colour inside the [min,max] of the
 colours its submesh already had. A midpoint is a blend of two existing colours and
 cannot leave their range.

Check 8 exists because check 7 is not enough. The first tessellator averaged GU_COLOR_5551
with 565 masks: it left every original vertex untouched and wrote widened green into the
midpoints, so the geometry invariants passed and hardware showed flat green roads. A
channel-mean comparison does not catch it either - the means move legitimately when the
triangles being split are not of average colour, which failed four honest tiles.

Usage: pmap_tess_verify.py <file.pmap> [...] [--max-edge 18] [--ref <dir>]
"""

import argparse
import math
import os
import struct
import sys

try:
    import lz4.block
except ImportError:
    sys.exit("needs python lz4: pip install lz4")

MAGIC = 0x50414D50
HDR = "<23I"
VERT = 12
LZ4_SCRATCH = 320 * 1024

# Matches src/platform_psp/pmap.h's own PMAP_VERSION_STRIPPED (5): a
# world-store stage 2a stripped tile (tools/world_store_build.py
# strip_tile). comp_flag (h[20]) is tested for truthiness only, never a
# specific value - a stripped tile's comp_flag (2) is truthy too, so
# that check alone does not gate this; version is what actually
# identifies the format.
PMAP_VERSION_STRIPPED = 5


def read_tables(path):
    """(buf, header, models, submeshes, model_comp, tex_comp) for one v3 file."""
    buf = open(path, "rb").read()
    h = struct.unpack_from(HDR, buf, 0)
    models = [struct.unpack_from("<IIffffff", buf, h[4] + i * 32) for i in range(h[3])]
    subs = [struct.unpack_from("<iIIII", buf, h[6] + i * 20) for i in range(h[5])]
    mcomp = [struct.unpack_from("<II", buf, h[21] + i * 8) for i in range(h[3])]
    tcomp = [struct.unpack_from("<II", buf, h[22] + i * 8) for i in range(h[7])]
    return buf, h, models, subs, mcomp, tcomp


def model_blob(buf, mcomp, mi):
    off, csize = mcomp[mi]
    if not csize or off + csize > len(buf):
        return None
    try:
        return lz4.block.decompress(buf[off:off + csize], uncompressed_size=LZ4_SCRATCH)
    except Exception:
        return None


def check_geometry(path, max_edge):
    errs = []
    buf, h, models, subs, mcomp, tcomp = read_tables(path)
    if h[0] != MAGIC:
        return ["not a pmap"], 0.0
    if h[1] == PMAP_VERSION_STRIPPED:
        return (["this is a STRIPPED tile (version=%d) - it belongs to the world "
                "store (tools/world_store_build.py), not this checker" % h[1]], 0.0)
    if not h[20]:
        return ["v2 file, this checker is for v3"], 0.0
    if h[2] != len(buf):
        errs.append("header file_size %d != actual %d" % (h[2], len(buf)))

    worst = 0.0
    for mi in range(len(models)):
        first, count, scale = models[mi][0], models[mi][1], models[mi][2]
        if not count or not mcomp[mi][1]:
            continue
        blob = model_blob(buf, mcomp, mi)
        if blob is None:
            errs.append("model %d does not decompress" % mi)
            continue
        if len(blob) > LZ4_SCRATCH:
            errs.append("model %d blob %d B over LZ4_SCRATCH" % (mi, len(blob)))

        s0 = subs[first]
        sN = subs[first + count - 1]
        vbase, ibase = s0[1], s0[3]
        vtotal = (sN[1] + sN[2]) - vbase
        itotal = (sN[3] + sN[4]) - ibase
        if vtotal * VERT + itotal * 2 != len(blob):
            errs.append("model %d: tables want %d B, blob is %d B"
                        % (mi, vtotal * VERT + itotal * 2, len(blob)))
            continue

        vcur, icur = vbase, ibase
        for k in range(count):
            si = first + k
            vf, vc, if_, ic = subs[si][1], subs[si][2], subs[si][3], subs[si][4]
            if vf != vcur or if_ != icur:
                errs.append("model %d submesh %d not contiguous" % (mi, si))
            vcur += vc
            icur += ic
            if ic % 3:
                errs.append("model %d submesh %d index count %d not a multiple of 3"
                            % (mi, si, ic))
            if not ic:
                continue
            ids = struct.unpack_from("<%dH" % ic, blob, vtotal * VERT + (if_ - ibase) * 2)
            over = [x for x in ids if x >= vc]
            if over:
                errs.append("model %d submesh %d: %d indices past vertex_count %d"
                            % (mi, si, len(over), vc))
                continue
            base = (vf - vbase) * VERT
            pos = [struct.unpack_from("<hhh", blob, base + j * VERT + 6) for j in range(vc)]
            for t in range(0, len(ids) - 2, 3):
                tri = (pos[ids[t]], pos[ids[t + 1]], pos[ids[t + 2]])
                for e in ((0, 1), (1, 2), (2, 0)):
                    d = math.dist(tri[e[0]], tri[e[1]]) * scale
                    if d > worst:
                        worst = d

    order = sorted([(o, c, "m%d" % i) for i, (o, c) in enumerate(mcomp) if c]
                   + [(o, c, "t%d" % i) for i, (o, c) in enumerate(tcomp) if c])
    prev_end, prev_name = 0, "-"
    for o, c, nm in order:
        if o < prev_end:
            errs.append("blob %s at %d overlaps %s ending %d" % (nm, o, prev_name, prev_end))
        if o + c > len(buf):
            errs.append("blob %s runs past EOF" % nm)
        prev_end, prev_name = o + c, nm
    return errs, worst


def check_colour(path, ref_path):
    """(submeshes_checked, first_problem_or_None). Checks 7 and 8 above."""
    nbuf, nh, nmod, nsub, nmc, _nt = read_tables(path)
    rbuf, rh, rmod, rsub, rmc, _rt = read_tables(ref_path)
    if len(nmod) != len(rmod) or len(nsub) != len(rsub):
        return 0, ("table sizes differ (models %d vs %d, submeshes %d vs %d)"
                   % (len(nmod), len(rmod), len(nsub), len(rsub)))

    checked = 0
    for mi in range(len(nmod)):
        first = nmod[mi][0]
        count = nmod[mi][1]
        if rmod[mi][0] != first or rmod[mi][1] != count:
            return checked, "model %d submesh range changed" % mi
        if not count:
            continue
        nblob = model_blob(nbuf, nmc, mi)
        rblob = model_blob(rbuf, rmc, mi)
        if nblob is None or rblob is None:
            continue
        nvbase = nsub[first][1]
        rvbase = rsub[first][1]

        for k in range(count):
            si = first + k
            nvf, nvc = nsub[si][1], nsub[si][2]
            rvf, rvc = rsub[si][1], rsub[si][2]
            if nvc < rvc:
                return checked, "submesh %d lost vertices (%d < %d)" % (si, nvc, rvc)
            noff = (nvf - nvbase) * VERT
            roff = (rvf - rvbase) * VERT
            if noff + nvc * VERT > len(nblob) or roff + rvc * VERT > len(rblob):
                return checked, "submesh %d runs past its blob" % si

            if nblob[noff:noff + rvc * VERT] != rblob[roff:roff + rvc * VERT]:
                for j in range(rvc):
                    a = nblob[noff + j * VERT: noff + (j + 1) * VERT]
                    b = rblob[roff + j * VERT: roff + (j + 1) * VERT]
                    if a != b:
                        return checked, ("submesh %d vertex %d ALTERED (colour %04x -> %04x)"
                                         % (si, j,
                                            struct.unpack_from("<H", b, 4)[0],
                                            struct.unpack_from("<H", a, 4)[0]))

            if nvc > rvc:
                lo = [31, 31, 31]
                hi = [0, 0, 0]
                for j in range(rvc):
                    c = struct.unpack_from("<H", rblob, roff + j * VERT + 4)[0]
                    for ch, sh in ((0, 0), (1, 5), (2, 10)):
                        v = (c >> sh) & 0x1F
                        if v < lo[ch]:
                            lo[ch] = v
                        if v > hi[ch]:
                            hi[ch] = v
                for j in range(rvc, nvc):
                    c = struct.unpack_from("<H", nblob, noff + j * VERT + 4)[0]
                    for ch, sh, chname in ((0, 0, "R"), (1, 5, "G"), (2, 10, "B")):
                        v = (c >> sh) & 0x1F
                        if v < lo[ch] or v > hi[ch]:
                            return checked, ("submesh %d midpoint %d has %s=%d outside the "
                                             "submesh range [%d,%d] - a blend cannot leave "
                                             "the range of what it blends (wrong colour "
                                             "format?)" % (si, j, chname, v, lo[ch], hi[ch]))
            checked += 1
    return checked, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-edge", type=float, default=24.0)
    ap.add_argument("--ref", default=None,
                    help="directory of untessellated originals; enables the colour checks")
    a = ap.parse_args()

    bad = 0
    for path in a.paths:
        name = os.path.basename(path)
        try:
            errs, worst = check_geometry(path, a.max_edge)
        except Exception as e:
            bad += 1
            print("%s: FAILED to read (%s)" % (name, e))
            continue
        if errs:
            bad += 1
            print("%s: %d PROBLEM(S)" % (name, len(errs)))
            for e in errs[:12]:
                print("   " + e)
            if len(errs) > 12:
                print("   ... and %d more" % (len(errs) - 12))
            continue

        note = ""
        if a.ref:
            ref = os.path.join(a.ref, name)
            if os.path.exists(ref):
                try:
                    nchk, problem = check_colour(path, ref)
                except Exception as e:
                    bad += 1
                    print("%s: colour check FAILED (%s)" % (name, e))
                    continue
                if problem:
                    bad += 1
                    print("%s: COLOUR - %s" % (name, problem))
                    continue
                note = ", %d submeshes colour-verified" % nchk
            else:
                note = ", no reference on disk"
        flag = "" if worst <= a.max_edge * 1.01 else "  <-- still over the threshold"
        print("%s: OK, longest world edge %.1fu%s%s" % (name, worst, flag, note))

    print("%d of %d files have problems" % (bad, len(a.paths)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
