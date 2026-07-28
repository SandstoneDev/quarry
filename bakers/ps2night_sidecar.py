#!/usr/bin/env python3
"""Build region_*.night sidecars from a night-coloured twin export.

The trick: run the world exporter twice - once normally (day colours in the
vertex pool) and once with --night (ps2world_pilot.py packs the night set into
the same geometry). The two .pmap files are byte-identical except the 5551
colour field of every pool vertex, so the night sidecar is just the colour
column of the night twin: u16[vertex_count], exactly what pmap_load_night eats
(plain array, size == vertex_bytes/12 entries).

Usage: ps2night_sidecar.py <nightDir> <dayDir>
    For every region_*.pmap present in BOTH dirs, writes dayDir/region_*.night.
"""
import os
import struct
import subprocess
import sys
import tempfile

# PMAP v2 header (psp_scene.write_scene order):
# magic, version, file_size, nModels, model_off, nSubmesh, submesh_off,
# nTex, texture_off, nInst, instance_off, grid_off,
# vertex_off, vertex_len, index_off, index_len, texel_off, texel_len,
# clut_off, clut_len
_HDR = struct.Struct("<4s19I")
VERTEX_SIZE = 12


def night_column(path):
    b = open(path, "rb").read()
    h = _HDR.unpack_from(b, 0)
    assert h[0] == b"PMAP", "not a pmap: %s" % path
    if h[1] >= 3:
        # v3 = lz4 pools; decompress to a temp twin first (colour column needs
        # the raw vertex pool). Run the chain BEFORE lz4 to avoid this hop.
        tools = os.path.dirname(os.path.abspath(__file__))
        tmp = os.path.join(tempfile.gettempdir(),
                           "nsc_" + os.path.basename(path))
        r = subprocess.run([sys.executable,
                            os.path.join(tools, "pmap_lz4_decompress.py"),
                            path, tmp])
        if r.returncode != 0:
            raise RuntimeError("lz4 decompress failed for %s" % path)
        b = open(tmp, "rb").read()
        os.unlink(tmp)
        h = _HDR.unpack_from(b, 0)
    vertex_off, vertex_len = h[12], h[13]
    nv = vertex_len // VERTEX_SIZE
    out = bytearray(nv * 2)
    for i in range(nv):
        c = b[vertex_off + i * VERTEX_SIZE + 4: vertex_off + i * VERTEX_SIZE + 6]
        out[i * 2: i * 2 + 2] = c
    return bytes(out), nv, vertex_len


def _lum5551(c):
    """r*3+g*4+b luminance on 8-bit expansions of a GU 5551 colour."""
    r = (c & 31) << 3
    g = ((c >> 5) & 31) << 3
    b = ((c >> 10) & 31) << 3
    return r * 3 + g * 4 + b


def nightd_runs(day_col, night_col, nv):
    """NDL2 glow runs from the day/night 5551 columns - same emissive filter
    as the battle night_delta_bake (lum >= 800 AND lum > day_lum * 0.7):
    only lit windows / neon get a run, dull tints stay on the global darken."""
    runs = []
    i = 0
    while i < nv:
        nc = day_col[i * 2] | (day_col[i * 2 + 1] << 8)
        gc = night_col[i * 2] | (night_col[i * 2 + 1] << 8)
        nl = _lum5551(gc)
        if nl >= 800 and nl > _lum5551(nc) * 0.7:
            j = i + 1
            while j < nv and j - i < 0xFFFF:
                gc2 = night_col[j * 2] | (night_col[j * 2 + 1] << 8)
                nc2 = day_col[j * 2] | (day_col[j * 2 + 1] << 8)
                nl2 = _lum5551(gc2)
                if gc2 != gc or not (nl2 >= 800 and nl2 > _lum5551(nc2) * 0.7):
                    break
                j += 1
            runs.append((i, j - i, gc))
            i = j
        else:
            i += 1
    return runs


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ndir, ddir = sys.argv[1], sys.argv[2]
    made = 0
    for fn in sorted(os.listdir(ndir)):
        if not (fn.startswith("region_") and fn.endswith(".pmap")):
            continue
        day_pmap = os.path.join(ddir, fn)
        if not os.path.exists(day_pmap):
            print("skip (no day twin):", fn)
            continue
        col, nv, vlen = night_column(os.path.join(ndir, fn))
        day_col, dnv, dvlen = night_column(day_pmap)
        # sanity: the day twin must have the SAME pool size or the engine's
        # size check (nv == vertex_bytes/12) rejects the sidecar at load.
        if dvlen != vlen:
            print("MISMATCH %s: day pool %d vs night pool %d - skipped"
                  % (fn, dvlen, vlen))
            continue
        out = os.path.join(ddir, fn[:-5] + ".night")
        open(out, "wb").write(col)
        runs = nightd_runs(day_col, col, nv)
        nd = bytearray(struct.pack("<II", 0x324C444E, len(runs)))   # 'NDL2'
        for vidx, n, c in runs:
            nd += struct.pack("<IHH", vidx, n, c)
        open(os.path.join(ddir, fn[:-5] + ".nightd"), "wb").write(nd)
        made += 1
        print("wrote %s (%d verts) + .nightd (%d runs)" % (out, nv, len(runs)))
    print("night sidecars:", made)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
