#!/usr/bin/env python3
"""Step 3 of the day/night pipeline: slice the tessellated whole-map night stream
(night_tess.bin, aligned to <in_tess.pmap>) into per-region `region_<rx>_<ry>.night`
side-files, using the EXACT same tile bucketing + model order as
`sa_export_pmap.build_grid_pmaps` so each region .night is 1:1 with that tile's
`region_<rx>_<ry>.pmap` vertex pool.

The night stream rides the global model-major / submesh / vertex order; read_scene gives
the models in that order, so we slice night_tess sequentially per submesh, then for each
tile concatenate the night of `sorted(used_models)` (build_grid_pmaps's order).

Usage: python night_slice_grid.py <in_tess.pmap> <night_tess.bin> <out_dir> [tile=900]
 (out_dir should be the data dir that already holds the region_*.pmap)
"""
import os
import struct
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene   # noqa: E402


def region_pmap_vcount(path):
    with open(path, "rb") as f:
        h = f.read(56)
    return struct.unpack_from("<I", h, 52)[0] // 12


def main():
    if len(sys.argv) < 4:
        print("usage: night_slice_grid.py <in_tess.pmap> <night_tess.bin> <out_dir> [tile=900]")
        return 1
    src = sys.argv[1]; night_in = sys.argv[2]; out_dir = sys.argv[3]
    tile = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0

    scene = psp_scene.read_scene(open(src, "rb").read())
    nb = open(night_in, "rb").read()
    night = struct.unpack("<%dH" % (len(nb) // 2), nb)
    print("scene models=%d instances=%d | night verts=%d"
          % (len(scene.models), len(scene.instances), len(night)))

    # slice night per submesh (sequential == the global model-major/submesh/vertex order)
    cursor = 0
    sm_night = []
    for model in scene.models:
        per = []
        for sm in model.submeshes:
            nv = len(sm.vertex_bytes) // 12
            per.append(night[cursor:cursor + nv]); cursor += nv
        sm_night.append(per)
    assert cursor == len(night), "night length %d != pool %d" % (len(night), cursor)

    # replicate build_grid_pmaps bucketing EXACTLY
    insts = scene.instances
    ox = min(i.pos[0] for i in insts); oy = min(i.pos[1] for i in insts)
    mx = max(i.pos[0] for i in insts); my = max(i.pos[1] for i in insts)
    nx = max(1, int((mx - ox) // tile) + 1); ny = max(1, int((my - oy) // tile) + 1)

    def tile_of(x, y):
        rx = int((x - ox) // tile); ry = int((y - oy) // tile)
        rx = 0 if rx < 0 else (nx - 1 if rx >= nx else rx)
        ry = 0 if ry < 0 else (ny - 1 if ry >= ny else ry)
        return rx, ry

    buckets = {}
    for inst in insts:
        buckets.setdefault(tile_of(inst.pos[0], inst.pos[1]), []).append(inst)

    nfiles = ok = mismatch = 0
    for (rx, ry), bi in sorted(buckets.items()):
        used = sorted({inst.model for inst in bi})
        out = bytearray()
        for g in used:
            for per in sm_night[g]:
                for nc in per:
                    out += struct.pack("<H", nc)
        path = os.path.join(out_dir, "region_%d_%d.night" % (rx, ry))
        with open(path, "wb") as f:
            f.write(out)
        nfiles += 1
        pm = os.path.join(out_dir, "region_%d_%d.pmap" % (rx, ry))
        if os.path.exists(pm):
            if region_pmap_vcount(pm) == len(out) // 2:
                ok += 1
            else:
                mismatch += 1
                print("  MISMATCH region_%d_%d: night %d vs pmap %d verts"
                      % (rx, ry, len(out) // 2, region_pmap_vcount(pm)))
    print("wrote %d region .night  (aligned-to-pmap: %d ok, %d mismatch)"
          % (nfiles, ok, mismatch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
