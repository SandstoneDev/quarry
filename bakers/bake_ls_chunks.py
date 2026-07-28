#!/usr/bin/env python3
"""Bake ONE Los-Santos-only chunk set at a given tile size.

Filters input_tess.pmap instances to the LS bbox, slices into region tiles, then
runs the rest of the chunk pipeline (tex-downscale, COL, LOD, lz4). Night is
skipped (g_loadNight=0 since build 170 -> .night is not loaded at runtime).

Usage: python bake_ls_chunks.py <tile_units> <out_dir>
"""
import os, sys, subprocess

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene
from gvcslib.work.sa_export_pmap import build_grid_pmaps
sys.path.insert(0, "")
from pmap_uv_split import split_scene_models   # striped-road fix (UV span > s16 range)

SRC   = ""
TOOLS = ""
CELL  = 400.0
# LS bbox = spawn (2471,-1674) +/- 2000, clamped to map extent.
BX0, BX1 = 471.0, 3306.0
BY0, BY1 = -2938.0, 326.0


def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("step failed: %s" % cmd)


def main():
    if len(sys.argv) < 3:
        print("usage: bake_ls_chunks.py <tile_units> <out_dir> [src.pmap]"); return 1
    tile = float(sys.argv[1]); out = sys.argv[2]
    global SRC
    if len(sys.argv) > 3:
        SRC = sys.argv[3]          # e.g. the 32u guard-band A/B source (b396)
    os.makedirs(out, exist_ok=True)

    print("reading", SRC)
    scene = psp_scene.read_scene(open(SRC, "rb").read())
    keep = [i for i in scene.instances if BX0 <= i.pos[0] <= BX1 and BY0 <= i.pos[1] <= BY1]
    print("instances: %d total -> %d in LS bbox" % (len(scene.instances), len(keep)))
    if not keep:
        raise SystemExit("LS bbox kept 0 instances - check bbox vs scene coords")

    split_scene_models(scene.models)   # striped-road fix: split UV spans > 15.5 tiles
    build_grid_pmaps(scene.models, scene.textures, keep, out, tile, CELL)
    print("slice done @ tile", tile, "->", out)

    py = sys.executable
    run([py, os.path.join(TOOLS, "pmap_tex_downscale.py"), out, "64"])
    run([py, os.path.join(TOOLS, "pmap_uv_recenter.py"), out])   # fix clamped tiled UVs (BEFORE lz4, raw v2)
    run([py, os.path.join(TOOLS, "col_bake.py"), "regions", out])
    run([py, os.path.join(TOOLS, "lod_bake_regions.py"), out])
    run([py, os.path.join(TOOLS, "pmap_lz4.py"), "--dir", out])
    print("BAKE COMPLETE:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
