#!/usr/bin/env python3
"""Bake the FULL-map chunkset with the tx128 ROAD TIER (b397).

Slices input_tx128.pmap (full map, 24u-tessellated, texels still at the 128 cap)
into 900u region tiles, then runs the production chunk pipeline with ONE change:
the texture downscale keeps wet_road-model textures at 128px and caps everything
else at the production 64px. 

Usage: python bake_tx128road_chunks.py [out_dir]
"""
import os, sys, subprocess

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene
from gvcslib.work.sa_export_pmap import build_grid_pmaps
sys.path.insert(0, "")
from pmap_uv_split import split_scene_models   # striped-road fix (UV span > s16 range)

SRC   = ""
OUT   = ""
TOOLS = ""
TILE  = 450.0   # MATCH PRODUCTION: chunks_v2_world_uvfix is a 13x9 grid of 450u tiles
                # (the first run sliced 900u -> 49 tiles = a different streaming profile,
                # useless as an A/B against prod)
CELL  = 400.0


def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("step failed: %s" % cmd)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    os.makedirs(out, exist_ok=True)

    print("reading", SRC)
    scene = psp_scene.read_scene(open(SRC, "rb").read())
    print("instances:", len(scene.instances))

    split_scene_models(scene.models)   # UV spans > 15.5 tiles -> split (s16 packing)
    build_grid_pmaps(scene.models, scene.textures, scene.instances, out, TILE, CELL)
    print("slice done ->", out)

    py = sys.executable
    run([py, os.path.join(TOOLS, "pmap_tex_downscale.py"), out, "64", "--road-tier", "128"])
    run([py, os.path.join(TOOLS, "pmap_uv_recenter.py"), out])   # BEFORE lz4 (raw v2)
    # graft the +351 production-only instances (bridge LOD sections, prop clusters --
    # present in prod tiles, absent from every surviving whole-map source; the tool
    # that once added them is lost). MUST run before the sidecar bakes (dyn needs
    # the grafted prop positions) and before lz4 (tiles must be raw v2).
    run([py, os.path.join(TOOLS, "pmap_graft.py"),
         "", out])
    run([py, os.path.join(TOOLS, "col_bake.py"), "regions", out])
    run([py, os.path.join(TOOLS, "lod_bake_regions.py"), out])
    run([py, os.path.join(TOOLS, "road_sidecar_bake.py"), out])  #.road flags (Z-bias sub-pass)
    run([py, os.path.join(TOOLS, "dyn_sidecar_bake.py"), out])   # knockable props
    run([py, os.path.join(TOOLS, "pmap_lz4.py"), "--dir", out])
    print("BAKE COMPLETE:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
