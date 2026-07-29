#!/usr/bin/env python3
"""pmap_model_graft - surgically replace GUTTED model records in a prod tile.

Background (detail-gap session 2026-07-21): every IPL instance is present in
production, but an early-era bin-mesh decoder bug left some MODELS with 2-33%
of their source triangles (always the alpha splits: telephone wires, agave /
minipalm bushes, planters, storm-drain overgrowth, plaza detail, the Grove
bridge). The transplant chain froze those corpses ("models bit-for-bat").

This tool grafts CONTENT only: for each census-flagged model in a prod tile it
finds the same model in a FRESH map-export-v2 bake (matched per-instance by
world position <=0.5u), swaps in the fresh model record (vertices / indices /
submeshes; fresh textures are appended to the prod texture table), and leaves
the trusted instance list, flags and everything else untouched. Round-trip via
pmap_tex_transplant.tile_to_scene + psp_scene.write_scene.

Fresh tiles: bake with
 python tools/map_export/export_map.py <out>/ --bbox X0 Y0 X1 Y1
(use the prod region's world box +24u margin; feed <out>/raw/*.pmap here).

Usage:
 pmap_model_graft.py <prod.pmap> <fresh_raw_dir> <census.json> <out.pmap>
census.json = model_census.py output; entry key = prod pmap basename.
"""
import glob
import json
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_scene

from pmap_tex_transplant import tile_to_scene


def _tris(model):
    return sum(len(sm.index_bytes) // 6 for sm in model.submeshes)


def main():
    prod_path, fresh_dir, census_path, out_path = sys.argv[1:5]
    census = json.load(open(census_path))
    key = os.path.basename(prod_path)
    wanted = census.get(key, [])
    if not wanted:
        print(f"  {key}: no census entries - nothing to graft")
        return

    prod = tile_to_scene(prod_path)

    # fresh index: rounded world pos -> (scene, model_index)
    fresh_at = {}
    fresh_scenes = []
    for fp in sorted(glob.glob(os.path.join(fresh_dir, "*.pmap"))):
        sc = tile_to_scene(fp)
        fresh_scenes.append(sc)
        for inst in sc.instances:
            k = (round(inst.pos[0] * 2), round(inst.pos[1] * 2),
                 round(inst.pos[2] * 2))
            fresh_at.setdefault(k, (sc, inst.model))
    if not fresh_scenes:
        print(f"  {key}: no fresh tiles in {fresh_dir}"); sys.exit(1)

    # prod model -> one instance position (for matching)
    prod_pos = {}
    for inst in prod.instances:
        prod_pos.setdefault(inst.model, inst.pos)

    tex_remap = {}          # (fresh_scene_id, fresh_tex_idx) -> prod tex idx
    grafted = missing = 0
    for entry in wanted:
        mi = entry["model"]
        pos = prod_pos.get(mi)
        if pos is None:
            missing += 1
            continue
        hit = None
        for dx in (0, 1, -1):
            for dy in (0, 1, -1):
                for dz in (0, 1, -1):
                    k = (round(pos[0] * 2) + dx, round(pos[1] * 2) + dy,
                         round(pos[2] * 2) + dz)
                    if k in fresh_at:
                        hit = fresh_at[k]; break
                if hit: break
            if hit: break
        if hit is None:
            print(f"    model {mi} ({entry.get('name','?')}): no fresh match at "
                  f"({pos[0]:.1f},{pos[1]:.1f}) - SKIP")
            missing += 1
            continue
        fsc, fmi = hit
        fmodel = fsc.models[fmi]
        if _tris(fmodel) <= entry["dep_tris"]:
            print(f"    model {mi} ({entry.get('name','?')}): fresh not richer "
                  f"({_tris(fmodel)} <= {entry['dep_tris']}) - SKIP")
            missing += 1
            continue
        # import the fresh model's textures into the prod table
        new_subs = []
        for sm in fmodel.submeshes:
            ti = sm.texture
            if ti < 0:
                nt = -1
            else:
                rk = (id(fsc), ti)
                if rk not in tex_remap:
                    prod.textures.append(fsc.textures[ti])
                    tex_remap[rk] = len(prod.textures) - 1
                nt = tex_remap[rk]
            new_subs.append(psp_scene.Submesh(
                texture=nt, vertex_bytes=sm.vertex_bytes,
                index_bytes=sm.index_bytes))
        prod.models[mi] = psp_scene.Model(
            submeshes=new_subs, scale=fmodel.scale, center=fmodel.center,
            bound_radius=fmodel.bound_radius,
            draw_dist=prod.models[mi].draw_dist)   # keep the trusted prod dd
        grafted += 1
        print(f"    model {mi} ({entry.get('name','?')}): "
              f"{entry['dep_tris']} -> {_tris(fmodel)} tris, "
              f"{len(new_subs)} subs")

    blob = psp_scene.write_scene(prod.models, prod.textures, prod.instances,
                                 prod.grid)
    tmp = tempfile.mktemp(suffix='.pmap')
    open(tmp, 'wb').write(blob)
    subprocess.check_call([sys.executable, os.path.join(TOOLS, 'pmap_lz4.py'),
                           tmp, out_path], stdout=subprocess.DEVNULL)
    os.remove(tmp)
    print(f"  {key}: grafted {grafted}, skipped {missing}, "
          f"textures +{len(tex_remap)} -> {out_path}")


if __name__ == "__main__":
    main()
