#!/usr/bin/env python3
"""export_map - THE map exporter CLI (map-export-v2, SAW-based, float-space).

  python tools/map_export/export_map.py out_pilot/ --tile-at 2471 -1674
  python tools/map_export/export_map.py out_ls/    --bbox 471 -2938 3306 326
  python tools/map_export/export_map.py out_world/ --all

Grid alignment: the region grid origin is ALWAYS the shipped chunks origin
(471,-2745) so tile indices match the chunks_small set for A/B comparison.
Raw v2 tiles are copied to <out>/raw/ before lz4 (tile_preview reads raw).
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_source
import geom
import pack as packm
import emit

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_scene, sa_txd_d3d9
from formats.dff import parse_dff       # SAW (path set by sa_source import)

GRID_OX, GRID_OY = 471.0, -2745.0
TILE = 450.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tile-at", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-post", action="store_true",
                    help="skip col/lod/lz4 passes (debug)")
    args = ap.parse_args()

    t0 = time.time()
    img = sa_source.open_img()
    defs = sa_source.load_defs()
    inst = sa_source.load_instances(defs, img)
    print("defs=%d inst=%d  (%.1fs)" % (len(defs), len(inst), time.time() - t0))

    if args.tile_at:
        tx = int((args.tile_at[0] - GRID_OX) // TILE)
        ty = int((args.tile_at[1] - GRID_OY) // TILE)
        bb = (GRID_OX + tx * TILE, GRID_OY + ty * TILE,
              GRID_OX + (tx + 1) * TILE, GRID_OY + (ty + 1) * TILE)
        print("pilot tile %d,%d bbox %s" % (tx, ty, bb))
    elif args.bbox:
        bb = tuple(args.bbox)
    else:
        bb = None
    keep = list(range(len(inst)))
    if bb:
        keep = [k for k in keep
                if bb[0] <= inst[k]["pos"][0] < bb[2]
                and bb[1] <= inst[k]["pos"][1] < bb[3]]
    print("instances in scope:", len(keep))
    if not keep:
        raise SystemExit("scope kept 0 instances")

    used = {}                                  # model_id -> global model idx
    models, failures = [], []
    texpool = packm.TexPool()
    txd_cache = {}
    scene_inst = []
    full_to_scene = {}                         # full inst index -> scene index
    scene_links_full = []                      # scene index -> full lod_ref
    done = 0
    for k in keep:
        i = inst[k]
        mid = i["model_id"]
        if mid not in used:
            used[mid] = -1
            d = defs.get(mid)
            if d:
                try:
                    blob = sa_source.img_read(img, d["dff"] + ".dff")
                    if blob:
                        dff = parse_dff(blob)
                        if d["txd"] not in txd_cache:
                            tblob = sa_source.img_read(img, d["txd"] + ".txd")
                            txd_cache[d["txd"]] = (
                                {k.lower(): v for k, v in
                                 sa_txd_d3d9.decode(tblob).items()}
                                if tblob else {})
                        parts = []
                        for a in dff.atomics:
                            parts += geom.process_geometry(
                                dff.geometries[a.geometry_index])
                        m = packm.pack_processed(parts, texpool,
                                                 txd_cache[d["txd"]], d["dd"],
                                                 txd_name=d["txd"])
                        if m is not None:
                            models.append(m)
                            used[mid] = len(models) - 1
                except Exception as e:
                    failures.append((d["dff"], str(e)))
            done += 1
            if done % 500 == 0:
                print("  ... %d models decoded (%.0fs)"
                      % (done, time.time() - t0))
        gi = used[mid]
        if gi < 0:
            continue
        full_to_scene[k] = len(scene_inst)
        scene_links_full.append(i.get("lod_ref", -1))
        scene_inst.append(psp_scene.Instance(
            model=gi, pos=i["pos"], quat=i["quat"],
            interior=i["is_lod"]))             # interior field IS the is_lod flag
    # lod links: full-list indices -> scene indices (missing target -> -1)
    scene_links = [full_to_scene.get(t, -1) for t in scene_links_full]
    print("models=%d tex=%d inst=%d failures=%d missing_tex=%d links=%d  (%.1fs)"
          % (len(models), len(texpool.list), len(scene_inst), len(failures),
             len(texpool.missing), sum(1 for t in scene_links if t >= 0),
             time.time() - t0))
    for nm, err in failures[:10]:
        print("  ! %s: %s" % (nm, err))

    os.makedirs(args.out, exist_ok=True)
    emit.emit_regions(models, texpool.list, scene_inst, args.out, TILE, 400.0)
    emit.write_lod_files(args.out, scene_inst, scene_links)
    raw = os.path.join(args.out, "raw")
    os.makedirs(raw, exist_ok=True)
    import glob
    for p in glob.glob(os.path.join(args.out, "region_*.pmap")):
        shutil.copy(p, raw)
    emit.budget_report(args.out)
    if not args.no_post:
        emit.post_passes(args.out)
    print("EXPORT DONE %.1fs -> %s" % (time.time() - t0, args.out))


if __name__ == "__main__":
    main()
