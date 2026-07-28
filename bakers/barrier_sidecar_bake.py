#!/usr/bin/env python3
"""barrier_sidecar_bake - per-region .barr sidecars flagging SA BARRIER models.

Clone of road_sidecar_bake. SA city/bridge/construction barriers are barriers.ide
model ids 966..998 (dedicated models -> every instance of such a model is a barrier).
The .pmap carries only a tile-local model index (no SA id), so this tool re-reads
each region_X_Y.pmap, matches instances BY POSITION against the SA IPL placements of
barrier models, collects the tile-local MODEL indices those instances use, and writes

  region_X_Y.barr:  'BARR' u16 modelCount u16 pad, then u16 localModelIdx[]

The runtime ORs PMAP_MODELFLAG_BARRIER into that model's per-model flag byte
(Streaming.build_model_flags), and the renderer skips those models' draws while the
"Barriers" toggle is off (collision is skipped in parallel via the col offset-68 tag).

Usage: python tools/barrier_sidecar_bake.py [chunks_dir]
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_export"))
import sa_source

CHUNKS = ""

BARR_LO, BARR_HI = 966, 998   # barriers.ide model id range


def barrier_positions():
    """{(x,y,z rounded 0.1)} for every IPL placement of a barrier model (id 966..998)."""
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    inst = sa_source.load_instances(defs, img)
    # b464: city-lock/bridge barriers = IDE txd 'barrierblk' (4510-4527: ce_fredbar, sfw/cn2/sfse
    # roadblocks - the between-cities locks) PLUS the barriers.ide construction range 966..998.
    barr_ids = {mid for mid, d in defs.items() if (d.get("txd") or "").lower() == "barrierblk"} \
             | {mid for mid in defs if BARR_LO <= mid <= BARR_HI}
    pos = set()
    for i in inst:
        if i["model_id"] in barr_ids:
            x, y, z = i["pos"]
            pos.add((round(x, 1), round(y, 1), round(z, 1)))
    return pos, len(barr_ids), len(inst)


def region_barrier_models(path, barr_pos):
    """The set of tile-local model indices used by barrier-positioned instances."""
    blob = open(path, "rb").read()
    inst_count = struct.unpack_from("<I", blob, 36)[0]
    inst_off = struct.unpack_from("<I", blob, 40)[0]
    models = set()
    for i in range(inst_count):
        o = inst_off + i * 36                        # sizeof(PmapInstance) = 36
        mi = struct.unpack_from("<I", blob, o)[0]    # local model index at record+0
        x, y, z = struct.unpack_from("<3f", blob, o + 4)  # pos at record+4
        if (round(x, 1), round(y, 1), round(z, 1)) in barr_pos:
            models.add(mi)
    return models


def main():
    chunks = sys.argv[1] if len(sys.argv) > 1 else CHUNKS
    barr_pos, n_barr, n_inst = barrier_positions()
    print(f"barrier models: {n_barr} | SA instances scanned: {n_inst} | "
          f"barrier placement keys: {len(barr_pos)}")

    total = files = 0
    for pmap in sorted(glob.glob(os.path.join(chunks, "region_*_*.pmap"))):
        base = os.path.splitext(pmap)[0]
        out = base + ".barr"
        models = sorted(region_barrier_models(pmap, barr_pos))
        if not models:
            if os.path.exists(out):
                os.remove(out)
            continue
        buf = b"BARR" + struct.pack("<HH", len(models), 0)
        for mi in models:
            buf += struct.pack("<H", mi)
        open(out, "wb").write(buf)
        files += 1
        total += len(models)
        print(f"  {os.path.basename(base)}.barr: {len(models)} barrier models")
    print(f"wrote {files} region .barr sidecars, {total} barrier models total")


if __name__ == "__main__":
    main()
