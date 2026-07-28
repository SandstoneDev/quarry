#!/usr/bin/env python3
"""road_sidecar_bake - per-region .road sidecars flagging wet_road MODELS.

Road tiles in SA are laid directly on the terrain surface (Delta-z ~0.03 at Grove
Street). That coplanar overlap Z-fights on the PSP's 16-bit reversed-Z depth
buffer at a grazing/standing angle - two opaque detail tiles the depth test
cannot separate, so pixels flip between road and terrain. Moving the geometry up
(a +Z lift) sinks the player's feet into the raised road and cannot separate the
interpenetrating sloped surfaces anyway, so instead the runtime draws road in a
short sub-pass with a reversed-Z bias toward the near plane - road wins the depth
test everywhere, geometry unmoved (collision + feet stay put).

wet_road is a per-MODEL IDE flag (bit 0), so every instance of a road model is
road. The .pmap carries only a tile-local model index (no SA id), so this tool
re-reads each region_X_Y.pmap, matches instances BY POSITION against the SA IPL
placements of wet_road models, collects the tile-local MODEL indices those
instances use, and writes

  region_X_Y.road:  'ROAD' u16 modelCount u16 pad, then u16 localModelIdx[]

The runtime ORs a road bit into that model's per-model flag byte
(Streaming.build_model_flags) and the renderer biases those models' opaque draws.

Usage: python tools/road_sidecar_bake.py [chunks_dir]
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_export"))
import sa_source

CHUNKS = ""


def wetroad_positions():
    """{(x,y,z rounded 0.1)} for every IPL placement of a wet_road model (IDE
    flag bit 0). sa_source handles both text + binary IPLs."""
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    inst = sa_source.load_instances(defs, img)
    wet = {mid for mid, d in defs.items() if (d.get("flags", 0) & 1)}
    pos = set()
    for i in inst:
        if i["model_id"] in wet:
            x, y, z = i["pos"]
            pos.add((round(x, 1), round(y, 1), round(z, 1)))
    return pos, len(wet), len(inst)


def region_road_models(path, road_pos):
    """The set of tile-local model indices used by road-positioned instances."""
    blob = open(path, "rb").read()
    inst_count = struct.unpack_from("<I", blob, 36)[0]
    inst_off = struct.unpack_from("<I", blob, 40)[0]
    models = set()
    for i in range(inst_count):
        o = inst_off + i * 36                        # sizeof(PmapInstance) = 36
        mi = struct.unpack_from("<I", blob, o)[0]    # local model index at record+0
        x, y, z = struct.unpack_from("<3f", blob, o + 4)  # pos at record+4
        if (round(x, 1), round(y, 1), round(z, 1)) in road_pos:
            models.add(mi)
    return models


def main():
    chunks = sys.argv[1] if len(sys.argv) > 1 else CHUNKS
    road_pos, n_wet, n_inst = wetroad_positions()
    print(f"wet_road models: {n_wet} | SA instances scanned: {n_inst} | "
          f"road placement keys: {len(road_pos)}")

    total = files = 0
    for pmap in sorted(glob.glob(os.path.join(chunks, "region_*_*.pmap"))):
        base = os.path.splitext(pmap)[0]
        out = base + ".road"
        models = sorted(region_road_models(pmap, road_pos))
        if not models:
            if os.path.exists(out):
                os.remove(out)
            continue
        buf = b"ROAD" + struct.pack("<HH", len(models), 0)
        for mi in models:
            buf += struct.pack("<H", mi)
        open(out, "wb").write(buf)
        files += 1
        total += len(models)
        print(f"  {os.path.basename(base)}.road: {len(models)} road models")
    print(f"wrote {files} region .road sidecars, {total} road models total")


if __name__ == "__main__":
    main()
