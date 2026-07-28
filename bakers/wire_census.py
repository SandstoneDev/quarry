#!/usr/bin/env python3
"""wire_census - census JSON of every *wire* model instance in the prod chunks.

The 568-model graft census only caught GUTTED geometry (tri-ratio < 0.5); the
telephone wires kept their triangles but lost their ALPHA in the ancient bake
(64px downscale averaged the thin line away, class left opaque) - so they were
never name-repaired. This maps every SA instance whose model name contains
'wire' to its prod region/model by position and writes the same census JSON
that fix_graft_texnames.py consumes ({file: [{model, name}]}), so the existing
name-driven repair re-pulls their textures at native size with true alpha.

Usage: wire_census.py <chunks_dir> <out.json>
"""
import json
import math
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
from pmap_tex_t4from128 import HDR, load_v2


def main():
    chunks_dir, out_path = sys.argv[1], sys.argv[2]
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts = sa_source.load_instances(defs, img)
    wires = [(i["name"].lower(), i["pos"]) for i in insts
             if "wire" in i["name"].lower()]
    print(f"SA wire instances: {len(wires)}")

    census = {}
    found = 0
    matched = set()
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.startswith("region_") or not fn.endswith(".pmap"):
            continue
        blob, _ = load_v2(os.path.join(chunks_dir, fn))
        h = HDR.unpack_from(blob, 0)
        ic, ioff = h[9], h[10]
        ents = []
        seen_models = set()
        for ii in range(ic):
            mi, px, py, pz = struct.unpack_from('<I3f', blob, ioff + 36*ii)
            for wi, (nm, p) in enumerate(wires):
                if abs(px-p[0]) < 2.0 and abs(py-p[1]) < 2.0 and abs(pz-p[2]) < 4.0:
                    matched.add(wi)
                    if mi not in seen_models:
                        seen_models.add(mi)
                        ents.append({"model": int(mi), "name": nm})
                    found += 1
                    break
        if ents:
            census[fn] = ents
    json.dump(census, open(out_path, "w"), indent=1)
    missing = [wires[i] for i in range(len(wires)) if i not in matched]
    print(f"matched {found} prod instances over {len(census)} regions; "
          f"{len(missing)} SA instances UNMATCHED")
    for nm, p in missing[:20]:
        print("  MISSING", nm, [round(v, 1) for v in p])


if __name__ == "__main__":
    main()
