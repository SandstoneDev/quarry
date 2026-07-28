#!/usr/bin/env python3
"""rewrite_tobj_flags - stamp the ADDITIVE bit into existing .tobj sidecars.

b586: SA renders a tobj additively ONLY when its IDE flags carry ADDITIVE(8);
_dy/_nt model swaps, floodbeams etc. draw normally. Our b576 sidecars only
stored {inst,on,off}; this walks every region_X_Y.tobj, resolves each entry's
model name (position match against the SA instances) and sets bit7 of the ON
byte when the IDE def is additive. Idempotent.

Usage: rewrite_tobj_flags.py <chunks_dir>
"""
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
from night_delta_bake import load_v2, HDR
from match_util import build_grid, match_all


def main():
    chunks_dir = sys.argv[1]
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts = sa_source.load_instances(defs, img)
    grid = build_grid(insts)
    additive = {d["dff"].lower() for d in defs.values()
                if d.get("time_on") is not None and (int(d.get("flags", 0)) & 8)}
    print(f"additive tobj defs: {len(additive)}")

    files = flips = 0
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.endswith(".tobj"):
            continue
        p = os.path.join(chunks_dir, fn)
        td = bytearray(open(p, "rb").read())
        magic, n = struct.unpack_from('<II', td, 0)
        blob = load_v2(os.path.join(chunks_dir, fn[:-5] + ".pmap"))
        h = HDR.unpack_from(blob, 0)
        ic, ioff = h[9], h[10]
        changed = 0
        for i in range(n):
            inst, on, off = struct.unpack_from('<HBB', td, 8 + 4 * i)
            if inst >= ic:
                continue
            _, px, py, pz = struct.unpack_from('<I3f', blob, ioff + 36 * inst)
            names = [nm.lower() for nm in match_all(grid, px, py, pz)]
            is_add = any(nm in additive for nm in names)
            new_on = ((on & 0x7F) | (0x80 if is_add else 0))
            if new_on != on:
                struct.pack_into('<HBB', td, 8 + 4 * i, inst, new_on, off)
                changed += 1
        if changed:
            open(p, "wb").write(bytes(td))
            files += 1; flips += changed
            print(f"  {fn}: {changed} flagged")
    print(f"DONE: {flips} entries flagged additive in {files} sidecars")


if __name__ == "__main__":
    main()
