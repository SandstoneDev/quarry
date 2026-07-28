#!/usr/bin/env python3
"""append_tobj_glow - keep injected time-objects (neon) bright at night.

Injected tobj models are world geometry now, so the night vertex darken
(pmap.c darken5551) kills their glow. This appends .nightd runs covering
every vertex of every tobj model with its own DAY colour: at night the
relight overlay lerps those verts back to authored brightness (that IS the
neon look), by day nothing changes. EBOOT untouched.

Idempotent: skips a region if its .nightd already covers the first tobj
vertex. Runs are appended (tobj verts sit at the pool tail, so vidx order
stays sorted for the runtime bsearch).

Usage: append_tobj_glow.py <chunks_dir>
"""
import os
import struct
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from pmap_tex_t4from128 import HDR, load_v2

MAGIC = 0x324C444E   # 'NDL2'


def main():
    chunks_dir = sys.argv[1]
    tot_runs = files = 0
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.endswith(".tobj"):
            continue
        base = fn[:-5]
        pmap_path = os.path.join(chunks_dir, base + ".pmap")
        nd_path = os.path.join(chunks_dir, base + ".nightd")
        td = open(os.path.join(chunks_dir, fn), "rb").read()
        magic, n = struct.unpack_from('<II', td, 0)
        tobj_inst = [struct.unpack_from('<HBB', td, 8 + 4*i)[0] for i in range(n)]
        if not tobj_inst:
            continue

        blob, _ = load_v2(pmap_path)
        h = HDR.unpack_from(blob, 0)
        mc, moff, sc, soff, ic, ioff, voff = h[3], h[4], h[5], h[6], h[9], h[10], h[12]
        models = [struct.unpack_from('<2I6f', blob, moff + 32*i) for i in range(mc)]
        subs = [struct.unpack_from('<i4I', blob, soff + 20*i) for i in range(sc)]

        # models used by tobj instances
        tobj_models = set()
        for ii in tobj_inst:
            if ii < ic:
                mi = struct.unpack_from('<I', blob, ioff + 36*ii)[0]
                if mi < mc:
                    tobj_models.add(mi)
        if not tobj_models:
            continue

        # existing nightd
        runs = []
        try:
            nd = open(nd_path, "rb").read()
            m2, rn = struct.unpack_from('<II', nd, 0)
            if m2 == MAGIC:
                runs = [struct.unpack_from('<IHH', nd, 8 + 8*i) for i in range(rn)]
        except Exception:
            runs = []
        covered = set(r[0] for r in runs)

        new_runs = []
        for mi in sorted(tobj_models):
            m = models[mi]
            for s in range(m[0], m[0] + m[1]):
                tex, vf, vc, if_, icnt = subs[s]
                if vc == 0 or vf in covered:
                    continue
                # per-vertex day colours -> coalesce equal-colour runs
                cols = np.frombuffer(blob, dtype='<u2', count=vc*6,
                                     offset=voff + vf*12).reshape(-1, 6)[:, 2]
                start = 0
                for i in range(1, vc + 1):
                    if i == vc or cols[i] != cols[start] \
                            or (i - start) >= 0xFFFF:
                        new_runs.append((vf + start, i - start, int(cols[start])))
                        start = i
        if not new_runs:
            continue
        all_runs = sorted(runs + new_runs)
        with open(nd_path, "wb") as f:
            f.write(struct.pack('<II', MAGIC, len(all_runs)))
            for vidx, cnt, col in all_runs:
                f.write(struct.pack('<IHH', vidx, cnt, col))
        files += 1; tot_runs += len(new_runs)
        print(f"  {base}: +{len(new_runs)} glow runs for {len(tobj_models)} tobj models",
              flush=True)
    print(f"DONE: {tot_runs} runs appended in {files} regions")


if __name__ == "__main__":
    main()
