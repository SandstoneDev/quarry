#!/usr/bin/env python3
"""model_census - find "gutted" models across all deployed regions.

Background (research/… detail-gap session 2026-07-21): every IPL instance is
present in the deployed set (775/775 on the Grove tile), but an early-era
bin-mesh decoder bug left some MODELS with only 2-33% of their source
triangles - always the alpha material splits: telephone wires, agave/minipalm
bushes, planter vegetation, storm-drain overgrowth, plaza detail, the Grove
bridge. The transplant/graft chain froze those corpses into production
("models bit-for-bat"). This tool measures the disease map-wide.

Method: source IPL instances (sa_source: names+positions) are position-matched
(<=0.5u) to each region's pmap instances; per pmap model, deployed triangle
count is compared to the DFF source count (destripped). Deployed can be larger
(24u/UV tessellation, ~1.5x typical); GUTTED = deployed < RATIO_BAD * source.

Usage: model_census.py <chunks_dir> <out.json> [--ratio 0.5] [--min-src 50]
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from formats.dff import parse_dff, _destrip

HDR = struct.Struct('<20I')


def load_v2(path):
    blob = open(path, 'rb').read()
    ver = struct.unpack_from('<I', blob, 4)[0]
    if ver == 2:
        return blob
    tmp = tempfile.mktemp(suffix='.pmap')
    subprocess.check_call([sys.executable,
                           os.path.join(TOOLS, 'pmap_lz4_decompress.py'),
                           path, tmp], stdout=subprocess.DEVNULL)
    blob = open(tmp, 'rb').read()
    os.remove(tmp)
    return blob


def main():
    argv = sys.argv[1:]
    ratio_bad = 0.5
    min_src = 50
    if "--ratio" in argv:
        k = argv.index("--ratio"); ratio_bad = float(argv[k+1]); del argv[k:k+2]
    if "--min-src" in argv:
        k = argv.index("--min-src"); min_src = int(argv[k+1]); del argv[k:k+2]
    chunks_dir, out_path = argv

    print("loading source defs+instances...", flush=True)
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts = sa_source.load_instances(defs, img)
    # position grid for 0.5u matching
    grid = {}
    for i in insts:
        x, y, z = i["pos"]
        grid.setdefault((round(x), round(y)), []).append((x, y, z, i["name"]))
    def match(x, y, z):
        for cx in (round(x)-1, round(x), round(x)+1):
            for cy in (round(y)-1, round(y), round(y)+1):
                for (sx, sy, sz, nm) in grid.get((cx, cy), ()):
                    if abs(sx-x) <= 0.5 and abs(sy-y) <= 0.5 and abs(sz-z) <= 0.5:
                        return nm
        return None

    src_tris_cache = {}
    def src_tris(name):
        if name in src_tris_cache:
            return src_tris_cache[name]
        n = -1
        blob = sa_source.img_read(img, name + ".dff")
        if blob is not None:
            try:
                dff = parse_dff(blob)
                n = 0
                for a in dff.atomics:
                    geo = dff.geometries[a.geometry_index]
                    for sp in geo.splits:
                        idx = _destrip(sp["indices"]) if sp["strip"] else list(sp["indices"])
                        n += len(idx) // 3
            except Exception:
                n = -1
        src_tris_cache[name] = n
        return n

    report = {}
    files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".pmap"))
    total_bad = 0
    for fi, fn in enumerate(files):
        blob = load_v2(os.path.join(chunks_dir, fn))
        h = HDR.unpack_from(blob, 0)
        mc, moff, sc, soff, ic, ioff = h[3], h[4], h[5], h[6], h[9], h[10]
        models = [struct.unpack_from('<2I6f', blob, moff + 32*i) for i in range(mc)]
        subs = [struct.unpack_from('<i4I', blob, soff + 20*i) for i in range(sc)]
        m_names = [None] * mc
        for i in range(ic):
            mi, px, py, pz = struct.unpack_from('<I3f', blob, ioff + 36*i)
            if mi < mc and m_names[mi] is None:
                m_names[mi] = match(px, py, pz)
        bad = []
        for mi, m in enumerate(models):
            nm = m_names[mi]
            if not nm:
                continue
            dep = sum(subs[s][4] for s in range(m[0], m[0]+m[1])) // 3
            st = src_tris(nm)
            if st >= min_src and dep < ratio_bad * st:
                bad.append({"model": mi, "name": nm, "dep_tris": dep,
                            "src_tris": st, "subs": m[1]})
        if bad:
            report[fn] = bad
            total_bad += len(bad)
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(files)} regions, {total_bad} gutted so far...",
                  flush=True)
    json.dump(report, open(out_path, "w"), indent=1)
    names = {}
    for fn, lst in report.items():
        for b in lst:
            names.setdefault(b["name"], [0, 0, 0])
            names[b["name"]][0] += 1
            names[b["name"]][1] = b["src_tris"]
            names[b["name"]][2] = b["dep_tris"]
    print(f"DONE: {total_bad} gutted model records in {len(report)} regions, "
          f"{len(names)} unique models -> {out_path}")
    for nm, (cnt, st, dt) in sorted(names.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {nm:24s} regions={cnt:3d} src={st:5d} dep={dt:5d}")


if __name__ == "__main__":
    main()
