#!/usr/bin/env python3
"""night_delta_bake - sparse per-vertex NIGHT colours for deployed regions.

SA ships a second prelight set per geometry (EXTENSION chunk 0x0253F2F9,
magic + RwRGBA[nvert]) - lit windows / neon at night. The full-buffer
region_*.night sidecars (~270KB/tile) were disabled in build 170 (heap);
this bakes the SPARSE alternative: only vertices whose night colour differs
from day. Typically a few % (windows), so a tile costs KBs, not hundreds.

Per region (run on the FINAL deployed pmap - vertex indices must match!):
 1. position-match pmap instances to source IPL rows (sa_source) -> DFF name
 2. parse the DFF's day+night vertex colours per geometry (local space)
 3. for each pmap vertex of the model: nearest source vertex (<=0.5u);
 if source night != day -> emit {global_vidx, night5551}
 (tessellation midpoints inherit the nearest corner's glow)
 4. coalesce consecutive same-colour vertices into RUNS (window quads share
 one colour across 4+ sequential verts) and write region_<rx>_<ry>.nightd:
 u32 magic 'NDL2' u32 run_count then run_count x {u32 vidx, u16 n, u16 col}

Usage: night_delta_bake.py <chunks_dir> [--only region_12_2] [--report]
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from formats.dff import parse_dff

NIGHT_CHUNK = 0x0253F2F9
HDR = struct.Struct('<20I')
MAGIC = 0x324C444E   # 'NDL2' (v2: colour runs)


def load_v2(path):
    blob = open(path, 'rb').read()
    if struct.unpack_from('<I', blob, 4)[0] == 2:
        return blob
    tmp = tempfile.mktemp(suffix='.pmap')
    subprocess.check_call([sys.executable,
                           os.path.join(TOOLS, 'pmap_lz4_decompress.py'),
                           path, tmp], stdout=subprocess.DEVNULL)
    blob = open(tmp, 'rb').read()
    os.remove(tmp)
    return blob


def to5551(r, g, b, a):
    return (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10) | ((1 if a >= 128 else 0) << 15)


def night_verts_of(dff):
    """[(x,y,z,night5551)] for every vertex whose night differs from day.
 Walks geometry extensions for chunk 0x0253F2F9 via the SAW parser's raw
 chunk tree (geo.extensions dict if present, else geo.night_colors)."""
    out = []
    for geo in dff.geometries:
        night = getattr(geo, "night_colors", None)
        if night is None:
            ext = getattr(geo, "extensions", None) or {}
            night = ext.get(NIGHT_CHUNK)
        if not night:
            continue
        pre = geo.prelit_colors
        if pre is None:
            continue
        n = min(len(night), len(geo.vertices), len(pre))
        for i in range(n):
            nr, ng_, nb, na = night[i]
            dr, dg, db, da = pre[i]
            # GLOW filter: the runtime's night base is a global darken
            # (day * ~0.44); most SA night colours are just that tint and need
            # no delta. Emit only verts noticeably BRIGHTER than the darkened
            # day - the emissive class (lit windows, neon, lamps).
            nl = nr * 3 + ng_ * 4 + nb
            dl = dr * 3 + dg * 4 + db
            # true emissive only: absolutely bright at night AND clearly above
            # what the global darken would give (dull tints stay base-darkened)
            if nl >= 800 and nl > dl * 0.7:
                x, y, z = geo.vertices[i]
                out.append((x, y, z, to5551(nr, ng_, nb, na)))
    return out


def main():
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        k = argv.index("--only"); only = argv[k+1]; del argv[k:k+2]
    report = "--report" in argv
    argv = [a for a in argv if a != "--report"]
    chunks_dir = argv[0]

    print("loading source defs+instances...", flush=True)
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts = sa_source.load_instances(defs, img)
    from match_util import build_grid, match_all, pick_by_verts
    grid = build_grid(insts)

    vc_cache = {}
    def dff_verts(name):
        if name in vc_cache:
            return vc_cache[name]
        n = 0
        blob = sa_source.img_read(img, name + ".dff")
        if blob is not None:
            try:
                d = parse_dff(blob)
                for geo in d.geometries:
                    n += len(geo.vertices)
            except Exception:
                n = 0
        vc_cache[name] = n
        return n

    night_cache = {}
    def night_of(name):
        if name in night_cache:
            return night_cache[name]
        v = []
        blob = sa_source.img_read(img, name + ".dff")
        if blob is not None:
            try:
                v = night_verts_of(parse_dff(blob))
            except Exception:
                v = []
        night_cache[name] = v
        return v

    files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".pmap"))
    if only:
        files = [f for f in files if f.startswith(only)]
    tot_files = tot_deltas = 0
    for fn in files:
        blob = load_v2(os.path.join(chunks_dir, fn))
        h = HDR.unpack_from(blob, 0)
        mc, moff, sc, soff, ic, ioff, voff = h[3], h[4], h[5], h[6], h[9], h[10], h[12]
        models = [struct.unpack_from('<2I6f', blob, moff + 32*i) for i in range(mc)]
        subs = [struct.unpack_from('<i4I', blob, soff + 20*i) for i in range(sc)]
        pverts = [0] * mc
        for mi2 in range(mc):
            m2 = models[mi2]
            for s in range(m2[0], m2[0] + m2[1]):
                pverts[mi2] += subs[s][2]
        m_name = [None] * mc
        for i in range(ic):
            mi, px, py, pz = struct.unpack_from('<I3f', blob, ioff + 36*i)
            if mi < mc and m_name[mi] is None:
                cands = match_all(grid, px, py, pz)
                if cands:
                    m_name[mi] = pick_by_verts(cands, pverts[mi], dff_verts)
        deltas = []
        for mi, m in enumerate(models):
            nm = m_name[mi]
            if not nm:
                continue
            nv_src = night_of(nm)
            if not nv_src:
                continue
            first, cnt, scale, cx, cy, cz = m[0], m[1], m[2], m[3], m[4], m[5]
            src = np.array([(x, y, z) for (x, y, z, c) in nv_src], np.float32)
            col = np.array([c for (_, _, _, c) in nv_src], np.uint16)
            for s in range(first, first + cnt):
                tex, vfirst, vcount, ifirst, icnt = subs[s]
                if vcount == 0:
                    continue
                v = np.frombuffer(blob, dtype=np.int16,
                                  count=vcount * 6, offset=voff + vfirst * 12)
                v = v.reshape(-1, 6)
                px_ = v[:, 3].astype(np.float32) * scale + cx
                py_ = v[:, 4].astype(np.float32) * scale + cy
                pz_ = v[:, 5].astype(np.float32) * scale + cz
                # nearest source night-vert per pmap vert (small src sets)
                d2 = ((px_[:, None] - src[None, :, 0]) ** 2
                      + (py_[:, None] - src[None, :, 1]) ** 2
                      + (pz_[:, None] - src[None, :, 2]) ** 2)
                j = d2.argmin(1)
                dm = d2[np.arange(len(j)), j]
                hit = dm <= 0.25          # <=0.5u
                for k in np.nonzero(hit)[0]:
                    deltas.append((vfirst + int(k), int(col[j[k]])))
        out_path = os.path.join(chunks_dir, fn[:-5] + ".nightd")
        deltas.sort()
        runs = []
        for vidx, c in deltas:
            if runs and runs[-1][2] == c and runs[-1][0] + runs[-1][1] == vidx \
                    and runs[-1][1] < 0xFFFF:
                runs[-1][1] += 1
            else:
                runs.append([vidx, 1, c])
        with open(out_path, "wb") as f:
            f.write(struct.pack('<II', MAGIC, len(runs)))
            for vidx, n, c in runs:
                f.write(struct.pack('<IHH', vidx, n, c))
        tot_files += 1
        tot_deltas += len(deltas)
        if report or len(deltas):
            print(f"  {fn[:-5]}.nightd: {len(deltas)} deltas in {len(runs)} runs "
                  f"({(8 + 8*len(runs))//1024}KB)", flush=True)
    print(f"DONE: {tot_files} sidecars, {tot_deltas} night deltas total")


if __name__ == "__main__":
    main()
