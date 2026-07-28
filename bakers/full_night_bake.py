#!/usr/bin/env python3
"""full_night_bake - FULL per-vertex night colours (region_X_Y.night) for the

Output format = what pmap_load_night() expects: a raw little-endian u16 5551
array, EXACTLY vertex_count entries, aligned to the .pmap vertex pool. No
header. The runtime lerps day->night per vertex by the DN balance; the sparse
.nightd glow runs still overlay after it (tobj neon keeps working).

Per vertex:
  - model position-matched to an SA instance whose DFF carries the second
    prelight set (chunk 0x0253F2F9): nearest source vertex (<=0.5u) -> its
    authored night colour (this is where lit windows come from).
  - everything else (no night chunk / unmatched / far vertex): the same
    global darken the runtime would apply (darken5551: r*2 g*3 b*5 >>4 in
    5-bit space), so the look degrades to exactly the current base.

Usage: full_night_bake.py <chunks_dir> [--only region_12_2]
"""
import os
import struct
import sys

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
from night_delta_bake import load_v2, HDR, NIGHT_CHUNK

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from formats.dff import parse_dff


def to5551(r, g, b):
    return (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)


def night_verts_all(dff):
    """(pos Nx3 float32, rgb Nx3 float32) for EVERY vertex with a night colour.
    v2: keep RGB888 floats - the IDW blend happens in RGB space, quantise later."""
    ps, cs = [], []
    for geo in dff.geometries:
        night = getattr(geo, "night_colors", None)
        if night is None:
            ext = getattr(geo, "extensions", None) or {}
            night = ext.get(NIGHT_CHUNK)
        if not night:
            continue
        n = min(len(night), len(geo.vertices))
        for i in range(n):
            nr, ng, nb, na = night[i]
            x, y, z = geo.vertices[i]
            ps.append((x, y, z))
            cs.append((nr, ng, nb))
    if not ps:
        return None, None
    return np.array(ps, np.float32), np.array(cs, np.float32)


def darken5551_vec(day):
    """replicate pmap.c darken5551 on a u16 numpy array (keeps the alpha bit)."""
    r = ((day & 31).astype(np.uint32) * 2) >> 4
    g = (((day >> 5) & 31).astype(np.uint32) * 3) >> 4
    b = (((day >> 10) & 31).astype(np.uint32) * 5) >> 4
    return (r | (g << 5) | (b << 10) | (day & 0x8000)).astype(np.uint16)


# v3: the PS2 building formula is tex x (prelit + ambient*surfAmb) - the night
# AMBIENT term is what keeps dark facades readable (lit-window TEXTURES visible
# on a near-black prelit). We fold it into the baked night colours offline (the
# runtime stays a pure modulate). SA night Amb row is ~(22,22,22) + LA haze.
AMB_NIGHT = (26, 28, 34)


def add_amb5551(packed):
    """packed u16 5551 + AMB_NIGHT per channel (clamped), alpha bit kept."""
    r = np.minimum(((packed & 31) << 3) + AMB_NIGHT[0], 255).astype(np.uint32)
    g = np.minimum((((packed >> 5) & 31) << 3) + AMB_NIGHT[1], 255).astype(np.uint32)
    b = np.minimum((((packed >> 10) & 31) << 3) + AMB_NIGHT[2], 255).astype(np.uint32)
    return ((r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)
            | (packed & 0x8000)).astype(np.uint16)


def main():
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        k = argv.index("--only"); only = argv[k + 1]; del argv[k:k + 2]
    chunks_dir = argv[0]

    print("loading source defs+instances...", flush=True)
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts = sa_source.load_instances(defs, img)
    from match_util import build_grid, match_all, pick_by_verts
    grid = build_grid(insts)

    # DFF total vertex count per name (for LOD-vs-HI disambiguation: a co-located
    # LOD instance used to hijack the first-hit match and feed the HI mesh the
    # LOD's dull night set -> authored lit windows lost map-wide).
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

    cache = {}
    def night_of(name):
        if name in cache:
            return cache[name]
        v = (None, None)
        blob = sa_source.img_read(img, name + ".dff")
        if blob is not None:
            try:
                v = night_verts_all(parse_dff(blob))
            except Exception:
                v = (None, None)
        cache[name] = v
        return v

    files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".pmap"))
    if only:
        files = [f for f in files if f.startswith(only)]
    done = 0
    for fn in files:
        blob = load_v2(os.path.join(chunks_dir, fn))
        h = HDR.unpack_from(blob, 0)
        mc, moff, sc, soff, ic, ioff = h[3], h[4], h[5], h[6], h[9], h[10]
        voff, vbytes = h[12], h[13]          # header: [12]=vertex_off [13]=vertex_bytes
        nv = vbytes // 12
        vv = np.frombuffer(blob, dtype='<i2', count=nv * 6, offset=voff).reshape(-1, 6)
        day = vv[:, 2].view(np.uint16)
        night = darken5551_vec(day)                        # base: global darken

        models = [struct.unpack_from('<2I6f', blob, moff + 32 * i) for i in range(mc)]
        subs = [struct.unpack_from('<i4I', blob, soff + 20 * i) for i in range(sc)]
        pverts = [0] * mc                                  # deployed vertex totals
        for mi2 in range(mc):
            m2 = models[mi2]
            for s in range(m2[0], m2[0] + m2[1]):
                pverts[mi2] += subs[s][2]
        m_name = [None] * mc
        for i in range(ic):
            mi = struct.unpack_from('<I3f', blob, ioff + 36 * i)
            if mi[0] < mc and m_name[mi[0]] is None:
                cands = match_all(grid, mi[1], mi[2], mi[3])
                if cands:
                    m_name[mi[0]] = pick_by_verts(cands, pverts[mi[0]], dff_verts)

        # v3: tobj models WITHOUT a night chunk keep their DAY prelit at night
        # (SA never darkens chunk-less models; our global darken killed the
        # _nt casino swaps / floodbeams whose light lives in day prelit+texture)
        tobj_names = {d["dff"].lower() for d in defs.values()
                      if d.get("time_on") is not None}
        lit = 0
        for mi, m in enumerate(models):
            nm = m_name[mi]
            if not nm:
                continue
            src, col = night_of(nm)
            if (src is None or cKDTree is None) and nm.lower() in tobj_names:
                first, cnt = m[0], m[1]
                for s in range(first, first + cnt):
                    tex, vfirst, vcount, ifirst, icnt = subs[s]
                    night[vfirst:vfirst + vcount] = day[vfirst:vfirst + vcount]
                    lit += vcount
                continue
            if src is None or cKDTree is None:
                continue
            tree = cKDTree(src)
            k = 3 if len(src) >= 3 else len(src)
            first, cnt, scale, cx, cy, cz = m[0], m[1], m[2], m[3], m[4], m[5]
            for s in range(first, first + cnt):
                tex, vfirst, vcount, ifirst, icnt = subs[s]
                if vcount == 0:
                    continue
                pv = vv[vfirst:vfirst + vcount, 3:6].astype(np.float32) * scale
                pv[:, 0] += cx; pv[:, 1] += cy; pv[:, 2] += cz
                # v2 IDW: blend the k nearest authored night colours by 1/d^2 --
                # NO distance threshold at all. The 0.5u cutoff (v1) left every
                # tessellation midpoint on the global darken = the dark blotches;
                # a 6u guard still cut the midpoints of BIG LOD/terrain triangles
                # (50-100u edges). The KD-tree only ever contains THIS model's
                # authored nodes (name+position matched), so there is no foreign
                # geometry to guard against - a far midpoint blending 3 corner
                # nodes IS the triangle's average colour, which is correct.
                dm, j = tree.query(pv, k=k)
                if k == 1:
                    dm = dm[:, None]; j = j[:, None]
                w = 1.0 / (dm * dm + 1e-4)
                blend = (col[j] * w[:, :, None]).sum(1) / w.sum(1)[:, None]
                r = np.clip(blend[:, 0], 0, 255).astype(np.uint16)
                g = np.clip(blend[:, 1], 0, 255).astype(np.uint16)
                b = np.clip(blend[:, 2], 0, 255).astype(np.uint16)
                packed = (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)
                night[vfirst:vfirst + vcount] = packed | (day[vfirst:vfirst + vcount] & 0x8000)
                lit += vcount
        out = os.path.join(chunks_dir, fn[:-5] + ".night")
        night = add_amb5551(night)      # v3: + PS2 night ambient term (see AMB_NIGHT)
        night.tofile(out)
        done += 1
        print(f"  {fn}: {nv} verts, {lit} authored-night, {nv*2//1024}KB", flush=True)
    print(f"DONE: {done} regions")


if __name__ == "__main__":
    main()
