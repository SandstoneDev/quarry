#!/usr/bin/env python3
"""tile_preview - offline render of a region_*.pmap (v2, PRE-lz4) to PNG.
Top-down + oblique, GE-faithful sampling (uv = s16/4096, REPEAT, nearest,
MODULATE by vertex colour). Usage:
  python tools/map_export/tile_preview.py <region.pmap> <out.png>
Writes <out>_top.png and <out>_ob.png.
"""
import os
import math
import sys

import numpy as np
from PIL import Image

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_scene, rwtex

GU_PSM_T4 = 4


def decode_tex(t):
    if t.format == GU_PSM_T4:
        wb_bytes = max(t.buffer_width // 2, 16)
    else:
        wb_bytes = max(t.buffer_width, 16)
    lvl0 = t.texel_bytes[:wb_bytes * t.height]
    lin = rwtex.unswizzle(lvl0, wb_bytes, t.height)
    rgba = np.zeros((t.height, t.width, 4), np.uint8)
    cl = t.clut_bytes
    for y in range(t.height):
        row = lin[y * wb_bytes:(y + 1) * wb_bytes]
        for x in range(t.width):
            if t.format == GU_PSM_T4:
                bt = row[x >> 1]
                pi = (bt & 0xF) if (x & 1) == 0 else ((bt >> 4) & 0xF)
            else:
                pi = row[x]
            s = pi * 4
            if s + 4 <= len(cl):
                rgba[y, x] = cl[s], cl[s+1], cl[s+2], cl[s+3]
    return rgba


def quat_mat(qx, qy, qz, qw):
    """Row-vector rotation matching the runtime build_inst_matrix convention
    (SA IPL stores the conjugate; the baker wrote quats verbatim, the engine
    builds the transposed RW basis - same math both sides, verified visually
    against PPSSPP renders)."""
    x2, y2, z2 = qx+qx, qy+qy, qz+qz
    xx, yy, zz = qx*x2, qy*y2, qz*z2
    xy, xz, yz = qx*y2, qx*z2, qy*z2
    wx, wy, wz = qw*x2, qw*y2, qw*z2
    return np.array([[1-(yy+zz), xy+wz,     xz-wy],
                     [xy-wz,     1-(xx+zz), yz+wx],
                     [xz+wy,     yz-wx,     1-(xx+yy)]], np.float32)


def load_tris(pmap_path):
    sc = psp_scene.read_scene(open(pmap_path, "rb").read())
    texs = [decode_tex(t) for t in sc.textures]
    tris = []
    for inst in sc.instances:
        if inst.interior:                  # LOD proxies would double-draw: skip
            continue
        md = sc.models[inst.model]
        R = quat_mat(*inst.quat)
        base = np.array(inst.pos, np.float32)
        ctr = np.array(md.center, np.float32)
        for sm in md.submeshes:
            v = np.frombuffer(sm.vertex_bytes, np.uint8).reshape(-1, 12)
            uv = v[:, 0:4].view("<i2").astype(np.float32) / 4096.0
            col = v[:, 4:6].view("<u2").astype(np.uint32)[:, 0]
            pos = v[:, 6:12].view("<i2").astype(np.float32) * md.scale + ctr
            wpos = pos @ R + base
            r = ((col & 31) << 3).astype(np.float32)
            g = (((col >> 5) & 31) << 3).astype(np.float32)
            b = (((col >> 10) & 31) << 3).astype(np.float32)
            idx = np.frombuffer(sm.index_bytes, np.uint16)
            tris.append((wpos, uv, np.stack([r, g, b], 1), idx, sm.texture))
    return tris, texs


def render(tris, texs, view, W=1024, H=1024):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 34, 40)
    zbuf = np.full((H, W), 1e9, np.float32)
    allp = np.concatenate([t[0] for t in tris])
    pv = allp @ view.T
    mn, mx = pv.min(0), pv.max(0)
    span = max(mx[0]-mn[0], mx[1]-mn[1]) * 1.02
    scpx = min(W, H) / span
    cx, cy = (mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5
    for (wpos, uv, rgb, idx, ti) in tris:
        v = wpos @ view.T
        sx = (v[:, 0]-cx)*scpx + W*0.5
        sy = (v[:, 1]-cy)*scpx + H*0.5
        sz = v[:, 2]
        tex = texs[ti] if 0 <= ti < len(texs) else None
        for t in range(0, len(idx)-2, 3):
            a, b, c = idx[t], idx[t+1], idx[t+2]
            xs = (sx[a], sx[b], sx[c])
            ys = (sy[a], sy[b], sy[c])
            minx = max(int(min(xs)), 0); maxx = min(int(max(xs))+1, W)
            miny = max(int(min(ys)), 0); maxy = min(int(max(ys))+1, H)
            if minx >= maxx or miny >= maxy:
                continue
            d = (xs[1]-xs[0])*(ys[2]-ys[0]) - (xs[2]-xs[0])*(ys[1]-ys[0])
            if abs(d) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(minx, maxx)+0.5,
                                 np.arange(miny, maxy)+0.5)
            w1 = ((gx-xs[0])*(ys[2]-ys[0]) - (xs[2]-xs[0])*(gy-ys[0])) / d
            w2 = ((xs[1]-xs[0])*(gy-ys[0]) - (gx-xs[0])*(ys[1]-ys[0])) / d
            w0 = 1.0 - w1 - w2
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            z = w0*sz[a] + w1*sz[b] + w2*sz[c]
            sub = zbuf[miny:maxy, minx:maxx]
            m &= z < sub
            if not m.any():
                continue
            sub[m] = z[m]
            u = w0*uv[a, 0] + w1*uv[b, 0] + w2*uv[c, 0]
            vv = w0*uv[a, 1] + w1*uv[b, 1] + w2*uv[c, 1]
            cr = (w0*rgb[a, 0] + w1*rgb[b, 0] + w2*rgb[c, 0])
            cg = (w0*rgb[a, 1] + w1*rgb[b, 1] + w2*rgb[c, 1])
            cb = (w0*rgb[a, 2] + w1*rgb[b, 2] + w2*rgb[c, 2])
            if tex is not None:
                th, tw = tex.shape[:2]
                tu = (np.floor(u*tw).astype(np.int64) % tw)
                tv = (np.floor(vv*th).astype(np.int64) % th)
                tt = tex[tv, tu]
                cr = cr * tt[..., 0] / 255.0 * 2.0     # x2: prelit is half-bright
                cg = cg * tt[..., 1] / 255.0 * 2.0
                cb = cb * tt[..., 2] / 255.0 * 2.0
            o = img[miny:maxy, minx:maxx]
            o[m] = np.stack([np.clip(cr, 0, 255)[m], np.clip(cg, 0, 255)[m],
                             np.clip(cb, 0, 255)[m]], 1).astype(np.uint8)
    return Image.fromarray(img)


def main():
    pmap, out = sys.argv[1], sys.argv[2]
    tris, texs = load_tris(pmap)
    print("tris source: %d submesh batches, %d textures" % (len(tris), len(texs)))
    top = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    a = math.radians(35)
    ob = np.array([[1, 0, 0],
                   [0, -math.sin(a), -math.cos(a)],
                   [0, -math.cos(a),  math.sin(a)]], np.float32)
    render(tris, texs, top).save(out.replace(".png", "_top.png"))
    print("saved", out.replace(".png", "_top.png"))
    render(tris, texs, ob).save(out.replace(".png", "_ob.png"))
    print("saved", out.replace(".png", "_ob.png"))


if __name__ == "__main__":
    main()
