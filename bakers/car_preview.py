#!/usr/bin/env python3
"""car_preview - offline software render of car.bin (CAR3), GE-faithful.

Rasterizes the baked car exactly the way the PSP GE will sample it:
  pos = s16 * scale + center (car space), uv = s16 / 4096 (TexScale 8.0 on GE),
  REPEAT wrap, nearest sample, MODULATE by the baked vertex colour.
Wheel mesh is instanced at the 4 mounts. Output: side + 3/4 PNG views.

Use to verify texture mapping / paint / damage meshes without a PPSSPP camera
dance: `python tools/car_preview.py [--dam] [--vlo]`.
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import rwtex

CAR = ""
OUT = ""


def read_car(path):
    b = open(path, "rb").read()
    o = 0
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o)
        o += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]
    assert b[:4] == b"CAR3", "not CAR3"
    o = 4
    rd("24f")                      # handling
    rd("3B3B2x")                   # colours
    rd("3f")                       # seat
    rd("2f")                       # wheelScale, wheelRadius
    mounts = [rd("3f") for _ in range(4)]
    ntex = rd("I")
    texs = []
    for _ in range(ntex):
        tw, th, nl, ce = rd("HHHH")
        tl, cl = rd("II")
        texels = b[o:o+tl]; o += tl
        clut = b[o:o+cl]; o += cl
        # level 0 plane -> unswizzle -> CLUT8888 -> RGBA ndarray
        wb = max(tw, 16)                       # T8 byte width (min swizzle block 16)
        lin = rwtex.unswizzle(texels[:wb*th], wb, th)
        rgba = np.zeros((th, tw, 4), np.uint8)
        for y in range(th):
            row = lin[y*wb:y*wb+tw]
            for x in range(tw):
                s = row[x] * 4
                rgba[y, x] = clut[s], clut[s+1], clut[s+2], clut[s+3]
        texs.append(rgba)
    ncomp = rd("I")
    comps = []
    for _ in range(ncomp):
        name = bytes(rd("16B")).split(b"\0")[0].decode()
        kind, axis, hasDam, _p = rd("BBBB")
        pivot = rd("3f")
        okS = rd("f"); okC = rd("3f"); okN = rd("I")
        dmS = rd("f"); dmC = rd("3f"); dmN = rd("I")
        comps.append(dict(name=name, okS=okS, okC=okC, okN=okN,
                          dmS=dmS, dmC=dmC, dmN=dmN))
    vloS = rd("f"); vloC = rd("3f"); vloN = rd("I")
    wS = rd("f"); wC = rd("3f"); wN = rd("I")

    def read_prims(n):
        out = []
        nonlocal o
        for _ in range(n):
            ti, am, _pad = rd("hBB")
            vb, ib = rd("II")
            verts = np.frombuffer(b[o:o+vb], np.uint8).reshape(-1, 12); o += vb
            idx = np.frombuffer(b[o:o+ib], np.uint16); o += ib
            uv = verts[:, 0:4].view("<i2").astype(np.float32) / 4096.0
            col = verts[:, 4:6].view("<u2").astype(np.uint32)[:, 0]
            pos = verts[:, 6:12].view("<i2").astype(np.float32)
            r = ((col & 31) << 3).astype(np.uint8)
            g = (((col >> 5) & 31) << 3).astype(np.uint8)
            bl = (((col >> 10) & 31) << 3).astype(np.uint8)
            out.append(dict(tex=ti, amode=am, uv=uv, pos=pos, idx=idx,
                            rgb=np.stack([r, g, bl], 1)))
        return out
    for c in comps:
        c["ok"] = read_prims(c["okN"])
        c["dam"] = read_prims(c["dmN"])
    vlo = read_prims(vloN)
    wheel = read_prims(wN)
    return dict(texs=texs, comps=comps, vlo=vlo, vloS=vloS, vloC=vloC,
                wheel=wheel, wS=wS, wC=wC, mounts=mounts)


def gather(car, dam=False, use_vlo=False):
    """[(worldPos Nx3, uv, rgb, idx, tex)] with pack scale/centre applied."""
    out = []
    if use_vlo:
        for p in car["vlo"]:
            out.append((p["pos"] * car["vloS"] + np.array(car["vloC"]), p))
        return out
    for c in car["comps"]:
        run, S, C = (c["dam"], c["dmS"], c["dmC"]) if dam and c["dmN"] else (c["ok"], c["okS"], c["okC"])
        for p in run:
            out.append((p["pos"] * S + np.array(C), p))
    for m in car["mounts"]:
        for p in car["wheel"]:
            out.append((p["pos"] * car["wS"] + np.array(car["wC"]) + np.array(m), p))
    return out


def render(tris, texs, view, W=760, H=380):
    """view: 3x3 rows -> screen (x right, y down, z depth)."""
    img = np.zeros((H, W, 3), np.uint8); img[:] = (40, 44, 52)
    zbuf = np.full((H, W), 1e9, np.float32)
    # project all to find fit
    allp = np.concatenate([w for (w, p) in tris])
    pv = allp @ view.T
    mn = pv.min(0); mx = pv.max(0)
    span = max(mx[0]-mn[0], mx[1]-mn[1]) * 1.10
    sc = min(W, H) / span
    cx = (mn[0]+mx[0])*0.5; cy = (mn[1]+mx[1])*0.5
    def to_scr(p):
        v = p @ view.T
        x = (v[:, 0]-cx)*sc + W*0.5
        y = (v[:, 1]-cy)*sc + H*0.5
        return np.stack([x, y, v[:, 2]], 1)
    for (wpos, p) in tris:
        scr = to_scr(wpos)
        tex = texs[p["tex"]] if p["tex"] >= 0 else None
        uv = p["uv"]; rgb = p["rgb"]; idx = p["idx"]
        for t in range(0, len(idx)-2, 3):
            a, b, c = idx[t], idx[t+1], idx[t+2]
            pa, pb, pc = scr[a], scr[b], scr[c]
            minx = max(int(min(pa[0], pb[0], pc[0])), 0)
            maxx = min(int(max(pa[0], pb[0], pc[0]))+1, W)
            miny = max(int(min(pa[1], pb[1], pc[1])), 0)
            maxy = min(int(max(pa[1], pb[1], pc[1]))+1, H)
            if minx >= maxx or miny >= maxy:
                continue
            d = (pb[0]-pa[0])*(pc[1]-pa[1]) - (pc[0]-pa[0])*(pb[1]-pa[1])
            if abs(d) < 1e-9:
                continue
            xs = np.arange(minx, maxx) + 0.5
            ys = np.arange(miny, maxy) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            w1 = ((gx-pa[0])*(pc[1]-pa[1]) - (pc[0]-pa[0])*(gy-pa[1])) / d
            w2 = ((pb[0]-pa[0])*(gy-pa[1]) - (gx-pa[0])*(pb[1]-pa[1])) / d
            w0 = 1.0 - w1 - w2
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            z = w0*pa[2] + w1*pb[2] + w2*pc[2]
            sub = zbuf[miny:maxy, minx:maxx]
            m &= z < sub
            if not m.any():
                continue
            sub[m] = z[m]
            u = w0*uv[a, 0] + w1*uv[b, 0] + w2*uv[c, 0]
            v = w0*uv[a, 1] + w1*uv[b, 1] + w2*uv[c, 1]
            cr = (w0*rgb[a, 0] + w1*rgb[b, 0] + w2*rgb[c, 0])
            cg = (w0*rgb[a, 1] + w1*rgb[b, 1] + w2*rgb[c, 1])
            cb = (w0*rgb[a, 2] + w1*rgb[b, 2] + w2*rgb[c, 2])
            if tex is not None:
                th, tw = tex.shape[:2]
                tu = (np.floor(u*tw).astype(np.int64) % tw)
                tv = (np.floor(v*th).astype(np.int64) % th)
                tt = tex[tv, tu]
                cr = cr * tt[..., 0] / 255.0
                cg = cg * tt[..., 1] / 255.0
                cb = cb * tt[..., 2] / 255.0
            out = img[miny:maxy, minx:maxx]
            out[m] = np.stack([np.clip(cr, 0, 255)[m],
                               np.clip(cg, 0, 255)[m],
                               np.clip(cb, 0, 255)[m]], 1).astype(np.uint8)
    return Image.fromarray(img)


def main():
    dam = "--dam" in sys.argv
    vlo = "--vlo" in sys.argv
    car = read_car(CAR)
    tris = gather(car, dam=dam, use_vlo=vlo)
    tag = "dam" if dam else ("vlo" if vlo else "ok")
    # side view: car +Y (fwd) -> screen -X ... look from +X (right side)
    side = np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]], np.float32)
    # 3/4 front-left ~ rotate 35deg + slight top-down
    import math
    a = math.radians(215); e = math.radians(18)
    fwd = np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), -math.sin(e)])
    rgt = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.cross(rgt, fwd)
    q34 = np.stack([rgt, -up, fwd]).astype(np.float32)
    render(tris, car["texs"], side).save(OUT % (tag + "_side"))
    render(tris, car["texs"], q34).save(OUT % (tag + "_34"))
    print("saved", OUT % (tag + "_side"), OUT % (tag + "_34"))


if __name__ == "__main__":
    main()
