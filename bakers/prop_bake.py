#!/usr/bin/env python3
"""prop_bake - bake ONE small static prop (money stack etc.) into a tiny
self-contained mesh blob for simple runtime renderers (Pickups).

  python prop_bake.py money dyn_cash data/money.bin

prop.bin ('PRP1', little-endian):
  u16 nvert, nidx, texW, texH, nlevels|amode<<8, clutEntries
  u32 texelLen, clutLen
  vert[nvert]: f32 u,v; u32 colorABGR; f32 x,y,z     (GE static vertex order)
  idx[nidx]:   u16
  texels[texelLen] swizzled T8 (all mips), clut[clutLen] RGBA8888
"""
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
import geom

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import sa_txd_d3d9, psp_tex
from formats.dff import parse_dff


def main():
    dffname = sys.argv[1] if len(sys.argv) > 1 else "money"
    txdname = sys.argv[2] if len(sys.argv) > 2 else "dyn_cash"
    outp = sys.argv[3] if len(sys.argv) > 3 else \
        ""
    # optional vertex colour (0xRRGGBB) -> modulates the texture (SA marker is a
    # WHITE gradient modulated yellow). Default white = texture as-is.
    vcolor = 0xFFFFFFFF
    if len(sys.argv) > 4:
        rgb = int(sys.argv[4], 16)
        r = (rgb >> 16) & 0xFF; g = (rgb >> 8) & 0xFF; b = rgb & 0xFF
        vcolor = (0xFF << 24) | (b << 16) | (g << 8) | r     # AABBGGRR

    # txd "NONE" -> UNTEXTURED glow: no texture, per-vertex alpha ramped by
    # height (top transparent -> bottom solid). CProp draws it as a plain
    # blended coloured mesh (the SA-marker translucency without the gradient tex).
    glow = (txdname.upper() == "NONE")

    img = sa_source.open_img()
    dff = parse_dff(sa_source.img_read(img, dffname + ".dff"))
    txd = {}
    if not glow:
        txd = {k.lower(): v for k, v in
               sa_txd_d3d9.decode(sa_source.img_read(img, txdname + ".txd")).items()}

    verts = []           # (u,v,color,x,y,z)
    vmap = {}
    idx = []
    texname = None
    zmin = 1e9; zmax = -1e9
    for a in dff.atomics:
        for part in geom.process_geometry(dff.geometries[a.geometry_index]):
            for tri in part["tris"]:
                for (pos, uv, col) in tri:
                    zmin = min(zmin, pos[2]); zmax = max(zmax, pos[2])
    for a in dff.atomics:
        for part in geom.process_geometry(dff.geometries[a.geometry_index]):
            m = part.get("mat")
            if texname is None and m is not None and getattr(m, "texture_name", ""):
                texname = m.texture_name.lower()
            for tri in part["tris"]:
                for (pos, uv, col) in tri:
                    key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                           round(uv[0], 5), round(uv[1], 5))
                    vi = vmap.get(key)
                    if vi is None:
                        vi = len(verts)
                        vmap[key] = vi
                        c = vcolor
                        if glow:
                            # alpha ramp: bottom (zmin) opaque, top (zmax) faint
                            t01 = (pos[2]-zmin)/(zmax-zmin) if zmax > zmin else 0.5
                            a8 = int(30 + (1.0-t01)*205)     # 235 bottom -> 30 top
                            c = (a8 << 24) | (vcolor & 0x00FFFFFF)
                        verts.append((uv[0], uv[1], c, pos[0], pos[1], pos[2]))
                    idx.append(vi)
    print(f"{dffname}: verts={len(verts)} idx={len(idx)} tex={'GLOW' if glow else texname}")

    if glow:
        texels = b""; clut = b""; w = h = 0; ce = 0; amode = 2
    else:
        entry = txd.get(texname or "") or next(iter(txd.values()))
        tw, th, rgba = entry
        t = psp_tex.author_psp_texture(rgba, tw, th, fmt="T8", mipmaps=False)
        texels, clut = t["texel_bytes"], t["clut_bytes"]
        w = t["width"]; h = t["height"]; ce = t["clut_entries"]; amode = t.get("alpha_mode", 0)

    nl_amode = (1 if glow else 1) & 0xFF
    buf = b"PRP1" + struct.pack("<6H2I", len(verts), len(idx), w, h,
                                (nl_amode) | ((amode & 3) << 8), ce,
                                len(texels), len(clut))
    for (u, v, c, x, y, z) in verts:
        buf += struct.pack("<2fI3f", u, v, c, x, y, z)
    for i in idx:
        buf += struct.pack("<H", i)
    buf += texels + clut
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "wb").write(buf)
    dep = "" + \
        os.path.basename(outp)
    try:
        open(dep, "wb").write(buf); d = "deployed"
    except OSError:
        d = ""
    print(f"{outp}: {len(buf)} bytes {d}")


if __name__ == "__main__":
    main()
