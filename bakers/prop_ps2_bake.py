#!/usr/bin/env python3
"""prop_ps2_bake - bake one small static prop from a PS2 disc into a PRP1 blob.

CProp (game_sa/Prop.c) draws a self-contained mesh with a single texture: the money
stack a dead ped drops, the save-point icon. The PC-sourced prop_bake.py cannot read
a PS2 disc, so this is the same job over ps2dff + the PS2 TXD codec.

A prop may use several materials (pickupsave has a metal disk and a logo face) while
CProp holds ONE texture, so the material carrying the most triangles wins and the
others reuse it - their UVs still address their own islands of the sheet.

PRP1 (little-endian), exactly what CProp_Load reads:
  'PRP1' u16 nVert, nIdx, texW, texH, levels|amode<<8, clutEntries
         u32 texelLen, clutLen
  vert[nVert]: f32 u, v; u32 colour (bytes R,G,B,A); f32 x, y, z
  idx[nIdx]:   u16
  texels (swizzled T8, all mip levels), clut (RGBA8888)

Usage: prop_ps2_bake.py <dff-name> <txd-name> <out.bin> [amode]
       gta3.img comes from SA_GTA3_IMG, else SA_ROOT/MODELS/gta3.img
"""
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))

import ps2dff
from gvcslib import sa_txd, psp_tex


def open_img():
    p = os.environ.get("SA_GTA3_IMG", "")
    if not p:
        p = os.path.join(os.environ.get("SA_ROOT", ""), "MODELS", "gta3.img")
    return p


def img_index(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:4] != b"VER2":
            raise SystemExit("%s is not a VER2 archive" % path)
        n = struct.unpack_from("<I", head, 4)[0]
        f.seek(0)
        hdr = f.read(8 + 32 * n)
    ents = {}
    for i in range(n):
        o = 8 + 32 * i
        off, sz, _ = struct.unpack_from("<IHH", hdr, o)
        nm = hdr[o + 8:o + 32].split(b"\0")[0].decode("ascii", "replace").lower()
        ents[nm] = (off * 2048, sz * 2048)
    return ents


def read_entry(path, ents, name):
    key = name.lower()
    if key not in ents:
        raise SystemExit("%s not in the archive" % name)
    off, sz = ents[key]
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(sz)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    dffname, txdname, out = sys.argv[1], sys.argv[2], sys.argv[3]
    amode = int(sys.argv[4]) if len(sys.argv) > 4 else 1        # 1 = alpha-test cutout

    img = open_img()
    if not os.path.exists(img):
        print("prop_ps2_bake: no archive at %s - skipped" % img)
        return 1
    ents = img_index(img)

    model = ps2dff.decode_sa(read_entry(img, ents, dffname + ".dff"))
    txd = {k.lower(): v for k, v in
           sa_txd.decode(read_entry(img, ents, txdname + ".txd")).items()}

    # the material with the most triangles owns the prop's single texture
    per_mat = {}
    for me in model.meshes:
        per_mat[me.material_index] = per_mat.get(me.material_index, 0) + len(me.triangles)
    if not per_mat:
        print("prop_ps2_bake: %s has no geometry" % dffname)
        return 1
    main_mat = max(per_mat, key=per_mat.get)
    texname = model.materials[main_mat]["texture_name"].lower()
    if texname not in txd:
        print("prop_ps2_bake: %s.txd has no %s" % (txdname, texname))
        return 1
    tw, th, rgba = txd[texname]
    tex = psp_tex.author_psp_texture(rgba, tw, th, fmt="T8", mipmaps=False)

    verts, idx, vmap = [], [], {}
    for me in model.meshes:
        for tri in me.triangles:
            for li in tri:
                p = me.positions[li]
                uv = me.uv[li] if li < len(me.uv) else (0.0, 0.0)
                c = me.colors[li] if li < len(me.colors) else 0xFFFFFFFF
                r, g, b, a = (c >> 24) & 0xFF, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF
                col = (a << 24) | (b << 16) | (g << 8) | r      # GE byte order R,G,B,A
                key = (round(p[0], 4), round(p[1], 4), round(p[2], 4),
                       round(uv[0], 5), round(uv[1], 5), col)
                vi = vmap.get(key)
                if vi is None:
                    vi = len(verts)
                    vmap[key] = vi
                    verts.append((uv[0], uv[1], col, p[0], p[1], p[2]))
                idx.append(vi)

    if len(verts) > 65535 or len(idx) > 65535:
        print("prop_ps2_bake: %s is too big for a prop (%d verts, %d indices)"
              % (dffname, len(verts), len(idx)))
        return 1

    blob = b"PRP1" + struct.pack("<6H2I", len(verts), len(idx),
                                 tex["width"], tex["height"],
                                 (tex["num_levels"] & 0xFF) | ((amode & 3) << 8),
                                 tex["clut_entries"],
                                 len(tex["texel_bytes"]), len(tex["clut_bytes"]))
    for u, v, c, x, y, z in verts:
        blob += struct.pack("<2fI3f", u, v, c, x, y, z)
    blob += struct.pack("<%dH" % len(idx), *idx)
    blob += tex["texel_bytes"] + tex["clut_bytes"]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "wb").write(blob)
    print("%s: %d verts %d tris, tex %s %dx%d -> %d bytes"
          % (os.path.basename(out), len(verts), len(idx) // 3, texname,
             tex["width"], tex["height"], len(blob)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
