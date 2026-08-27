#!/usr/bin/env python3
"""weapon_render - draw a baked data/weapons/w<type>.bin the way the GE will.

An instrument, not part of the build. It reads the PRP1 exactly as CProp_Load does,
de-swizzles the T8 plane exactly as the GE does, and rasterises the triangles with
the SAME texture addressing the engine sets in prop_body(). Run it with --clamp and
with --repeat and the difference IS the bug: every weapon whose UVs leave [0,1]
collapses onto one row of texels under GU_CLAMP.

 GVCS_ROOT=... python tools/weapon_render.py 34 36 42 46 --out <dir> [--clamp]
"""
import argparse, os, struct, sys, zlib

G = os.environ.get("GVCS_ROOT", "")
if G:
    sys.path.insert(0, G); sys.path.insert(0, os.path.join(G, "gvcslib"))
from gvcslib import psp_tex


def png(path, w, h, px):
    raw = b"".join(b"\x00" + bytes(px[y * w * 4:(y + 1) * w * 4]) for y in range(h))
    def ch(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n"
                           + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                           + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b""))


def load_prp1(path):
    d = open(path, "rb").read()
    assert d[:4] == b"PRP1", path
    nv, ni, tw, th, la, ce = struct.unpack_from("<6H", d, 4)
    tl, cl = struct.unpack_from("<2I", d, 16)
    o = 24
    verts = [struct.unpack_from("<ffIfff", d, o + i * 24) for i in range(nv)]
    o += nv * 24
    idx = list(struct.unpack_from("<%dH" % ni, d, o)); o += ni * 2
    tex = {"gu_pixfmt": psp_tex.GU_PSM_T8, "gu_clutfmt": psp_tex.GU_PSM_8888,
           "width": tw, "height": th, "swizzle": 1, "clut_entries": ce,
           "num_levels": la & 0xff, "alpha_mode": (la >> 8) & 3,
           "texel_bytes": d[o:o + tl], "clut_bytes": d[o + tl:o + tl + cl]}
    return verts, idx, tex, psp_tex.decode_psp_texture(tex)


def render(verts, idx, tex, rgba, size=256, clamp=True, axis=(0, 2)):
    tw, th = tex["width"], tex["height"]
    ax, ay = axis
    pos = [(v[3], v[4], v[5]) for v in verts]
    lo = [min(p[i] for p in pos) for i in range(3)]
    hi = [max(p[i] for p in pos) for i in range(3)]
    span = max(hi[ax] - lo[ax], hi[ay] - lo[ay]) or 1.0
    sc = (size - 16) / span

    def proj(p):
        return (8 + (p[ax] - lo[ax]) * sc, size - 8 - (p[ay] - lo[ay]) * sc)

    img = bytearray(size * size * 4)
    for i in range(0, size * size):
        img[i * 4:i * 4 + 4] = b"\x20\x20\x28\xff"
    zbuf = [1e30] * (size * size)

    def sample(u, v):
        if clamp:
            x = min(tw - 1, max(0, int(u * tw)))
            y = min(th - 1, max(0, int(v * th)))
        else:
            x = int(u * tw) % tw
            y = int(v * th) % th
        s = (y * tw + x) * 4
        return rgba[s:s + 4]

    for t in range(0, len(idx) - 2, 3):
        tri = [verts[idx[t + k]] for k in range(3)]
        p = [proj((v[3], v[4], v[5])) for v in tri]
        depth = sum(v[3 + (3 - ax - ay)] for v in tri) / 3.0
        xs = [q[0] for q in p]; ys = [q[1] for q in p]
        x0, x1 = int(max(0, min(xs))), int(min(size - 1, max(xs)) + 1)
        y0, y1 = int(max(0, min(ys))), int(min(size - 1, max(ys)) + 1)
        d = ((p[1][1] - p[2][1]) * (p[0][0] - p[2][0])
             + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]))
        if abs(d) < 1e-9:
            continue
        for py in range(y0, y1):
            for px in range(x0, x1):
                fx, fy = px + 0.5, py + 0.5
                w0 = ((p[1][1] - p[2][1]) * (fx - p[2][0]) + (p[2][0] - p[1][0]) * (fy - p[2][1])) / d
                w1 = ((p[2][1] - p[0][1]) * (fx - p[2][0]) + (p[0][0] - p[2][0]) * (fy - p[2][1])) / d
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                o = py * size + px
                if depth >= zbuf[o]:
                    continue
                zbuf[o] = depth
                u = w0 * tri[0][0] + w1 * tri[1][0] + w2 * tri[2][0]
                v = w0 * tri[0][1] + w1 * tri[1][1] + w2 * tri[2][1]
                img[o * 4:o * 4 + 4] = sample(u, v)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("types", nargs="+", type=int)
    ap.add_argument("--dir", default="assets_build/weapons")
    ap.add_argument("--out", default=".")
    ap.add_argument("--clamp", action="store_true", help="GU_CLAMP (what prop_body does today)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--sheet", default="", help="write one contact sheet here instead")
    ap.add_argument("--cols", type=int, default=7)
    a = ap.parse_args()
    if a.sheet:
        cells, labels = [], []
        for t in a.types:
            src = os.path.join(a.dir, "w%d.bin" % t)
            if not os.path.isfile(src):
                continue
            verts, idx, tex, rgba = load_prp1(src)
            cells.append(render(verts, idx, tex, rgba, size=a.size, clamp=a.clamp))
            labels.append(t)
        cols = a.cols
        rows = (len(cells) + cols - 1) // cols
        W, H = cols * a.size, rows * a.size
        out = bytearray(W * H * 4)
        for i in range(W * H):
            out[i * 4:i * 4 + 4] = bytes((0x18, 0x18, 0x1e, 0xff))
        for n, img in enumerate(cells):
            cx, cy = (n % cols) * a.size, (n // cols) * a.size
            for y in range(a.size):
                d = ((cy + y) * W + cx) * 4
                out[d:d + a.size * 4] = img[y * a.size * 4:(y + 1) * a.size * 4]
        png(a.sheet, W, H, out)
        print("sheet: %d weapons %dx%d -> %s" % (len(cells), W, H, a.sheet))
        print("order: " + " ".join("w%d" % t for t in labels))
        return
    tag = "clamp" if a.clamp else "repeat"
    for t in a.types:
        p = os.path.join(a.dir, "w%d.bin" % t)
        verts, idx, tex, rgba = load_prp1(p)
        img = render(verts, idx, tex, rgba, size=a.size, clamp=a.clamp)
        dst = os.path.join(a.out, "render_w%d_%s.png" % (t, tag))
        png(dst, a.size, a.size, img)
        print("w%-3d %s -> %s" % (t, tag, dst))


if __name__ == "__main__":
    main()
