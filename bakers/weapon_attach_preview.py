#!/usr/bin/env python3
"""weapon_attach_preview - where does an attached weapon actually END UP on the ped?

The engine draws a held weapon at a bone matrix, and the only way that has ever been
checked is to boot and look. This does it on the host: it replays hero.bin's BIND pose
with the same maths SkinAnim.c uses (row-vector, `local * parentWorld`, the quaternion
laid out exactly as quat_pos_to_m4 lays it out), takes the bone the engine would take,
applies the same offset/rotation, and draws CJ and the weapon together from two views.

It exists because of the parachute: SA hangs it on bone 3 with a 90 deg turn about Y,
and RW pre-concats onto a row-vector matrix - the SIGN of that turn is the one thing a
reading of the disassembly does not hand you for free. Two silhouettes settle it.

 GVCS_ROOT=... python tools/weapon_attach_preview.py 46 --out preview.png
"""
import argparse
import math
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import weapon_render as WR                      # load_prp1 + png

SA_BONE_RIGHTHAND, SA_BONE_SPINE2 = 24, 3
WEAPON_PARACHUTE = 46


def load_hero(path):
    d = open(path, "rb").read()
    hasNrm = d[:4] == b"HRO2"
    nb, nc, nv, ns, nt, up = struct.unpack_from("<6H", d, 4)
    o = 16
    bones = []
    for _ in range(nb):
        parent, nodeId = struct.unpack_from("<2h", d, o); o += 4
        q = struct.unpack_from("<4f", d, o); o += 16
        p = struct.unpack_from("<3f", d, o); o += 12
        o += 64                                  # invBind, unused here
        bones.append((parent, nodeId, q, p))
    vstride = 12 + 8 + 4 + 4 + 16 + (12 if hasNrm else 0)
    verts = [struct.unpack_from("<3f", d, o + i * vstride) for i in range(nv)]
    o += nv * vstride
    subs = []
    for _ in range(ns):
        tex, _pad, first, count = struct.unpack_from("<hHII", d, o); o += 12
        subs.append((first, count))
    nidx = max((f + c) for f, c in subs) if subs else 0
    idx = list(struct.unpack_from("<%dH" % nidx, d, o))
    return bones, verts, idx


def quat_pos_to_m4(q, p):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n > 1e-8:
        s = 1.0 / math.sqrt(n); x *= s; y *= s; z *= s; w *= s
    return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0.0,
            2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0.0,
            2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0.0,
            p[0], p[1], p[2], 1.0]


def m4_mul(A, B):
    return [sum(A[r * 4 + k] * B[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def bind_world(bones):
    """s_world[b] at the bind pose - the same composition SkinAnim.c does."""
    W = []
    for b, (parent, _nid, q, p) in enumerate(bones):
        local = quat_pos_to_m4(q, p)
        W.append(m4_mul(local, W[parent]) if 0 <= parent < b else local)
    return W


def bone_mtx(bones, W, nodeId):
    for b, (_p, nid, _q, _t) in enumerate(bones):
        if nid == nodeId:
            M = W[b]
            return ([M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]],
                    [M[12], M[13], M[14]])
    return None, None


def attach(R, T, weaponType, spin_sign):
    """Exactly what CWeaponModels_DrawInHand does, with the 90 deg sign switchable."""
    if weaponType != WEAPON_PARACHUTE:
        return R, T
    ox, oy = 0.1, -0.15                                   # ELF 0x5F9548
    T = [T[0] + ox * R[0] + oy * R[3],
         T[1] + ox * R[1] + oy * R[4],
         T[2] + ox * R[2] + oy * R[5]]
    x0, x1, x2 = R[0], R[1], R[2]
    s = spin_sign
    R = list(R)
    R[0] = -s * R[6]; R[1] = -s * R[7]; R[2] = -s * R[8]
    R[6] = s * x0;    R[7] = s * x1;    R[8] = s * x2
    return R, T


def draw(tris, size, axis, tex=None, flip=False):
    """Painter-algorithm view. tris = [((p0,p1,p2), rgb_or_uvs)].

 A triangle whose colour slot holds three (u, v) pairs is TEXTURED from `tex` - that is what makes the weapon's own orientation readable (the parachute's blue
 label has to face away from the back, and a flat silhouette cannot show that).
 """
    ax, ay, adepth = axis
    pts = [p for t, _ in tris for p in t]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    span = max(hi[ax] - lo[ax], hi[ay] - lo[ay]) or 1.0
    sc = (size - 20) / span
    cx = (size - (hi[ax] - lo[ax]) * sc) * 0.5

    img = bytearray(size * size * 4)
    for i in range(size * size):
        img[i * 4:i * 4 + 4] = bytes((0x14, 0x16, 0x1c, 0xff))

    def proj(p):
        u = (p[ax] - lo[ax]) * sc
        if flip:
            u = (hi[ax] - lo[ax]) * sc - u
        return (cx + u, size - 10 - (p[ay] - lo[ay]) * sc)

    sgn = 1.0 if flip else -1.0
    order = sorted(range(len(tris)),
                   key=lambda i: sgn * sum(tris[i][0][k][adepth] for k in range(3)))
    for i in order:
        tri, rgb = tris[i]
        p = [proj(v) for v in tri]
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
                if w0 < 0 or w1 < 0 or 1.0 - w0 - w1 < 0:
                    continue
                o = (py * size + px) * 4
                if tex is not None and isinstance(rgb[0], tuple):
                    w2 = 1.0 - w0 - w1
                    u = w0 * rgb[0][0] + w1 * rgb[1][0] + w2 * rgb[2][0]
                    v = w0 * rgb[0][1] + w1 * rgb[1][1] + w2 * rgb[2][1]
                    tw, th, px8 = tex
                    s2 = ((int(v * th) % th) * tw + (int(u * tw) % tw)) * 4
                    img[o:o + 4] = px8[s2:s2 + 3] + b"\xff"
                else:
                    img[o:o + 4] = bytes(rgb) + b"\xff"
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("types", nargs="*", type=int, default=[WEAPON_PARACHUTE])
    ap.add_argument("--hero", default="assets_build/hero.bin")
    ap.add_argument("--dir", default="assets_build/weapons")
    ap.add_argument("--out", default="attach.png")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--sign", type=int, default=1, help="sign of the 90 deg Y turn")
    a = ap.parse_args()

    bones, hverts, hidx = load_hero(a.hero)
    W = bind_world(bones)

    for wt in a.types:
        node = SA_BONE_SPINE2 if wt == WEAPON_PARACHUTE else SA_BONE_RIGHTHAND
        R, T = bone_mtx(bones, W, node)
        if R is None:
            sys.exit("nodeId %d not in %s" % (node, a.hero))
        print("w%d -> bone nodeId %d  T=(%.3f, %.3f, %.3f)" % (wt, node, T[0], T[1], T[2]))
        R, T = attach(R, T, wt, a.sign)

        wverts, widx, wtex, wrgba = WR.load_prp1(os.path.join(a.dir, "w%d.bin" % wt))
        tex = (wtex["width"], wtex["height"], wrgba)
        tris = []
        for i in range(0, len(hidx) - 2, 3):
            tri = tuple(hverts[hidx[i + k]] for k in range(3))
            tris.append((tri, (0x8a, 0x8f, 0x99)))
        for i in range(0, len(widx) - 2, 3):
            tri = []
            uvs = []
            for k in range(3):
                v = wverts[widx[i + k]]
                px, py, pz = v[3], v[4], v[5]
                uvs.append((v[0], v[1]))
                tri.append((px * R[0] + py * R[3] + pz * R[6] + T[0],
                            px * R[1] + py * R[4] + pz * R[7] + T[1],
                            px * R[2] + py * R[5] + pz * R[8] + T[2]))
            tris.append((tuple(tri), tuple(uvs)))

        # X is the ped's FORWARD axis in this rig and Y is the side axis, so:
        # (X,Z) is the side view and (Y,Z) is the front/back view.
        # Third entry looks at the BACK (mirrored X,Z) - that is where the pack's
        # label must be readable if the 90 deg turn has the right sign.
        views = [((0, 2, 1), False), ((1, 2, 0), False), ((1, 2, 0), True)]
        imgs = [draw(tris, a.size, v, tex, fl) for v, fl in views]
        Wd, H = a.size * len(imgs), a.size
        out = bytearray(Wd * H * 4)
        for n, im in enumerate(imgs):
            for y in range(a.size):
                d = (y * Wd + n * a.size) * 4
                out[d:d + a.size * 4] = im[y * a.size * 4:(y + 1) * a.size * 4]
        dst = a.out if len(a.types) == 1 else a.out.replace(".png", "_w%d.png" % wt)
        WR.png(dst, Wd, H, out)
        print("  -> %s  (left = front view along +Y, right = side view, +Y to the right)" % dst)


if __name__ == "__main__":
    main()
