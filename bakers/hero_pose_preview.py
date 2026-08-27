#!/usr/bin/env python3
"""hero_pose_preview - what does a CLIP actually look like on our rig?

Poses hero.bin (or a streamed blk_*.bin clip) with the same maths SkinAnim.c uses - bind pose where a bone has no track, `local * parentWorld` down the hierarchy, skin =
invBind * boneWorld, linear blend over four influences - and draws the result. Two
clips can be put side by side, which is the point: a clip that "looks strange" is only
diagnosable against the one it replaced.

 GVCS_ROOT=... python tools/hero_pose_preview.py IDLE_STANCE muscleidle --phase 0.0

Clip names are resolved in hero.bin first, then in every blk_*.bin under --blocks.
"""
import argparse
import glob
import math
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import weapon_render as WR                       # png


# ---- readers -----------------------------------------------------------------------
def read_hero(path):
    d = open(path, "rb").read()
    hasNrm = d[:4] == b"HRO2"
    nb, nc, nv, ns, nt, up = struct.unpack_from("<6H", d, 4)
    o = 16
    bones = []
    for _ in range(nb):
        parent, nodeId = struct.unpack_from("<2h", d, o); o += 4
        q = struct.unpack_from("<4f", d, o); o += 16
        p = struct.unpack_from("<3f", d, o); o += 12
        inv = struct.unpack_from("<16f", d, o); o += 64
        bones.append({"parent": parent, "node": nodeId, "q": q, "p": p, "inv": inv})
    verts = []
    for _ in range(nv):
        pos = struct.unpack_from("<3f", d, o); o += 12
        o += 8 + 4                                    # uv, rgba
        bidx = struct.unpack_from("<4B", d, o); o += 4
        bw = struct.unpack_from("<4f", d, o); o += 16
        if hasNrm:
            o += 12
        verts.append((pos, bidx, bw))
    subs = []
    for _ in range(ns):
        tex, _pad, first, count = struct.unpack_from("<hHII", d, o); o += 12
        subs.append((first, count))
    nidx = max((f + c) for f, c in subs) if subs else 0
    idx = list(struct.unpack_from("<%dH" % nidx, d, o)); o += nidx * 2
    for _ in range(nt):                               # textures
        tw, th, nl, ce = struct.unpack_from("<4H", d, o); o += 8
        tl, cl = struct.unpack_from("<2I", d, o); o += 8
        o += tl + cl
    clips = read_clips(d, o, nc)
    return bones, verts, idx, clips


def read_clips(d, o, count):
    out = {}
    for _ in range(count):
        if o + 32 > len(d):
            break
        nm = d[o:o + 24].split(b"\0")[0].decode("latin-1"); o += 24
        dur, nt, _p = struct.unpack_from("<fHH", d, o); o += 8
        tracks = []
        for _t in range(nt):
            bone, hasT, _pad, nk = struct.unpack_from("<hBBH", d, o); o += 6
            keys = []
            for _k in range(nk):
                q = struct.unpack_from("<4h", d, o); o += 8
                t16 = struct.unpack_from("<h", d, o)[0]; o += 2
                tr = (0, 0, 0)
                if hasT:
                    tr = struct.unpack_from("<3h", d, o); o += 6
                keys.append((q, t16, tr))
            tracks.append({"bone": bone, "hasTrans": hasT, "keys": keys})
        out[nm.lower()] = {"name": nm, "dur": dur, "tracks": tracks}
    return out


def read_block(path):
    d = open(path, "rb").read()
    if d[:4] != b"ABLK":
        return {}
    _v, n, _r = struct.unpack_from("<HHI", d, 4)
    return read_clips(d, 12, n)


# ---- posing (mirrors SkinAnim.c) ---------------------------------------------------
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


def sample(track, t):
    """Nearest-key sample - enough to see a pose; the engine slerps between the same keys."""
    keys = track["keys"]
    best = keys[0]
    for k in keys:
        if k[1] / 60.0 <= t:
            best = k
        else:
            break
    q = tuple(c / 4096.0 for c in best[0])
    tr = tuple(c / 1024.0 for c in best[2]) if track["hasTrans"] else None
    return q, tr


def qmul(a, b):
    """a * b, xyzw - the same order SkinAnim.c's postconcat uses."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def torso_aim_q(q, yaw, pitch):
    """CPedIK::PointGunInDirection's two POSTCONCATs, byte-for-byte with SkinAnim.c:
 pitch about +Z (headAngle approximated 0), then yaw about +X, q <- q_extra * q."""
    hp, hy = pitch * 0.5, yaw * 0.5
    qe = (0.0, 0.0, math.sin(hp), math.cos(hp))          # pitch about +Z
    qy = (math.sin(hy), 0.0, 0.0, math.cos(hy))          # yaw about +X
    return qmul(qmul(qy, qe), q)


def pose(bones, clip, phase, torso=None, torso_node=3):
    """`torso` = (yawRad, pitchRad) applied to the bone whose nodeId is `torso_node`,
 exactly as the engine does - so a still image can answer 'does the aim actually
 move the gun?' without a build."""
    t = (clip["dur"] * phase) if clip else 0.0
    byb = {tr["bone"]: tr for tr in clip["tracks"]} if clip else {}
    W = []
    for b, bd in enumerate(bones):
        tr = byb.get(b)
        if tr is not None:
            q, p = sample(tr, t)
            if p is None:
                p = bd["p"]
        else:
            q, p = bd["q"], bd["p"]
        if torso is not None and bd.get("node", b) == torso_node:
            q = torso_aim_q(q, torso[0], torso[1])
        local = quat_pos_to_m4(q, p)
        par = bd["parent"]
        W.append(m4_mul(local, W[par]) if 0 <= par < b else local)
    return [m4_mul(list(bones[b]["inv"]), W[b]) for b in range(len(bones))]


def skin(verts, S):
    out = []
    for pos, bidx, bw in verts:
        x = y = z = 0.0
        for i in range(4):
            w = bw[i]
            if w <= 0.0:
                continue
            m = S[bidx[i]]
            x += w * (pos[0] * m[0] + pos[1] * m[4] + pos[2] * m[8] + m[12])
            y += w * (pos[0] * m[1] + pos[1] * m[5] + pos[2] * m[9] + m[13])
            z += w * (pos[0] * m[2] + pos[1] * m[6] + pos[2] * m[10] + m[14])
        out.append((x, y, z))
    return out


def draw(pts, idx, size, axis, lo, hi, rgb=(0x8a, 0x8f, 0x99)):
    ax, ay, adepth = axis
    span = max(hi[ax] - lo[ax], hi[ay] - lo[ay]) or 1.0
    sc = (size - 20) / span
    cx = (size - (hi[ax] - lo[ax]) * sc) * 0.5
    img = bytearray(size * size * 4)
    for i in range(size * size):
        img[i * 4:i * 4 + 4] = bytes((0x14, 0x16, 0x1c, 0xff))

    def proj(p):
        return (cx + (p[ax] - lo[ax]) * sc, size - 10 - (p[ay] - lo[ay]) * sc)

    tris = [(idx[i], idx[i + 1], idx[i + 2]) for i in range(0, len(idx) - 2, 3)]
    tris.sort(key=lambda t: -sum(pts[v][adepth] for v in t))
    for tri in tris:
        p = [proj(pts[v]) for v in tri]
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
                img[o:o + 4] = bytes(rgb) + b"\xff"
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--hero", default="assets_build/hero.bin")
    ap.add_argument("--blocks", default="assets_build/anim/blocks")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=280)
    ap.add_argument("--out", default="pose.png")
    ap.add_argument("--torso", default="",
                    help="apply the CPedIK gun aim as 'yawDeg,pitchDeg' to bone node 3, "
                         "the same postconcat the engine does. Repeat per clip with ';' "
                         "to compare aims of the SAME clip: --torso 0,0;0,40;40,0")
    a = ap.parse_args()

    bones, verts, idx, clips = read_hero(a.hero)
    for f in sorted(glob.glob(os.path.join(a.blocks, "blk_*.bin"))):
        for k, v in read_block(f).items():
            clips.setdefault(k, v)
    print("clips available: %d" % len(clips))

    torsos = [None]
    if a.torso:
        torsos = []
        for part in a.torso.split(";"):
            y, _, pch = part.partition(",")
            torsos.append((math.radians(float(y)), math.radians(float(pch or 0.0))))

    cells = []
    # one shared bounding box so the two poses are drawn at the SAME scale - a pose that
    # is subtly off looks fine when each frame is normalised on its own.
    allpts = []
    posed = []
    for name in a.clips:
        c = clips.get(name.lower())
        if c is None:
            sys.exit("clip %r not found" % name)
        pts = skin(verts, pose(bones, c, a.phase, torsos[len(posed) % len(torsos)]))
        posed.append((name, c, pts))
        allpts += pts
    lo = [min(p[i] for p in allpts) for i in range(3)]
    hi = [max(p[i] for p in allpts) for i in range(3)]
    for name, c, pts in posed:
        print("  %-22s dur=%.2f tracks=%2d  z=[%.3f,%.3f]"
              % (c["name"], c["dur"], len(c["tracks"]),
                 min(p[2] for p in pts), max(p[2] for p in pts)))
        for view in ((1, 2, 0), (0, 2, 1)):        # front (Y,Z) and side (X,Z)
            cells.append(draw(pts, idx, a.size, view, lo, hi))

    W, H = a.size * len(cells), a.size
    out = bytearray(W * H * 4)
    for n, im in enumerate(cells):
        for y in range(a.size):
            d = (y * W + n * a.size) * 4
            out[d:d + a.size * 4] = im[y * a.size * 4:(y + 1) * a.size * 4]
    WR.png(a.out, W, H, out)
    print("-> %s  (per clip: front, side)" % a.out)


if __name__ == "__main__":
    main()
