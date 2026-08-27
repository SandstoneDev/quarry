#!/usr/bin/env python3
"""jetpack_bake - the jetpack as THREE props, because its nozzles move.

pickup_bake writes p370.bin with merge=True, which welds every atomic into one mesh. That
is right for a pickup lying on the ground and wrong for a worn one: SA tilts the two nozzle
FRAMES independently (CTaskSimpleJetPack::RenderJetPack, 0x67F6A0) - by the thrust angle,
by opposite halves of the strafe, and in hover it walks them round a circle with a 500 ms
period - and a welded mesh has no frames left to tilt.

jetpack.dff carries exactly the three the original names:

 geometry 0 frame 'jetpack' 820 tris the body, at the origin
 geometry 1 frame 'jetball1' 98 tris left nozzle, offset (-0.387, -0.229, +0.175)
 geometry 2 frame 'jetball2' 98 tris right nozzle, offset (+0.387, -0.229, +0.175)

Rotation on both nozzle frames is identity, so the offsets above are the whole attachment
and the engine can rebuild the hierarchy from three flat props.

Output (same 'PRP1' container CProp_Load already reads):
 pickups/p370.bin the body
 pickups/p370_jb1.bin left nozzle
 pickups/p370_jb2.bin right nozzle
The nozzle offsets are emitted alongside as pickups/p370.ofs so the engine never has to
hardcode numbers this script measured.

Run: SA_ROOT=<extracted disc> python tools/jetpack_bake.py [--out data]
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutprops_bake
import ps2dff
from sa_img import SaImg

SA_ROOT = os.environ.get("SA_ROOT", "")
WANT = (("jetpack", "p370.bin"), ("jetball1", "p370_jb1.bin"), ("jetball2", "p370_jb2.bin"))


def emit(verts, idx, tex):
    buf = bytearray(b"PRP1")
    buf += struct.pack("<HHHHHH", len(verts), len(idx), tex["width"], tex["height"],
                       tex["num_levels"] | (tex.get("alpha_mode", 0) << 8),
                       tex["clut_entries"])
    buf += struct.pack("<II", len(tex["texel_bytes"]), len(tex["clut_bytes"]))
    for (u, v, c, x, y, z) in verts:
        buf += struct.pack("<ffIfff", u, v, c, x, y, z)
    for i in idx:
        buf += struct.pack("<H", i)
    buf += tex["texel_bytes"] + tex["clut_bytes"]
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--root", default=SA_ROOT)
    a = ap.parse_args()
    if not a.root:
        sys.exit("jetpack_bake: set SA_ROOT (or pass --root)")

    img = SaImg(os.path.join(a.root, "models", "gta3.img"))
    raw = img.extract("jetpack.dff")
    model = ps2dff.load_dff(bytes(raw))
    by_frame = {getattr(g, "frame_name", None): g for g in model.geometries}

    missing = [nm for nm, _ in WANT if nm not in by_frame]
    if missing:
        sys.exit("jetpack_bake: jetpack.dff has no frame(s) %s - it carries %s"
                 % (", ".join(missing), ", ".join(repr(k) for k in by_frame)))

    outdir = os.path.join(a.out, "pickups")
    os.makedirs(outdir, exist_ok=True)

    # One TXD decode shared by all three: they use the same texture.
    from gvcslib import psp_tex
    txd = {k.lower(): v for k, v in cutprops_bake._decode_txd(img.extract("jetpack.txd")).items()}

    offsets = {}
    for frame, fname in WANT:
        geo = by_frame[frame]
        texname, _n = cutprops_bake._geo_texture(geo)
        entry = txd.get(texname) or next(iter(txd.values()))
        tw, th, rgba = entry
        tex = psp_tex.author_psp_texture(rgba, tw, th, fmt="T8", mipmaps=False)

        # Bake the geometry IN ITS OWN FRAME (do not apply frame_ltm): the engine places the
        # nozzles by the offsets below, exactly as SA's hierarchy does, so baking the offset
        # into the vertices too would apply it twice.
        verts, vmap, idx = [], {}, []
        for (i0, i1, i2, _mat) in geo.tris:
            for vi_src in (i0, i1, i2):
                pos = geo.verts[vi_src]
                uv = geo.uvs[vi_src] if vi_src < len(geo.uvs) else (0.0, 0.0)
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                       round(uv[0], 5), round(uv[1], 5))
                j = vmap.get(key)
                if j is None:
                    j = len(verts); vmap[key] = j
                    verts.append((uv[0], uv[1], 0xFFFFFFFF, pos[0], pos[1], pos[2]))
                idx.append(j)

        ltm = getattr(geo, "frame_ltm", None)
        offsets[frame] = (0.0, 0.0, 0.0) if ltm is None else tuple(ltm[1])

        blob = emit(verts, idx, tex)
        with open(os.path.join(outdir, fname), "wb") as f:
            f.write(blob)
        print("  %-9s %-14s %4d vert %5d idx  %dx%d  %6d B  ofs %.3f %.3f %.3f"
              % (frame, fname, len(verts), len(idx), tex["width"], tex["height"], len(blob),
                 offsets[frame][0], offsets[frame][1], offsets[frame][2]))

    with open(os.path.join(outdir, "p370.ofs"), "wb") as f:
        f.write(b"JPOF")
        for frame in ("jetball1", "jetball2"):
            f.write(struct.pack("<3f", *offsets[frame]))
    print("jetpack_bake: 3 props + p370.ofs -> %s" % outdir)


if __name__ == "__main__":
    main()
