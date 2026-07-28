#!/usr/bin/env python3
"""pmap_to_obj - export a region_*.pmap (v2) as Wavefront OBJ + MTL + PNG
textures for inspection in Blender. Instances are baked to world space (SA Z-up,
which is Blender Z-up). LOD proxies (interior!=0) are skipped. Each pmap texture
becomes one material with map_Kd + map_d (alpha) so foliage/wire cutouts show.

Usage: pmap_to_obj.py <region.pmap> <out_dir>
"""
import os
import sys

import numpy as np
from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene
import tile_preview as TP


def main():
    pmap, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    sc = psp_scene.read_scene(open(pmap, "rb").read())

    used = sorted({sm.texture for md in sc.models for sm in md.submeshes
                   if 0 <= sm.texture < len(sc.textures)})
    for ti in used:
        rgba = TP.decode_tex(sc.textures[ti])          # HxWx4
        Image.fromarray(rgba, "RGBA").save(os.path.join(outdir, "t%d.png" % ti))
    with open(os.path.join(outdir, "grove.mtl"), "w") as m:
        for ti in used:
            m.write("newmtl t%d\nKa 1 1 1\nKd 1 1 1\nd 1\nillum 1\n"
                    "map_Kd t%d.png\nmap_d t%d.png\n\n" % (ti, ti, ti))

    obj = open(os.path.join(outdir, "grove.obj"), "w")
    obj.write("mtllib grove.mtl\n")
    vb = 1                                              # OBJ is 1-indexed
    nverts = ntris = 0
    for ii, inst in enumerate(sc.instances):
        if inst.interior:
            continue                                   # LOD proxy -> skip (double-draw)
        md = sc.models[inst.model]
        R = TP.quat_mat(*inst.quat)
        base = np.array(inst.pos, np.float32)
        ctr = np.array(md.center, np.float32)
        for si, sm in enumerate(md.submeshes):
            v = np.frombuffer(sm.vertex_bytes, np.uint8).reshape(-1, 12)
            uv = v[:, 0:4].view("<i2").astype(np.float32) / 4096.0
            pos = v[:, 6:12].view("<i2").astype(np.float32) * md.scale + ctr
            wpos = pos @ R + base                       # world space (Z-up)
            idx = np.frombuffer(sm.index_bytes, np.uint16)
            if len(wpos) == 0 or len(idx) < 3:
                continue
            obj.write("o inst%d_sm%d\n" % (ii, si))
            for p in wpos:
                obj.write("v %.3f %.3f %.3f\n" % (p[0], p[1], p[2]))
            for u, w in uv:
                obj.write("vt %.5f %.5f\n" % (u, 1.0 - w))
            obj.write("usemtl t%d\n" % sm.texture if sm.texture >= 0 else "usemtl none\n")
            for t in range(0, len(idx) - 2, 3):
                a, b, c = int(idx[t]) + vb, int(idx[t + 1]) + vb, int(idx[t + 2]) + vb
                obj.write("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, c, c))
                ntris += 1
            vb += len(wpos)
            nverts += len(wpos)
    obj.close()
    print("OBJ: %d verts, %d tris, %d textures -> %s/grove.obj"
          % (nverts, ntris, len(used), outdir))


if __name__ == "__main__":
    main()
