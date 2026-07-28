#!/usr/bin/env python3
"""grass_bake - bake GRASS collision triangles per region for CPlantMgr-lite.

SA seeds procedural grass on COL faces whose surface type is one of the six
grass materials (eSurfaceType 9..14). Our runtime world_col.bin dropped the
surface byte, so this tool goes back to the SA source: every world instance
(sa_source, text+binary IPL, conjugated quaternion like CFileLoader) whose
ColModel carries grass faces contributes world-space triangles, clipped into
the production region grid and written as region_X_Y.grass sidecars.

Format (little-endian):
  u32 magic 'GRS1' (0x31535247), u32 tri_count
  tri (20B): s16 x1,y1,z1,x2,y2,z2,x3,y3,z3 (world * 10, i.e. 0.1u quant)
             u8 surface (9..14), u8 pad
Runtime cost: ~KBs-100KB per region, loaded with the region like .nightd.

Usage: grass_bake.py <chunks_dir> [--only region_12_2]
"""
import math
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
import sa_col
from night_delta_bake import load_v2, HDR

GRASS_SURF = {9, 10, 11, 12, 13, 14}
MAGIC = 0x31535247  # 'GRS1'


def quat_to_matrix(qx, qy, qz, qw):
    # SA conjugates the IPL quaternion (CFileLoader::LoadObjectInstance)
    x, y, z, w = -qx, -qy, -qz, qw
    return (
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    )


def main():
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        k = argv.index("--only"); only = argv[k + 1]; del argv[k:k + 2]
    chunks_dir = argv[0]

    print("indexing COL library...", flush=True)
    img = sa_col.ImgArchive(sa_col.IMG)   # gta3.img via SA_GTA3_IMG env (like col_bake/dyn_sidecar/light); the chain sets it
    col_index, _libs = sa_col.build_index(img)

    print("loading instances...", flush=True)
    defs = sa_source.load_defs()
    simg = sa_source.open_img()
    insts = sa_source.load_instances(defs, simg)

    # region grid from the deployed pmaps: region world box per file
    regions = []
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.startswith("region_") or not fn.endswith(".pmap"):
            continue
        if only and not fn.startswith(only):
            continue
        blob = load_v2(os.path.join(chunks_dir, fn))
        h = HDR.unpack_from(blob, 0)
        ic, ioff = h[9], h[10]
        if ic == 0:
            continue
        xs = []; ys = []
        for i in range(0, ic, max(1, ic // 400)):
            _, px, py, pz = struct.unpack_from('<I3f', blob, ioff + 36 * i)
            xs.append(px); ys.append(py)
        regions.append([fn[:-5], min(xs) - 30, min(ys) - 30,
                        max(xs) + 30, max(ys) + 30, []])
    print(f"{len(regions)} regions", flush=True)

    col_cache = {}
    def col_of(name):
        key = name.lower()
        if key in col_cache:
            return col_cache[key]
        cm = None
        try:
            cm = col_index.get(key)
        except Exception:
            cm = None
        col_cache[key] = cm
        return cm

    tot = 0
    for inst in insts:
        cm = col_of(inst["name"])
        if cm is None or not cm.faces:
            continue
        gfaces = [f for f in cm.faces if f[3] in GRASS_SURF]
        if not gfaces:
            continue
        px, py, pz = inst["pos"]
        qx, qy, qz, qw = inst["quat"]
        m = quat_to_matrix(qx, qy, qz, qw)
        vs = cm.verts
        wv = {}
        def world(vi):
            if vi in wv:
                return wv[vi]
            x, y, z = vs[vi]
            out = (m[0]*x + m[1]*y + m[2]*z + px,
                   m[3]*x + m[4]*y + m[5]*z + py,
                   m[6]*x + m[7]*y + m[8]*z + pz)
            wv[vi] = out
            return out
        for (a, b, c, surf) in gfaces:
            wa, wb, wc = world(a), world(b), world(c)
            cx = (wa[0] + wb[0] + wc[0]) / 3.0
            cy = (wa[1] + wb[1] + wc[1]) / 3.0
            for r in regions:
                if r[1] <= cx <= r[3] and r[2] <= cy <= r[4]:
                    r[5].append((wa, wb, wc, surf))
                    tot += 1
                    break

    files = 0
    for r in regions:
        tris = r[5]
        out = os.path.join(chunks_dir, r[0] + ".grass")
        if not tris:
            if os.path.exists(out):
                os.remove(out)
            continue
        with open(out, "wb") as f:
            f.write(struct.pack('<II', MAGIC, len(tris)))
            for (wa, wb, wc, surf) in tris:
                f.write(struct.pack('<9hBB',
                    int(round(wa[0]*10)), int(round(wa[1]*10)), int(round(wa[2]*10)),
                    int(round(wb[0]*10)), int(round(wb[1]*10)), int(round(wb[2]*10)),
                    int(round(wc[0]*10)), int(round(wc[1]*10)), int(round(wc[2]*10)),
                    surf, 0))
        files += 1
        print(f"  {r[0]}: {len(tris)} grass tris ({(8+20*len(tris))//1024}KB)",
              flush=True)
    print(f"DONE: {tot} grass triangles over {files} regions")


if __name__ == "__main__":
    main()
