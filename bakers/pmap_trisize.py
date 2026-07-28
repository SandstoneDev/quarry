#!/usr/bin/env python3
"""Measure world-space triangle sizes in input.pmap to diagnose the PSP guard-band
hole bug (#1): the Allegrex GE drops an ENTIRE triangle if any vertex leaves the
guard band, so large near tiles vanish when the camera rotates. If the map has
many large triangles, tessellation is required.

vertex (PmapVertex, 12B): s16 u,v | u16 color | s16 x,y,z
world edge length = |pos_i16_a - pos_i16_b| * model.scale   (center cancels)
"""
import struct
import sys

PMAP = ""


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else PMAP
    blob = open(p, "rb").read()
    (magic, ver, fsize, mcount, moff, scount, soff, tcount, toff,
     icount, ioff, grid_off, voff, vbytes, idxoff, idxbytes,
     texoff, texbytes, clutoff, clutbytes) = struct.unpack_from("<20I", blob, 0)
    assert magic == 0x50414D50, "bad magic %08x" % magic
    print("pmap v%d  models=%d submeshes=%d verts=%dB idx=%dB" % (ver, mcount, scount, vbytes, idxbytes))

    # models: first_submesh,submesh_count,scale,cx,cy,cz,radius,draw_dist (8*4=32B)
    models = [struct.unpack_from("<II f ffff f", blob, moff + i*32) for i in range(mcount)]
    # submesh -> model scale
    smesh_scale = [1.0] * scount
    for (fs, sc, scale, cx, cy, cz, rad, dd) in models:
        for s in range(fs, min(fs+sc, scount)):
            smesh_scale[s] = scale
    # submeshes: tex,vfirst,vcount,ifirst,icount (5*4=20B)
    SUB = [struct.unpack_from("<iIIII", blob, soff + i*20) for i in range(scount)]

    VBASE = voff       # vertex pool byte base; vertex i at VBASE + i*12, +6 = x,y,z s16
    IBASE = idxoff

    buckets = {10: 0, 20: 0, 40: 0, 80: 0, 160: 0}
    total_tris = 0
    biggest = []   # (edge, smesh, model_scale)
    for si, (tex, vf, vc, if_, ic) in enumerate(SUB):
        scale = smesh_scale[si]
        # read this submesh's indices, vertices positions
        # indices are submesh-local (relative to vf)
        for t in range(0, ic, 3):
            io = IBASE + (if_ + t) * 2
            a, b, c = struct.unpack_from("<3H", blob, io)
            pa = struct.unpack_from("<3h", blob, VBASE + (vf+a)*12 + 6)
            pb = struct.unpack_from("<3h", blob, VBASE + (vf+b)*12 + 6)
            pc = struct.unpack_from("<3h", blob, VBASE + (vf+c)*12 + 6)
            def elen(p, q):
                dx=(p[0]-q[0])*scale; dy=(p[1]-q[1])*scale; dz=(p[2]-q[2])*scale
                return (dx*dx+dy*dy+dz*dz) ** 0.5
            e = max(elen(pa,pb), elen(pb,pc), elen(pc,pa))
            total_tris += 1
            for thr in buckets:
                if e > thr: buckets[thr] += 1
            if len(biggest) < 20:
                biggest.append((e, si, scale))
            elif e > biggest[-1][0]:
                biggest[-1] = (e, si, scale); biggest.sort(reverse=True)

    print("total triangles: %d" % total_tris)
    print("triangles with max-edge >  N world units (a 480px screen ~ object spanning ~tens of units up close):")
    for thr in sorted(buckets):
        pct = 100.0*buckets[thr]/total_tris if total_tris else 0
        print("   > %3d u : %8d  (%.2f%%)" % (thr, buckets[thr], pct))
    biggest.sort(reverse=True)
    print("largest 10 triangle edges (world units):")
    for (e, si, sc) in biggest[:10]:
        print("   edge=%.1f u  submesh=%d scale=%.4f" % (e, si, sc))


if __name__ == "__main__":
    main()
