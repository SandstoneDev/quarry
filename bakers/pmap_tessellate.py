#!/usr/bin/env python3
"""Tessellate large triangles in a .pmap so no triangle exceeds THRESHOLD world
units per edge. Fixes the PSP Allegrex GE guard-band hole bug (#1): the GE drops
an entire triangle when any vertex leaves the guard band, so big near tiles vanish
when the camera rotates.

Only the vertex + index pools and the submesh table are rebuilt; models, textures,
instances, grid and the (huge) texel/clut pools are copied verbatim. Subdivision is
crack-free 1->4 with a per-submesh edge-midpoint cache (an edge shared by two tris
gets a single shared midpoint -> no T-junctions inside a submesh). Indices stay
submesh-local; per-model submesh contiguity is preserved so pmap.c's
vstart..vend model range stays valid.

Usage: python pmap_tessellate.py [in.pmap] [out.pmap] [threshold_units] [night_in] [night_out]

If night_in (a raw u16-5551-per-vertex stream aligned to in.pmap, e.g. night_pre.bin from
night_bake.py) is given, a PARALLEL night stream is carried through the SAME 1->4 subdivision
+ avg5551 for midpoints and written to night_out, aligned to out.pmap's vertex pool.
"""
import struct
import sys

IN  = ""
OUT = ""
THRESHOLD = 24.0
U16MAX = 65535


def avg5551(c1, c2):
    r = ((c1 & 31) + (c2 & 31)) >> 1
    g = (((c1 >> 5) & 31) + ((c2 >> 5) & 31)) >> 1
    b = (((c1 >> 10) & 31) + ((c2 >> 10) & 31)) >> 1
    a = ((c1 >> 15) + (c2 >> 15)) >> 1
    return (a << 15) | (b << 10) | (g << 5) | r


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else IN
    dst = sys.argv[2] if len(sys.argv) > 2 else OUT
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else THRESHOLD
    night_in  = sys.argv[4] if len(sys.argv) > 4 else None
    night_out = sys.argv[5] if len(sys.argv) > 5 else None
    blob = bytearray(open(src, "rb").read())

    night = None                 # u16 5551 per ORIGINAL vertex, aligned to src
    if night_in:
        nb = open(night_in, "rb").read()
        night = list(struct.unpack("<%dH" % (len(nb) // 2), nb))

    H = list(struct.unpack_from("<20I", blob, 0))
    (magic, ver, fsize, mcount, moff, scount, soff, tcount, toff,
     icount, ioff, grid_off, voff, vbytes, idxoff, idxbytes,
     texoff, texbytes, clutoff, clutbytes) = H
    assert magic == 0x50414D50

    models = [list(struct.unpack_from("<II f ffff f", blob, moff + i*32)) for i in range(mcount)]
    subs   = [list(struct.unpack_from("<iIIII", blob, soff + i*20)) for i in range(scount)]
    smesh_scale = [1.0] * scount
    for (fs, sc, scale, cx, cy, cz, rad, dd) in models:
        for s in range(fs, min(fs+sc, scount)):
            smesh_scale[s] = scale

    new_v = bytearray()      # rebuilt vertex pool
    new_i = bytearray()      # rebuilt index pool (u16, submesh-local)
    new_night = bytearray() if night is not None else None   # parallel night stream
    vcursor = 0              # running vertex count (pool units)
    icursor = 0              # running index count
    split_tris = 0
    tris_in = 0
    tris_out = 0
    over_warn = 0

    for si in range(scount):
        tex, vf, vc, if_, ic = subs[si]
        scale = smesh_scale[si]
        thr_q2 = (thr / scale) ** 2 if scale > 0 else 1e30

        # load submesh vertices (mutable list of [u,v,color,x,y,z]) and tris
        verts = []
        for v in range(vc):
            u, vv, col, x, y, z = struct.unpack_from("<hhHhhh", blob, voff + (vf+v)*12)
            verts.append([u, vv, col, x, y, z])
        # parallel night colour per submesh vertex (vf+v indexes the ORIGINAL pool)
        nverts = [night[vf+v] for v in range(vc)] if night is not None else None
        tris = []
        for t in range(0, ic, 3):
            io = idxoff + (if_ + t) * 2
            tris.append(struct.unpack_from("<3H", blob, io))
        tris_in += len(tris)

        midcache = {}
        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            m = midcache.get(key)
            if m is not None:
                return m
            va, vb = verts[a], verts[b]
            verts.append([(va[0]+vb[0])//2, (va[1]+vb[1])//2, avg5551(va[2], vb[2]),
                          (va[3]+vb[3])//2, (va[4]+vb[4])//2, (va[5]+vb[5])//2])
            if nverts is not None:                       # midpoint night = avg of edge ends
                nverts.append(avg5551(nverts[a], nverts[b]))
            m = len(verts) - 1
            midcache[key] = m
            return m

        def big(a, b):
            dx = verts[a][3]-verts[b][3]; dy = verts[a][4]-verts[b][4]; dz = verts[a][5]-verts[b][5]
            return (dx*dx + dy*dy + dz*dz) > thr_q2

        out = []
        stack = list(tris)
        while stack:
            a, b, c = stack.pop()
            if big(a, b) or big(b, c) or big(c, a):
                mab = midpoint(a, b); mbc = midpoint(b, c); mca = midpoint(c, a)
                stack.append((a, mab, mca)); stack.append((mab, b, mbc))
                stack.append((mca, mbc, c)); stack.append((mab, mbc, mca))
                split_tris += 1
            else:
                out.append((a, b, c))
        tris_out += len(out)

        if len(verts) > U16MAX:
            over_warn += 1   # would overflow u16 local indices; left as-is would corrupt

        # emit submesh verts + local indices (+ parallel night, same vertex order)
        for vtx in verts:
            new_v += struct.pack("<hhHhhh", vtx[0], vtx[1], vtx[2] & 0xFFFF, vtx[3], vtx[4], vtx[5])
        if nverts is not None:
            for nc in nverts:
                new_night += struct.pack("<H", nc & 0xFFFF)
        for (a, b, c) in out:
            new_i += struct.pack("<3H", a, b, c)

        subs[si][1] = vcursor          # vertex_first
        subs[si][2] = len(verts)       # vertex_count
        subs[si][3] = icursor          # index_first
        subs[si][4] = len(out) * 3     # index_count
        vcursor += len(verts)
        icursor += len(out) * 3

    # pad pools to 16B alignment (matches gvcslib layout discipline)
    def pad16(ba):
        while len(ba) % 16: ba += b"\x00"
        return ba
    new_v = pad16(new_v); new_i = pad16(new_i)
    texels = bytes(blob[texoff:texoff+texbytes])
    cluts  = bytes(blob[clutoff:clutoff+clutbytes])

    # rebuild index prefix (header + tables + grid) - same sizes, then new pools.
    prefix = bytearray(blob[0:voff])          # everything before old vertex pool
    # patch submesh table inside prefix
    for i in range(scount):
        struct.pack_into("<iIIII", prefix, soff + i*20, *subs[i])
    # new offsets
    new_voff = len(prefix)
    new_ioff = new_voff + len(new_v)
    new_texoff = new_ioff + len(new_i)
    new_clutoff = new_texoff + len(texels)
    new_fsize = new_clutoff + len(cluts)

    out = bytearray()
    out += prefix
    out += new_v
    out += new_i
    out += texels
    out += cluts
    # patch header
    struct.pack_into("<I", out, 8, new_fsize)          # file_size
    struct.pack_into("<I", out, 48, new_voff)          # vertex_off (idx 12)
    struct.pack_into("<I", out, 52, len(new_v))        # vertex_bytes
    struct.pack_into("<I", out, 56, new_ioff)          # index_off
    struct.pack_into("<I", out, 60, len(new_i))        # index_bytes
    struct.pack_into("<I", out, 64, new_texoff)        # texel_off
    struct.pack_into("<I", out, 68, len(texels))       # texel_bytes
    struct.pack_into("<I", out, 72, new_clutoff)       # clut_off
    struct.pack_into("<I", out, 76, len(cluts))        # clut_bytes

    open(dst, "wb").write(out)
    if new_night is not None and night_out:
        open(night_out, "wb").write(new_night)
        nv_out = len(new_v) // 12
        print("night stream: %d -> %d verts (%d bytes) -> %s  [%s]"
              % (len(night), len(new_night)//2, len(new_night), night_out,
                 "ALIGNED" if len(new_night)//2 == nv_out else "MISALIGNED!"))
    print("tessellate thr=%.0fu : tris %d -> %d (split ops %d)" % (thr, tris_in, tris_out, split_tris))
    print("vertex pool %.1fMB -> %.1fMB   index pool %.1fMB -> %.1fMB"
          % (vbytes/1e6, len(new_v)/1e6, idxbytes/1e6, len(new_i)/1e6))
    print("file %.1fMB -> %.1fMB   wrote %s" % (len(blob)/1e6, new_fsize/1e6, dst))
    if over_warn:
        print("  !! %d submeshes exceeded 65535 verts (u16 index overflow) - lower not split or raise thr" % over_warn)


if __name__ == "__main__":
    main()
