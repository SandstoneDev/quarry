#!/usr/bin/env python3
"""pmap_lz4.py - compress a v2 .pmap region tile into a v3 (lz4-streamed) tile.

The streaming loader (platform_psp/pmap.c) reads each model's geometry blob
(verts||indices) and each texture's blob (texels||clut) on demand off the Memory
Stick. v2 stores those pools raw; v3 stores them as per-model / per-texture LZ4
blocks, so the disk read is smaller (faster streaming, shorter strmq backlog) and
the bg thread inflates each blob into the cache buffer.

v3 = v2 with:
 * header version 3, +3 u32 at the end: comp_flag, comp_model_off, comp_tex_off
 * the resident prefix (header..grid+cell tables..instances) copied verbatim,
 every header *_off shifted by the 12-byte header growth
 * two comp tables in the resident prefix: PmapComp{u32 off; u32 csize}
 [model_count] and [texture_count]; off = absolute file offset of the blob
 * the raw vertex/index/texel/clut pools REPLACED by the concatenated LZ4 blobs
 * vertex_off = start of the compressed-blob region (== end of resident prefix)

Decompressed sizes are NOT stored: the loader recomputes them exactly as v2 did
(model span from its submeshes; texture from texel_bytes + clut_entries*4).

Usage:
 python pmap_lz4.py <in_v2.pmap> <out_v3.pmap>
 python pmap_lz4.py --dir <data_dir> # compress every region_*.pmap in place (.pmap)
"""
import os
import sys
import struct
import glob

import lz4.block

HDR_V2 = 80          # 20 u32
HDR_V3 = 92          # +3 u32 (comp_flag, comp_model_off, comp_tex_off)
MODEL_STRIDE   = 32  # first_submesh,submesh_count,scale,cx,cy,cz,bound_r,draw_dist
SUBMESH_STRIDE = 20  # texture(i32),vfirst,vcount,ifirst,icount
TEX_STRIDE     = 32  # w(u16),h(u16),format,texel_first,texel_bytes,bufw,clut_first,clut_entries,num_levels


def align(n, a=16):
    return (n + a - 1) & ~(a - 1)


def compress(path_in, path_out):
    data = open(path_in, "rb").read()
    h = struct.unpack_from("<20I", data, 0)
    (magic, version, file_size,
     model_count, model_off,
     submesh_count, submesh_off,
     texture_count, texture_off,
     instance_count, instance_off,
     grid_off,
     vertex_off, vertex_bytes,
     index_off, index_bytes,
     texel_off, texel_bytes,
     clut_off, clut_bytes) = h

    if magic != 0x50414D50:
        raise SystemExit("%s: bad magic" % path_in)
    if version == 3:
        print("  %s already v3, skip" % os.path.basename(path_in))
        return False
    if version != 2:
        raise SystemExit("%s: unexpected version %d" % (path_in, version))

    verts   = data[vertex_off:vertex_off + vertex_bytes]
    indices = data[index_off:index_off + index_bytes]
    texels  = data[texel_off:texel_off + texel_bytes]
    clut    = data[clut_off:clut_off + clut_bytes]

    submeshes = [struct.unpack_from("<i4I", data, submesh_off + i * SUBMESH_STRIDE)
                 for i in range(submesh_count)]

    # ---- build per-model compressed blobs (verts||indices), loader span rules ----
    model_blobs = []
    max_need = 0
    for mi in range(model_count):
        first, scount = struct.unpack_from("<2I", data, model_off + mi * MODEL_STRIDE)
        if scount == 0:
            model_blobs.append(b"")
            continue
        s0 = submeshes[first]
        sN = submeshes[first + scount - 1]
        vstart = s0[1]; istart = s0[3]
        vbytes = (sN[1] + sN[2] - vstart) * 12
        ibytes = (sN[3] + sN[4] - istart) * 2
        blob = verts[vstart * 12: vstart * 12 + vbytes] + indices[istart * 2: istart * 2 + ibytes]
        max_need = max(max_need, len(blob))
        model_blobs.append(lz4.block.compress(blob, mode="high_compression", store_size=False))

    # ---- per-texture compressed blobs (texels||clut) ----
    tex_blobs = []
    for ti in range(texture_count):
        w, ht, fmt, tfirst, tbytes, bufw, cfirst, centries, nlev = \
            struct.unpack_from("<HHIIIIIII", data, texture_off + ti * TEX_STRIDE)
        blob = texels[tfirst:tfirst + tbytes] + clut[cfirst:cfirst + centries * 4]
        max_need = max(max_need, len(blob))
        tex_blobs.append(lz4.block.compress(blob, mode="high_compression", store_size=False))

    # ---- assemble v3 ----
    # body = everything after the header in the prefix, copied verbatim (tables,
    # instances, grid, cell lists). Offsets are absolute -> shift by +12.
    body = data[HDR_V2:vertex_off]
    comp_model_off = HDR_V3 + len(body)
    comp_tex_off   = comp_model_off + model_count * 8
    blobs_off      = align(comp_tex_off + texture_count * 8)

    comp_model_tbl = bytearray()
    comp_tex_tbl   = bytearray()
    blob_region    = bytearray()
    cur = blobs_off
    for b in model_blobs:
        if b:
            comp_model_tbl += struct.pack("<II", cur, len(b))
            blob_region += b
            cur += len(b)
        else:
            comp_model_tbl += struct.pack("<II", 0, 0)
    for b in tex_blobs:
        if b:
            comp_tex_tbl += struct.pack("<II", cur, len(b))
            blob_region += b
            cur += len(b)
        else:
            comp_tex_tbl += struct.pack("<II", 0, 0)

    new_vertex_off = blobs_off
    total = blobs_off + len(blob_region)

    out = bytearray(blobs_off)
    # header v3
    struct.pack_into("<20I", out, 0,
                     magic, 3, total,
                     model_count, model_off + 12,
                     submesh_count, submesh_off + 12,
                     texture_count, texture_off + 12,
                     instance_count, instance_off + 12,
                     grid_off + 12,
                     new_vertex_off, vertex_bytes,  # vertex_off repurposed = blob start; *_bytes kept (night uses vertex_bytes)
                     0, index_bytes,      # index_off unused (raw pool gone), index_bytes kept
                     0, texel_bytes,      # texel_off unused, texel_bytes kept
                     0, clut_bytes)       # clut_off unused, clut_bytes kept
    struct.pack_into("<3I", out, HDR_V2, 1, comp_model_off, comp_tex_off)  # comp_flag, tables
    out[HDR_V3:HDR_V3 + len(body)] = body
    out[comp_model_off:comp_model_off + len(comp_model_tbl)] = comp_model_tbl
    out[comp_tex_off:comp_tex_off + len(comp_tex_tbl)] = comp_tex_tbl
    out += blob_region

    open(path_out, "wb").write(out)
    raw_pools = vertex_bytes + index_bytes + texel_bytes + clut_bytes
    print("  %s  %d -> %d KB  (pools %d->%d KB, max_need %d)" % (
        os.path.basename(path_out), len(data) >> 10, len(out) >> 10,
        raw_pools >> 10, len(blob_region) >> 10, max_need))
    if new_vertex_off > 320 * 1024:
        print("  WARN: resident prefix %d > PMAP_REGION_INDEX_MAX 320KB" % new_vertex_off)
    return max_need


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args[0] == "--dir":
        d = args[1]
        files = sorted(glob.glob(os.path.join(d, "region_*.pmap")))
        if not files:
            print("no region_*.pmap in", d); return 1
        gmax = 0
        for f in files:
            tmp = f + ".v3tmp"
            r = compress(f, tmp)
            if r is False:
                if os.path.exists(tmp):
                    os.remove(tmp)
                continue
            gmax = max(gmax, r)
            os.replace(tmp, f)   # overwrite in place
        print("done. global max_need (uncompressed blob) = %d bytes (%.1f KB)" % (gmax, gmax / 1024.0))
        print("-> set LZ4 scratch >= %d in pmap.c" % gmax)
        return 0
    if len(args) < 2:
        print("usage: pmap_lz4.py <in.pmap> <out.pmap>  |  --dir <dir>")
        return 1
    compress(args[0], args[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
