#!/usr/bin/env python3
"""pmap_lz4_decompress.py - INVERSE of pmap_lz4.py: inflate a v3 (lz4-streamed)
.pmap region tile back into a raw v2 tile.

v3 stores each model's geometry blob (verts||indices) and each texture's blob
(texels||clut) as a per-model / per-texture LZ4 block; v2 stores those pools raw
and contiguous.  v3-only tools (pmap_tex_downscale.py, which reads through
gvcslib.psp_scene) choke on v3 ("unsupported PMAP version 3"); this restores a
byte-exact v2 so the v2 toolchain can edit the tile, after which pmap_lz4.py
re-compresses it to v3.

This is the exact inverse of pmap_lz4.compress():
  * header version 3 -> 2, the +3 u32 (comp_flag/comp_model_off/comp_tex_off)
    dropped, every prefix *_off un-shifted by the 12-byte header shrink
  * the resident prefix body (model/submesh/texture/instance tables + grid + cell
    lists) copied verbatim - it holds NO absolute file offsets, only pool-local
    indices, so it is version-agnostic
  * each LZ4 block inflated (uncompressed size RECOMPUTED exactly as the engine
    does in pmap.c: model span = (last submesh v/i end - first submesh v/i start);
    texture = texel_bytes + clut_entries*4) and scattered back into the raw pools
    at its pool-local position (verts@vfirst*12, idx@ifirst*2, texels@texel_first,
    clut@clut_first) - the inter-texture alignment padding stays zero, matching
    what psp_scene.write_scene emitted originally.

A decompress -> pmap_lz4.py re-compress reproduces the ORIGINAL v3 byte-for-byte
(--validate proves it per tile); that is the correctness guarantee.

Usage:
  python pmap_lz4_decompress.py <in_v3.pmap> <out_v2.pmap>
  python pmap_lz4_decompress.py --dir <data_dir>        # v3 -> v2 in place
  python pmap_lz4_decompress.py --validate <v3.pmap>    # round-trip vs pmap_lz4.py
  python pmap_lz4_decompress.py --validate --dir <dir>  # round-trip every tile
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

PMAP_MAGIC = 0x50414D50


def align(n, a=16):
    return (n + a - 1) & ~(a - 1)


def decompress(path_in, path_out):
    """Inflate a v3 tile into a raw v2 tile. Returns True on success."""
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

    if magic != PMAP_MAGIC:
        raise SystemExit("%s: bad magic" % path_in)
    if version == 2:
        print("  %s already v2, skip" % os.path.basename(path_in))
        return False
    if version == 4:
        raise SystemExit("%s: version 4 (UVR) carries a per-submesh UV-range table "
                         "that v2 cannot hold; pmap_lz4.py would drop it on re-compress. "
                         "Refusing - this pipeline is v3-only." % path_in)
    if version != 3:
        raise SystemExit("%s: unexpected version %d" % (path_in, version))

    comp_flag, comp_model_off, comp_tex_off = struct.unpack_from("<3I", data, HDR_V2)
    if not comp_flag:
        raise SystemExit("%s: v3 header but comp_flag=0 (no LZ4 pools)" % path_in)

    # ---- resident prefix body: tables + grid, copied verbatim (v3->v2 identical) ----
    # data[HDR_V3 : comp_model_off] is exactly what compress() wrote from v2's
    # data[HDR_V2 : vertex_off]. The comp tables live between the body and the blobs.
    body = data[HDR_V3:comp_model_off]

    # ---- comp tables (absolute file offset + compressed size per blob) ----
    comp_models = [struct.unpack_from("<II", data, comp_model_off + i * 8)
                   for i in range(model_count)]
    comp_tex = [struct.unpack_from("<II", data, comp_tex_off + i * 8)
                for i in range(texture_count)]

    # ---- table records (read at their absolute v3 offsets, all inside `body`) ----
    submeshes = [struct.unpack_from("<i4I", data, submesh_off + i * SUBMESH_STRIDE)
                 for i in range(submesh_count)]

    # ---- reconstruct the four raw pools ----
    vertex_pool = bytearray(vertex_bytes)
    index_pool  = bytearray(index_bytes)
    texel_pool  = bytearray(texel_bytes)
    clut_pool   = bytearray(clut_bytes)

    # models: inflate (verts||indices), scatter to pool-local positions
    for mi in range(model_count):
        first, scount = struct.unpack_from("<2I", data, model_off + mi * MODEL_STRIDE)
        off, csize = comp_models[mi]
        if scount == 0:
            if csize:
                raise SystemExit("%s: model %d empty but has a blob" % (path_in, mi))
            continue
        s0 = submeshes[first]
        sN = submeshes[first + scount - 1]
        vstart = s0[1]; istart = s0[3]
        vbytes = (sN[1] + sN[2] - vstart) * 12
        ibytes = (sN[3] + sN[4] - istart) * 2
        need = vbytes + ibytes
        if not csize:
            raise SystemExit("%s: model %d has geometry (need=%d) but csize=0" % (path_in, mi, need))
        raw = lz4.block.decompress(data[off:off + csize], uncompressed_size=need)
        if len(raw) != need:
            raise SystemExit("%s: model %d inflate %d != need %d" % (path_in, mi, len(raw), need))
        vertex_pool[vstart * 12: vstart * 12 + vbytes] = raw[:vbytes]
        index_pool[istart * 2: istart * 2 + ibytes]    = raw[vbytes:]

    # textures: inflate (texels||clut), scatter to pool-local positions
    for ti in range(texture_count):
        w, ht, fmt, tfirst, tbytes, bufw, cfirst, centries, nlev = \
            struct.unpack_from("<HHIIIIIII", data, texture_off + ti * TEX_STRIDE)
        off, csize = comp_tex[ti]
        cbytes = centries * 4
        need = tbytes + cbytes
        if need == 0:
            if csize:
                raise SystemExit("%s: texture %d empty but has a blob" % (path_in, ti))
            continue
        if not csize:
            raise SystemExit("%s: texture %d has texels (need=%d) but csize=0" % (path_in, ti, need))
        raw = lz4.block.decompress(data[off:off + csize], uncompressed_size=need)
        if len(raw) != need:
            raise SystemExit("%s: texture %d inflate %d != need %d" % (path_in, ti, len(raw), need))
        texel_pool[tfirst:tfirst + tbytes] = raw[:tbytes]
        if cbytes:
            clut_pool[cfirst:cfirst + cbytes] = raw[tbytes:]

    # ---- assemble v2 (offsets exactly as psp_scene.write_scene lays them out) ----
    new_vertex_off = HDR_V2 + len(body)   # == original v2 vertex_off (16-aligned)
    if new_vertex_off & 15:
        raise SystemExit("%s: reconstructed vertex_off %d not 16-aligned (format misread)"
                         % (path_in, new_vertex_off))
    new_index_off = align(new_vertex_off + vertex_bytes)
    new_texel_off = align(new_index_off + index_bytes)
    new_clut_off  = align(new_texel_off + texel_bytes)
    total = align(new_clut_off + clut_bytes)

    out = bytearray(total)
    struct.pack_into("<20I", out, 0,
                     magic, 2, total,
                     model_count, model_off - 12,
                     submesh_count, submesh_off - 12,
                     texture_count, texture_off - 12,
                     instance_count, instance_off - 12,
                     grid_off - 12,
                     new_vertex_off, vertex_bytes,
                     new_index_off, index_bytes,
                     new_texel_off, texel_bytes,
                     new_clut_off, clut_bytes)
    out[HDR_V2:HDR_V2 + len(body)] = body
    out[new_vertex_off:new_vertex_off + vertex_bytes] = vertex_pool
    out[new_index_off:new_index_off + index_bytes]    = index_pool
    out[new_texel_off:new_texel_off + texel_bytes]    = texel_pool
    out[new_clut_off:new_clut_off + clut_bytes]       = clut_pool

    open(path_out, "wb").write(out)
    return True


def validate(path_v3, verbose=True):
    """Round-trip: v3 -> decompress -> v2 -> pmap_lz4.compress -> v3'.
    Byte-identical v3' proves the format is understood and the v2 is correct.
    Returns True on match."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pmap_lz4

    orig = open(path_v3, "rb").read()
    tmp_v2 = path_v3 + ".rtv2.tmp"
    tmp_v3 = path_v3 + ".rtv3.tmp"
    try:
        decompress(path_v3, tmp_v2)
        # v2 must be parseable by the v2 toolchain (gvcslib.psp_scene)
        try:
            sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
            from gvcslib import psp_scene
            psp_scene.read_scene(open(tmp_v2, "rb").read())
            v2_parses = True
        except Exception as e:
            v2_parses = False
            v2_err = str(e)
        pmap_lz4.compress(tmp_v2, tmp_v3)
        rt = open(tmp_v3, "rb").read()
    finally:
        for t in (tmp_v2, tmp_v3):
            if os.path.exists(t):
                os.remove(t)

    ok = (rt == orig)
    if verbose:
        name = os.path.basename(path_v3)
        if ok and v2_parses:
            print("  OK   %-22s v3->v2->v3' byte-identical (%d B), v2 parses" % (name, len(orig)))
        elif ok and not v2_parses:
            print("  WARN %-22s v3' identical but v2 read_scene FAILED: %s" % (name, v2_err))
        else:
            n = min(len(rt), len(orig))
            diff = next((i for i in range(n) if rt[i] != orig[i]), n)
            print("  FAIL %-22s v3'(%d) != v3(%d), first diff @%d" % (name, len(rt), len(orig), diff))
    return ok and v2_parses


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    do_validate = False
    if args and args[0] == "--validate":
        do_validate = True
        args = args[1:]

    if args and args[0] == "--dir":
        d = args[1]
        files = sorted(glob.glob(os.path.join(d, "region_*.pmap")))
        if not files:
            print("no region_*.pmap in", d); return 1
        if do_validate:
            nok = nfail = 0
            for f in files:
                if validate(f):
                    nok += 1
                else:
                    nfail += 1
            print("validate: %d OK, %d FAIL / %d tiles" % (nok, nfail, len(files)))
            return 0 if nfail == 0 else 2
        for f in files:
            tmp = f + ".v2tmp"
            if decompress(f, tmp):
                os.replace(tmp, f)   # overwrite in place
            elif os.path.exists(tmp):
                os.remove(tmp)
        print("done. %d tiles -> v2" % len(files))
        return 0

    if do_validate:
        if len(args) < 1:
            print("usage: pmap_lz4_decompress.py --validate <v3.pmap>"); return 1
        return 0 if validate(args[0]) else 2

    if len(args) < 2:
        print("usage: pmap_lz4_decompress.py <in_v3.pmap> <out_v2.pmap>  |  --dir <dir>  |  --validate <v3.pmap>")
        return 1
    decompress(args[0], args[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
