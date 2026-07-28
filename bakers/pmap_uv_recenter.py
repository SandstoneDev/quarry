#!/usr/bin/env python3
"""pmap_uv_recenter.py - fix clamped tiled UVs in v2 region .pmap tiles.

!!! DEPRECATED (b572): built on the FALSE "signed +-8" model. The GE reads 16-bit
UVs UNSIGNED ([0,16)-tile window, wrap mod 16) - centring spans around 0 CAUSES
the striped-texture bug. Use tools/pmap_uv_unsign.py instead.
(research/striped_textures_rootcause_and_fix.md)

The gvcslib baker stores UVs as s16 = round(uv * 4096) (UV_FIXED_ONE=4096), so the
representable range is [-8, +8). A tiled texture whose UV runs past 8 (long roads,
big walls, fences) gets CLAMPED at 8.0 -> the UV gradient breaks -> garbled texture.

gvcslib is read-only, so fix it downstream: per submesh, subtract a WHOLE number of
texture periods (round(mean_uv) tiles) from every vertex UV. With GU_REPEAT this is
visually identical (uv and uv-N for integer N sample the same texel) but re-centres
the UVs near 0 so they fit [-8, 8) and no longer clamp. Vertices are not shared
between submeshes (each submesh owns a contiguous vertex slice), so this is safe.

Run on the RAW v2 tiles BEFORE pmap_lz4.py. Usage:
  python pmap_uv_recenter.py <region_dir>
"""
import os, sys, struct, glob

UV_ONE = 4096          # s16 per 1.0 uv (must match gvcslib UV_FIXED_ONE)
S16_MAX, S16_MIN = 32767, -32768
SUB_STRIDE = 20        # texture(i32), vfirst,vcount,ifirst,icount (u32)
VTX_STRIDE = 12        # u,v(s16) color(u16) x,y,z(s16)


def clampi(x):
    return S16_MAX if x > S16_MAX else (S16_MIN if x < S16_MIN else x)


def recenter(path):
    d = bytearray(open(path, "rb").read())
    h = struct.unpack_from("<20I", d, 0)
    if h[1] != 2:
        return None   # not raw v2 (already lz4'd?) -> skip
    sub_cnt, sub_off = h[5], h[6]
    voff, vbytes = h[12], h[13]
    nverts = vbytes // VTX_STRIDE
    moved = 0
    span_overflow = 0
    for s in range(sub_cnt):
        _tex, vf, vc, _if, _ic = struct.unpack_from("<i4I", d, sub_off + s * SUB_STRIDE)
        if vc == 0:
            continue
        # mean UV of this submesh (in s16 units)
        su = sv = 0
        for i in range(vf, vf + vc):
            u, v = struct.unpack_from("<hh", d, voff + i * VTX_STRIDE)
            su += u; sv += v
        mu = su // vc; mv = sv // vc
        # nearest whole-tile offset (keeps GU_REPEAT appearance)
        ou = int(round(mu / UV_ONE)) * UV_ONE
        ov = int(round(mv / UV_ONE)) * UV_ONE
        if ou == 0 and ov == 0:
            continue
        for i in range(vf, vf + vc):
            u, v = struct.unpack_from("<hh", d, voff + i * VTX_STRIDE)
            nu = clampi(u - ou); nv = clampi(v - ov)
            if nu in (S16_MAX, S16_MIN) or nv in (S16_MAX, S16_MIN):
                span_overflow += 1   # submesh UV span > 16 -> still clamps (rare, huge tiling)
            struct.pack_into("<hh", d, voff + i * VTX_STRIDE, nu, nv)
        moved += 1
    open(path, "wb").write(d)
    return (sub_cnt, moved, span_overflow)


def main():
    if len(sys.argv) < 2:
        print("usage: pmap_uv_recenter.py <region_dir>"); return 1
    files = sorted(glob.glob(os.path.join(sys.argv[1], "region_*.pmap")))
    if not files:
        print("no region_*.pmap in", sys.argv[1]); return 1
    tot_sub = tot_moved = tot_overflow = 0
    for f in files:
        r = recenter(f)
        if r is None:
            print("  %s: not v2 (skip)" % os.path.basename(f)); continue
        sc, mv, ov = r
        tot_sub += sc; tot_moved += mv; tot_overflow += ov
        if mv:
            print("  %s: recentred %d/%d submeshes%s" %
                  (os.path.basename(f), mv, sc, (" (%d still clamp: span>16)" % ov) if ov else ""))
    print("done: %d submeshes recentred (of %d), %d residual-clamp verts (huge tiling)" %
          (tot_moved, tot_sub, tot_overflow))
    return 0


if __name__ == "__main__":
    sys.exit(main())
