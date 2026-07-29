#!/usr/bin/env python3
"""pmap_tex_mipgen.py - add mipmap chains to T8 world textures in a v2 .pmap tile.

WHY: the striped/moire ground+roads bug. The world textures are small (64x64)
paletted (T8) tiles repeated 6-11x across each ground/road patch and shipped with
NO mipmaps (num_levels==1). A 64px texture minified ~8x at a grazing angle with no
mip chain aliases into the fanning diagonal "stripes" the player sees. The runtime
ALREADY filters mips (Renderer.c draw_submesh: GU_LINEAR_MIPMAP_LINEAR + AUTO LOD
when num_levels>1); only the DATA never carried them.

WHAT: for each T8 texture (fmt 5) with num_levels==1 and a mippable size, append a
mip chain to its swizzled level-0 texels. The chain is CLAMPED so every level stays
a valid PSP T8 swizzle block: width>=16 AND multiple of 16, height>=8 AND multiple
of 8. The <16px levels are EXACTLY what corrupted the build-199 mip attempt (T8
swizzle needs a 16-byte row stride), so we stop before them - 64->{64,32,16}.
Level 0 is left byte-identical; each mip is a BOX downsample of level 0 (decoded to
RGBA through the CLUT) re-indexed to the SAME palette by nearest colour, so the
CLUT is untouched and every level shares it (one texture record, one palette).

v2 in -> v2 out. Pipeline per tile:
 pmap_lz4_decompress.py (v3 -> v2)
 pmap_tex_mipgen.py (v2 -> v2', THIS)
 pmap_lz4.py (v2' -> v3')

Usage:
 python pmap_tex_mipgen.py <in_v2.pmap> <out_v2.pmap>
 python pmap_tex_mipgen.py --selftest # swizzle round-trip + level rule
"""
import struct
import sys

import numpy as np
from PIL import Image

T8 = 5
TEX_STRIDE = 32   # w(u16),h(u16),format,texel_first,texel_bytes,bufw,clut_first,clut_entries,num_levels


def unswizzle_t8(swz, w, h):
    """Swizzled T8 (16-byte x 8-row blocks, buffer width == w) -> linear indices."""
    out = bytearray(w * h)
    bx = w // 16
    for byb in range(h // 8):
        for bxb in range(bx):
            for r in range(8):
                base = ((byb * bx + bxb) * 8 + r) * 16
                di = (byb * 8 + r) * w + bxb * 16
                out[di:di + 16] = swz[base:base + 16]
    return bytes(out)


def swizzle_t8(idx, w, h):
    """Linear T8 indices -> swizzled (exact inverse of unswizzle_t8)."""
    out = bytearray(w * h)
    bx = w // 16
    for byb in range(h // 8):
        for bxb in range(bx):
            for r in range(8):
                base = ((byb * bx + bxb) * 8 + r) * 16
                di = (byb * 8 + r) * w + bxb * 16
                out[base:base + 16] = idx[di:di + 16]
    return bytes(out)


def mip_sizes(w, h):
    """Level dimensions, clamped so each stays a valid T8 swizzle block."""
    lv = [(w, h)]
    cw, ch = w, h
    while (cw // 2) >= 16 and (ch // 2) >= 8 and (cw // 2) % 16 == 0 and (ch // 2) % 8 == 0:
        cw //= 2
        ch //= 2
        lv.append((cw, ch))
    return lv


def build_chain(swz0, w, h, pal_rgba):
    """Return the concatenated swizzled levels (level0 unchanged + generated mips),
 or None if this texture yields no mip level. pal_rgba is (256,4) uint8."""
    sizes = mip_sizes(w, h)
    if len(sizes) == 1:
        return None
    idx0 = np.frombuffer(unswizzle_t8(swz0, w, h), dtype=np.uint8).reshape(h, w)
    rgba0 = pal_rgba[idx0]                      # (h,w,4)
    img0 = Image.fromarray(rgba0, "RGBA")
    pal = pal_rgba.astype(np.int32)             # (256,4)
    blobs = [swz0]                              # level 0: byte-exact
    for (lw, lh) in sizes[1:]:
        im = img0.resize((lw, lh), Image.BOX)   # box-average downsample
        arr = np.asarray(im, dtype=np.int32).reshape(-1, 4)   # (N,4) RGBA
        # nearest palette entry by squared RGBA distance
        d = ((arr[:, None, :] - pal[None, :, :]) ** 2).sum(2)  # (N,256)
        idx = d.argmin(1).astype(np.uint8).reshape(lh, lw)
        blobs.append(swizzle_t8(idx.tobytes(), lw, lh))
    return b"".join(blobs), len(sizes)


def mipgen(inp, outp):
    d = open(inp, "rb").read()
    hh = struct.unpack_from("<20I", d, 0)
    (magic, ver, fsize, mc, moff, smc, smoff, tc, toff, ic, ioff, goff,
     voff, vbytes, ioff2, ibytes, texeloff, texelbytes, clutoff, clutbytes) = hh
    if magic != 0x50414D50:
        raise SystemExit("%s: bad magic" % inp)
    if ver != 2:
        raise SystemExit("%s: need v2 (got %d) - decompress first" % (inp, ver))

    texels = d[texeloff:texeloff + texelbytes]
    clut = d[clutoff:clutoff + clutbytes]
    recs = [list(struct.unpack_from("<HHIIIIIII", d, toff + ti * TEX_STRIDE)) for ti in range(tc)]

    new_texels = bytearray()
    n_mip = 0
    for ti, (w, ht, fmt, tf, tb, bw, cf, ce, nl) in enumerate(recs):
        while len(new_texels) % 16:            # keep every texture 16-aligned (matches original)
            new_texels.append(0)
        new_tf = len(new_texels)
        blob = texels[tf:tf + tb]              # default: unchanged
        if fmt == T8 and nl == 1 and ce > 0:
            pal = np.frombuffer(clut[cf:cf + ce * 4], dtype=np.uint8).reshape(ce, 4)
            if ce < 256:                        # index space is 0..255; pad unused entries
                pal = np.vstack([pal, np.zeros((256 - ce, 4), np.uint8)])
            res = build_chain(blob, w, ht, pal)
            if res is not None:
                blob, levels = res
                recs[ti][8] = levels            # num_levels
                n_mip += 1
        new_texels += blob
        recs[ti][3] = new_tf                    # texel_first
        recs[ti][4] = len(blob)                 # texel_bytes
    while len(new_texels) % 16:
        new_texels.append(0)
    new_texelbytes = len(new_texels)

    # texel_off is unchanged (pools before it did not move); clut_off shifts after the bigger pool.
    new_clutoff = (texeloff + new_texelbytes + 15) & ~15
    total = (new_clutoff + clutbytes + 15) & ~15

    out = bytearray(total)
    out[0:texeloff] = d[0:texeloff]                          # header + tables + grid + vertex/index pools
    out[texeloff:texeloff + new_texelbytes] = new_texels
    out[new_clutoff:new_clutoff + clutbytes] = clut
    struct.pack_into("<I", out, 2 * 4, total)                # file_size
    struct.pack_into("<I", out, 17 * 4, new_texelbytes)      # texel_bytes
    struct.pack_into("<I", out, 18 * 4, new_clutoff)         # clut_off
    for ti, rec in enumerate(recs):                          # rewrite texture table (inside copied region)
        struct.pack_into("<HHIIIIIII", out, toff + ti * TEX_STRIDE, *rec)

    open(outp, "wb").write(out)
    grow = 100 * (new_texelbytes - texelbytes) // max(texelbytes, 1)
    name = inp.replace("\\", "/").split("/")[-1]
    print("  %-22s mipped %d/%d T8, texel %d->%d KB (+%d%%), file %d->%d KB"
          % (name, n_mip, tc, texelbytes >> 10, new_texelbytes >> 10, grow, fsize >> 10, total >> 10))
    return n_mip


def selftest():
    rng = np.random.default_rng(1)
    for (w, h) in [(64, 64), (64, 32), (32, 32), (32, 16), (16, 16), (128, 16)]:
        idx = rng.integers(0, 256, size=w * h, dtype=np.uint8).tobytes()
        assert unswizzle_t8(swizzle_t8(idx, w, h), w, h) == idx, "roundtrip %dx%d" % (w, h)
    assert mip_sizes(64, 64) == [(64, 64), (32, 32), (16, 16)]
    assert mip_sizes(64, 32) == [(64, 32), (32, 16), (16, 8)]
    assert mip_sizes(32, 32) == [(32, 32), (16, 16)]
    assert mip_sizes(16, 16) == [(16, 16)]
    assert mip_sizes(48, 48) == [(48, 48)]     # 24 not %16 -> no mip (guarded)
    print("selftest OK: swizzle round-trip + mip level rule")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        selftest()
        return 0
    if len(args) < 2:
        print(__doc__)
        return 1
    mipgen(args[0], args[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
