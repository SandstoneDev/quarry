#!/usr/bin/env python3
"""pmap_tex_t4from128 - upgrade ground/road textures to 128px T4 (16-colour).

Experiment C of the striped-textures session: the prod set carries roads at
64x64 T8 (downscaled from the tx128 transplant); the pre-downscale backup set
(chunks_prod_tx128_bak128) still has the same textures at 128px T8. For every
texture whose backup is exactly 2x the prod size and T8-opaque, decode the
128px texels, median-cut quantize to a 16-colour CLUT, swizzle as T4 and
splice into the CURRENT (UV-fixed) prod region. 2x linear resolution for
2x texel bytes (T4 halves bpp), CLUT shrinks 1024->64B. Friend's reference
port ships 90% of its world as 128x128 T4 - proven look on real HW.

UVs are untouched (resolution-independent); only the texel/clut pools and the
texture table change. Textures with alpha (num_levels alpha byte != 0) are
skipped - 16-colour quantization of alpha edges is a quality risk.

Usage:
 pmap_tex_t4from128.py <prod.pmap> <bak128.pmap> <out.pmap> [--report]
v3 inputs are decompressed / the output recompressed automatically.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))

HDR = struct.Struct('<20I')
TEX = struct.Struct('<HH7I')


def load_v2(path):
    blob = open(path, 'rb').read()
    ver = struct.unpack_from('<I', blob, 4)[0]
    if ver == 2:
        return blob, 2
    assert ver == 3, ver
    tmp = tempfile.mktemp(suffix='.pmap')
    subprocess.check_call([sys.executable,
                           os.path.join(TOOLS, 'pmap_lz4_decompress.py'),
                           path, tmp], stdout=subprocess.DEVNULL)
    blob = open(tmp, 'rb').read()
    os.remove(tmp)
    return blob, 3


def unswizzle(data, w_bytes, h):
    out = bytearray(w_bytes * h)
    pos = 0
    for by in range((h + 7) // 8):
        for bx in range(w_bytes // 16):
            for y in range(8):
                dy = by * 8 + y
                if dy >= h:
                    pos += 16
                    continue
                dst = dy * w_bytes + bx * 16
                out[dst:dst + 16] = data[pos:pos + 16]
                pos += 16
    return bytes(out)


def swizzle(lin, w_bytes, h):
    hpad = (h + 7) // 8 * 8
    out = bytearray(w_bytes * hpad)
    pos = 0
    for by in range(hpad // 8):
        for bx in range(w_bytes // 16):
            for y in range(8):
                sy = by * 8 + y
                src = sy * w_bytes + bx * 16
                if sy < h:
                    out[pos:pos + 16] = lin[src:src + 16]
                pos += 16
    return bytes(out)


def decode_t8(blob, h20, t):
    (w, hh, fmt, texel_first, texel_bytes, bufw, clut_first, clut_entries,
     nlev) = t
    texel_off, clut_off = h20[16], h20[18]
    wb = max(bufw, 16)
    lvl0 = blob[texel_off + texel_first: texel_off + texel_first + wb * hh]
    lin = unswizzle(lvl0, wb, hh)
    clut = blob[clut_off + clut_first: clut_off + clut_first + clut_entries * 4]
    idx = np.frombuffer(lin, np.uint8).reshape(hh, wb)[:, :w]
    pal = np.frombuffer(clut, np.uint8).reshape(-1, 4)
    return pal[idx]                       # h x w x 4 RGBA


def encode_t4(rgba):
    """RGBA (h x w x 4, opaque) -> (swizzled t4 bytes, clut64 bytes)."""
    h, w = rgba.shape[:2]
    img = Image.fromarray(rgba[..., :3], 'RGB')
    q = img.quantize(colors=16, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    pal = q.getpalette()[:48]
    idx = np.asarray(q, np.uint8)
    clut = bytearray()
    for i in range(16):
        clut += bytes((pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2], 255))
    wb = w // 2                           # 4bpp row bytes
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)  # low nibble = even x
    lin = packed.tobytes()
    return swizzle(lin, wb, h), bytes(clut)


def main():
    report = '--report' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prod_path, bak_path, out_path = args
    prod, prod_ver = load_v2(prod_path)
    bak, _ = load_v2(bak_path)
    hp = HDR.unpack_from(prod, 0)
    hb = HDR.unpack_from(bak, 0)
    tc = hp[7]
    assert tc == hb[7], "texture count mismatch (different bake lineage)"
    tex_off = hp[8]
    tp = [TEX.unpack_from(prod, tex_off + 32 * i) for i in range(tc)]
    tb = [TEX.unpack_from(bak, hb[8] + 32 * i) for i in range(tc)]

    upgrades = {}
    for i in range(tc):
        (pw, ph, pf, _, _, _, _, _, pn) = tp[i]
        (bw_, bh_, bf, _, _, _, _, _, _) = tb[i]
        if pf != 5 or bf != 5:
            continue
        if bw_ != pw * 2 or bh_ != ph * 2 or bw_ < 32:
            continue
        if (pn >> 8) & 0xFF:              # alpha_mode folded into byte 1: skip
            continue
        rgba = decode_t8(bak, hb, tb[i])
        texels, clut = encode_t4(rgba)
        upgrades[i] = (bw_, bh_, texels, clut)
    if report:
        print(f"{os.path.basename(prod_path)}: {len(upgrades)} upgradable textures")

    # rebuild texel + clut pools (they are the LAST two sections of the file)
    texel_pool = bytearray()
    clut_pool = bytearray()
    new_tex = []
    for i in range(tc):
        (w, h, fmt, texel_first, texel_bytes, bufw, clut_first, clut_entries,
         nlev) = tp[i]
        if i in upgrades:
            nw, nh, texels, clut = upgrades[i]
            nt = (nw, nh, 4,              # PMAP_FMT_T4
                  len(texel_pool), len(texels), nw,
                  len(clut_pool), 16, (nlev & 0xFFFFFF00) | 1)
            texel_pool += texels
            clut_pool += clut
        else:
            texels = prod[hp[16] + texel_first: hp[16] + texel_first + texel_bytes]
            clut = prod[hp[18] + clut_first: hp[18] + clut_first + clut_entries * 4]
            nt = (w, h, fmt, len(texel_pool), texel_bytes, bufw,
                  len(clut_pool), clut_entries, nlev)
            texel_pool += texels
            clut_pool += clut
        new_tex.append(nt)
        while len(texel_pool) % 16:
            texel_pool.append(0)
        while len(clut_pool) % 16:
            clut_pool.append(0)

    out = bytearray(prod[:hp[16]])        # everything up to the texel pool
    texel_off = hp[16]
    clut_off = texel_off + len(texel_pool)
    out += texel_pool
    out += clut_pool
    hl = list(hp)
    hl[2] = len(out)                      # file_size
    hl[17] = len(texel_pool)              # texel_bytes
    hl[18] = clut_off
    hl[19] = len(clut_pool)
    HDR.pack_into(out, 0, *hl)
    for i, nt in enumerate(new_tex):
        TEX.pack_into(out, tex_off + 32 * i, *nt)

    if prod_ver == 3:
        tmp = tempfile.mktemp(suffix='.pmap')
        open(tmp, 'wb').write(bytes(out))
        subprocess.check_call([sys.executable,
                               os.path.join(TOOLS, 'pmap_lz4.py'),
                               tmp, out_path], stdout=subprocess.DEVNULL)
        os.remove(tmp)
    else:
        open(out_path, 'wb').write(bytes(out))
    print(f"  {os.path.basename(out_path)}: upgraded {len(upgrades)} tex, "
          f"texel_pool {hp[17]}->{len(texel_pool)}B "
          f"clut_pool {hp[19]}->{len(clut_pool)}B "
          f"file {len(prod)}->{len(out)}B (v2 sizes)")


if __name__ == '__main__':
    main()
