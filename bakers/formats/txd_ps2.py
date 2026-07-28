"""PS2/console-native TXD raster decoder (TextureNative platformId 4).

the source game-PS2 (and VCS console) textures are stored as PlayStation 2 Graphics
Synthesizer rasters: indexed PSMT8 (8bpp + 256-CLUT) or PSMT4 (4bpp + 16-CLUT),
occasionally direct PSMCT32. Index planes are stored **swizzled**, and the
PSMT8 CLUT is column-interleaved. There is no DXT on PS2. Format/dimensions
come from the GS TEX0 register (PSM field), with the header depth field
taking priority when they disagree (modded repacks write garbage tex0).

Swizzle truth (2026-07-21 fix; the old page/block/column VRAM model here was
WRONG): RenderWare serializes the host->GS *transfer stream*, uploading
swizzled rasters as a wider pixel format at halved dimensions (GIF TRXREG on
disk = w/2 x h/2). PSMT8 is a PSMCT32 transfer (byte units), PSMT4 a PSMCT16
transfer (nibble units, low nibble first); the closed-form texel mapping is
librw ps2raster.cpp swizzle() (== classic Sparky unswizzle8), with RW
transferMinSize row strides max(w,16) for PSMT8 / max(w,32) for PSMT4.
Mirrors gvcslib.sa_txd (validated vs PC references, corr 0.96-0.99).

Layout: ✅ SDK-verification + 
renderware_ps2_sdk_reference.md. derived in gvcslib (validated against real
SA-PS2 TXDs); this is an independent re-implementation on top of core.rwstream.

TextureNative (0x15) children:
 STRING(name), STRING(mask), STRUCT(tiny 8B: u32 platform, u32 filterAddr),
 STRUCT(big) -> STRUCT(raster header 14×u32) + STRUCT(GIF packet stream)
Raster header u32[14]: width, height, depth, rasterFormat, tex0_lo, tex0_hi,
 tex1_lo, tex1_hi, miptbp1_lo/hi, miptbp2? , texelDataSize, paletteSize (last fields).
GIF stream: IMAGE packets - first = mip0 pixels, last = CLUT.
"""
from __future__ import annotations

import struct
from typing import List, Optional

from core import rwstream as rw

PSM_PSMCT32 = 0x00
PSM_PSMT8 = 0x13
PSM_PSMT4 = 0x14

# GS swizzled-transfer address mapping (librw ps2raster.cpp swizzle).
def _gs_transfer_addr(x: int, y: int, w: int) -> int:
    """Unit index in the swizzled transfer stream for texel (x, y).

 Units are bytes for PSMT8 (PSMCT32 transfer) and nibbles for PSMT4
 (PSMCT16 transfer). w is the stored row stride in texel units - max(width, 16) for PSMT8 / max(width, 32) for PSMT4 (RW min transfer).
 """
    xx = x ^ ((((y >> 1) ^ (y >> 2)) & 1) << 2)    # half-word swap rows
    nx = (xx & 7) | ((x >> 1) & ~7)                # CT word/unit x
    ny = (y & 1) | ((y >> 1) & ~1)                 # CT row (2 per 4-texel strip)
    lane = ((y >> 1) & 1) | (((x >> 3) & 1) << 1)  # texel lane within CT unit
    return lane | (nx << 2) | ny * 2 * w


def _unswizzle_psmt8(data: bytes, w: int, h: int) -> bytes:
    """PSMCT32 transfer of max(w,16)/2 x max(h,4)/2 -> linear w*h indices."""
    ww = max(w, 16)
    out = bytearray(w * h)
    n = len(data)
    for y in range(h):
        row = y * w
        for x in range(w):
            a = _gs_transfer_addr(x, y, ww)
            if a < n:
                out[row + x] = data[a]
    return bytes(out)


def _unswizzle_psmt4(data: bytes, w: int, h: int) -> bytes:
    """PSMCT16 transfer of max(w,32)/2 x max(h,4)/2, nibble units (low first)."""
    ww = max(w, 32)
    out = bytearray(w * h)
    n = len(data)
    for y in range(h):
        row = y * w
        for x in range(w):
            a = _gs_transfer_addr(x, y, ww)        # nibble index
            byte_addr = a >> 1
            if byte_addr < n:
                b = data[byte_addr]
                out[row + x] = (b >> 4) & 0xF if (a & 1) else b & 0xF
    return bytes(out)


def _deswizzle_clut8(raw: bytes) -> bytes:
    """Restore linear order of a 256-entry PSMT8 CLUT (column-interleave in VRAM)."""
    out = bytearray(256 * 4)
    for i in range(256):
        j = (i & ~0x18) | ((i & 0x08) << 1) | ((i & 0x10) >> 1)
        out[j * 4:j * 4 + 4] = raw[i * 4:i * 4 + 4]
    return bytes(out)


def _gif_images(d: bytes) -> List[bytes]:
    """Raw bytes of every IMAGE-mode GIF packet (flg==2); skip PACKED/REGLIST setup."""
    images = []
    off = 0
    n = len(d)
    while off + 16 <= n:
        qw = struct.unpack_from("<Q", d, off)[0]
        nloop = qw & 0x3FFF
        flg = (qw >> 58) & 0x3
        nreg = (qw >> 60) & 0xF
        if flg == 2:
            nbytes = nloop * 16
            images.append(d[off + 16:off + 16 + nbytes])
            off += 16 + nbytes
        else:
            off += 16 + nloop * max(nreg, 1) * 16
    return images


def _rgba_from_indices(indices: bytes, clut: bytes, n_clut: int, n_pixels: int) -> bytes:
    """Map palette indices → RGBA8888. PS2 alpha 0-128 → 0-255 (x2, clamp)."""
    rgba = bytearray(n_pixels * 4)
    mask = n_clut - 1
    for p in range(n_pixels):
        idx = (indices[p] & mask) * 4
        r, g, b, a = clut[idx], clut[idx + 1], clut[idx + 2], clut[idx + 3]
        o = p * 4
        rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = min(255, a * 2)
    return bytes(rgba)


def decode_texturenative(data: bytes, tn: rw.ChunkHeader) -> Optional[dict]:
    """Decode ONE PS2-native TEXTURENATIVE (0x15) chunk.

 Returns {name, mask, width, height, depth, psm, fmt, has_alpha, rgba} or None
 if the raster can't be decoded (unknown PSM, missing packets, bad dims).
 `rgba` is w*h*4 bytes, R,G,B,A order, alpha 0-255.
 """
    # name / mask from STRING children
    name = mask = ""
    strings = [c for c in rw.iter_chunks(data, tn.body_offset, tn.end) if c.type == rw.STRING]
    if strings:
        name = data[strings[0].body_offset:strings[0].end].split(b"\x00", 1)[0].decode("latin-1", "replace")
    if len(strings) > 1:
        mask = data[strings[1].body_offset:strings[1].end].split(b"\x00", 1)[0].decode("latin-1", "replace")

    # the raster lives in the LARGEST STRUCT child (the tiny 8B one is platform/filter)
    structs = [c for c in rw.iter_chunks(data, tn.body_offset, tn.end) if c.type == rw.STRUCT]
    if not structs:
        return None
    big = max(structs, key=lambda c: c.size)

    inner = [c for c in rw.iter_chunks(data, big.body_offset, big.end) if c.type == rw.STRUCT]
    if len(inner) < 2:
        return None
    hdr, blk = inner[0], inner[1]
    if hdr.size < 56:
        return None

    fields = struct.unpack_from("<14I", data, hdr.body_offset)
    w, h, depth, raster_format = fields[0], fields[1], fields[2], fields[3]
    tex0 = (fields[5] << 32) | fields[4]
    psm = (tex0 >> 20) & 0x3F
    # depth wins over tex0 when they disagree (modded repacks write garbage tex0)
    if depth == 8 and psm != PSM_PSMT8:
        psm = PSM_PSMT8
    elif depth == 4 and psm != PSM_PSMT4:
        psm = PSM_PSMT4
    if w == 0 or h == 0 or (w & (w - 1)) or (h & (h - 1)):
        return None

    images = _gif_images(data[blk.body_offset:blk.end])
    if len(images) < 1:
        return None
    mip0 = images[0]
    clut_raw = images[-1] if len(images) >= 2 else b""

    if psm == PSM_PSMT8:
        if len(clut_raw) < 256 * 4:
            return None
        indices = _unswizzle_psmt8(mip0, w, h)
        clut = _deswizzle_clut8(clut_raw[:256 * 4])
        rgba = _rgba_from_indices(indices, clut, 256, w * h)
        fmt = "PS2_PSMT8"
    elif psm == PSM_PSMT4:
        if len(clut_raw) < 16 * 4:
            return None
        indices = _unswizzle_psmt4(mip0, w, h)
        rgba = _rgba_from_indices(indices, clut_raw[:16 * 4], 16, w * h)
        fmt = "PS2_PSMT4"
    elif psm == PSM_PSMCT32:
        expected = w * h * 4
        raw32 = (mip0 + bytes(expected))[:expected]
        rgba = bytearray(expected)
        for i in range(0, expected, 4):
            r, g, b, a = raw32[i:i + 4]
            rgba[i:i + 4] = bytes([r, g, b, min(255, a * 2)])
        rgba = bytes(rgba)
        fmt = "PS2_PSMCT32"
    else:
        return None

    has_alpha = any(rgba[i] != 255 for i in range(3, len(rgba), 4))
    return {
        "name": name, "mask": mask, "width": w, "height": h, "depth": depth,
        "psm": psm, "raster_format": raster_format, "fmt": fmt,
        "has_alpha": has_alpha, "rgba": rgba,
    }
