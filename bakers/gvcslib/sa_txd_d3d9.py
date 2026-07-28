"""the source game PC (Direct3D 9) TXD decoder -> RGBA8888.

PC SA textures are RW "D3D9 native" rasters (platform id 9): DXT1/3/5 block
compression, raw 16/32-bit, or 8-bit palette.  The PSP GE has no DXT and a 2 MB
VRAM budget, so we decode everything to RGBA8888 here and hand it to psp_tex
(swizzle + T4/T8 CLUT + mipmaps) exactly like the PS2 path.

D3D9 TexNative STRUCT (after the 12-byte chunk header):
    u32 platform            (== 9)
    u32 filterAddrFlags      (low byte filter, nibbles 8-11 U addr, 12-15 V addr)
    char name[32]
    char mask[32]
    u32 rasterFormat         (format nibble in bits 8-11; 0x2000 = pal8, 0x4000 = pal4)
    u32 d3dFormat            (FOURCC 'DXTn' or a D3DFMT enum)
    u16 width
    u16 height
    u8  depth                (bits per texel)
    u8  numLevels            (mip count)
    u8  rasterType           (== 4)
    u8  flags                (0x01 hasAlpha, 0x02 cube, 0x04 autoMip, 0x08 compressed)
    [ palette ]              (pal8: 256*RGBA8888; pal4: 32*RGBA8888 - BGRA on disk)
    per mip level: u32 dataSize, u8 data[dataSize]   (level 0 first)

Returns from decode(): { name(lower): (w, h, rgba8888_bytes) }  (R high byte),
matching gvcslib.sa_txd so the export driver is format-agnostic.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

STRUCT = 0x01
EXTENSION = 0x03
TEXNATIVE = 0x15      # RwTextureNative
TEXDICT = 0x16        # RwTexDictionary

# rasterFormat format nibble (bits 8-11)
_FMT_1555 = 0x1
_FMT_565 = 0x2
_FMT_4444 = 0x3
_FMT_LUM8 = 0x4
_FMT_8888 = 0x5
_FMT_888 = 0x6
_FMT_555 = 0xa

_FOURCC_DXT1 = 0x31545844
_FOURCC_DXT3 = 0x33545844
_FOURCC_DXT5 = 0x35545844


# --------------------------------------------------------------------------
# chunk walk (RW: u32 type, u32 size, u32 libid ; then body)
# --------------------------------------------------------------------------
def _find_all(b: bytes, o: int, end: int, want: int) -> List[Tuple[int, int]]:
    """Return (data_off, size) for every top-level chunk of type `want` in [o,end)."""
    out = []
    while o + 12 <= end:
        typ, sz, _lib = struct.unpack_from("<III", b, o)
        body = o + 12
        if typ == want:
            out.append((body, sz))
        o = body + sz
    return out


def _first_child(b: bytes, data_off: int, size: int, want: int):
    return _find_all(b, data_off, data_off + size, want)[:1]


# --------------------------------------------------------------------------
# pixel expanders -> RGBA8888 (R high byte)
# --------------------------------------------------------------------------
def _pack(r, g, b, a):
    return bytes((r, g, b, a))


def _expand_raw(data: bytes, w: int, h: int, fmt: int, has_alpha: bool) -> bytes:
    out = bytearray(w * h * 4)
    n = w * h
    if fmt in (_FMT_8888, _FMT_888):
        # stored BGRA (D3D little-endian)
        for i in range(n):
            bb, gg, rr, aa = data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]
            o = i*4
            out[o] = rr; out[o+1] = gg; out[o+2] = bb
            out[o+3] = aa if (fmt == _FMT_8888 and has_alpha) else 0xFF
    elif fmt == _FMT_565:
        for i in range(n):
            v = data[i*2] | (data[i*2+1] << 8)
            r = (v >> 11) & 0x1F; g = (v >> 5) & 0x3F; bl = v & 0x1F
            o = i*4
            out[o] = (r << 3) | (r >> 2)
            out[o+1] = (g << 2) | (g >> 4)
            out[o+2] = (bl << 3) | (bl >> 2)
            out[o+3] = 0xFF
    elif fmt in (_FMT_1555, _FMT_555):
        for i in range(n):
            v = data[i*2] | (data[i*2+1] << 8)
            a1 = (v >> 15) & 1
            r = (v >> 10) & 0x1F; g = (v >> 5) & 0x1F; bl = v & 0x1F
            o = i*4
            out[o] = (r << 3) | (r >> 2)
            out[o+1] = (g << 3) | (g >> 2)
            out[o+2] = (bl << 3) | (bl >> 2)
            out[o+3] = 0xFF if (fmt == _FMT_555 or a1) else 0x00
    elif fmt == _FMT_4444:
        for i in range(n):
            v = data[i*2] | (data[i*2+1] << 8)
            a = (v >> 12) & 0xF; r = (v >> 8) & 0xF; g = (v >> 4) & 0xF; bl = v & 0xF
            o = i*4
            out[o] = (r << 4) | r
            out[o+1] = (g << 4) | g
            out[o+2] = (bl << 4) | bl
            out[o+3] = ((a << 4) | a) if has_alpha else 0xFF
    elif fmt == _FMT_LUM8:
        for i in range(n):
            l = data[i]
            o = i*4
            out[o] = out[o+1] = out[o+2] = l
            out[o+3] = 0xFF
    else:
        raise ValueError("unsupported raw format nibble 0x%x" % fmt)
    return bytes(out)


def _expand_palette(data: bytes, pal: bytes, w: int, h: int, pal_n: int) -> bytes:
    """8-bit (pal_n=256) or 4-bit (pal_n=16) palette indices; pal is BGRA on disk."""
    out = bytearray(w * h * 4)
    # palette -> RGBA
    prgba = bytearray(pal_n * 4)
    for i in range(pal_n):
        bb, gg, rr, aa = pal[i*4], pal[i*4+1], pal[i*4+2], pal[i*4+3]
        prgba[i*4] = rr; prgba[i*4+1] = gg; prgba[i*4+2] = bb; prgba[i*4+3] = aa
    n = w * h
    if pal_n == 256:
        for i in range(n):
            idx = data[i] * 4
            o = i*4
            out[o:o+4] = prgba[idx:idx+4]
    else:  # 4-bit, two indices per byte (low nibble first)
        for i in range(n):
            byte = data[i >> 1]
            idx = (byte & 0xF) if (i & 1) == 0 else (byte >> 4)
            idx *= 4
            o = i*4
            out[o:o+4] = prgba[idx:idx+4]
    return bytes(out)


# --------------------------------------------------------------------------
# DXT block decoders
# --------------------------------------------------------------------------
def _c565(v):
    r = (v >> 11) & 0x1F; g = (v >> 5) & 0x3F; b = v & 0x1F
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def _dxt_colors(c0, c1, dxt1):
    r0, g0, b0 = _c565(c0); r1, g1, b1 = _c565(c1)
    col = [(r0, g0, b0, 255), (r1, g1, b1, 255), None, None]
    if dxt1 and c0 <= c1:
        col[2] = ((r0 + r1)//2, (g0 + g1)//2, (b0 + b1)//2, 255)
        col[3] = (0, 0, 0, 0)               # transparent
    else:
        col[2] = ((2*r0 + r1)//3, (2*g0 + g1)//3, (2*b0 + b1)//3, 255)
        col[3] = ((r0 + 2*r1)//3, (g0 + 2*g1)//3, (b0 + 2*b1)//3, 255)
    return col


def _decode_dxt(data: bytes, w: int, h: int, kind: int) -> bytes:
    """kind: 1=DXT1, 3=DXT3, 5=DXT5."""
    out = bytearray(w * h * 4)
    bx = (w + 3) // 4
    by = (h + 3) // 4
    blocksz = 8 if kind == 1 else 16
    p = 0
    for byi in range(by):
        for bxi in range(bx):
            blk = data[p:p+blocksz]; p += blocksz
            if kind == 1:
                c0, c1 = struct.unpack_from("<HH", blk, 0)
                bits = struct.unpack_from("<I", blk, 4)[0]
                col = _dxt_colors(c0, c1, True)
                alpha = None
            elif kind == 3:
                # 16x 4-bit explicit alpha
                a64 = struct.unpack_from("<Q", blk, 0)[0]
                c0, c1 = struct.unpack_from("<HH", blk, 8)
                bits = struct.unpack_from("<I", blk, 12)[0]
                col = _dxt_colors(c0, c1, False)
                alpha = [((a64 >> (4*i)) & 0xF) * 17 for i in range(16)]
            else:  # DXT5
                a0 = blk[0]; a1 = blk[1]
                abits = int.from_bytes(blk[2:8], "little")
                c0, c1 = struct.unpack_from("<HH", blk, 8)
                bits = struct.unpack_from("<I", blk, 12)[0]
                col = _dxt_colors(c0, c1, False)
                if a0 > a1:
                    at = [a0, a1] + [((((7-i)*a0 + (i)*a1)//7)) for i in range(1, 7)]
                    # canonical: a2..a7 = ((6-k)*a0 + (k+1)*a1)/7
                    at = [a0, a1]
                    for k in range(1, 7):
                        at.append(((7-k)*a0 + k*a1)//7)
                else:
                    at = [a0, a1]
                    for k in range(1, 5):
                        at.append(((5-k)*a0 + k*a1)//5)
                    at.append(0); at.append(255)
                alpha = [at[(abits >> (3*i)) & 7] for i in range(16)]
            for i in range(16):
                px = bxi*4 + (i & 3)
                py = byi*4 + (i >> 2)
                if px >= w or py >= h:
                    continue
                ci = (bits >> (2*i)) & 3
                r, g, b, a = col[ci]
                if alpha is not None:
                    a = alpha[i]
                o = (py*w + px) * 4
                out[o] = r; out[o+1] = g; out[o+2] = b; out[o+3] = a
    return bytes(out)


# --------------------------------------------------------------------------
# one TexNative -> (name, w, h, rgba8888)
# --------------------------------------------------------------------------
def _decode_texnative(b: bytes, data_off: int, size: int):
    o = data_off
    st = _first_child(b, data_off, size, STRUCT)
    if not st:
        return None
    so, ss = st[0]
    platform = struct.unpack_from("<I", b, so)[0]
    if platform != 9:
        return None  # not D3D9
    name = b[so+8:so+8+32].split(b"\x00", 1)[0].decode("latin1", "replace")
    raster_fmt = struct.unpack_from("<I", b, so+72)[0]
    d3dfmt = struct.unpack_from("<I", b, so+76)[0]
    w, h = struct.unpack_from("<HH", b, so+80)
    depth = b[so+84]
    num_levels = b[so+85]
    _rtype = b[so+86]
    flags = b[so+87]
    has_alpha = bool(flags & 0x01)
    compressed = bool(flags & 0x08) or d3dfmt in (_FOURCC_DXT1, _FOURCC_DXT3, _FOURCC_DXT5)
    fmt_nibble = (raster_fmt >> 8) & 0xF
    is_pal8 = bool(raster_fmt & 0x2000)
    is_pal4 = bool(raster_fmt & 0x4000)

    p = so + 88
    pal = b""
    if is_pal8:
        pal = b[p:p+256*4]; p += 256*4
    elif is_pal4:
        pal = b[p:p+32*4]; p += 32*4

    # level 0
    lvl0_size = struct.unpack_from("<I", b, p)[0]; p += 4
    data = b[p:p+lvl0_size]

    if compressed:
        if d3dfmt == _FOURCC_DXT1:
            rgba = _decode_dxt(data, w, h, 1)
        elif d3dfmt == _FOURCC_DXT3:
            rgba = _decode_dxt(data, w, h, 3)
        elif d3dfmt == _FOURCC_DXT5:
            rgba = _decode_dxt(data, w, h, 5)
        else:
            raise ValueError("compressed but unknown d3dFormat 0x%x" % d3dfmt)
    elif is_pal8:
        rgba = _expand_palette(data, pal, w, h, 256)
    elif is_pal4:
        rgba = _expand_palette(data, pal, w, h, 16)
    else:
        rgba = _expand_raw(data, w, h, fmt_nibble, has_alpha)

    return name, w, h, rgba


def decode(blob) -> Dict[str, Tuple[int, int, bytes]]:
    """Decode a PC (D3D9) TXD blob -> {name_lower: (w, h, rgba8888_bytes)}."""
    b = bytes(blob)
    out: Dict[str, Tuple[int, int, bytes]] = {}
    # TexDictionary 0x16 -> its body holds STRUCT + N TextureNative 0x15
    tds = _find_all(b, 0, len(b), TEXDICT)
    search = tds if tds else [(0, len(b))]
    for tdo, tdsz in search:
        for to, tsz in _find_all(b, tdo, tdo + tdsz, TEXNATIVE):
            try:
                r = _decode_texnative(b, to, tsz)
            except Exception:
                r = None
            if r:
                name, w, h, rgba = r
                out[name.lower()] = (w, h, rgba)
    return out
