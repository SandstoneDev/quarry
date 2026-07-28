"""RW-PSP texture full codec (the source console title, PSP).

This module decodes the *embedded* RW-PSP rasters that the streaming system
packs inside the world ``<ZONE>.IMG`` files (reached via the ``.LVZ`` texture
grid), as well as decoding a general RW-PSP raster given its descriptor byte
pair.  It complements :mod:`gvcslib.xtx`, which handles the *standalone*
single-texture ``.XTX`` files; this module is for the dictionary blobs that
live inside an IMG cell.

IMG-cell texture dictionary (``AERA`` container)
------------------------------------------------
A texture-grid cell (see :mod:`gvcslib.lvz`, ``TexCell``) names a byte range of
the sibling IMG.  That range is a 0x20-byte relocatable container with magic
``AERA`` (see :mod:`gvcslib.container`).  Inside the relocated payload sits a
texture dictionary whose individual rasters are introduced by a 0x10-byte
*texture descriptor*:

    +0x00 u8   descByte0
    +0x01 u8   descByte1   (log2 width  = low nibble, log2 height = high nibble)
    +0x02 u16  marker      == 0x0012  (the "RW-PSP raster" tag)
    +0x04 u32  texel_ptr   == (descriptor offset + 0x10): the texels follow
    +0x08 u32  pad[2]      (zero)

``descByte0`` packs:

    bits [2:0]  TPSM format  (0=565, 1=5551, 2=4444, 3=8888, 4=T4, 5=T8)
    bit  [3]    swizzle flag
    bits [6:4]  maxMipLevel  -> mip count = (descByte0 & 0x70) >> 4 (>=1)
    bit  [7]    mip-enable

After the descriptor come the texels: the FULL mip chain (every level packed
contiguously, level 0 first), then a fixed 0x20-byte name/flags trailer, then
the CLUT (palette).  For palettised formats the CLUT is 256 (T8) or 16 (T4)
RGBA8888 entries, alpha LAST byte.  ``decode`` returns one :class:`RwTexture`
per descriptor in the blob.

Swizzle
-------
The PSP GE stores texels in 16-byte by 8-row blocks, blocks laid out
left-to-right then top-to-bottom, each block written as contiguous rows.  The
swizzle operates on the *byte* image whose width in bytes is::

    T4   -> max(W // 2, 16)      (4 bpp, 2 texels/byte)
    T8   -> W                    (8 bpp)
    565/5551/4444 (16bpp) -> W * 2
    8888 (32bpp)          -> W * 4

A mip level whose height is < 8 (or whose byte width is < 16) stores a partial
final block-row; :func:`unswizzle`/:func:`swizzle` clamp the row count per
block-row accordingly, so they remain an exact involution for every level.

Round-trip contract
--------------------
``encode(decode(blob))`` reproduces the original blob byte-exact: every texel
mip level is re-swizzled from the decoded index/pixel planes and the CLUT is
repacked in place, while every non-texel byte (container header, descriptors,
name trailers, padding, other resources) is preserved verbatim from the source
blob.
"""
import struct

from ._io import u8, u16, u32
from .container import Container

# AERA container magic ('AERA' little-endian).
MAGIC_AERA = 0x41455241

# u16 tag that introduces an RW-PSP raster descriptor (the "0x0012" marker).
RASTER_MARKER = 0x0012

# texture descriptor is 0x10 bytes; texels immediately follow it.
DESC_SIZE = 0x10

# fixed-size name/flags trailer between the mip chain and the CLUT.
NAME_TRAILER_SIZE = 0x20

# PSP swizzle block geometry (bytes wide x rows tall).
BLOCK_W = 16
BLOCK_H = 8

# TPSM formats.
FMT_565 = 0
FMT_5551 = 1
FMT_4444 = 2
FMT_8888 = 3
FMT_T4 = 4
FMT_T8 = 5

FMT_NAMES = {
    FMT_565: "565", FMT_5551: "5551", FMT_4444: "4444",
    FMT_8888: "8888", FMT_T4: "T4", FMT_T8: "T8",
}

# bits-per-pixel of each TPSM format.
FMT_BPP = {
    FMT_565: 16, FMT_5551: 16, FMT_4444: 16, FMT_8888: 32,
    FMT_T4: 4, FMT_T8: 8,
}

# palette entry count (0 = non-palettised).
FMT_CLUT_ENTRIES = {FMT_T4: 16, FMT_T8: 256}

CLUT_ENTRY_BYTES = 4  # RGBA8888


# --------------------------------------------------------------------------
# swizzle
# --------------------------------------------------------------------------
def byte_width(fmt, w):
    """Width in *bytes* of one texel row for ``fmt`` at pixel width ``w``."""
    if fmt == FMT_T4:
        return max(w // 2, BLOCK_W)
    if fmt == FMT_T8:
        return w
    if fmt == FMT_8888:
        return w * 4
    # 16bpp: 565 / 5551 / 4444
    return w * 2


def unswizzle(src, wbytes, h):
    """De-swizzle a PSP texel byte plane (``wbytes`` x ``h``) into linear order.

    The plane is read as 16-byte x 8-row blocks (rows-within-block contiguous,
    blocks left-to-right then top-to-bottom).  A final partial block-row
    (``h`` not a multiple of 8) is handled by clamping its row count.
    """
    out = bytearray(wbytes * h)
    bxn = wbytes // BLOCK_W
    i = 0
    by = 0
    while by * BLOCK_H < h:
        rows = min(BLOCK_H, h - by * BLOCK_H)
        for bx in range(bxn):
            for row in range(rows):
                dst = (by * BLOCK_H + row) * wbytes + bx * BLOCK_W
                out[dst:dst + BLOCK_W] = src[i:i + BLOCK_W]
                i += BLOCK_W
        by += 1
    return bytes(out)


def swizzle(src, wbytes, h):
    """Swizzle a linear texel byte plane (``wbytes`` x ``h``) into PSP order.

    Exact inverse of :func:`unswizzle`.
    """
    out = bytearray(wbytes * h)
    bxn = wbytes // BLOCK_W
    i = 0
    by = 0
    while by * BLOCK_H < h:
        rows = min(BLOCK_H, h - by * BLOCK_H)
        for bx in range(bxn):
            for row in range(rows):
                s = (by * BLOCK_H + row) * wbytes + bx * BLOCK_W
                out[i:i + BLOCK_W] = src[s:s + BLOCK_W]
                i += BLOCK_W
        by += 1
    return bytes(out)


# --------------------------------------------------------------------------
# mip-chain geometry
# --------------------------------------------------------------------------
def mip_level_bytes(fmt, w, h):
    """Stored byte size of one mip level (== byte_width * h)."""
    return byte_width(fmt, w) * h


def mipchain_bytes(fmt, w, h, mipcount):
    """Total stored byte size of a full mip chain (level 0 .. mipcount-1)."""
    total = 0
    for lvl in range(mipcount):
        lw = max(w >> lvl, 1)
        lh = max(h >> lvl, 1)
        total += mip_level_bytes(fmt, lw, lh)
    return total


# --------------------------------------------------------------------------
# pixel unpacking (palettised + direct-colour)
# --------------------------------------------------------------------------
def _expand5(v):
    return (v << 3) | (v >> 2)


def _expand6(v):
    return (v << 2) | (v >> 4)


def _expand4(v):
    return (v << 4) | v


def _decode565(buf, n):
    out = bytearray(n * 4)
    for p in range(n):
        c = struct.unpack_from("<H", buf, p * 2)[0]
        r = _expand5(c & 0x1f)
        g = _expand6((c >> 5) & 0x3f)
        b = _expand5((c >> 11) & 0x1f)
        out[p * 4:p * 4 + 4] = bytes((r, g, b, 0xff))
    return bytes(out)


def _decode5551(buf, n):
    out = bytearray(n * 4)
    for p in range(n):
        c = struct.unpack_from("<H", buf, p * 2)[0]
        r = _expand5(c & 0x1f)
        g = _expand5((c >> 5) & 0x1f)
        b = _expand5((c >> 10) & 0x1f)
        a = 0xff if (c >> 15) & 1 else 0x00
        out[p * 4:p * 4 + 4] = bytes((r, g, b, a))
    return bytes(out)


def _decode4444(buf, n):
    out = bytearray(n * 4)
    for p in range(n):
        c = struct.unpack_from("<H", buf, p * 2)[0]
        r = _expand4(c & 0x0f)
        g = _expand4((c >> 4) & 0x0f)
        b = _expand4((c >> 8) & 0x0f)
        a = _expand4((c >> 12) & 0x0f)
        out[p * 4:p * 4 + 4] = bytes((r, g, b, a))
    return bytes(out)


# --------------------------------------------------------------------------
# decoded texture object
# --------------------------------------------------------------------------
class RwTexture:
    """One decoded RW-PSP raster.

    Attributes:
        w, h:        base (level 0) pixel dimensions.
        fmt:         TPSM format id (see FMT_* constants).
        mipcount:    number of stored mip levels.
        indices:     for palettised formats, level-0 UN-swizzled palette
                     indices (bytes, length w*h, one byte each).  None for
                     direct-colour formats.
        pixels:      for direct-colour formats, level-0 UN-swizzled raw texels
                     (bytes, length w*h*bpp/8).  None for palettised formats.
        palette:     CLUT bytes (entries*4 RGBA8888, alpha last), or b"" for
                     direct-colour formats.
        csa:         clut start address offset applied (entries, usually 0).
        desc_off:    blob offset of the 0x10-byte descriptor.
        texel_off:   blob offset of the level-0 texels.
        clut_off:    blob offset of the CLUT (or texel-region end for direct).
        mip_offsets: list of (blob_offset, byte_size) per stored mip level.
        raw:         the source blob (preserved for byte-exact encode).
    """

    def __init__(self, w, h, fmt, mipcount, indices, pixels, palette, csa,
                 desc_off, texel_off, clut_off, mip_offsets, raw):
        self.w = w
        self.h = h
        self.fmt = fmt
        self.mipcount = mipcount
        self.indices = indices
        self.pixels = pixels
        self.palette = palette
        self.csa = csa
        self.desc_off = desc_off
        self.texel_off = texel_off
        self.clut_off = clut_off
        self.mip_offsets = mip_offsets
        self.raw = bytes(raw)

    @property
    def fmt_name(self):
        return FMT_NAMES.get(self.fmt, "?%d" % self.fmt)

    @property
    def is_palettised(self):
        return self.fmt in FMT_CLUT_ENTRIES

    def __repr__(self):
        return ("RwTexture(%dx%d %s mips=%d desc=0x%x texel=0x%x clut=0x%x)"
                % (self.w, self.h, self.fmt_name, self.mipcount,
                   self.desc_off, self.texel_off, self.clut_off))

    def decode_rgba(self):
        """Return level-0 RGBA8888 pixels (row-major, alpha last), length w*h*4."""
        n = self.w * self.h
        if self.is_palettised:
            pal = self.palette
            csa = self.csa * CLUT_ENTRY_BYTES
            out = bytearray(n * 4)
            idx = self.indices
            for p in range(n):
                s = csa + idx[p] * CLUT_ENTRY_BYTES
                out[p * 4:p * 4 + 4] = pal[s:s + 4]
            return bytes(out)
        if self.fmt == FMT_8888:
            return bytes(self.pixels[:n * 4])
        if self.fmt == FMT_565:
            return _decode565(self.pixels, n)
        if self.fmt == FMT_5551:
            return _decode5551(self.pixels, n)
        if self.fmt == FMT_4444:
            return _decode4444(self.pixels, n)
        raise ValueError("unsupported fmt %d" % self.fmt)

    def to_png(self, path):
        """Write the level-0 texture to a PNG file using PIL."""
        from PIL import Image
        img = Image.frombytes("RGBA", (self.w, self.h), self.decode_rgba())
        img.save(path)
        return path


# --------------------------------------------------------------------------
# descriptor discovery + decode
# --------------------------------------------------------------------------
def _find_descriptors(blob):
    """Return the blob offsets of every RW-PSP raster descriptor.

    A genuine descriptor has the 0x0012 marker at +0x02 *and* a self-pointer
    ``texel_ptr == desc_off + 0x10`` at +0x04 (the texels always immediately
    follow the 0x10-byte descriptor).  This strict pairing rejects the
    incidental 0x0012 half-words that occur inside texel data.
    """
    offs = []
    marker = struct.pack("<H", RASTER_MARKER)
    pos = blob.find(marker, 2)
    while pos != -1:
        desc_off = pos - 2
        if desc_off >= 0 and desc_off + DESC_SIZE <= len(blob):
            if u32(blob, desc_off + 4) == desc_off + DESC_SIZE:
                offs.append(desc_off)
        pos = blob.find(marker, pos + 2)
    return offs


def _decode_one(blob, desc_off):
    """Decode a single RW-PSP raster at ``desc_off`` into an :class:`RwTexture`."""
    b0 = u8(blob, desc_off)
    b1 = u8(blob, desc_off + 1)
    fmt = b0 & 0x07
    if fmt not in FMT_NAMES:
        raise ValueError("bad TPSM format %d at 0x%x" % (fmt, desc_off))
    mipcount = (b0 & 0x70) >> 4
    if mipcount == 0:
        mipcount = 1
    w = 1 << (b1 & 0x0f)
    h = 1 << ((b1 >> 4) & 0x0f)

    texel_off = desc_off + DESC_SIZE

    # gather every mip level's (offset, size).
    mip_offsets = []
    off = texel_off
    for lvl in range(mipcount):
        lw = max(w >> lvl, 1)
        lh = max(h >> lvl, 1)
        sz = mip_level_bytes(fmt, lw, lh)
        mip_offsets.append((off, sz))
        off += sz
    mipchain_end = off

    indices = None
    pixels = None
    palette = b""
    csa = 0
    clut_off = mipchain_end

    if fmt in FMT_CLUT_ENTRIES:
        # full mip chain, then a fixed 0x20-byte name/flags trailer, then CLUT.
        clut_off = mipchain_end + NAME_TRAILER_SIZE
        entries = FMT_CLUT_ENTRIES[fmt]
        clut_bytes = entries * CLUT_ENTRY_BYTES
        if clut_off + clut_bytes > len(blob):
            raise ValueError("CLUT overruns blob at 0x%x" % clut_off)
        palette = blob[clut_off:clut_off + clut_bytes]
        # level-0 indices (un-swizzled, then nibble-expanded for T4).
        wb = byte_width(fmt, w)
        lin = unswizzle(blob[texel_off:texel_off + wb * h], wb, h)
        # Level-0 indices, read with the STORED ROW STRIDE wb (bytes/row). For T4
        # narrow widths byte_width pads wb up to >=16, so each row holds w//2 real
        # index bytes followed by padding; a DENSE read (lin[p>>1]) walks into that
        # padding and shears every row after the first. Strided read is correct for
        # all widths (mirrors gvcslib.resident_raster.decode_resident_raster).
        if fmt == FMT_T8:
            idx = bytearray(w * h)
            for y in range(h):
                row = y * wb
                for x in range(w):
                    idx[y * w + x] = lin[row + x]
            indices = bytes(idx)
        else:  # T4: low nibble = even pixel, high nibble = odd pixel.
            idx = bytearray(w * h)
            for y in range(h):
                row = y * wb
                for x in range(w):
                    byte = lin[row + (x >> 1)]
                    idx[y * w + x] = byte & 0x0f if (x & 1) == 0 else (byte >> 4) & 0x0f
            indices = bytes(idx)
    else:
        # direct-colour: un-swizzle level 0 into linear pixels.
        wb = byte_width(fmt, w)
        if texel_off + wb * h > len(blob):
            raise ValueError("texels overrun blob at 0x%x" % texel_off)
        pixels = unswizzle(blob[texel_off:texel_off + wb * h], wb, h)

    return RwTexture(w, h, fmt, mipcount, indices, pixels, palette, csa,
                     desc_off, texel_off, clut_off, mip_offsets, blob)


def decode(blob, all=False):
    """Decode RW-PSP texture(s) from an IMG-cell blob.

    ``blob`` may be the raw bytes of an ``AERA`` container, a
    :class:`~gvcslib.container.Container`, or already an inflated payload.
    Returns the first :class:`RwTexture` by default; pass ``all=True`` to get
    the full list of every raster descriptor found in the blob.
    """
    if isinstance(blob, Container):
        data = blob.payload
    else:
        data = bytes(blob)
        # accept a zlib-wrapped or raw container; AERA blobs are raw.
        if data[:2] in (b"\x78\xda", b"\x78\x9c", b"\x78\x01"):
            data = Container.load(data).payload

    offs = _find_descriptors(data)
    if not offs:
        raise ValueError("no RW-PSP raster descriptors found in blob")
    texs = [_decode_one(data, o) for o in offs]
    return texs if all else texs[0]


def decode_all(blob):
    """Convenience: decode every RW-PSP raster in the blob (list)."""
    return decode(blob, all=True)


# --------------------------------------------------------------------------
# encode (byte-exact for the texel + CLUT region)
# --------------------------------------------------------------------------
def _repack_palettised(tex, out):
    """Re-swizzle level-0 indices and repack CLUT into ``out`` (bytearray)."""
    fmt = tex.fmt
    w, h = tex.w, tex.h
    wb = byte_width(fmt, w)
    if fmt == FMT_T8:
        lin = tex.indices
    else:  # T4: pack two indices per byte (low nibble = even pixel), ROW-STRIDED.
        # the byte plane is padded to wb>=16 per row; preserve the original padding
        # bytes, overwriting only the meaningful nibbles at the stored row stride
        # (must match the strided decode read so the round-trip stays byte-exact).
        packed = bytearray(unswizzle(tex.raw[tex.texel_off:tex.texel_off + wb * h], wb, h))
        for y in range(h):
            row = y * wb
            for x in range(w):
                bi = row + (x >> 1)
                v = tex.indices[y * w + x] & 0x0f
                if (x & 1) == 0:
                    packed[bi] = (packed[bi] & 0xf0) | v
                else:
                    packed[bi] = (packed[bi] & 0x0f) | (v << 4)
        lin = bytes(packed)
    out[tex.texel_off:tex.texel_off + wb * h] = swizzle(lin, wb, h)
    out[tex.clut_off:tex.clut_off + len(tex.palette)] = tex.palette


def encode(tex):
    """Re-encode a decoded texture (or list) back to whole-blob bytes.

    Byte-exact: starts from ``tex.raw`` and rewrites only each texture's
    level-0 texel plane (re-swizzled from the decoded indices/pixels) and CLUT
    region.  Lower mip levels, descriptors, name trailers, padding and any
    other resources in the blob are preserved verbatim.
    """
    texs = tex if isinstance(tex, (list, tuple)) else [tex]
    out = bytearray(texs[0].raw)
    for t in texs:
        if t.is_palettised:
            _repack_palettised(t, out)
        else:
            wb = byte_width(t.fmt, t.w)
            out[t.texel_off:t.texel_off + wb * t.h] = swizzle(t.pixels, wb, t.h)
    return bytes(out)
