"""Decode an LVZ-resident RW-PSP raster to RGBA8888 pixels (the source game: the source console title, PSP).

A draw-record *material id* (see :mod:`gvcslib.submesh`) is a SLOT into
``lvz.decode(<zone>.lvz).all_records``. That record's ``desc_ptr`` is a
payload offset of a RESIDENT RW-PSP raster descriptor that lives directly in
the (relocated) LVZ payload - NOT inside an ``AERA`` IMG-cell container.

The on-payload layout at ``desc_off`` is identical to the AERA-cell descriptor
that :mod:`gvcslib.rwtex` already decodes (verified byte-for-byte on BEACH /
MAINLA), so this module deliberately reuses ``rwtex``'s swizzle / mip-geometry /
pixel-unpack primitives instead of re-deriving them:

 +0x00 u8 descByte0
 bits[2:0] TPSM format (0=565 1=5551 2=4444 3=8888 4=T4 5=T8)
 bit [3] swizzle flag (informational; texels are ALWAYS
 stored in PSP 16x8 block order on disk, so we always
 un-swizzle - matches the shipping AERA decoder)
 bits[6:4] maxMipLevel -> stored mip count = max(that, 1)
 bit [7] mip-enable
 +0x01 u8 descByte1 log2(width) = low nibble, log2(height) = high nibble
 +0x02 u16 marker == 0x0012 (the RW-PSP raster tag)
 +0x04 u32 texel_ptr (relocated absolute payload offset; == desc_off+0x10)
 +0x08 u32 pad[2]

Immediately after the 0x10-byte descriptor:
 - the full mip chain (level 0 first, every level contiguous);
 - a fixed 0x20-byte name/flags trailer (holds the ASCII texture name);
 - for palettised formats (T4/T8), the CLUT: 16 (T4) or 256 (T8) RGBA8888
 entries, ALPHA IN THE LAST BYTE.

CLUT alpha convention
---------------------
the console title resident CLUTs store alpha full-range 0..255 in the last byte (observed
values include 0xfe and 0xff); they are NOT the half-range 0..128 form, so no
x2 scaling is applied - the byte is used verbatim, matching ``rwtex``.

T4 nibble order
---------------
Within each index byte the LOW nibble is the even (left) pixel and the HIGH
nibble is the odd (right) pixel - again matching ``rwtex``.
"""
import struct

from . import rwtex
from ._io import u8, u16, u32

# u16 tag that introduces an RW-PSP raster descriptor.
RASTER_MARKER = 0x0012
# descriptor is 0x10 bytes; texels immediately follow.
DESC_SIZE = 0x10
# fixed name/flags trailer between the mip chain and the CLUT.
NAME_TRAILER_SIZE = 0x20

# PSP GE swizzle block width in bytes (texels are stored in 16x8 blocks).
BLOCK_W = 16


def _swizzle_stride(fmt, w):
    """Stored row stride in bytes: logical byte width padded up to 16 bytes.

 ``rwtex.byte_width`` already pads T4 to >=16; this additionally rounds up
 to a 16-byte multiple so the swizzle block geometry is always exact. For
 every raster format/width that occurs in the console title this equals
 ``rwtex.byte_width(fmt, w)`` (verified), so the change is observationally
 inert on real data and only fixes the synthetic narrow-T8 swizzle case.
 """
    wb = rwtex.byte_width(fmt, w)
    return (wb + BLOCK_W - 1) // BLOCK_W * BLOCK_W


def is_resident_raster(payload, desc_off):
    """True if ``desc_off`` in ``payload`` is a genuine RW-PSP raster descriptor.

 Requires the 0x0012 marker at +0x02 *and* the self-pointer
 ``texel_ptr == desc_off + 0x10`` at +0x04, and a known TPSM format. This
 is the same strict pairing :func:`rwtex._find_descriptors` uses, so it
 rejects the incidental 0x0012 half-words inside texel data and the
 geometry resources that share the LVZ master-record array.
 """
    if desc_off < 0 or desc_off + DESC_SIZE > len(payload):
        return False
    if u16(payload, desc_off + 2) != RASTER_MARKER:
        return False
    if u32(payload, desc_off + 4) != desc_off + DESC_SIZE:
        return False
    return (u8(payload, desc_off) & 0x07) in rwtex.FMT_NAMES


def raster_extent(payload, desc_off):
    """Total byte length of the raster at ``desc_off`` (descriptor..end of CLUT).

 Direct-colour formats: descriptor + mip0 texels. Palettised (T4/T8):
 descriptor + full mip chain + 0x20 name trailer + CLUT. Used to bound an
 in-place overwrite so it cannot clobber the next resource.
 """
    if not is_resident_raster(payload, desc_off):
        raise ValueError("no RW-PSP raster descriptor at payload+0x%x" % desc_off)
    b0 = u8(payload, desc_off)
    b1 = u8(payload, desc_off + 1)
    fmt = b0 & 0x07
    mipcount = (b0 & 0x70) >> 4 or 1
    w = 1 << (b1 & 0x0f)
    h = 1 << ((b1 >> 4) & 0x0f)
    off = desc_off + DESC_SIZE
    if fmt not in rwtex.FMT_CLUT_ENTRIES:
        return (off + _swizzle_stride(fmt, w) * h) - desc_off
    for lvl in range(mipcount):
        lw = max(w >> lvl, 1)
        lh = max(h >> lvl, 1)
        off += _swizzle_stride(fmt, lw) * lh
    off += NAME_TRAILER_SIZE
    off += rwtex.FMT_CLUT_ENTRIES[fmt] * rwtex.CLUT_ENTRY_BYTES
    return off - desc_off


def decode_resident_raster(payload, desc_off):
    """Decode the LVZ-resident raster at ``desc_off`` to level-0 RGBA8888.

 Args:
 payload: the *relocated* LVZ container payload bytes
 (``lvz.decode(...).payload`` / ``Container.payload``).
 desc_off: payload offset of the 0x10-byte raster descriptor, i.e. a
 used ``ResourceRecord.desc_ptr`` reached from a draw-record
 material id.

 Returns:
 ``(w, h, fmt_name, rgba_bytes)`` where ``rgba_bytes`` is length
 ``w*h*4`` row-major RGBA8888 with alpha last.

 Raises:
 ValueError if the descriptor is malformed or its texels/CLUT overrun
 the payload.
 """
    if not is_resident_raster(payload, desc_off):
        raise ValueError("no RW-PSP raster descriptor at payload+0x%x" % desc_off)

    b0 = u8(payload, desc_off)
    b1 = u8(payload, desc_off + 1)
    fmt = b0 & 0x07
    mipcount = (b0 & 0x70) >> 4 or 1
    w = 1 << (b1 & 0x0f)
    h = 1 << ((b1 >> 4) & 0x0f)

    texel_off = desc_off + DESC_SIZE
    # The PSP GE swizzle operates on 16-byte-wide blocks, so the *stored* row
    # stride is the logical byte width rounded up to a multiple of 16. For T4
    # this is always already a multiple of 16 (byte_width pads to >=16), so this
    # is a no-op for every raster that occurs in the console title; it only matters for the
    # synthetic narrow T8 (w<16) edge case, keeping the swizzle self-consistent.
    wb = _swizzle_stride(fmt, w)

    # ---- direct-colour formats: un-swizzle level 0, then unpack ------------
    if fmt not in rwtex.FMT_CLUT_ENTRIES:
        end = texel_off + wb * h
        if end > len(payload):
            raise ValueError("texels overrun payload at 0x%x" % texel_off)
        pixels = rwtex.unswizzle(payload[texel_off:end], wb, h)
        n = w * h
        if fmt == rwtex.FMT_8888:
            rgba = bytes(pixels[:n * 4])
        elif fmt == rwtex.FMT_565:
            rgba = rwtex._decode565(pixels, n)
        elif fmt == rwtex.FMT_5551:
            rgba = rwtex._decode5551(pixels, n)
        elif fmt == rwtex.FMT_4444:
            rgba = rwtex._decode4444(pixels, n)
        else:  # pragma: no cover - guarded by is_resident_raster
            raise ValueError("unsupported fmt %d" % fmt)
        return w, h, rwtex.FMT_NAMES[fmt], rgba

    # ---- palettised formats (T4 / T8) --------------------------------------
    # skip the FULL mip chain, then the 0x20 name trailer, to reach the CLUT.
    # Each level's stored size uses the 16-byte-padded swizzle stride.
    off = texel_off
    for lvl in range(mipcount):
        lw = max(w >> lvl, 1)
        lh = max(h >> lvl, 1)
        off += _swizzle_stride(fmt, lw) * lh
    clut_off = off + NAME_TRAILER_SIZE

    entries = rwtex.FMT_CLUT_ENTRIES[fmt]
    clut_bytes = entries * rwtex.CLUT_ENTRY_BYTES
    if clut_off + clut_bytes > len(payload):
        raise ValueError("CLUT overruns payload at 0x%x" % clut_off)
    clut = payload[clut_off:clut_off + clut_bytes]

    # level-0 indices: un-swizzle the byte plane, then read it with the stored
    # ROW STRIDE ``wb`` (bytes/row). For T4 narrow widths ``wb`` is padded up
    # to 16 (``byte_width``), so each row has ``w//2`` real index bytes followed
    # by padding - indexing densely (lin[p>>1]) would walk into that padding
    # and corrupt every row after the first. Strided indexing is correct for
    # all widths and degenerates to the dense case when wb == w (T8) / w//2 (T4).
    end = texel_off + wb * h
    if end > len(payload):
        raise ValueError("texels overrun payload at 0x%x" % texel_off)
    lin = rwtex.unswizzle(payload[texel_off:end], wb, h)

    n = w * h
    rgba = bytearray(n * 4)
    ce = rwtex.CLUT_ENTRY_BYTES
    if fmt == rwtex.FMT_T8:
        for y in range(h):
            row = y * wb
            o = (y * w) * 4
            for x in range(w):
                s = lin[row + x] * ce
                rgba[o:o + 4] = clut[s:s + 4]
                o += 4
    else:  # T4: low nibble = even pixel, high nibble = odd pixel.
        for y in range(h):
            row = y * wb
            o = (y * w) * 4
            for x in range(w):
                byte = lin[row + (x >> 1)]
                idx = byte & 0x0f if (x & 1) == 0 else (byte >> 4) & 0x0f
                s = idx * ce
                rgba[o:o + 4] = clut[s:s + 4]
                o += 4

    return w, h, rwtex.FMT_NAMES[fmt], bytes(rgba)
