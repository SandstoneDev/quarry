"""XTX - the original publisher Leeds PSP texture (the source game: the source console title).

An XTX file is the shared 0x20-byte 'tex' (0x00746578) relocatable container
(see gvcslib.container) followed by one (single-texture) or many (dictionary)
texture descriptors. The file is NOT zlib-compressed; the header sits at file
offset 0.

Single-texture, 8bpp palettized layout (the byte-exact target of this codec):

 +0x00 0x20-byte container header
 ... one texture descriptor (in-header for SPLASH1; in a footer block
 after the palette for the LOADSC*/MPLOAD* "raw" variant)
 pix_off .. pix_off + W*H PSP-swizzled 8bpp index pixels
 pix_off + W*H .. + 0x400 256 * RGBA8888 palette (alpha LAST byte)
 reloc_off u32 fixup-site list (reloc_count entries)

The texture descriptor is located generically via the relocation table: the
LAST reloc site (index reloc_count-1) is the offset of the descriptor's
``pix_off`` field (a relocated pointer to the pixel data). At that field:

 +0x00 u32 pix_off (absolute payload offset of the pixels)
 +0x06 u8 W exponent (width = 1 << byte)
 +0x07 u8 H exponent (height = 1 << byte)

PSP 8bpp swizzle: pixels are stored as 16-byte-wide x 8-row blocks, blocks
laid out left-to-right then top-to-bottom, each block stored as 8 contiguous
16-byte rows.

decode(data) -> Texture(w, h, indices, palette, ...). encode(tex) -> bytes,
byte-exact for the single-texture variants by preserving every non-pixel byte
verbatim and only re-swizzling the index plane back into the pixel region.

The dictionary variant (e.g. EMPHUD, container +0x14 reloc_count != 7) is
decoded best-effort only and is NOT covered by the byte-exact gate.
"""
import struct

from ._io import u8, u32
from .container import Container, MAGIC_TEX

# Single-texture XTX files in the corpus all carry exactly this many reloc
# sites; the dictionary variant carries many more.
SINGLE_RELOC_COUNT = 7

PALETTE_ENTRIES = 256
PALETTE_BYTES = PALETTE_ENTRIES * 4

# PSP 8bpp swizzle block geometry.
BLOCK_W = 16  # bytes (== pixels for 8bpp) per swizzle block row
BLOCK_H = 8   # rows per swizzle block


def unswizzle(src, w, h):
    """De-swizzle a PSP 8bpp pixel plane into a linear w*h index buffer."""
    out = bytearray(w * h)
    bx_n = w // BLOCK_W
    by_n = h // BLOCK_H
    i = 0
    for by in range(by_n):
        for bx in range(bx_n):
            for row in range(BLOCK_H):
                dst = (by * BLOCK_H + row) * w + bx * BLOCK_W
                out[dst:dst + BLOCK_W] = src[i:i + BLOCK_W]
                i += BLOCK_W
    return bytes(out)


def swizzle(src, w, h):
    """Swizzle a linear w*h index buffer into a PSP 8bpp pixel plane."""
    out = bytearray(w * h)
    bx_n = w // BLOCK_W
    by_n = h // BLOCK_H
    i = 0
    for by in range(by_n):
        for bx in range(bx_n):
            for row in range(BLOCK_H):
                s = (by * BLOCK_H + row) * w + bx * BLOCK_W
                out[i:i + BLOCK_W] = src[s:s + BLOCK_W]
                i += BLOCK_W
    return bytes(out)


class Texture:
    """A single decoded XTX texture.

 Attributes:
 w, h: pixel dimensions.
 indices: bytes, length w*h, UN-swizzled 8bpp palette indices (row-major).
 palette: bytes, 256*4 RGBA8888 (alpha is the LAST byte of each entry).
 pix_off: payload offset of the (swizzled) pixel plane.
 desc_off: payload offset of the descriptor pix_off field.
 raw: the original whole-file bytes (preserved for byte-exact encode).
 """

    def __init__(self, w, h, indices, palette, pix_off, desc_off, raw):
        self.w = w
        self.h = h
        self.indices = bytes(indices)
        self.palette = bytes(palette)
        self.pix_off = pix_off
        self.desc_off = desc_off
        self.raw = bytes(raw)

    def __repr__(self):
        return "Texture(w=%d, h=%d, pix_off=0x%x)" % (self.w, self.h, self.pix_off)

    def decode_rgba(self):
        """Return bytes of length w*h*4: RGBA8888 pixels (row-major, alpha last)."""
        pal = self.palette
        idx = self.indices
        out = bytearray(self.w * self.h * 4)
        for p, ci in enumerate(idx):
            s = ci * 4
            out[p * 4:p * 4 + 4] = pal[s:s + 4]
        return bytes(out)

    def to_png(self, path):
        """Write the decoded texture to a PNG file using PIL."""
        from PIL import Image
        img = Image.frombytes("RGBA", (self.w, self.h), self.decode_rgba())
        img.save(path)
        return path


def _read_descriptor(data):
    """Locate the single-texture descriptor via the reloc table.

 Returns (pix_off, w, h, desc_off). Raises ValueError for the dictionary
 variant or anything that does not look like a single 8bpp texture.
 """
    c = Container.load(data)
    if c.magic != MAGIC_TEX:
        raise ValueError("not an XTX 'tex' container: magic=0x%08x" % c.magic)
    if c.reloc_count != SINGLE_RELOC_COUNT:
        raise ValueError(
            "not a single-texture XTX (reloc_count=%d, expected %d; likely the "
            "dictionary variant)" % (c.reloc_count, SINGLE_RELOC_COUNT))
    sites = c.reloc_sites()
    desc_off = sites[-1]
    pix_off = u32(data, desc_off)
    w = 1 << u8(data, desc_off + 6)
    h = 1 << u8(data, desc_off + 7)
    if w % BLOCK_W or h % BLOCK_H:
        raise ValueError("texture %dx%d not block-aligned" % (w, h))
    if pix_off + w * h + PALETTE_BYTES > len(data):
        raise ValueError("pixels+palette overrun file")
    return pix_off, w, h, desc_off


def decode(data):
    """Decode a single-texture XTX into a Texture (un-swizzled indices + palette)."""
    data = bytes(data)
    pix_off, w, h, desc_off = _read_descriptor(data)
    swz = data[pix_off:pix_off + w * h]
    indices = unswizzle(swz, w, h)
    palette = data[pix_off + w * h:pix_off + w * h + PALETTE_BYTES]
    return Texture(w, h, indices, palette, pix_off, desc_off, data)


def encode(tex):
    """Re-encode a Texture to whole-file bytes.

 Byte-exact for the single-texture variants: every non-pixel byte (header,
 descriptor, palette region, footer, reloc table) is preserved from
 ``tex.raw``; only the swizzled pixel plane is rebuilt from ``tex.indices``.
 """
    out = bytearray(tex.raw)
    pix = swizzle(tex.indices, tex.w, tex.h)
    out[tex.pix_off:tex.pix_off + tex.w * tex.h] = pix
    # Refresh the palette region too, in case the caller edited it.
    pal_off = tex.pix_off + tex.w * tex.h
    out[pal_off:pal_off + PALETTE_BYTES] = tex.palette
    return bytes(out)
