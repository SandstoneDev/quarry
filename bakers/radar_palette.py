"""One radar tile between RGBA and 4-bit indices with its own palette.

The map is nearly flat in colour: measured across all 144 tiles on the disc, the busiest
uses 15 colours. Fifteen fits a 4-bit palette exactly, so the tile keeps its native
128x128 resolution AND its exact colours, where the old path stitched everything into one
atlas and threw away two thirds of the resolution to fit the PSP's 512-texel limit.

Kept free of I/O so the transform can be tested without a disc image.
"""
import numpy as np

MAX_COLOURS = 16          # a 4-bit index


def pack_tile(rgba):
    """(h, w, 4) uint8 -> (clut, packed). Raises if the tile needs more than 16 colours.

 Refusing is deliberate. Quantising here would silently dull the map, and a bake that
 stops with a number in the message is easier to act on than a map nobody can explain.
 """
    h, w = rgba.shape[:2]
    if w % 2:
        raise ValueError("tile width %d is odd; two pixels share a byte" % w)
    flat = rgba.reshape(-1, 4)
    clut, inverse = np.unique(flat, axis=0, return_inverse=True)
    if len(clut) > MAX_COLOURS:
        raise ValueError("tile needs %d colours, a 4-bit palette holds %d"
                         % (len(clut), MAX_COLOURS))
    idx = inverse.reshape(h, w).astype(np.uint8)
    # low nibble is the even x - same order the world texture pipeline uses
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    return clut.astype(np.uint8), packed


def unpack_tile(clut, packed, w, h):
    """Inverse of pack_tile, for the test and for anyone checking a baked file."""
    idx = np.zeros((h, w), dtype=np.uint8)
    idx[:, 0::2] = packed & 0x0F
    idx[:, 1::2] = packed >> 4
    return clut[idx]
