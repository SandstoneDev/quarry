#!/usr/bin/env python3
"""Bake SA sky textures from particle.txd into clouds.bin (RGBA8888 planes the
PSP engine fopen-uploads like font.bin). Port-side wrapper over the READ-ONLY
gvcslib codecs.

clouds.bin layout (little-endian):
  'CLDS'                4
  count                 u32
  per texture:
    name                16 bytes (zero-padded)
    w, h                2x u16
    w*h*4 RGBA8888 bytes (row-major)

We bake cloud1 (SA's gpCloudTex low-cloud puff, drawn as individual ring sprites),
cloudhigh (the high streaks) and coronamoon (the night moon disc).
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (device id 6)

# Source is the disc's own particle.txd, read with the PS2 codec. The PC codec that
# used to be wired here cannot read a device-id-6 dictionary at all.
SA_ROOT = os.environ.get("SA_ROOT", "")
TXD  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SA_ROOT, "models", "particle.txd")
OUT  = sys.argv[2] if len(sys.argv) > 2 else "clouds.bin"
# b613: +coronastar (the SA sun/corona sprite with the RAYS - CCoronas
# gpCoronaTexture[CORONATYPE_SHINYSTAR]; our procedural radial had no rays,
WANT = ["cloudhigh", "coronamoon", "cloud1", "coronastar", "coronaringa"]


def main():
    texs = {k.lower(): v for k, v in sa_txd.decode(open(TXD, "rb").read()).items()}

    buf = bytearray(b"CLDS")
    have = [n for n in WANT if n in texs]
    buf += struct.pack("<I", len(have))
    for name in have:
        w, h, rgba = texs[name]
        assert len(rgba) == w * h * 4, "%s: %d != %d" % (name, len(rgba), w*h*4)
        # b624: coronastar ships RAW again (rays in RGB, opaque alpha) - the
        # runtime draws it with a pure ONE:ONE additive (black cancels itself),
        # which is how the PS2 renders it; the b618 alpha=luminance remap DILUTED
        if name == "cloud1":
            # cloud1 ships its shape in luminance over a black field with an opaque
            # alpha; alpha-blended that draws a black rectangle. Remap to white RGB +
            # alpha = luminance, so the cloud SHAPE comes from alpha and the COLOUR
            # comes entirely from the per-vertex tint (timecycle low-cloud + rain).
            rgba = bytearray(rgba)
            for p in range(0, len(rgba), 4):
                lum = (rgba[p]*77 + rgba[p+1]*150 + rgba[p+2]*29) >> 8
                rgba[p] = rgba[p+1] = rgba[p+2] = 255
                rgba[p+3] = lum
            rgba = bytes(rgba)
        buf += name.encode("ascii")[:16].ljust(16, b"\x00")
        buf += struct.pack("<HH", w, h)
        buf += rgba
        print("  baked %s %dx%d" % (name, w, h))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "wb").write(buf)
    print("wrote %s (%d bytes, %d textures)" % (OUT, len(buf), len(have)))


if __name__ == "__main__":
    main()
