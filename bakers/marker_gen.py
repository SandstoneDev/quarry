#!/usr/bin/env python3
"""marker_gen - generate the entry/exit marker cone (effects/marker.bin).

The engine draws a translucent cone over every enterable door. The mesh is a plain
12-segment cone with a vertex-colour alpha gradient (solid at the base, clear at the
tip) and no texture at all, so it is generated here rather than carried around as a
blob: nothing about it comes from, or needs, the game disc.

PRP1 (CProp_Load): magic[4], u16 nv, ni, texW, texH, alphaMode, clutEntries,
u32 texelLen, clutLen, then nv * (float u, v; u32 rgba; float x, y, z) and ni u16
indices. texW/texH/texelLen/clutLen are 0 - untextured.

Usage: python marker_gen.py <out marker.bin>
"""
import math
import struct
import sys

SEGMENTS = 12
RADIUS = 0.43          # world units; the engine scales per marker
TIP_Z = -1.02          # the marker points DOWN at the doorway, tip below the ring
RING_Z = 0.29
LEVELS_AMODE = 0x0201  # levels(1) | amode(2 = alpha blend) << 8, as CProp reads it

RGB = (0xFF, 0xEE, 0x00)   # marker yellow
TIP_ALPHA = 0xEB           # near-solid where it points
RING_ALPHA = 0x1E          # fading out at the wide end


def _rgba(a):
    r, g, b = RGB
    return (a << 24) | (b << 16) | (g << 8) | r


def build():
    # vertex 0 is the tip, then one ring. Untextured, so the UVs are all zero and
    # the shape is carried entirely by the vertex-colour alpha ramp.
    verts = [(0.0, 0.0, _rgba(TIP_ALPHA), 0.0, 0.0, TIP_Z)]
    for i in range(SEGMENTS):
        th = 2.0 * math.pi * i / SEGMENTS
        verts.append((0.0, 0.0, _rgba(RING_ALPHA),
                      RADIUS * math.cos(th), RADIUS * math.sin(th), RING_Z))

    idx = []
    for i in range(SEGMENTS):                   # cone wall, tip -> ring
        idx += [0, 1 + i, 1 + (i + 1) % SEGMENTS]
    for i in range(1, SEGMENTS - 1):            # cap the wide end
        idx += [1, 1 + i + 1, 1 + i]

    blob = b"PRP1" + struct.pack("<6H2I", len(verts), len(idx), 0, 0,
                                 LEVELS_AMODE, 0, 0, 0)
    for u, v, c, x, y, z in verts:
        blob += struct.pack("<2fI3f", u, v, c, x, y, z)
    blob += struct.pack("<%dH" % len(idx), *idx)
    return blob, len(verts), len(idx) // 3


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "marker.bin"
    blob, nv, ntri = build()
    open(out, "wb").write(blob)
    print("marker.bin: %d verts %d tris %d bytes -> %s" % (nv, ntri, len(blob), out))


if __name__ == "__main__":
    main()
