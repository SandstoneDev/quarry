#!/usr/bin/env python3
"""water_bake - DATA/water.dat -> data/world/water.bin (the sea/lake surface).

water.dat is the original's text water table: one zone per line, each vertex a
7-float group (x y z then four flow/wave parameters this port does not use yet),
followed by a trailing flags int. A line carries 4 vertices for a quad or 3 for a
triangle. Quad corners come in grid order - (x0,y0) (x1,y0) (x0,y1) (x1,y1) --
so the two triangles are (0,1,2) and (1,3,2), which keeps their winding the same.

water.bin (little-endian):
  'WATR' u32 nTris, then nTris * 9 f32 (three vertices, x y z each)

Usage: water_bake.py <water.dat> <out water.bin>
"""
import struct
import sys

MAGIC = b"WATR"
VERT_FLOATS = 7          # x y z + 4 unused flow/wave parameters


def parse(path):
    """water.dat -> [(x,y,z) * 3] per triangle."""
    tris = []
    quads = triangles = skipped = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("processed") or line[0] in "#;":
                continue
            tok = line.split()
            try:
                vals = [float(t) for t in tok]
            except ValueError:
                skipped += 1
                continue
            # trailing flags int, then a whole number of 7-float vertex groups
            nvert = (len(vals) - 1) // VERT_FLOATS
            if nvert not in (3, 4):
                skipped += 1
                continue
            v = [tuple(vals[i * VERT_FLOATS: i * VERT_FLOATS + 3]) for i in range(nvert)]
            if nvert == 4:
                tris.append((v[0], v[1], v[2]))
                tris.append((v[1], v[3], v[2]))
                quads += 1
            else:
                tris.append((v[0], v[1], v[2]))
                triangles += 1
    return tris, quads, triangles, skipped


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    tris, nq, nt, skipped = parse(src)
    if not tris:
        print("water_bake: no water zones parsed from %s" % src)
        return 1
    blob = MAGIC + struct.pack("<I", len(tris))
    for a, b, c in tris:
        blob += struct.pack("<9f", *a, *b, *c)
    open(dst, "wb").write(blob)
    print("water.bin: %d quads + %d triangles -> %d tris, %d bytes%s"
          % (nq, nt, len(tris), len(blob),
             ("  (%d lines skipped)" % skipped) if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
