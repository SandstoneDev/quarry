#!/usr/bin/env python3
"""cull_air_bake - extract EXTRA_AIR_RESISTANCE (flag 0x4000) cull zones from cull.ipl
and write data/cull_air.bin: the freeway speed-cap zones (SA CCullZones).

Format (little-endian):  u32 magic 'CULA' | u32 count | count x {f32 minX,minY,minZ,maxX,maxY,maxZ}

A regular CULL row is 11 fields: cx cy cz, v1x v1y minZ, v2x v2y maxZ, flags, flags2.
SA builds an oriented parallelogram (corner + 2 edges); 23/29 retail air zones are
axis-aligned and 6 slightly rotated, so we bake the tight AABB enclosing the 4 XY corners
(a hair larger on the 6 - fine for a demake speed-cap).

Usage: python cull_air_bake.py <cull.ipl> <out cull_air.bin>
"""
import struct, sys

EXTRA_AIR = 0x4000
MAGIC = 0x414C5543  # 'CULA'


def parse_cull_rows(path):
    rows, sec = [], False
    for raw in open(path, errors="replace"):
        s = raw.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low == "cull":
            sec = True; continue
        if low == "end":
            sec = False; continue
        if not sec:
            continue
        rows.append(s.replace(",", " ").split())
    return rows


def zone_aabb(t):
    """11-field regular row -> (minx,miny,minz,maxx,maxy,maxz). corner = pos-v1-v2,
    edges = 2*v1, 2*v2; AABB over the 4 XY corners."""
    x, y = float(t[0]), float(t[1])
    v1x, v1y = float(t[3]), float(t[4])
    minz = float(t[5])
    v2x, v2y = float(t[6]), float(t[7])
    maxz = float(t[8])
    cx, cy = x - v1x - v2x, y - v1y - v2y
    e1x, e1y, e2x, e2y = 2*v1x, 2*v1y, 2*v2x, 2*v2y
    xs = [cx, cx+e1x, cx+e2x, cx+e1x+e2x]
    ys = [cy, cy+e1y, cy+e2y, cy+e1y+e2y]
    return min(xs), min(ys), minz, max(xs), max(ys), maxz


def main():
    if len(sys.argv) != 3:
        print(__doc__); return 1
    rows = parse_cull_rows(sys.argv[1])
    zones = []
    for t in rows:
        if len(t) != 11:                 # 14 = mirror zone, skip
            continue
        try:
            flags = int(float(t[9]))
        except ValueError:
            continue
        if flags & EXTRA_AIR:
            zones.append(zone_aabb(t))
    if len(zones) > 64:                  # engine CULLZONE_MAX - loader rejects a bigger file
        print("WARNING: %d zones > engine cap 64; raise CULLZONE_MAX in CullZones.c or the "
              "loader will treat cull_air.bin as invalid (feature inert)" % len(zones))
    with open(sys.argv[2], "wb") as f:
        f.write(struct.pack("<II", MAGIC, len(zones)))
        for z in zones:
            f.write(struct.pack("<6f", *z))
    print("cull_air.bin: %d EXTRA_AIR_RESISTANCE zone(s)" % len(zones))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
