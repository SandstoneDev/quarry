#!/usr/bin/env python3
"""Bake the PS2 rom's data/info.zon into data/hud/zones.bin for the b616

zones.bin: 'ZON1' + u32 count + per zone:
 key[8] zero-padded GXT key (the label; CText resolves it, else shown raw)
 min[3]f max[3]f
Smallest-area containing zone wins at runtime (SA nests zones).
"""
import struct
import sys

# argv override (the Quarry converter drives this); defaults keep the dev loop.
SRC = sys.argv[1] if len(sys.argv) > 1 else (
    "data/info.zon")
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    ""


def main():
    zones = []
    for raw in open(SRC, "r", errors="replace").read().splitlines():
        t = raw.strip()
        if not t or t.lower() in ("zone", "end") or t.startswith("/"):
            continue
        p = [x.strip() for x in t.split(",")]
        if len(p) < 10:
            continue
        # name, type, x1,y1,z1, x2,y2,z2, level, textkey
        key = p[9].upper()[:8]
        x1, y1, z1 = float(p[2]), float(p[3]), float(p[4])
        x2, y2, z2 = float(p[5]), float(p[6]), float(p[7])
        zones.append((key, min(x1,x2), min(y1,y2), min(z1,z2),
                           max(x1,x2), max(y1,y2), max(z1,z2)))
    buf = b"ZON1" + struct.pack("<I", len(zones))
    for k, ax, ay, az, bx, by, bz in zones:
        buf += k.encode("ascii").ljust(8, b"\x00")
        buf += struct.pack("<6f", ax, ay, az, bx, by, bz)
    open(OUT, "wb").write(buf)
    print("wrote %s (%d zones, %d bytes)" % (OUT, len(zones), len(buf)))


if __name__ == "__main__":
    main()
