#!/usr/bin/env python3
"""Bake timecycP.dat into data/timecyc.bin - the RUNTIME timecycle table.

Replaces the old timecyc_bake.py (which emitted src/game_sa/timecyc_table.h,
compiling the game's colour data INTO the engine binary - a legalization
violation: the engine must ship asset-free, all game data arrives via the
Quarry converter from the user's own disc).

Source column order mirrors CTimeCycle::Initialise (52 columns, 23 weathers x
8 sampled hours = 184 data lines; timecycP alphas are already PS2 half-range,
the weather-16@20h row is intact here unlike PC timecyc.dat - see the old
script's header for the full story).

timecyc.bin layout (little-endian):
 'TCY1' u8 hours=8 u8 weathers=23 u16 reserved=0
 then hours x weathers rows, 46 bytes each, [hour][weather] order:
 u8 amb[3] ambObj[3] skyTop[3] skyBot[3] sunCore[3] sunCorona[3] (18)
 i16 far i16 fog (4)
 f32 sunSize (4)
 u8 lowCloud[3] topCloud[3] water[4] (10)
 u8 postfx1[4] postfx2[4] (8)
 u8 highlight u8 dirMult (2)

Usage: timecyc_bin_bake.py <timecycP.dat> <out timecyc.bin>
"""
import struct
import sys

NW = 23
NH = 8


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out = sys.argv[1], sys.argv[2]

    rows = []
    for raw in open(src, "r", errors="replace").read().splitlines():
        t = raw.strip()
        if not t or t[0] == "/":
            continue
        nums = t.replace("\t", " ").split()
        if len(nums) < 40:
            continue
        rows.append(nums)
    assert len(rows) >= NW * NH, "expected >=184 data lines, got %d" % len(rows)

    def ci(r, i):    return int(round(float(r[i - 1])))
    def cf(r, i):    return float(r[i - 1])

    # source is weather-major (8 hour lines per weather); table is [hour][weather]
    table = [[None] * NW for _ in range(NH)]
    k = 0
    for w in range(NW):
        for hi in range(NH):
            r = rows[k]; k += 1
            assert len(r) == 52, "timecycP row %d has %d cols" % (k - 1, len(r))
            table[hi][w] = r

    buf = bytearray(b"TCY1")
    buf += struct.pack("<BBH", NH, NW, 0)
    for hi in range(NH):
        for w in range(NW):
            r = table[hi][w]
            u8s = [ci(r, 1), ci(r, 2), ci(r, 3),          # amb
                   ci(r, 4), ci(r, 5), ci(r, 6),          # ambObj
                   ci(r, 10), ci(r, 11), ci(r, 12),       # skyTop
                   ci(r, 13), ci(r, 14), ci(r, 15),       # skyBot
                   ci(r, 16), ci(r, 17), ci(r, 18),       # sunCore
                   ci(r, 19), ci(r, 20), ci(r, 21)]       # sunCorona
            buf += struct.pack("<18B", *u8s)
            buf += struct.pack("<hh", int(round(cf(r, 28))), int(round(cf(r, 29))))
            buf += struct.pack("<f", cf(r, 22))
            buf += struct.pack("<10B",
                               ci(r, 31), ci(r, 32), ci(r, 33),           # lowCloud
                               ci(r, 34), ci(r, 35), ci(r, 36),           # topCloud
                               int(round(cf(r, 37))), int(round(cf(r, 38))),
                               int(round(cf(r, 39))), int(round(cf(r, 40))))  # water RGBA
            # PS2 ColourFilter: file order a1 R1 G1 B1 a2 R2 G2 B2 -> stored RGBA
            buf += struct.pack("<8B",
                               ci(r, 42), ci(r, 43), ci(r, 44), ci(r, 41),
                               ci(r, 46), ci(r, 47), ci(r, 48), ci(r, 45))
            buf += struct.pack("<BB", ci(r, 50), int(round(cf(r, 52) * 100.0)))
    open(out, "wb").write(buf)
    print("wrote %s (%d bytes, %dx%d rows)" % (out, len(buf), NH, NW))
    return 0


if __name__ == "__main__":
    sys.exit(main())
