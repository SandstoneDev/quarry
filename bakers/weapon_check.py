#!/usr/bin/env python3
"""weapon_check - read data/weapon.bin the way the engine reads it and print the
numbers every later layer is measured against.

This exists because the weapon table is the one place where a silent parse error
would not look like a parse error: it would look like "the guns feel wrong" three
layers later. Everything here is recomputed from the FILE, not from weapon.dat, so a
baker bug cannot hide behind the source it was baked from.

 python tools/weapon_check.py [path/to/weapon.bin]
"""
import os
import struct
import sys

WEAPON_REC = "<Iffffffffffffffffffhh HHH BBBBBBBxxx".replace(" ", "")
AIM_REC = "<ffffhhhh"
assert struct.calcsize(WEAPON_REC) == 96
assert struct.calcsize(AIM_REC) == 24

NUM_WEAPONS = 47
FIRST_SKILLED, LAST_SKILLED = 22, 32
NUM_SKILLED = LAST_SKILLED - FIRST_SKILLED + 1

FIRE = ["MELEE", "INSTANT", "PROJ", "AREA", "CAMERA", "USE"]
SKILL = ["POOR", "STD", "PRO", "COP"]

FLAG_RELOAD, FLAG_TWIN, FLAG_LONG_RELOAD = 0x1000, 0x800, 0x8000


def idx(wtype, skill):
    """The engine's CWeaponInfo_Get, transcribed. If this and the C disagree, one of
 the two is wrong and the guns will silently read a neighbour's stats."""
    if skill == 1 or not (FIRST_SKILLED <= wtype <= LAST_SKILLED):
        return wtype
    lane = {0: 0, 2: 1, 3: 2}[skill]
    return NUM_WEAPONS + (wtype - FIRST_SKILLED) + lane * NUM_SKILLED


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("assets_build", "weapon.bin")
    d = open(path, "rb").read()
    assert d[:4] == b"WEAP", "bad magic"
    ver, ni, na, nn = struct.unpack_from("<HHHH", d, 4)
    print("%s  v%d  %d records  %d aim offsets  %d names  %d B"
          % (path, ver, ni, na, nn, len(d)))

    off = 12
    infos = [struct.unpack_from(WEAPON_REC, d, off + i * 96) for i in range(ni)]
    off += ni * 96
    aims = [struct.unpack_from(AIM_REC, d, off + i * 24) for i in range(na)]
    off += na * 24
    names = [d[off + i * 16: off + i * 16 + 16].split(b"\0")[0].decode() for i in range(nn)]

    def rec(i):
        (flags, trng, wrng, acc, mspd, ox, oy, oz, l0s, l0e, l0f, l1s, l1e, l1f,
         brk, spd, rad, life, spr, m1, m2, clip, dmg, req,
         ft, slot, sk, grp, aimi, bc, nc) = infos[i]
        return dict(flags=flags, trng=trng, wrng=wrng, acc=acc, mspd=mspd,
                    loop=(l0s, l0e, l0f), loop2=(l1s, l1e, l1f), brk=brk,
                    m1=m1, clip=clip, dmg=dmg, req=req, ft=ft, slot=slot,
                    sk=sk, grp=grp, aimi=aimi, bc=bc, nc=nc)

    def reload_ms(r):
        if r["flags"] & FLAG_RELOAD:
            return 2000 if (r["flags"] & FLAG_TWIN) else 1000
        if r["flags"] & FLAG_LONG_RELOAD:
            return 1000
        a = aims[r["aimi"]] if r["aimi"] < len(aims) else (0,) * 8
        return max(400, max(a[4], a[5], a[6], a[7]) + 100)

    print()
    print("%-16s %-8s %4s %5s %5s %5s %5s %6s %6s %4s %5s"
          % ("weapon", "fire", "slot", "dmg", "clip", "trng", "wrng", "delay", "reload", "grp", "acc"))
    for t in range(NUM_WEAPONS):
        r = rec(idx(t, 1))
        delay = int(900.0 * (r["loop"][1] - r["loop"][0]))
        print("%-16s %-8s %4d %5d %5d %5.1f %5.1f %5dms %5dms %4d %5.2f"
              % (names[t], FIRE[r["ft"]] if r["ft"] < 6 else "?", r["slot"],
                 r["dmg"], r["clip"], r["trng"], r["wrng"], delay, reload_ms(r),
                 r["grp"], r["acc"]))

    print()
    print("skill lanes (the index math is the thing being checked here):")
    for t in (22, 24, 25, 30, 32):
        line = "  %-16s" % names[t]
        for sk in range(4):
            r = rec(idx(t, sk))
            line += "  %s dmg%-4d clip%-4d acc%.2f%s" % (
                SKILL[sk], r["dmg"], r["clip"], r["acc"],
                " TWIN" if (r["flags"] & FLAG_TWIN) else "")
        print(line)

    # Every skill lane must actually differ from STD somewhere: identical rows mean the
    # lane was never written and every skilled weapon is silently reading STD.
    bad = []
    for t in range(FIRST_SKILLED, LAST_SKILLED + 1):
        std = infos[idx(t, 1)]
        for sk in (0, 2, 3):
            if infos[idx(t, sk)] == std:
                bad.append((names[t], SKILL[sk]))
    print()
    if bad:
        print("!! %d skill lanes identical to STD: %s" % (len(bad), bad[:6]))
        return 1
    print("ok: all 33 skill lanes differ from their STD row")

    # And the melee rows must be MELEE fire type with a combo, or the melee half of the
    # table never got parsed at all.
    melee = [t for t in range(16) if rec(t)["ft"] == 0]
    print("ok: %d melee rows, base combos %s"
          % (len(melee), sorted({rec(t)["bc"] for t in melee})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
