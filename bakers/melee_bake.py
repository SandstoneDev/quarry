#!/usr/bin/env python3
"""melee_bake - data/melee.dat -> data/melee.bin ('MELE').

The whole hand-to-hand system is DATA on the disc, which is the reason this is a parser
and not a reversing job. `CTaskSimpleFight::LoadMeleeData` (PC 0x5BEDC0) reads
data/melee.dat; every combo names an ANIMGROUP out of the animation group table and
carries six attack rows:

 ATTACK1/2/3 the three-hit chain
 AGROUND the stomp on a downed target (has an extra groundLoop column)
 AMOVING the attack thrown while walking
 ABLOCK the guard (hit + chain only)

Each row is `hit chain radius hitLevel damage hitAnim altHit [groundLoop]`, where
`hit` and `chain` are ANIMATION FRAMES: hit is when the blow lands, chain is the latest
frame at which pressing attack again continues the combo. 99.0 means "never" - that is
how KICK_STD and PISTOL_WHIP express having only two real attacks.

Which combo a weapon uses comes from CWeaponInfo, not from this file: `baseCombo` is a
ONE-BASED index and it names the LAST entry of the weapon's run, so the run is
`base-num+1 .. base`. Fits all 47 rows: the bat/golfclub/nightstick/shovel/poolcue all
land on BBALLBAT, the katana and cane on SWORD, and UNARMED's base=4 num=4 covers
UNARMED_1..4, the four gym fighting styles. See sa-melee-combo-table in the vault.

 SA_ROOT=... python tools/melee_bake.py [--out data]
"""
import argparse
import os
import struct
import sys

SA_ROOT = os.environ.get("SA_ROOT", "")

# `hitLevel` column. H/L/G/B are high/low/ground/behind; the doubled forms appear only on
# AGROUND and AMOVING rows and are kept distinct rather than folded, because nothing here
# knows yet whether the engine treats "L" and "LL" the same.
LEVELS = ("N", "H", "L", "G", "B", "HL", "LL", "GL")

ROWS = ("ATTACK1", "ATTACK2", "ATTACK3", "AGROUND", "AMOVING", "ABLOCK")
NROW = len(ROWS)


class Attack:
    __slots__ = ("hit", "chain", "radius", "level", "damage", "hitAnim", "altHit", "ground")

    def __init__(self):
        self.hit = self.chain = 99.0
        self.radius = 0.0
        self.level = 0
        self.damage = self.hitAnim = self.altHit = 0
        self.ground = 0.0


class Combo:
    __slots__ = ("name", "group", "rng", "flags", "rows")

    def __init__(self, name):
        self.name = name
        self.group = ""
        self.rng = 1.6
        self.flags = 0
        self.rows = [Attack() for _ in range(NROW)]


def parse(path):
    combos, cur = [], None
    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            key = tok[0].upper()
            if key == "START_COMBO":
                cur = Combo(tok[1])
                continue
            if key == "START_LEVELS":
                # The hit-level table. It shares END_COMBO as its terminator, which is why
                # a naive reader ends up one record out of step - and being one out of
                # step is exactly what would silently give every weapon its neighbour's
                # combo. Consumed and discarded here on purpose.
                cur = None
                continue
            if key == "END_COMBO":
                if cur is not None:
                    combos.append(cur)
                cur = None
                continue
            if cur is None:
                continue
            if key == "ANIMGROUP":
                cur.group = tok[1]
            elif key == "RANGES":
                cur.rng = float(tok[1])
            elif key == "FLAGS":
                cur.flags = int(tok[1], 16)
            elif key in ROWS:
                a = cur.rows[ROWS.index(key)]
                a.hit = float(tok[1])
                a.chain = float(tok[2])
                if key == "ABLOCK":
                    continue                      # guard: hit + chain only
                a.radius = float(tok[3])
                lv = tok[4].upper()
                if lv not in LEVELS:
                    sys.exit("melee_bake: unknown hit level %r in %s" % (lv, cur.name))
                a.level = LEVELS.index(lv)
                a.damage = int(float(tok[5]))
                a.hitAnim = int(float(tok[6]))
                a.altHit = int(float(tok[7]))
                if len(tok) > 8:
                    a.ground = float(tok[8])
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--root", default=SA_ROOT)
    a = ap.parse_args()
    if not a.root:
        sys.exit("melee_bake: set SA_ROOT (or pass --root)")

    src = os.path.join(a.root, "data", "melee.dat")
    if not os.path.isfile(src):
        sys.exit("melee_bake: %s not found" % src)
    combos = parse(src)
    if not combos:
        sys.exit("melee_bake: parsed no combos - refusing to write an empty table")

    buf = bytearray(b"MELE")
    buf += struct.pack("<HHHH", 1, len(combos), NROW, 0)
    for c in combos:
        buf += c.name.encode("latin-1")[:15].ljust(16, b"\0")
        buf += c.group.encode("latin-1")[:15].ljust(16, b"\0")
        buf += struct.pack("<fI", c.rng, c.flags)
        for r in c.rows:
            buf += struct.pack("<fffBBBBf", r.hit, r.chain, r.radius,
                               r.level, min(r.damage, 255), min(r.hitAnim, 255),
                               min(r.altHit, 255), r.ground)

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "melee.bin")
    with open(dst, "wb") as f:
        f.write(buf)

    print("melee_bake: %d combos -> %s (%d B)" % (len(combos), dst, len(buf)))
    for i, c in enumerate(combos, 1):          # ONE-BASED: that is how baseCombo indexes
        r = c.rows[0]
        print("  %2d %-12s group=%-10s rng=%.1f flags=0x%03x  a1 hit=%.0f chain=%.0f dmg=%d"
              % (i, c.name, c.group or "-", c.rng, c.flags, r.hit, r.chain, r.damage))
    # The two the weapon table leans on hardest: if the list ever shifts, every melee
    # weapon quietly swings someone else's combo, so assert the anchors rather than trust.
    want = {5: "BBALLBAT", 6: "KNIFE", 8: "SWORD", 9: "CHAINSAW", 10: "DILDO", 11: "FLOWERS"}
    for idx, nm in want.items():
        if idx > len(combos) or combos[idx - 1].name != nm:
            sys.exit("melee_bake: combo %d is %r, expected %r - the one-based mapping "
                     "CWeaponInfo.baseCombo relies on has shifted"
                     % (idx, combos[idx - 1].name if idx <= len(combos) else None, nm))
    print("  anchors OK (5=BBALLBAT 6=KNIFE 8=SWORD 9=CHAINSAW 10=DILDO 11=FLOWERS)")


if __name__ == "__main__":
    main()
