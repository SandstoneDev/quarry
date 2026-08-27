#!/usr/bin/env python3
"""weapon_bake - bake SA's weapon table into data/weapon.bin ('WEAP').

The whole weapon system hangs off one file: DATA/WEAPON.DAT. It carries 47 weapon
types in three skill levels (80 records in all), the twenty gun-aiming offsets, and
every timing the firing animation is paced by. sections 1 and 10 for where each column
goes and why the numbers are what they are.

Two things here are NOT obvious and must not be "tidied up":

 * the animation loop columns are FRAMES and the original quantises them at load:
 end = start + floor(0.1 + (end - start) * 50) / 50 - 0.006
 reproduced verbatim in loop_times() below;
 * the 80 records are laid out [STD 0..46][POOR 47..57][PRO 58..68][COP 69..79],
 and the index math that reads them (CWeaponInfo::GetWeaponInfo) only works if
 the layout is exactly that.

A note on the PS2 build: it contains no string "WEAPON.DAT" - it contains no data
file name at all, because it assembles "\\DATA\\" + "%s.DAT" at runtime. handling.cfg
is equally absent and is unquestionably live, so the missing string proves nothing.
The disc's weapon.dat IS the shipped data.

 SA_ROOT="…/(v2.01)" python tools/weapon_bake.py [--out data]

Format 'WEAP' v1, little-endian:
 'WEAP' u16 version u16 nInfo u16 nAim u16 nName
 info [nInfo] 96 B (see WEAPON_REC below, mirrors CWeaponInfo field for field)
 aim [nAim] 24 B AimX AimZ DuckX DuckZ (f32) RLoadA RLoadB CrouchA CrouchB (s16)
 name [nName] 16 B NUL-padded weapon name, index == eWeaponType
"""
import argparse
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SA_ROOT", "")

NUM_WEAPONS = 47                  # WEAPON_UNARMED.. WEAPON_PARACHUTE
FIRST_SKILLED = 22                # WEAPON_PISTOL
LAST_SKILLED = 32                 # WEAPON_TEC9
NUM_SKILLED = LAST_SKILLED - FIRST_SKILLED + 1     # 11
NUM_INFOS = NUM_WEAPONS + NUM_SKILLED * 3          # 80
# 21, not 20. The original declares g_GunAimingOffsets[20] but indexes it with
# (animGroup - ANIM_GROUP_PYTHON), and PYTHON..SPRAYCAN is twenty-ONE groups - the
# `% spraycan` row of weapon.dat writes one entry past the end of R*'s array. We keep
# the data and drop the overflow.
NUM_AIM = 21

ANIM_GROUP_PYTHON = 11
ANIM_GROUP_SPRAYCAN = 31

# CWeaponInfo::ms_aWeaponNames, recovered at 0x5F8350 in the PS2 build. The order IS
# eWeaponType, so a name's index is its type.
WEAPON_NAMES = [
    "UNARMED", "BRASSKNUCKLE", "GOLFCLUB", "NIGHTSTICK", "KNIFE", "BASEBALLBAT",
    "SHOVEL", "POOLCUE", "KATANA", "CHAINSAW", "DILDO1", "DILDO2", "VIBE1", "VIBE2",
    "FLOWERS", "CANE", "GRENADE", "TEARGAS", "MOLOTOV", "ROCKET", "ROCKET_HS",
    "FREEFALL_BOMB", "PISTOL", "PISTOL_SILENCED", "DESERT_EAGLE", "SHOTGUN",
    "SAWNOFF", "SPAS12", "MICRO_UZI", "MP5", "AK47", "M4", "TEC9", "COUNTRYRIFLE",
    "SNIPERRIFLE", "RLAUNCHER", "RLAUNCHER_HS", "FTHROWER", "MINIGUN",
    "SATCHEL_CHARGE", "DETONATOR", "SPRAYCAN", "EXTINGUISHER", "CAMERA",
    "NIGHTVISION", "INFRARED", "PARACHUTE",
]
assert len(WEAPON_NAMES) == NUM_WEAPONS

# CWeaponInfo::FindWeaponFireType. Anything unrecognised is INSTANT_HIT, exactly as
# the original does - that is not a fallback, it is the documented behaviour.
FIRE_TYPES = {"MELEE": 0, "INSTANT_HIT": 1, "PROJECTILE": 2,
              "AREA_EFFECT": 3, "CAMERA": 4, "USE": 5}

# GetBaseComboByName - eMeleeCombo. Unrecognised means UNARMED_1.
BASE_COMBOS = {"UNARMED": 4, "BBALLBAT": 5, "KNIFE": 6, "GOLFCLUB": 7,
               "SWORD": 8, "CHAINSAW": 9, "DILDO": 10, "FLOWERS": 11}

# 96-byte record. Field order chosen so every float is 4-aligned on the PSP; the
# SET is CWeaponInfo's, the ORDER is not (SA's 0x70 layout has the CVector in the
# middle and would need padding here for nothing).
WEAPON_REC = "<Iffff fff fff fff fffff hhHHH BBBBBBB xxx"
WEAPON_REC = WEAPON_REC.replace(" ", "")
assert struct.calcsize(WEAPON_REC) == 96, struct.calcsize(WEAPON_REC)

AIM_REC = "<ffffhhhh"
assert struct.calcsize(AIM_REC) == 24


def group_ids():
    """AssocGroupId by name, from the table recovered by tools/extract_anim_groups.py."""
    with open(os.path.join(HERE, "data", "sa_anim_groups.json"), encoding="utf-8") as f:
        doc = json.load(f)
    return {g["name"].lower(): g["id"] for g in doc["groups"]}


def loop_times(start_frames, end_frames, fire_frames):
    """weapon.dat's animation columns are frames; CWeaponInfo::LoadWeaponData converts
 to seconds at 30 fps and then snaps the END onto a 1/50 grid, minus 6 ms. Keep it
 literal - the shot pacing (900 * (end - start)) is measured off this number."""
    start = start_frames / 30.0
    end = end_frames / 30.0
    fire = fire_frames / 30.0
    end = start + math.floor(0.1 + (end - start) * 50.0) / 50.0 - 0.006
    return start, end, fire


def info_index(wtype, skill):
    """CWeaponInfo::GetWeaponInfoIndex. STD is the identity; the other three skills
 live in 11-wide blocks after the 47 STD records."""
    if skill == 1:
        return wtype
    assert FIRST_SKILLED <= wtype <= LAST_SKILLED, (wtype, skill)
    lane = {0: 0, 2: 1, 3: 2}[skill]
    return NUM_WEAPONS + (wtype - FIRST_SKILLED) + lane * NUM_SKILLED


class Info(object):
    __slots__ = ("flags", "target_range", "weapon_range", "accuracy", "move_speed",
                 "ofs", "loop", "loop2", "breakout", "speed", "radius", "lifespan",
                 "spread", "model1", "model2", "clip", "damage", "req_stat",
                 "fire_type", "slot", "skill", "anim_group", "aim_index",
                 "base_combo", "num_combos")

    def __init__(self):
        # CWeaponInfo::Initialise's defaults, so an unlisted record reads sanely.
        self.flags = 0
        self.target_range = self.weapon_range = 0.0
        self.accuracy = self.move_speed = 1.0
        self.ofs = (0.0, 0.0, 0.0)
        self.loop = (0.0, 0.0, 0.0)
        self.loop2 = (0.0, 0.0, 0.0)
        self.breakout = self.speed = self.radius = self.lifespan = self.spread = 0.0
        self.model1 = self.model2 = -1
        self.clip = self.damage = self.req_stat = 0
        self.fire_type = 0
        self.slot = 0xFF
        self.skill = 1
        self.anim_group = 0
        self.aim_index = 0
        self.base_combo = 4
        self.num_combos = 1

    def pack(self):
        return struct.pack(
            WEAPON_REC,
            self.flags, self.target_range, self.weapon_range, self.accuracy,
            self.move_speed,
            self.ofs[0], self.ofs[1], self.ofs[2],
            self.loop[0], self.loop[1], self.loop[2],
            self.loop2[0], self.loop2[1], self.loop2[2],
            self.breakout, self.speed, self.radius, self.lifespan, self.spread,
            self.model1, self.model2,
            self.clip, self.damage, self.req_stat,
            self.fire_type, self.slot & 0xFF, self.skill, self.anim_group,
            self.aim_index, self.base_combo, self.num_combos)


def parse(path, groups):
    infos = [Info() for _ in range(NUM_INFOS)]
    aim = [[0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0] for _ in range(NUM_AIM)]
    by_name = {n: i for i, n in enumerate(WEAPON_NAMES)}
    nguns = nmelee = naim = 0

    # The melee marker is 0xA3 ('£' in Windows-1252). Read as latin-1 so the byte
    # survives whatever the file's real encoding is.
    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if "ENDWEAPONDATA" in line:
                break
            if not line:
                continue
            tag = line[0]
            tok = line.split()
            if tag == "$":
                # $ NAME FIRETYPE trng wrng m1 m2 slot animGrp clip dmg ox oy oz
                # skill reqStat acc mspd a1s a1e a1f a2s a2e a2f breakout flags
                # [speed radius life spread]
                if len(tok) < 26:
                    continue
                wtype = by_name.get(tok[1])
                if wtype is None:
                    print("weapon_bake: unknown weapon '%s', skipped" % tok[1])
                    continue
                skill = int(tok[14])
                if not (FIRST_SKILLED <= wtype <= LAST_SKILLED):
                    skill = 1               # only 22..32 have skill lanes
                rec = infos[info_index(wtype, skill)]
                rec.fire_type = FIRE_TYPES.get(tok[2], 1)
                rec.target_range = float(tok[3])
                rec.weapon_range = float(tok[4])
                rec.model1 = int(tok[5])
                rec.model2 = int(tok[6])
                rec.slot = int(tok[7])
                rec.clip = int(tok[9])
                rec.damage = int(tok[10])
                rec.ofs = (float(tok[11]), float(tok[12]), float(tok[13]))
                rec.skill = skill
                rec.req_stat = min(int(tok[15]), 0xFFFF)
                rec.accuracy = float(tok[16])
                rec.move_speed = float(tok[17])
                rec.loop = loop_times(int(tok[18]), int(tok[19]), int(tok[20]))
                rec.loop2 = loop_times(int(tok[21]), int(tok[22]), int(tok[23]))
                rec.breakout = int(tok[24]) / 30.0
                rec.flags = int(tok[25], 16)
                if len(tok) >= 30:
                    rec.speed = float(tok[26])
                    rec.radius = float(tok[27])
                    rec.lifespan = float(tok[28])
                    rec.spread = float(tok[29])
                grp = tok[8]
                if not grp.startswith("null"):
                    gid = groups.get(grp.lower())
                    if gid is None:
                        print("weapon_bake: unknown anim group '%s'" % grp)
                    else:
                        rec.anim_group = gid
                        if ANIM_GROUP_PYTHON <= gid <= ANIM_GROUP_SPRAYCAN:
                            rec.aim_index = gid - ANIM_GROUP_PYTHON
                nguns += 1
            elif tag == "%":
                # % animGrp AimX AimZ DuckX DuckZ RLoadA RLoadB CrouchA CrouchB
                if len(tok) < 10:
                    continue
                gid = groups.get(tok[1].lower())
                if gid is None or not (ANIM_GROUP_PYTHON <= gid <= ANIM_GROUP_SPRAYCAN):
                    print("weapon_bake: aim offset for unknown group '%s'" % tok[1])
                    continue
                aim[gid - ANIM_GROUP_PYTHON] = [
                    float(tok[2]), float(tok[3]), float(tok[4]), float(tok[5]),
                    int(tok[6]), int(tok[7]), int(tok[8]), int(tok[9])]
                naim += 1
            elif ord(tag) == 0xA3:
                # £ NAME MELEE trng wrng m1 m2 slot baseCombo numCombos flags stealthGrp
                if len(tok) < 12:
                    continue
                wtype = by_name.get(tok[1])
                if wtype is None:
                    continue
                rec = infos[wtype]          # melee rows are STD-only
                rec.fire_type = FIRE_TYPES.get(tok[2], 1)
                rec.target_range = float(tok[3])
                rec.weapon_range = float(tok[4])
                rec.model1 = int(tok[5])
                rec.model2 = int(tok[6])
                rec.slot = int(tok[7])
                rec.base_combo = BASE_COMBOS.get(tok[8], 4)
                rec.num_combos = int(tok[9])
                rec.flags = int(tok[10], 16)
                if not tok[11].startswith("null"):
                    gid = groups.get(tok[11].lower())
                    if gid is not None:
                        rec.anim_group = gid
                nmelee += 1
    return infos, aim, nguns, nmelee, naim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    if not args.root:
        sys.exit("weapon_bake: set SA_ROOT (or pass --root) to the extracted PS2 disc")

    src = os.path.join(args.root, "data", "weapon.dat")
    if not os.path.isfile(src):
        sys.exit("weapon_bake: %s not found" % src)

    groups = group_ids()
    infos, aim, nguns, nmelee, naim = parse(src, groups)

    # A silent zero table is worse than a loud failure: the whole feature reads from
    # this file and a bad parse would look like "guns feel wrong" three layers later.
    pistol = infos[22]
    if not (pistol.damage == 25 and pistol.clip == 17 and abs(pistol.weapon_range - 35.0) < 1e-3):
        sys.exit("weapon_bake: sanity check failed - PISTOL reads dmg=%d clip=%d range=%g"
                 % (pistol.damage, pistol.clip, pistol.weapon_range))
    if nguns < 40 or nmelee < 15 or naim < 15:
        sys.exit("weapon_bake: only %d gun / %d melee / %d aim rows parsed" % (nguns, nmelee, naim))

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "weapon.bin")
    with open(dst, "wb") as f:
        f.write(b"WEAP")
        f.write(struct.pack("<HHHH", 1, NUM_INFOS, NUM_AIM, NUM_WEAPONS))
        for rec in infos:
            f.write(rec.pack())
        for a in aim:
            f.write(struct.pack(AIM_REC, a[0], a[1], a[2], a[3],
                                a[4], a[5], a[6], a[7]))
        for n in WEAPON_NAMES:
            f.write(n.encode("ascii")[:15].ljust(16, b"\0"))

    print("weapon_bake: %s - %d records (%d gun rows, %d melee rows), %d aim offsets, %d B"
          % (dst, NUM_INFOS, nguns, nmelee, naim, os.path.getsize(dst)))
    print("  PISTOL   dmg %d clip %d range %.1f accuracy %.2f loop %.4f..%.4f fire %.4f"
          % (pistol.damage, pistol.clip, pistol.weapon_range, pistol.accuracy,
             pistol.loop[0], pistol.loop[1], pistol.loop[2]))
    ak = infos[30]
    print("  AK47     dmg %d clip %d range %.1f shotdelay %d ms"
          % (ak.damage, ak.clip, ak.weapon_range, int(900.0 * (ak.loop[1] - ak.loop[0]))))


if __name__ == "__main__":
    main()
