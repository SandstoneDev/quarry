#!/usr/bin/env python3
"""anim_group_bake - bake the SA animation-group tables into data/anim/groups.bin.

This is the mechanism behind "entering a Rustler is not entering a Sentinel". Nothing
about a plane's boarding sequence is coded in the original: a vehicle carries an anim
group id, that group names two AssocGroupIds and eighteen selector bits, and each of
the forty vehicle animations is fetched from the first or the second group according to
its bit. Everything here is recovered from the game's own data.

Inputs (all under SA_ROOT, i.e. the user's extracted PS2 disc):
 data/handling.cfg '^' rows -> the 30 vehicle anim groups (two group ids, 18 selector
 bits, 5 general timings, 2x4 open/close timings, flags)
 vehicle rows, last column (the file's own legend calls it
 "(aj) vehicle anim group") -> model -> group
 data/vehicles.ide -> model id -> handling name
 data/animgrp.dat -> the 21 walkcycle groups, AssocGroupId 118..138

plus tools/data/sa_anim_groups.json, the 118 hard-coded groups recovered once from a
retail binary by tools/extract_anim_groups.py. They cannot come off the disc: the PS2
build assembles that table at runtime into .bss.

 python tools/anim_group_bake.py [--out <dir>]

Format 'AGRP' v1, little-endian:
 'AGRP' u16 version u16 nGroups u16 nAnims u16 nVehGroups u16 nVehMap u16 pad
 groups [nGroups] char name[16]; char block[16]; u16 firstAnim; u16 nAnims (36 B)
 anims [nAnims] u16 animId; u16 flags; char clip[24] (28 B)
 vehGroup [nVehGroups] u8 first; u8 second; u8 pad[2]; u32 animFlags;
 u32 specialFlags; f32 timing[5]; f32 inout[2][4] (64 B)
 vehMap [nVehMap] u16 modelId; u16 vehGroup ( 4 B)
 selector [40] u8 bit index per vehicle anim slot (0xFF = always first) (40 B)

`firstAnim` is the group's ANIM ID OFFSET as well as its slice start: SA looks an
animation up as m_Anims[animId - m_IdOffset], which is why ANIM_ID_WALK == 0 means a
different clip in every walkcycle group.
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = os.environ.get("SA_ROOT", "")
OUT = "data/anim"

CAR_ANIM_FIRST = 351      # ANIM_ID_CAR_ALIGN_LHS
CAR_ANIM_COUNT = 40       #.. ANIM_ID_CAR_DOORLOCKED_RHS
ANIM_GROUP_CARS_BEGIN = 88

# CVehicleAnimGroup::GetGroup(animId) is a flat switch: each animation family reads one
# bit of m_animFlags and takes the second group when it is set. Baked as a table so the
# runtime is a lookup. Index = animId - 351. 0xFF = not in the switch (always first).
SELECTOR_BIT = [
    0, 0, 0, 0,      # CAR_ALIGN_LHS/RHS, CAR_ALIGNHI_LHS/RHS bAlign
    1, 1,            # CAR_OPEN_LHS/RHS bOpenFrontDoorsOnExit
    2, 2,            # CAR_OPEN_LHS_1/RHS_1 bOpenRearDoorsOnExit
    3, 3,            # CAR_GETIN_LHS_0/RHS_0 bCanEnterFrontDoors
    4, 4,            # CAR_GETIN_LHS_1/RHS_1 bCanEnterRearDoors
    3,               # CAR_GETIN_BIKE_FRONT bCanEnterFrontDoors
    5, 5, 5,         # CAR_PULLOUT_LHS/RHS, UNKNOWN_15 bCanPulloutPed
    6, 6,            # CAR_CLOSEDOOR_LHS_0/RHS_0 bCloseFrontDoorWhenInside
    7, 7,            # CAR_CLOSEDOOR_LHS_1/RHS_1 bCloseRearDoorWhenInside
    8, 8,            # CAR_SHUFFLE_RHS_0/1 bShuffle
    9, 9,            # CAR_GETOUT_LHS_0/RHS_0 bCanExitFrontDoors
    10, 10,          # CAR_GETOUT_LHS_1/RHS_1 bCanExitRearDoors
    0xFF,            # UNKNOWN_26 - absent from the switch
    11, 11,          # CAR_JACKEDLHS/RHS bPlayerCanBeJacked
    12, 12,          # CAR_CLOSE_LHS_0/RHS_0 bCloseFrontDoorWhenOutside
    13, 13,          # CAR_CLOSE_LHS_1/RHS_1 bCloseRearDoorWhenOutside
    14, 14,          # CAR_ROLLOUT_LHS/RHS bCanJumpOutOfVehicle
    15,              # CAR_ROLLDOOR bRollDownWindowOnDoorClose
    16, 16,          # CAR_FALLOUT_LHS/RHS bPlayerCanFallOut
    17, 17,          # CAR_DOORLOCKED_LHS/RHS bDoorLocked
]
assert len(SELECTOR_BIT) == CAR_ANIM_COUNT


def _disc(*parts):
    return os.path.join(ROOT, *parts)


def load_hardcoded():
    """The 118 groups compiled into the game, as data (see the module docstring)."""
    p = os.path.join(HERE, "data", "sa_anim_groups.json")
    with open(p, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return blob["groups"]


def load_animgrp(std_anims):
    """DATA/ANIMGRP.DAT -> the walkcycle groups, AssocGroupId 118 upward.

 Format is documented in the file itself: `name, ifp block, walkcycle, count`, then
 `count` animation names, then `end`. They reuse aStdAnimDescs, so the flags for
 animation ids 0..5 are the ones group 0 already carries."""
    path = _disc("data", "animgrp.dat")
    if not os.path.exists(path):
        print("  ! %s absent - no walkcycle groups" % path)
        return []
    std_flags = {a[0]: a[1] for a in std_anims}
    out, cur = [], None
    for raw in open(path, "r", errors="replace"):
        s = raw.split("#")[0].strip()
        if not s:
            continue
        if s.lower() == "end":
            if cur:
                out.append(cur)
            cur = None
            continue
        if cur is None:
            f = [x.strip() for x in s.split(",")]
            if len(f) < 4:
                continue
            cur = {"name": f[0], "block": f[1], "model": 7, "anims": []}
            continue
        aid = len(cur["anims"])
        cur["anims"].append([aid, std_flags.get(aid, 0), s])
    if cur:
        out.append(cur)
    return out


def load_veh_anim_groups():
    """DATA/HANDLING.CFG '^' rows -> the 30 CVehicleAnimGroup records.

 Column order after the id: first group, second group, then EIGHTEEN selector flags,
 five general timings, four start/stop timing pairs, and the special flags. The two
 group columns are vehicle-group indices; the loader adds ANIM_GROUP_CARS_BEGIN.

 NOTE the flag order. The disc file documents it across two staggered comment lines
 and the real order is their zip - Align, OpenOutF, OpenOutR, GetInF, GetInR, Jack,
 CloseInsF, CloseInsR, Shuffle, GetOutF, GetOutR, BeJacked, CloseOutF, CloseOutR,
 JumpOut, CloseRoll, FallDie, OpenLocked - which maps 1:1 onto the bitfield. Reading
 the two comment lines sequentially instead gives a plausible but wrong order."""
    path = _disc("data", "handling.cfg")
    rows = {}
    for raw in open(path, "r", errors="replace"):
        s = raw.strip()
        if not s.startswith("^"):
            continue
        f = s[1:].split()
        if len(f) < 35:
            continue
        gid = int(f[0])
        first, second = int(f[1]), int(f[2])
        flags = 0
        for i in range(18):
            if int(f[3 + i]):
                flags |= 1 << i
        n = 21
        timing = [float(f[n + i]) for i in range(5)]          # GetIn Jump GetOut Jack Fall
        n += 5
        pairs = [float(f[n + i]) for i in range(8)]           # start/stop, interleaved
        special = int(f[n + 8])
        # pairs are OpenOutStart, OpenOutStop, CloseInStart, CloseInStop,
        # OpenInStart, OpenInStop, CloseOutStart, CloseOutStop
        start = [pairs[0], pairs[2], pairs[4], pairs[6]]
        stop = [pairs[1], pairs[3], pairs[5], pairs[7]]
        rows[gid] = dict(first=first + ANIM_GROUP_CARS_BEGIN,
                         second=second + ANIM_GROUP_CARS_BEGIN,
                         flags=flags, special=special,
                         timing=timing, start=start, stop=stop)
    return [rows[i] for i in sorted(rows)]


def load_model_to_veh_group():
    """model id -> vehicle anim group, via handling name.

 The last numeric column of a normal handling.cfg vehicle row is the anim group; the
 file's own legend calls it "(aj) vehicle anim group"."""
    aj = {}
    for raw in open(_disc("data", "handling.cfg"), "r", errors="replace"):
        s = raw.strip()
        if not s or s[0] in ";#^$%!":
            continue
        f = s.split()
        if len(f) < 30:
            continue
        try:
            aj[f[0].upper()] = int(f[-1])
        except ValueError:
            continue
    out = {}
    for raw in open(_disc("data", "vehicles.ide"), "r", errors="replace"):
        s = raw.split("#")[0].strip()
        if not s or s.lower().startswith(("end", "cars", "boats")):
            continue
        f = [x for x in s.replace(",", " ").split() if x]
        if len(f) < 5:
            continue
        try:
            mid = int(f[0])
        except ValueError:
            continue
        g = aj.get(f[4].upper())
        if g is not None:
            out[mid] = g
    return out


def bake(outdir=OUT):
    if not ROOT:
        raise SystemExit("SA_ROOT is not set - point it at the extracted disc")
    groups = load_hardcoded()
    std_anims = groups[0]["anims"]
    walk = load_animgrp(std_anims)
    groups = groups + walk
    vgs = load_veh_anim_groups()
    vmap = load_model_to_veh_group()

    anims, grec = [], []
    for g in groups:
        first = len(anims)
        for aid, flags, clip in g["anims"]:
            anims.append((aid & 0xFFFF, flags & 0xFFFF, clip))
        grec.append((g["name"], g["block"], first, len(g["anims"])))

    buf = bytearray(b"AGRP")
    buf += struct.pack("<6H", 1, len(grec), len(anims), len(vgs), len(vmap), 0)
    for name, block, first, n in grec:
        buf += name.encode("latin1")[:15].ljust(16, b"\0")
        buf += block.encode("latin1")[:15].ljust(16, b"\0")
        buf += struct.pack("<HH", first, n)
    for aid, flags, clip in anims:
        buf += struct.pack("<HH", aid, flags)
        buf += clip.encode("latin1")[:23].ljust(24, b"\0")
    for v in vgs:
        buf += struct.pack("<BBBB", v["first"], v["second"], 0, 0)
        buf += struct.pack("<II", v["flags"], v["special"])
        buf += struct.pack("<5f", *v["timing"])
        buf += struct.pack("<4f", *v["start"])
        buf += struct.pack("<4f", *v["stop"])
    for mid in sorted(vmap):
        buf += struct.pack("<HH", mid, vmap[mid])
    buf += bytes(SELECTOR_BIT)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "groups.bin")
    with open(path, "wb") as f:
        f.write(buf)
    print("groups.bin: %d groups (%d hard-coded + %d walkcycle), %d anim slots, "
          "%d vehicle groups, %d model mappings -> %d bytes"
          % (len(grec), len(grec) - len(walk), len(walk), len(anims), len(vgs), len(vmap), len(buf)))
    for gid in (88, 91, 102, 114, 115, 117):
        if gid < len(grec):
            print("    group %-3d %-18s block=%s" % (gid, grec[gid][0], grec[gid][1]))
    return path


if __name__ == "__main__":
    out = OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    bake(out)
