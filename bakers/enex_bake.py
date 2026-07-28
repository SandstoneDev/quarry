#!/usr/bin/env python3
"""enex_bake - bake SA entry/exit (interior door) markers for the PSP port.

Parses the `enex` sections of every TEXT IPL (all 376 SA enexes live there;
research/interior_enex_system.md), pairs the two lines that share a name
(world side <-> interior side, CEntryExitManager::PostEntryExitsCreation),
and writes the LS phase-1 set to data/enex.bin.

enex IPL line (18 cols):
  x, y, z, enterAngle, sizeX, sizeY, sizeZ,
  exitX, exitY, exitZ, exitAngle, targetInterior, flags, name,
  sky, numPedsToSpawn, timeOn, timeOff

enex.bin: 'ENEX' u16 count u16 pad, then per DOOR (one record per SIDE):
  f32 entX,entY,entZ, entHeading(rad)      the walk-into rectangle centre
  f32 sizeX, sizeY                          rectangle half-extents-ish (SA sizes)
  f32 spawnX, spawnY, spawnZ, spawnHeading  where THIS side's transition lands you
  u8  areaHere, areaTarget, sky, flags8     area codes: which world this side lives in
  char name[20]                             interior name (== interior_<name>.pmap)
"""
import glob
import os
import struct
import sys

# INPUT: SA_ROOT points at the user's extracted PS2 disc (Quarry sets it); the enex
# markers are a pure TEXT parse of the disc's IPL enex sections - no codec.
SA = os.environ.get("SA_ROOT", "")
MAPS = SA + "/data/maps"
# OUTPUT: default writes into <data>/interiors; the converter passes its own dir on
# argv (StepBakeEnex). No memstick/SA_PSP deploy from the baker.
OUT = ""

# ALL doors are emitted (set to None) or a specific subset. None = every pair
# that has a matching interior geometry pocket in gta_int.img.
PHASE1 = None


def parse_enex_lines():
    rows = []
    ipls = glob.glob(MAPS + "/**/*.ipl", recursive=True) + \
           glob.glob(MAPS + "/**/*.IPL", recursive=True)
    for ipl in sorted(set(os.path.normcase(p) for p in ipls)):
        sec = None
        for line in open(ipl, "r", errors="replace"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low == "enex":
                sec = "enex"; continue
            if low == "end":
                sec = None; continue
            if sec != "enex":
                continue
            p = [c.strip() for c in s.split(",")]
            if len(p) < 14:
                continue
            try:
                rows.append(dict(
                    x=float(p[0]), y=float(p[1]), z=float(p[2]),
                    ang=float(p[3]), sx=float(p[4]), sy=float(p[5]), sz=float(p[6]),
                    ex=float(p[7]), ey=float(p[8]), ez=float(p[9]), eang=float(p[10]),
                    interior=int(p[11]), flags=int(p[12]),
                    name=p[13].strip('"').upper(),
                    sky=int(p[14]) if len(p) > 14 else 0,
                ))
            except ValueError:
                continue
    return rows


def main():
    rows = parse_enex_lines()
    print(f"enex lines: {len(rows)}")
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    import math
    out = []
    names = sorted(by_name) if PHASE1 is None else sorted(PHASE1)
    for name in names:
        sides = by_name.get(name, [])
        if len(sides) < 2:
            continue                                  # need a pair (world + interior)
        # world side = flags bit2 (CREATE_LINKED_PAIR, research §1); pick by it,
        # fall back to interior field 0 vs !0.
        world = next((s for s in sides if s["flags"] & 4), None)
        inner = next((s for s in sides if s is not world), None)
        if world is None:
            world = next((s for s in sides if s["interior"] == 0), sides[0])
            inner = next((s for s in sides if s is not world), sides[1])
        if inner is None or inner["interior"] == 0:
            continue                                  # no real interior side
        # each record: walking into THIS side's rect lands you at the PAIR's exit
        for here, there in ((world, inner), (inner, world)):
            out.append(dict(
                ent=(here["x"], here["y"], here["z"]),
                heading=math.radians(here["ang"]),
                size=(max(1.0, here["sx"]), max(1.0, here["sy"])),
                spawn=(there["ex"], there["ey"], there["ez"]),
                spawnHeading=math.radians(there["eang"]),
                areaHere=(0 if here is world else here["interior"]),
                areaTarget=(0 if there is world else there["interior"]),
                sky=here["sky"], flags=here["flags"] & 0xFF,
                name=name,
            ))
    print(f"emitted {len(out)} sides ({len(out)//2} doors)")

    buf = b"ENEX" + struct.pack("<HH", len(out), 0)
    for r in out:
        nm = r["name"].encode()[:19].ljust(20, b"\0")
        buf += struct.pack("<4f2f4f4B",
                           r["ent"][0], r["ent"][1], r["ent"][2], r["heading"],
                           r["size"][0], r["size"][1],
                           r["spawn"][0], r["spawn"][1], r["spawn"][2], r["spawnHeading"],
                           r["areaHere"] & 0xFF, r["areaTarget"] & 0xFF,
                           r["sky"] & 0xFF, r["flags"]) + nm
    # OUTPUT dir via argv: a directory -> write enex.bin into it, else an explicit path.
    out_path = OUT
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        out_path = os.path.join(arg, "enex.bin") if not arg.lower().endswith(".bin") else arg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "wb").write(buf)
    print(f"enex.bin: {len(out)} records, {len(buf)} bytes -> {out_path}")


if __name__ == "__main__":
    main()
