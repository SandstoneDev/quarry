#!/usr/bin/env python3
"""ped_bake - bake the phase-1 ambient civilians into peds.bin.

peds.bin = 'PEDS' + u32 count + count HRO2 model streams back-to-back
(the exact hero.bin format, loaded by CSkelAnim_LoadPeds into models 1..N).

Clips per ped: IDLE_stance + WALK_civi + run_civi (locomotion slots 0/1/2 --
the runtime blends by move ratio R; ambient peds use R<=1 in phase 1).
Female models use the WOMAN_* clip set.

Model picks (LS street civs, checked against pedgrp): 2 male + 2 female.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hero_bake

OUT    = ""
# every data/ dir that holds a build (PPSSPP memstick + the real PSP on F: + the local
# deploy tree). Only existing ones are written - keeps the on-device peds.bin in sync so
# a stale CLST copy can't quietly re-arm the ped-GE garbage path.
DEPLOY = [
    "",
    "",
    "",
]

# clip slots are FIXED by index (runtime relies on them):
#   0 idle, 1 walk, 2 run, 3 HIT_front, 4 HIT_back, 5 KO_skid_front (death),
#   6 CHAT (IDLE_chat - SA standing "talk with hand gestures", 16 seqs/41 frames)
MALE_CLIPS   = ["IDLE_stance", "WALK_civi", "run_civi",
                "HIT_front", "HIT_back", "KO_skid_front", "IDLE_chat"]
FEMALE_CLIPS = ["woman_idlestance", "WOMAN_walknorm", "woman_run",
                "HIT_front", "HIT_back", "KO_skid_front", "IDLE_chat"]

# 8 models (spawn_tables §res): Grove fam, Ballas, business, casual m/f, rich.
# male01 stays index 0 as a safe fallback. anti-dup wants >=8 distinct models.
PEDS = [
    ("fam1",    MALE_CLIPS),     # Grove Street gang
    ("ballas1", MALE_CLIPS),     # Ballas gang
    ("bmost",   MALE_CLIPS),     # black male street (casual)
    ("wmost",   MALE_CLIPS),     # white male street
    ("hmori",   MALE_CLIPS),     # hispanic male worker
    ("wfyst",   FEMALE_CLIPS),   # white female young street
    ("bfyst",   FEMALE_CLIPS),   # black female young street
    ("wmybe",   MALE_CLIPS),     # beach male
]


def main():
    # argv[1] = explicit output path (Quarry passes <OutDir>/peds/peds.bin). When given
    # we write ONLY there and skip the dev-loop deploy mirror.
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    quarry = len(sys.argv) > 1

    # The ambient peds live ONLY in gta3.img, which on the PS2 disc stores them as PS2-NATIVE
    # skinned DFFs (native VIF geometry + native skin plugin). hero_bake now decodes those via
    # tools/ps2skin, so they bake on the disc as well as on the PC dev loop. Each bake stays
    # guarded: a ped that fails to decode is skipped, not fatal (a valid 'PEDS' container is
    # still written; the engine's CSkelAnim_LoadPeds treats count 0 / absent as hero-only).
    streams = []
    for name, clips in PEDS:
        print("--- baking %s ---" % name)
        try:
            streams.append(hero_bake.bake_model(name, clips=clips, emit_clst=False))
        except (SystemExit, Exception) as e:
            print("  ! %s skipped: %s" % (name, e))
    buf = bytearray()
    buf += b"PEDS"
    buf += struct.pack("<I", len(streams))
    for s in streams:
        buf += s
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(buf)
    if quarry:
        print("=== peds.bin: %d models, %d KB -> %s ===" % (len(streams), len(buf)//1024, out))
        if not streams:
            print("    (0 ambient peds decoded; PS2 gta3.img peds route through tools/ps2skin - "
                  "hero-only is still playable)")
        return
    n = 0
    for d in DEPLOY:
        if os.path.isdir(os.path.dirname(d)):
            try:
                open(d, "wb").write(buf); n += 1
            except OSError:
                pass
    print("=== peds.bin: %d models, %d KB, deployed to %d dir(s) ===" % (len(streams), len(buf)//1024, n))


if __name__ == "__main__":
    main()
