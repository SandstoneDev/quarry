#!/usr/bin/env python3
"""cutscene_bake - Phase 1: bake ONE intro1a cutscene actor (cssmoke = Big Smoke) into
cutscene.bin. Reuses hero_bake.bake_model(cut=...) with the ANPK clip. Output = the same
'PEDS' container the runtime CSkelAnim loader reads (magic + count + HRO2 streams), so the
one animated actor can be rendered by the existing skeletal-anim path.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "")
import hero_bake, sa_ifp_anpk
from sa_img import SaImg

# SA_ROOT env override: Quarry points this at the user's extracted PS2 disc. cuts.img (the
# ANPK clip) and player.img (the csplay/CJ cutscene actor - platform-NEUTRAL skinned DFF,
# byte-identical to PC, only its TXDs are PS2-native and handled by hero_bake._decode_txd)
# come from it. cutscene.img holds the cssmoke actor, which on the PS2 disc is a PS2-NATIVE VIF
# skinned DFF (flags 0x01010037, native bit 0x01000000); hero_bake now routes it through the
# PS2 native-skin codec (tools/ps2skin) so Big Smoke bakes on the disc exactly like on PC.
# Defaults keep the PC dev loop (col_bake.py uses the same idiom).
SA_ROOT      = os.environ.get("SA_ROOT", "")
CUTS_IMG     = SA_ROOT + "/anim/cuts.img"
CUTSCENE_IMG = SA_ROOT + "/models/cutscene.img"
OUT    = ""
DEPLOY = [
    "",
    "",
    "",
]

# intro1a skinned actors to bake. cssmoke = a full 61-bone cutscene ped (DFF from cutscene.img,
# index-mapped). csplay (CJ) = the REAL cutscene CJ, assembled by hero_bake from player.img
# cs_head + cs_hands (61-bone, full face 5001-5026) + the clothed body re-indexed onto the
# 61-bone rig, then the csplay ANPK index-mapped like cssmoke -> correct arms + lip-sync (was a
# hero-rig NAME-retarget with mangled arms + no lip-sync). Each entry: (actorName, mode).
ACTORS = [("cssmoke", "index"), ("csplay", "cjcut")]


def main():
    # argv[1] = explicit output path (Quarry passes <OutDir>/cutscene/cutscene.bin). When
    # given we write ONLY there and skip the dev-loop memstick mirror (ped_bake idiom).
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    quarry = len(sys.argv) > 1

    print("=== decode intro1a.ifp (ANPK) ===")
    anpk = sa_ifp_anpk.decode(SaImg(CUTS_IMG).extract("intro1a.ifp").rstrip(b"\x00"))
    streams = []
    for actor, mode in ACTORS:
        print("--- baking cutscene actor %s (%s) ---" % (actor, mode))
        cutd = {"img": CUTSCENE_IMG, "actor": actor, "anpk": anpk}
        if mode == "cjcut":
            cutd["cjCut"] = True
        # cssmoke (index) reads cutscene.img: a PS2-NATIVE VIF skinned DFF on the disc, decoded
        # by hero_bake through tools/ps2skin (and platform-neutral on the PC dev loop). csplay
        # (cjcut) is assembled from the platform-neutral PLAYER.IMG. The guard stays a safety net
        # - a genuinely undecodable actor is skipped, not fatal - but both bake now.
        try:
            s = hero_bake.bake_model(cut=cutd, emit_clst=False)
            streams.append(s)
        except (SystemExit, Exception) as e:
            print("  ! %s skipped: %s" % (actor, e))
    buf = bytearray(b"PEDS")
    buf += struct.pack("<I", len(streams))
    for s in streams:
        buf += s
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(buf)
    if quarry:
        print("=== cutscene.bin: %d/%d actor(s), %d KB -> %s ===" % (len(streams), len(ACTORS), len(buf)//1024, out))
        if len(streams) < len(ACTORS):
            print("    (an actor failed to decode; PS2-native VIF skinned DFFs route through "
                  "tools/ps2skin, PLAYER.IMG actors are platform-neutral)")
        return
    n = 0
    for d in DEPLOY:
        if os.path.isdir(os.path.dirname(os.path.dirname(d))):
            try:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                open(d, "wb").write(buf); n += 1
            except OSError:
                pass
    print("=== cutscene.bin: %d actor(s), %d KB, deployed to %d dir(s) ===" % (len(streams), len(buf)//1024, n))


if __name__ == "__main__":
    main()
