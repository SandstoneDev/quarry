#!/usr/bin/env python3
"""audio_bake - bake the source game SFX banks -> the PSP port's v2 sound arena.

Reads the `AUDIO/CONFIG` + `AUDIO/SFX` banks (see sa_audio.py for the on-disk spec),
slices every sound out of the chosen banks, transcodes 16-bit PCM -> Sony VAG ADPCM
where the source is a PC install (a PS2 disc body is already VAG and passes straight
through), and emits four files:

 sfx_index.bin bank + sound directory (format owned by sa_sfxpack.py)
 sfx_res.bin resident bank bodies - always in RAM, never reloaded
 sfx_banks.bin cell-loadable bank bodies - streamed into an arena cell on demand
 vehaud.bin per-model vehicle audio settings (format owned by sa_vehaud.py)

The v1 pool was one always-resident `sfx.bin` holding six hardcoded engine-category
banks. The arena instead keeps resident everything that must never be missing
(footsteps, collisions, doors, frontend, horn, rain, CJ pain, and EVERY dummy engine
bank so any traffic car sounds the moment it spawns) and leaves the ~48 player engine
banks cell-loadable, so the player's own car gets its real engine instead of the
nearest of six. arena_split() below is the whole policy.

We bake ONLY the environment / hero / collision / frontend / vehicle-engine banks - NO speech (SPC_*), NO radio streams, NO mission script banks.

Run: python tools/audio_bake.py # bake + deploy to all data/ dirs
 python tools/audio_bake.py measure # parse + sizes only
"""
import fnmatch
import os
import sys

import sa_audcurve
import sa_audio
import sa_sfxpack
import sa_vehaud

# SA_ROOT env override: Quarry points this at the user's extracted PS2 disc; sa_audio
# resolves the PS2 '<base>01.pak' bank files and reads their native VAG bodies. Default
# keeps the PC dev loop. Windows' case-insensitive fs maps '/audio' onto the disc 'AUDIO'.
SA_ROOT = os.environ.get("SA_ROOT", "")
SA  = SA_ROOT + "/audio"
CFG = SA + "/CONFIG"
SFX = SA + "/SFX"

# The PS2 executable, which carries gVehicleAudioSettings. There is no PC equivalent
# (the PC layout differs and sa_vehaud only decodes the PS2 one), so the arena split
# is a PS2-disc-only bake: without this table there is no list of player/dummy engine
# banks and therefore no split to make. A missing or foreign ELF must fail the bake
# loudly rather than fall back to the six-category v1 pool.
#
# Found by pattern, not by name: the converter runs on whatever disc the user owns, and
# the executable is named per region - SLES_525.41 (PAL), SLUS_209.46 (NTSC-U), SLPM/
# SLPS (NTSC-J). Hardcoding one of those fails every other user exactly the way the
# hardcoded TABLE_VA did before sa_vehaud learned to locate the table by signature.
ELF_GLOB = "SL[EUP][SM]_*"


class ElfNotFound(Exception):
    pass


def find_elf(root):
    """-> the path of the single PS2 executable in `root`.

 Matching on the name alone is not enough: a disc that has been opened in has
 `.id0`/`.id1`/`.nam` sidecars sitting next to the executable, named after it and
 therefore matching the same glob. The ELF magic is what separates the real one.
 Zero matches or more than one both raise - picking arbitrarily would bake the
 vehicle table out of a file nobody chose.
 """
    try:
        names = sorted(os.listdir(root)) if root else []
    except OSError as e:
        raise ElfNotFound("cannot list SA_ROOT %r: %s" % (root, e)) from e
    found = []
    for fn in names:
        if not fnmatch.fnmatch(fn.upper(), ELF_GLOB):
            continue
        p = os.path.join(root, fn)
        try:
            with open(p, "rb") as f:
                if f.read(4) == b"\x7fELF":
                    found.append(p)
        except OSError:
            continue                       # unreadable sidecar - not our executable
    if not found:
        raise ElfNotFound(
            "no PS2 executable found in SA_ROOT %r: nothing matching %s there is an "
            "ELF. The arena split needs the vehicle audio table out of the disc's "
            "executable; point SA_ROOT at the root of an extracted PS2 disc."
            % (root, ELF_GLOB))
    if len(found) > 1:
        raise ElfNotFound(
            "%d PS2 executables match %s in SA_ROOT %r: %s - cannot tell which disc "
            "this is, so the vehicle audio table would be read from an arbitrary one. "
            "Leave exactly one." % (len(found), ELF_GLOB, root, ", ".join(found)))
    return found[0]


# Deploy targets (each holds a data/ subfolder). Only existing ones are written.
DEPLOY = [
    "",
    "",
    "",
]

# Seed of the resident set: (bank_id, slotHint, label). Only bank_id is baked into the
# v2 pack - the arena addresses every bank by id, so BankRec has a `where` field instead
# of a slotHint. The slot column is kept as documentation of what the v1 pool preloaded
# where; the label is what the per-bank bake log prints.
BANKS = [
    (0,   41, "FEET_GENERIC"),      # SND_BANK_SLOT_FOOTSTEPS_GENERIC
    (1,   -1, "FEET_GRASS"),
    (2,   -1, "FEET_GRAVEL"),
    (3,   -1, "FEET_METAL"),
    (4,   -1, "FEET_SAND"),
    (5,   -1, "FEET_TILE"),
    (6,   -1, "FEET_WOOD"),
    (27,   3, "GENRL_BULLET_HITS"), # SND_BANK_SLOT_BULLET_HITS
    (39,   2, "GENRL_COLLISIONS"),  # SND_BANK_SLOT_COLLISIONS
    (51,  31, "GENRL_DOORS"),       # SND_BANK_SLOT_DOORS
    # b836: GENRL_EXPLOSIONS. CAEExplosionAudioEntity::AddAudioEvent (PS2 sub_58C3F0) gates
    # the whole thing on GetBankSlot(52, 5) and plays FIVE voices out of that one slot for a
    # single explosion: sounds 4, 3 and 2 at the blast position, and sound 1 TWICE at
    # {-1,0,0} and {+1,0,0} relative - a hard-panned pair that gives the boom its width.
    # Bank 52 was never in the pack at all, which is why the port had no explosion sound;
    # the same class of miss as the radio's "0 stations".
    (52,   5, "GENRL_EXPLOSIONS"), # SND_BANK_SLOT_EXPLOSIONS
    (59,   0, "GENRL_FRONTEND_GAME"),  # SND_BANK_SLOT_FRONTEND_GAME
    (60,   1, "GENRL_FRONTEND_MENU"),  # SND_BANK_SLOT_FRONTEND_MENU
    (74,  17, "GENRL_HORN"),        # SND_BANK_SLOT_HORN_AND_SIREN
    (105,  6, "GENRL_RAIN"),        # SND_BANK_SLOT_WEATHER
    (128, 32, "GENRL_SWIMMING"),    # SND_BANK_SLOT_SWIMMING
    (138, -1, "GENRL_VEHICLE_GEN"), # b442: tyre-skid loops only (TARSKIDTWIN1/2 = ids 24/25,
                                    # 11025 Hz, loop whole); the rest silenced via BANK_KEEP
    # b904: GENRL_WEAPONS. CAEWeaponAudioEntity::WeaponFire plays every gun out of this
    # one bank - and it was never in the pack, so EVERY firearm in the port was silent.
    # Third instance of the same class of miss as GENRL_EXPLOSIONS above and the radio's
    # "0 stations": the sound code was right, the bank behind it simply was not shipped.
    # 274 KB whole; BANK_KEEP trims it to the 31 ids Fire.c actually names.
    (143, -1, "GENRL_WEAPONS"),
    (144, -1, "PAIN_A_CARL"),       # CJ grunt/pain/breath (physical, NOT speech)
    # --- vehicle ENGINE banks (CAEVehicleAudioEntity): P-bank = 3 samples
    # (0 accel loop, 1 cruise loop, 2 off/decel), D-bank = 2 (0 rev, 1 idle).
    # v1 = 6 category pairs (sedan/sport/truck/van/scooter/sportbike); the runtime
    # maps each model to the nearest via vehicleAudioSettings. bankId-addressed. --
    (8,   -1, "ENG_90S_P"),         # sedan player (bravura)
    (7,   -1, "ENG_90S_D"),         # sedan dummy
    (38,  -1, "ENG_COBRA_P"),       # sport player (infernus)
    (37,  -1, "ENG_COBRA_D"),
    (84,  -1, "ENG_MACK_P"),        # truck player (linerun)
    (83,  -1, "ENG_MACK_D"),
    (137, -1, "ENG_VAN_P"),         # van player (ambulance/moonbeam)
    (136, -1, "ENG_VAN_D"),
    (119, -1, "ENG_SCOOTER_P"),     # scooter player (faggio)
    (118, -1, "ENG_SCOOTER_D"),
    (125, -1, "ENG_SPORTBIKE_P"),   # sportbike player (pcj600)
    (124, -1, "ENG_SPORTBIKE_D"),
]

BANK_LABELS = dict((b[0], b[2]) for b in BANKS)

# Banks referenced only by an enum constant - no surviving producer ever plays them
# (no guns, no swimming yet). Dropped from the pack entirely; find_bank returns NULL
# for these and every caller already guards it. arena_split asserts that the vehicle
# audio table names neither of them, so this list can never silently outrank the disc.
# HORN (74) used to be listed here and deliberately is NOT any more: the arena carries
# it resident because P3 gives the player's car a horn, and a bank that never made it
# into sfx_res.bin cannot be faded in later without re-baking the user's disc. Nothing
# plays it yet, so this changes what is *shipped*, not what is *audible*.
BANK_DROP = { 27, 128 }            # GENRL_BULLET_HITS, GENRL_SWIMMING

# --- v2 arena split -------------------------------------------------------------
# Resident = what must never be missing: footsteps (all 7 surfaces), horns, the shared
# vehicle-generic bank, and EVERY dummy engine bank so any traffic car can sound the
# moment it spawns. The PS2 rotated ten 18 KB slots for those because SPU2 had to hold
# everything else too; we have main RAM, so the rotation is simply unnecessary.
# Cell-loadable = the player engine banks, one at a time, for the car CJ is driving.
FEET_BANKS = [0, 1, 2, 3, 4, 5, 6]
HORN_BANK = 74
VEHICLE_GEN_BANK = 138


class BankRoleConflict(Exception):
    pass


class DroppedBankInUse(Exception):
    pass


def arena_split(vehaud_recs):
    """-> (resident_ids, cell_ids) as sorted lists of bank ids.

 A player engine bank is ALWAYS cell-loadable, even though six of them sit in the
 legacy BANKS whitelist from when the engine had six hardcoded categories. Leaving
 them resident costs 318 KB measured and defeats the point of the cell.

 Two invariants the disc has to satisfy for that subtraction to be safe. Both hold
 on the discs measured here; neither is guaranteed for a user's disc, and both fail
 as SILENCE rather than as a crash, which is the one failure mode the arena exists
 to prevent - so they are checked, not assumed.
 """
    resident = set(b[0] for b in BANKS)
    resident.update(FEET_BANKS)
    resident.add(HORN_BANK)
    resident.add(VEHICLE_GEN_BANK)
    dummies, cells = set(), set()
    for r in vehaud_recs:
        if r.dummy_bank >= 0:
            dummies.add(r.dummy_bank)
        if r.player_bank >= 0:
            cells.add(r.player_bank)

    # 1. No bank may be one model's dummy and another's player. `resident -= cells`
    # below is unconditional, so such a bank would be pulled OUT of the resident
    # set - and every traffic car that uses it as its dummy would then be silent
    # until some unrelated cell load happened to bring it in.
    both = dummies & cells
    if both:
        raise BankRoleConflict(
            "bank(s) %s are named as BOTH a dummy and a player engine bank by the "
            "vehicle audio table. The split gives the cell priority, which would take "
            "them out of the resident set and leave traffic that uses them as a dummy "
            "silent. Resolve before baking." % sorted(both))

    # 2. BANK_DROP is a static list written against the banks the port's producers
    # play. If the disc's vehicle table names a dropped bank, the drop would win
    # and that vehicle would lose its engine - again silently.
    dropped_in_use = BANK_DROP & (dummies | cells)
    if dropped_in_use:
        raise DroppedBankInUse(
            "BANK_DROP names bank(s) %s, which the vehicle audio table uses as engine "
            "banks. They would be dropped from the pack and those vehicles would be "
            "silent. Remove them from BANK_DROP." % sorted(dropped_in_use))

    resident |= dummies
    resident -= cells          # the cell wins: a player bank is never resident
    return sorted(resident), sorted(cells)


# per-bank cap on baked sounds (keep RAM down: the producers only use a few). 0 = all.
BANK_MAXSOUNDS = { 144: 16 }       # PAIN_A: CPedAudio_Pain uses ids 0..8 -> 16 is plenty

# Keep ONLY these soundIds within a bank; every other index up to max(keep) becomes a
# 1-frame silent VAG so soundId lookups stay index-aligned, and the tail past max(keep)
# is dropped. Trims the two whale banks whose producers touch a handful of dozens.
# CollisionAudio -> {0x02 metal-scrape, 0x1D carped/thud, 0x21 solid-wood, 0x22 concrete};
# MenuManager -> FRONTEND_GAME id 25 (AE_FRONTEND_START), FRONTEND_MENU ids 0/4/6. ~520 KB.
BANK_KEEP = {
    39: {0x02, 0x1D, 0x21, 0x22},  # GENRL_COLLISIONS (339 KB -> ~15 KB)
    52: {1, 2, 3, 4, 10},               # GENRL_EXPLOSIONS: the five voices sub_58C3F0 plays
                                    # (1 is used twice, panned) - ★ every other sound in
                                    # the bank belongs to producers we have not ported.
    # GENRL_FRONTEND_GAME. 25=AE_FRONTEND_START, 29/30=mission passed/failed jingles
    # (44.1 kHz ~1.5 s). b906 adds the PICKUP pairs out of CAEFrontendAudioEntity::
    # AddAudioEvent (0x4DD4A0): every frontend pickup plays TWO sounds hard-panned left
    # and right, {-1,0,0} and {+1,0,0}, with a 5-frame rate limit --
    # AE_FRONTEND_PICKUP_WEAPON (and the bomb-fit / purchase events) -> 27 + 28
    # AE_FRONTEND_PICKUP_MONEY/HEALTH/ADRENALINE/BODY_ARMOUR -> 16 + 17
    # Weapon pickups were playing bank 143 sound 6 - the Desert Eagle's shot layer --
    # because the AUDIO EVENT id 6 was mistaken for a sound id in the weapon bank.
    # b96x adds 23: AE_FRONTEND_RADIO_CLICK_ON/OFF (Radio.c's retune cue) - unlooped,
    # 2096 bytes, -2 dB in EventVol.dat. Was never baked, so retune had a real sound
    # to call and no data behind it; see radio_click's comment in Radio.c.
    # b97x adds 1, 2: AE_FRONTEND_RADIO_RETUNE_START/STOP - the twin-loop "tuning
    # static" bed CAETwinLoopSoundEntity plays alongside the click (both events fire
    # together at the on/off boundary on disc). BOTH loop (frame 1, per BankLkup's own
    # SPU loop offset - sa_audio.bank_loop_frames reads it automatically, no override
    # needed here): snd1 = 15360 B/12000 Hz (~2.24 s/pass), snd2 = 9648 B/10000 Hz
    # (~1.69 s/pass). Same class of miss as CLICK just above - the sound code named a
    # real id, but nothing baked it, so it would have keyed on a silent 2-frame VAG.
    # See radio_static_start's comment in Radio.c for the full derivation.
    59: {1, 2, 16, 17, 23, 25, 27, 28, 29, 30},
    60: {0, 4, 6},                  # GENRL_FRONTEND_MENU (37 KB -> ~8 KB)
    # GENRL_VEHICLE_GEN. 24/25 are the TARSKIDTWIN1/2 skid loops (b442); 10, 11, 26 and
    # 14 are the JET turbine (b831). ProcessGenericJet (PS2 sub_57B8D0) plays exactly
    # those four out of bank 138 in slot 20, one per sound slot, and they are the ONLY
    # engine sound the four jets have - Shamal, Hydra, AT-400 and Andromada carry
    # P = D = -1 in the vehicle audio table, so the bank-pair path can never make a
    # sound for them. ~41 KB for the set, all four already looping on disc.
    138: {10, 11, 14, 24, 25, 26},
    # GENRL_WEAPONS. 85 sounds on disc, 274 KB; the port's producer is Fire.c's k_gunSfx
    # table, which names exactly these 31 - the shot L/R pairs, the tails, the lows, the
    # dry-fire clicks, and the handful of non-gun one-shots (extinguisher, spray can,
    # detonator beep, goggles, flamethrower trigger). Anything else in the bank belongs to
    # producers that are not ported (the minigun state machine, the rocket launchers).
    # THIS SET IS COUPLED TO k_gunSfx: adding a soundId there without adding it here
    # gives that weapon a silent VAG, not an error.
    143: {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
          0x11, 0x12, 0x15, 0x16, 0x17, 0x18, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
          0x21, 0x31, 0x34, 0x35, 0x40, 0x49, 0x4A, 0x4C, 0x4D, 0x53},
}

# empty/silenced slot filler: a 2-frame silent VAG (priming + one-shot end) so soundId
# lookups stay index-aligned when a bank slot is dropped or whitelisted-out.
SILENT_VAG = bytes([0x00, 0x07] + [0] * 14) + bytes([0x00, 0x01] + [0] * 14)


def _write_audio_file(out_dir, name, blob):
    """Write one baked file. Quarry passes out_dir; the legacy dev loop fans the file
 out to every existing DEPLOY target, or next to the script if there is none."""
    if out_dir:
        ad = os.path.join(out_dir, "audio")
        os.makedirs(ad, exist_ok=True)
        dst = os.path.join(ad, name)
        with open(dst, "wb") as f:
            f.write(blob)
        print("  wrote ->", dst)
        return 1
    n = 0
    for d in DEPLOY:
        if os.path.isdir(d):
            os.makedirs(os.path.join(d, "audio"), exist_ok=True)
            dst = os.path.join(d, "audio", name)
            with open(dst, "wb") as f:
                f.write(blob)
            print("  deployed ->", dst)
            n += 1
    if not n:
        with open(name, "wb") as f:
            f.write(blob)
        print("  no deploy dir found; wrote ./" + name)
    return n


def _log_bank(bank_id, tag, label, sounds, codec, running):
    """One line per bank, in one format, folded into one running total per set.

 Every bank goes through here - including the mission jingle, which used to print
 a bare line with no size and no tag and was added after the counter stopped, so
 the log's last resident total and the headline differed by its 193 KB with nothing
 on screen accounting for the gap.
 """
    body = sum(len(s.vag) for s in sounds)
    running[tag] += body
    print("  bank %3d %-4s %-20s sounds=%-3d %s %6.1f KB  (%s total %.1f KB)"
          % (bank_id, tag, label, len(sounds), codec, body / 1024.0,
             tag, running[tag] / 1024.0))


def _bake_bank(bank_id, bl, pk):
    """Read one bank off the disc, apply the pool trims, and return its sounds as
 [sa_sfxpack.Sound, ...] plus (is_vag, decoded-equivalent PCM bytes) for the log.

 The trims (BANK_MAXSOUNDS / BANK_KEEP / SILENT_VAG) are what keep the resident set
 inside its budget: untrimmed, bank 39 alone is 334 KB and bank 144 is 717 KB. A
 silenced slot still emits a sound record, because soundId is an INDEX into the
 bank - skipping one would shift every id after it.
 """
    b = sa_audio.read_bank(CFG, SFX, bank_id, bl, pk)
    cap = BANK_MAXSOUNDS.get(bank_id, 0)
    keep = min(cap, len(b.sounds)) if cap else len(b.sounds)
    b.sounds = b.sounds[:keep]
    keepset = BANK_KEEP.get(bank_id)
    if keepset:                                       # drop the tail past the last kept id
        b.sounds = b.sounds[:min(len(b.sounds), max(keepset) + 1)]
    sounds = []
    pcm_equiv = 0
    for i, se in enumerate(b.sounds):
        silenced = keepset is not None and i not in keepset   # whitelisted bank: silence non-kept
        if b.is_vag:
            # PS2: the body is ALREADY native Sony PS-ADPCM -> drop the frames straight
            # into the pack (no PCM decode, no re-encode). The one genuine PS2 codec path;
            # target is a PSP, so its VAG == our VAG. Loop point read from the SPU field.
            raw = b"" if silenced else sa_audio.bank_vag(b, i)
            if len(raw) < sa_audio.FRAME:
                vag = SILENT_VAG
                loop_frames = sa_audio.NO_LOOP
            else:
                vag = raw
                loop_frames = sa_audio.NO_LOOP if silenced else sa_audio.bank_loop_frames(b, i)
            pcm_equiv += (len(vag) // sa_audio.FRAME) * sa_audio.SAMPLES_PER_FRAME * 2
        else:
            # PC: 16-bit PCM body -> transcode PCM->VAG (brute-force ADPCM encoder).
            pcm = b"" if silenced else sa_audio.bank_pcm(b, i)
            pcm_equiv += len(pcm)
            if len(pcm) < 2:
                vag = SILENT_VAG                      # empty slot -> silent VAG, indices stay aligned
                loop_frames = sa_audio.NO_LOOP
            else:
                vag, loop_blk = sa_audio.encode_vag(pcm, se.loop)
                loop_frames = sa_audio.NO_LOOP if loop_blk < 0 else loop_blk
        sounds.append(sa_sfxpack.Sound(vag=vag, rate=se.rate, loop_frame=loop_frames,
                                       headroom=se.headroom))
    return sounds, b.is_vag, pcm_equiv


def bake(measure_only=False, out_dir=None):
    bl = sa_audio.load_banklkup(CFG + "/BankLkup.dat")
    pk = sa_audio.load_pakfiles(CFG + "/PakFiles.dat")

    # The split decides WHICH banks get baked, so it has to run BEFORE the read loop,
    # not after it: only the vehicle audio table names the 43 dummy engine banks that
    # become resident and the player engine banks that become the cell set. The legacy
    # BANKS whitelist is just the seed.
    elf = find_elf(SA_ROOT)
    veh = sa_vehaud.read_from_elf(elf)
    # The two level curves ride in the same executable and CONFIG folder, and the
    # mixer cannot build a correct volume without them, so they are read here beside
    # the vehicle table - one disc, one failure point, no half-baked level model.
    atten = sa_audcurve.atten_from_elf(elf)
    eventvol, eventvol_on_disc = sa_audcurve.eventvol_from_dat(CFG + "/EventVol.dat")
    # The surface tables live in data/, not AUDIO/CONFIG - they are shared with the
    # physics side of the game and only their audio columns are read here.
    surfclass, n_surf, n_classified = sa_audcurve.surface_classes(
        SA_ROOT + "/data/surfinfo.dat", SA_ROOT + "/data/surfaud.dat")
    resident_ids, cell_ids = arena_split(veh)
    resident_set = set(resident_ids) - BANK_DROP
    cell_set = set(cell_ids) - BANK_DROP
    table_dummies = set(r.dummy_bank for r in veh if r.dummy_bank >= 0)

    collected = {}          # bankId -> [sa_sfxpack.Sound,...], insertion-ordered
    total_pcm = 0
    running = {"res": 0, "cell": 0}

    # Resident first, then cells, so the running totals in the log read as two budgets
    # rather than one. The two sets are disjoint (arena_split subtracts one from the
    # other), so no bank is read or emitted twice.
    for bank_id in sorted(resident_set) + sorted(cell_set):
        tag = "res" if bank_id in resident_set else "cell"
        # A bank that is neither in BANKS nor a player bank can only be a dummy engine
        # bank - arena_split adds nothing else beyond the whitelist.
        label = BANK_LABELS.get(bank_id,
                                "ENG_PLAYER" if tag == "cell" else "ENG_DUMMY")
        try:
            sounds, is_vag, pcm_equiv = _bake_bank(bank_id, bl, pk)
        except Exception as e:
            # The loop is fed by the disc's own vehicle table now, not by the curated
            # BANKS whitelist, so an id with no bank behind it is reachable - and
            # sa_audio indexes BankLkup directly, which reports it as a bare
            # "IndexError: list index out of range" naming neither the bank, nor the
            # table, nor the executable it came from.
            if bank_id in cell_set:
                src = "player engine bank named by the vehicle audio table in %s" % elf
            elif bank_id in table_dummies:
                src = "dummy engine bank named by the vehicle audio table in %s" % elf
            else:
                src = "from the BANKS whitelist"
            raise RuntimeError("bank %d (%s) could not be read from the disc: %s: %s"
                               % (bank_id, src, type(e).__name__, e)) from e
        total_pcm += pcm_equiv
        collected[bank_id] = sounds
        _log_bank(bank_id, tag, label, sounds, "VAG" if is_vag else "PCM", running)

    # --- custom bank 250: MISSION passed/failed jingle from the BEATS MUSIC STREAM (b542).
    # The mission-passed sound is NOT an SFX bank (b540 grabbed the wrong one -> noise); per the
    # modding docs it is BEATS track 182 (Mission
    # Complete #1 = passed) / 183 (#2 = failed). radio_bake extracts + decrypts the stream track
    # -> OGG; soundfile decodes -> mono 16-bit PCM -> VAG. Sound 0 = passed, 1 = failed.
    # PS2 DISC PATH FIRST. The stream elements are already SPU ADPCM, which is precisely
    # what a VAG body in this bank is, so the jingle needs no decode, no resample and no
    # encoder - the bytes go in as they come off the disc. That also means no numpy and no
    # soundfile, which is why this used to be skipped on a PS2 convert and the mission
    # end-sound never played. BEATS 182 = passed, 183 = failed, both 9.6 s at 24 kHz.
    jingle = []
    jingle_codec = "VAG"
    jingle_done = False
    try:
        import sa_ps2_stream as _S
        if os.path.isdir(os.path.join(SA, "STREAMS")):
            _packs, _tracks = _S.load_index(SA)
            for tid in (182, 183):
                h = _S.read_header(SA, _packs, _tracks, tid)
                with open(h["path"], "rb") as _f:
                    _f.seek(h["offset"] + _S.DATA)
                    raw = _f.read(h["size"])
                # Channel 0's own block, wherever the layout puts it - BEATS is a
                # two-channel element so that is +0, but ask rather than assume.
                body = _S.channel_bytes(raw, h["blk_off"][0], h["blk_size"][0],
                                        h["bytes_per_ch"])   # mono is enough for a sting
                keep = int(7.2 * h["rate"] / _S.SAMPLES_PER_FRAME) * _S.FRAME
                # Floor to a whole ADPCM frame: a short stream element would otherwise
                # hand sa_sfxpack a body that is not a FRAME multiple and abort the bake.
                n = min(keep, len(body))
                body = bytearray(body[:n - (n % _S.FRAME)])
                if len(body) < _S.FRAME:
                    raise RuntimeError("BEATS %d empty" % tid)
                body[-_S.FRAME + 1] |= 0x01           # mark the last frame END so playback stops there
                jingle.append(sa_sfxpack.Sound(vag=bytes(body), rate=h["rate"],
                                               loop_frame=sa_audio.NO_LOOP, headroom=0))
            jingle_done = True
    except Exception as _e:
        jingle = []
        print("  mission jingle (PS2 path) skipped:", _e)

    try:
        if jingle_done:
            raise RuntimeError("already baked from the PS2 stream")
        # DEP GATE for the PC dev loop: that path decodes an OGG and needs radio_bake +
        # numpy + soundfile (heavy). Quarry sets QUARRY_SFX_NO_JINGLE=1 to keep a bake on
        # the stdlib alone; the engine guards a missing bank 250.
        if os.environ.get("QUARRY_SFX_NO_JINGLE") == "1":
            raise RuntimeError("QUARRY_SFX_NO_JINGLE=1 - mission jingle deferred to the radio pass")
        import radio_bake, soundfile as _sf, io as _io, numpy as _np
        GAME = os.path.dirname(SA)
        _packs, _lut = radio_bake.load_lookups(GAME)
        for tid in (182, 183):
            _, ogg = radio_bake.extract_track(GAME, _packs, _lut, tid)
            arr, srate = _sf.read(_io.BytesIO(ogg), dtype="int16", always_2d=True)
            mono = arr.astype("float32").mean(axis=1) if arr.shape[1] > 1 else arr[:, 0].astype("float32")
            # Native 32 kHz, full quality (64-bit mixer cursor handles long sounds). Play the whole
            # sting to its NATURAL end (loud 0-5.2s then a reverb decay to silence by ~7s - that
            # peak = as loud as possible with no clipping (the music is mastered quieter than SFX).
            mono = mono[:int(7.2 * srate)]
            pk = float(_np.max(_np.abs(mono))) or 1.0
            mono = mono * (0.95 * 32767.0 / pk)
            mono = _np.clip(mono, -32768, 32767).astype("int16")
            pcm = mono.tobytes()
            total_pcm += len(pcm)
            vag, _lb = sa_audio.encode_vag(pcm, False)
            jingle.append(sa_sfxpack.Sound(vag=vag, rate=srate,
                                           loop_frame=sa_audio.NO_LOOP, headroom=0))
        jingle_codec = "PCM"                  # PC path: OGG -> PCM -> VAG, not passthrough
    except Exception as _e:
        # Silent when the PS2 path already delivered it - that success is visible in
        # bank 250's own log line, and "SKIPPED" shouting next to it read like a fault.
        if not jingle_done:
            jingle = []
            print("  mission jingle bake SKIPPED:", _e)

    if jingle:
        # Two sounds, played once at mission end - resident: nothing would ever pay to
        # stream a cell in for it, and a mission that ends in silence reads as a bug.
        # By far the largest single resident item (193 KB, 18% of the set), so it is
        # logged like every other bank rather than as a bare footnote.
        collected[250] = jingle
        resident_set.add(250)
        _log_bank(250, "res", "MISSION_JINGLE", jingle, jingle_codec, running)

    # --- pack -------------------------------------------------------------------
    packs = [sa_sfxpack.Bank(bank_id=bid, resident=(bid in resident_set), sounds=snd)
             for bid, snd in collected.items()]
    extras = sa_audcurve.pack_extras(atten, eventvol, surfclass)
    index, res_blob, cell_blob = sa_sfxpack.build(packs, extras)
    veh_blob = sa_vehaud.pack(veh)

    n_res = sum(1 for b in packs if b.resident)
    n_sounds = sum(len(b.sounds) for b in packs)
    worst = max([(b.data_bytes, b.bank_id) for b in packs if not b.resident] or [(0, -1)])
    print("--- arena: resident %.1f KB (%d banks) of %.1f KB budget, cells %.1f KB "
          "(%d banks), index %.1f KB, %d sounds"
          % (len(res_blob) / 1024.0, n_res, sa_sfxpack.RESIDENT_LIMIT / 1024.0,
             len(cell_blob) / 1024.0, len(packs) - n_res, len(index) / 1024.0, n_sounds))
    print("--- worst cell bank: %d at %.1f KB; PCM in %.1f MB -> VAG %.1f MB; vehaud %d records"
          % (worst[1], worst[0] / 1024.0, total_pcm / 1048576.0,
             (len(res_blob) + len(cell_blob)) / 1048576.0, len(veh)))
    n_muted = sum(1 for v in eventvol if v == sa_audcurve.EVENTVOL_MUTED)
    print("--- curves: distance attenuation %d entries (flat 0 dB to normalised %.1f), "
          "event volumes %d of %d on disc (%d muted), extras %.1f KB"
          % (len(atten), (next(i for i, v in enumerate(atten) if v < 0.0) - 1) / 10.0,
             len(eventvol), eventvol_on_disc, n_muted, len(extras) / 1024.0))
    print("--- surfaces: %d of %d carry an audio class (the rest stay on the generic "
          "footstep bank, as the disc leaves them)" % (n_classified, n_surf))
    if measure_only:
        return

    for name, blob in (("sfx_index.bin", index), ("sfx_res.bin", res_blob),
                       ("sfx_banks.bin", cell_blob), ("vehaud.bin", veh_blob)):
        _write_audio_file(out_dir, name, blob)


if __name__ == "__main__":
    # Usage: audio_bake.py [measure] [<outDataDir>]
    # measure -> parse + sizes only, no write
    # <outDataDir> -> write the four files into <dir>/audio/ (Quarry); omit for the
    # legacy deploy list
    _args = sys.argv[1:]
    _measure = "measure" in _args
    _outs = [a for a in _args if a != "measure"]
    bake(measure_only=_measure, out_dir=(_outs[0] if _outs else None))
