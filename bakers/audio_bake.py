#!/usr/bin/env python3
"""audio_bake - bake the source game SFX banks -> data/sfx.bin for the PSP port (sceSas VAG pool).

Reads the PC `AUDIO/CONFIG` + `AUDIO/SFX` banks (see sa_audio.py for the on-disk spec),
slices every sound out of the chosen banks, transcodes 16-bit PCM -> Sony VAG ADPCM
(sceSas-native), and writes a single resident pool `sfx.bin` the runtime mmaps once.

We bake ONLY the environment / hero / collision / frontend banks - NO speech (SPC_*),
NO radio streams, NO mission script banks. The selection mirrors what the surviving
producers actually trigger (footsteps by surface, collision impacts, bullet hits, doors,
horn, rain, swim, frontend UI, CJ pain grunts).

sfx.bin (little-endian) ========================================================
Header @0x00 (0x20):
 u32 magic 'SFXB' = 0x42584653
 u32 version = 1
 u32 nBanks, nSounds
 u32 blobOff # file offset of the VAG blob
 u32 blobSize
 u32 sampleRateMax # (info) highest source rate baked
 u32 reserved
BankRec[nBanks] @0x20 (0x10):
 u16 bankId # eSoundBank
 i16 slotHint # eSoundBankSlot to preload into (-1 = addressed by bankId only)
 u32 firstSound # index into SoundRec[]
 u16 numSounds
 u16 pad
 u32 pad2
SoundRec[nSounds] (0x18):
 u32 vagOff # offset into the VAG blob (relative to blobOff)
 u32 vagBytes # ADPCM byte length (multiple of 16)
 u32 rate # Hz
 u32 loopStart # in VAG frames into the blob (0xFFFFFFFF = no loop)
 i16 headroom # gain trim (hundredths dB); GetSoundHeadroom = headroom/100
 u16 pad
 u32 pad2
VAG blob @ blobOff: concatenated VAG ADPCM bodies (each 16-byte aligned).
================================================================================

Run: python tools/audio_bake.py # bake + deploy to all data/ dirs
 python tools/audio_bake.py measure # parse + sizes only
"""
import os
import sys
import struct

import sa_audio

# SA_ROOT env override: Quarry points this at the user's extracted PS2 disc; sa_audio
# resolves the PS2 '<base>01.pak' bank files and reads their native VAG bodies. Default
# keeps the PC dev loop. Windows' case-insensitive fs maps '/audio' onto the disc 'AUDIO'.
SA  = os.environ.get("SA_ROOT", "") + "/audio"
CFG = SA + "/CONFIG"
SFX = SA + "/SFX"

# Deploy targets (each holds a data/ subfolder). Only existing ones are written.
DEPLOY = [
    "",
    "",
    "",
]

# Banks to bake: (bank_id, slotHint, label). slotHint = the eSoundBankSlot the runtime
# preloads it into (so PlaySound(slot,id) resolves); -1 = resident, addressed by bankId
# (the surface-feet banks + pain, which the producer binds to a slot on demand).
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
    (59,   0, "GENRL_FRONTEND_GAME"),  # SND_BANK_SLOT_FRONTEND_GAME
    (60,   1, "GENRL_FRONTEND_MENU"),  # SND_BANK_SLOT_FRONTEND_MENU
    (74,  17, "GENRL_HORN"),        # SND_BANK_SLOT_HORN_AND_SIREN
    (105,  6, "GENRL_RAIN"),        # SND_BANK_SLOT_WEATHER
    (128, 32, "GENRL_SWIMMING"),    # SND_BANK_SLOT_SWIMMING
    (138, -1, "GENRL_VEHICLE_GEN"), # b442: tyre-skid loops only (TARSKIDTWIN1/2 = ids 24/25,
                                    # 11025 Hz, loop whole); the rest silenced via BANK_KEEP
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

# per-bank cap on baked sounds (keep RAM down: the producers only use a few). 0 = all.
BANK_MAXSOUNDS = { 144: 16 }       # PAIN_A: CPedAudio_Pain uses ids 0..8 -> 16 is plenty

# Banks referenced only by an enum constant - no surviving producer ever plays them
# (no guns / player horn / swimming yet). Dropped from the pool entirely; find_bank
# returns NULL for these and every caller already guards it. ~208 KB reclaimed.
BANK_DROP = { 27, 74, 128 }        # GENRL_BULLET_HITS, GENRL_HORN, GENRL_SWIMMING

# Keep ONLY these soundIds within a bank; every other index up to max(keep) becomes a
# 1-frame silent VAG so soundId lookups stay index-aligned, and the tail past max(keep)
# is dropped. Trims the two whale banks whose producers touch a handful of dozens.
# CollisionAudio -> {0x02 metal-scrape, 0x1D carped/thud, 0x21 solid-wood, 0x22 concrete};
# MenuManager -> FRONTEND_GAME id 25 (AE_FRONTEND_START), FRONTEND_MENU ids 0/4/6. ~520 KB.
BANK_KEEP = {
    39: {0x02, 0x1D, 0x21, 0x22},  # GENRL_COLLISIONS (339 KB -> ~15 KB)
    59: {25, 29, 30},               # GENRL_FRONTEND_GAME: 25=AE_FRONTEND_START, 29/30=mission passed/failed jingles (44.1kHz ~1.5s)
    60: {0, 4, 6},                  # GENRL_FRONTEND_MENU (37 KB -> ~8 KB)
    138: {24, 25},                  # GENRL_VEHICLE_GEN: TARSKIDTWIN1/2 skid loops (b442)
}

MAGIC = 0x42584653  # 'SFXB'

# empty/silenced slot filler: a 2-frame silent VAG (priming + one-shot end) so soundId
# lookups stay index-aligned when a bank slot is dropped or whitelisted-out.
SILENT_VAG = bytes([0x00, 0x07] + [0] * 14) + bytes([0x00, 0x01] + [0] * 14)


def bake(measure_only=False, out_dir=None):
    bl = sa_audio.load_banklkup(CFG + "/BankLkup.dat")
    pk = sa_audio.load_pakfiles(CFG + "/PakFiles.dat")

    bank_recs = []     # (bankId, slotHint, firstSound, numSounds)
    sound_recs = []    # (vagOff, vagBytes, rate, loopFrames, headroom)
    blob = bytearray()
    rate_max = 0
    total_pcm = 0

    for bank_id, slot, label in BANKS:
        if bank_id in BANK_DROP:
            continue
        b = sa_audio.read_bank(CFG, SFX, bank_id, bl, pk)
        first = len(sound_recs)
        cap = BANK_MAXSOUNDS.get(bank_id, 0)
        keep = min(cap, len(b.sounds)) if cap else len(b.sounds)
        b.sounds = b.sounds[:keep]
        keepset = BANK_KEEP.get(bank_id)
        if keepset:                                   # drop the tail past the last kept id
            b.sounds = b.sounds[:min(len(b.sounds), max(keepset) + 1)]
        for i, se in enumerate(b.sounds):
            silenced = keepset is not None and i not in keepset   # whitelisted bank: silence non-kept
            if b.is_vag:
                # PS2: the body is ALREADY native Sony PS-ADPCM -> drop the frames straight
                # into the pool (no PCM decode, no re-encode). The one genuine PS2 codec path;
                # target is a PSP, so its VAG == our VAG. Loop point read from the SPU field.
                raw = b"" if silenced else sa_audio.bank_vag(b, i)
                if len(raw) < sa_audio.FRAME:
                    vag = SILENT_VAG
                    loop_frames = sa_audio.NO_LOOP
                else:
                    vag = raw
                    loop_frames = sa_audio.NO_LOOP if silenced else sa_audio.bank_loop_frames(b, i)
                total_pcm += (len(vag) // sa_audio.FRAME) * sa_audio.SAMPLES_PER_FRAME * 2  # decoded-equiv
            else:
                # PC: 16-bit PCM body -> transcode PCM->VAG (brute-force ADPCM encoder).
                pcm = b"" if silenced else sa_audio.bank_pcm(b, i)
                total_pcm += len(pcm)
                if len(pcm) < 2:
                    vag = SILENT_VAG                  # empty slot -> silent VAG, indices stay aligned
                    loop_frames = sa_audio.NO_LOOP
                else:
                    vag, loop_blk = sa_audio.encode_vag(pcm, se.loop)
                    loop_frames = sa_audio.NO_LOOP if loop_blk < 0 else loop_blk
            off = len(blob)
            blob += vag
            if len(blob) & 0xF:                       # keep 16-byte alignment
                blob += b"\x00" * (0x10 - (len(blob) & 0xF))
            sound_recs.append((off, len(vag), se.rate, loop_frames, se.headroom))
            rate_max = max(rate_max, se.rate)
        bank_recs.append((bank_id, slot, first, len(b.sounds)))
        print("  bank %3d %-20s sounds=%-3d  %s  vag so far=%.1f KB"
              % (bank_id, label, len(b.sounds), "VAG" if b.is_vag else "PCM", len(blob)/1024.0))

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
    jingle_done = False
    try:
        import sa_ps2_stream as _S
        if os.path.isdir(os.path.join(SA, "STREAMS")):
            _packs, _tracks = _S.load_index(SA)
            jfirst = len(sound_recs)
            for tid in (182, 183):
                h = _S.read_header(SA, _packs, _tracks, tid)
                with open(h["path"], "rb") as _f:
                    _f.seek(h["offset"] + _S.DATA)
                    raw = _f.read(h["size"])
                body = _S.channel_bytes(raw, 0, h["bytes_per_ch"])     # mono is enough for a sting
                keep = int(7.2 * h["rate"] / _S.SAMPLES_PER_FRAME) * _S.FRAME
                body = bytearray(body[:min(keep, len(body))])
                if len(body) < _S.FRAME:
                    raise RuntimeError("BEATS %d empty" % tid)
                body[-_S.FRAME + 1] |= 0x01           # mark the last frame END so playback stops there
                off = len(blob); blob += bytes(body)
                if len(blob) & 0xF:
                    blob += b"\x00" * (0x10 - (len(blob) & 0xF))
                sound_recs.append((off, len(body), h["rate"], sa_audio.NO_LOOP, 0))
                rate_max = max(rate_max, h["rate"])
            bank_recs.append((250, -1, jfirst, 2))
            print("  bank 250 MISSION_JINGLE      sounds=2 (BEATS 182 passed / 183 failed, PS2 ADPCM passthrough)")
            jingle_done = True
    except Exception as _e:
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
        jfirst = len(sound_recs)
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
            off = len(blob); blob += vag
            if len(blob) & 0xF:
                blob += b"\x00" * (0x10 - (len(blob) & 0xF))
            sound_recs.append((off, len(vag), srate, sa_audio.NO_LOOP, 0))
            rate_max = max(rate_max, srate)
        bank_recs.append((250, -1, jfirst, 2))     # MISSION_JINGLE
        print("  bank 250 MISSION_JINGLE      sounds=2 (BEATS 182 passed / 183 failed)  vag=%.1f KB" % (len(blob)/1024.0))
    except Exception as _e:
        print("  mission jingle bake SKIPPED:", _e)

    nB, nS = len(bank_recs), len(sound_recs)
    hdr_sz = 0x20 + nB*0x10 + nS*0x18
    blob_off = (hdr_sz + 0xF) & ~0xF

    out = bytearray()
    out += struct.pack("<IIIIIIII", MAGIC, 1, nB, nS, blob_off, len(blob), rate_max, 0)
    for bankId, slot, first, num in bank_recs:
        out += struct.pack("<hhIHHI", bankId, slot, first, num, 0, 0)
    for off, vb, rate, loop, hr in sound_recs:
        out += struct.pack("<IIIIhHI", off, vb, rate, loop, hr, 0, 0)
    if len(out) < blob_off:
        out += b"\x00" * (blob_off - len(out))
    out += blob

    print("--- sfx.bin: %d banks, %d sounds, PCM in %.1f MB -> VAG %.1f MB (%.2fx), file %.1f MB"
          % (nB, nS, total_pcm/1048576.0, len(blob)/1048576.0,
             len(blob)/max(1,total_pcm), len(out)/1048576.0))
    if measure_only:
        return
    # Quarry path: write the single pool into <out_dir>/audio/sfx.bin (no memstick deploy).
    if out_dir:
        ad = os.path.join(out_dir, "audio")
        os.makedirs(ad, exist_ok=True)
        dst = os.path.join(ad, "sfx.bin")
        with open(dst, "wb") as f:
            f.write(out)
        print("  wrote ->", dst)
        return
    # legacy dev loop: deploy to every existing memstick/data dir.
    n = 0
    for d in DEPLOY:
        if os.path.isdir(d):
            os.makedirs(os.path.join(d, "audio"), exist_ok=True)
            with open(os.path.join(d, "audio", "sfx.bin"), "wb") as f:
                f.write(out)
            print("  deployed ->", os.path.join(d, "audio", "sfx.bin"))
            n += 1
    if not n:
        # fall back: write next to the script
        with open("sfx.bin", "wb") as f:
            f.write(out)
        print("  no deploy dir found; wrote ./sfx.bin")


if __name__ == "__main__":
    # Usage: audio_bake.py [measure] [<outDataDir>]
    # measure -> parse + sizes only, no write
    # <outDataDir> -> write <dir>/audio/sfx.bin (Quarry); omit for the legacy deploy list
    _args = sys.argv[1:]
    _measure = "measure" in _args
    _outs = [a for a in _args if a != "measure"]
    bake(measure_only=_measure, out_dir=(_outs[0] if _outs else None))
