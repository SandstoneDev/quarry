#!/usr/bin/env python3
"""loadtune_bake.py - the ORIGINAL SA loading-screen music -> data/audio/loadtuneN.wav.

SA plays a random one of 4 loading tunes over the loading screens: frontend event
AE_FRONTEND_LOADING_TUNE_START picks seed 0..3 and plays sounds {2*seed, 2*seed+1}
of SND_BANK_GENRL_LOADING (bank 82) as a stereo L/R pair (the reference notes
AEFrontendAudioEntity.cpp:20-22, 686-724). This tool extracts the 8 mono PCM
sounds, interleaves each pair into one stereo 16-bit WAV, and deploys them; the
engine (LoadMusic.c) streams a random one during the load.

Usage: python tools/loadtune_bake.py [outDataDir ...]
 (no args = deploy to F: + PPSSPP memstick + deploy_psp data dirs)
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_audio

# SA_ROOT env override: Quarry points this at the extracted PS2 disc (sa_audio resolves
# the PS2 '<base>01.pak' bank file and decodes its native VAG body). Default = PC dev loop.
SA  = os.environ.get("SA_ROOT", "") + "/audio"
CFG = SA + "/CONFIG"
SFX = SA + "/SFX"
BANK_LOADING = 82          # SND_BANK_GENRL_LOADING

DEFAULT_OUT = [
    "",
    "",
    "",
]

SRC_RATES = (44100,)   # b375: plain sceAudio channel (SRC dropped engine-wide) -> always 44.1k

def resample_pcm16(mono, src, dst):
    """linear-interp mono PCM16 bytes src->dst rate (bank tunes are 28000 Hz,
 which sceAudioSRC does NOT accept - snap to the next legal rate)."""
    import array
    a = array.array("h"); a.frombytes(mono[:len(mono) & ~1])
    n = len(a)
    m = int(n * dst / src)
    out = array.array("h", bytes(2 * m))
    for i in range(m):
        f = i * src / dst
        j = int(f); t = f - j
        s0 = a[j] if j < n else 0
        s1 = a[j + 1] if j + 1 < n else s0
        out[i] = int(s0 + (s1 - s0) * t)
    return out.tobytes()

def write_wav(path, left, right, rate):
    n = min(len(left), len(right)) // 2          # samples (PCM16 mono each)
    data = bytearray(n * 4)
    for i in range(n):
        data[i*4:i*4+2]   = left[i*2:i*2+2]
        data[i*4+2:i*4+4] = right[i*2:i*2+2]
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI",
                      b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16,
                      1, 2, rate, rate * 4, 4, 16, b"data", len(data))
    with open(path, "wb") as f:
        f.write(hdr); f.write(data)
    return n / float(rate), len(data)

def main():
    outs = sys.argv[1:] or DEFAULT_OUT
    bl = sa_audio.load_banklkup(CFG + "/BankLkup.dat")
    pf = sa_audio.load_pakfiles(CFG + "/PakFiles.dat")
    bank = sa_audio.read_bank(CFG, SFX, BANK_LOADING, bl, pf)
    ns = len(bank.sounds)
    print("bank 82 GENRL_LOADING: %d sounds" % ns)
    tunes = []
    for seed in range(ns // 2):
        if bank.is_vag:            # PS2 body is native Sony PS-ADPCM -> decode to PCM16 (stdlib)
            L = sa_audio.decode_vag(sa_audio.bank_vag(bank, 2 * seed))
            R = sa_audio.decode_vag(sa_audio.bank_vag(bank, 2 * seed + 1))
        else:                      # PC body is already 16-bit PCM
            L = sa_audio.bank_pcm(bank, 2 * seed)
            R = sa_audio.bank_pcm(bank, 2 * seed + 1)
        rate = bank.sounds[2 * seed].rate
        if rate not in SRC_RATES:                 # 28000 in retail -> resample to 32000
            dst = min((r for r in SRC_RATES if r >= rate), default=48000)
            L = resample_pcm16(L, rate, dst)
            R = resample_pcm16(R, rate, dst)
            rate = dst
        tunes.append((L, R, rate))
    for d in outs:
        if not os.path.isdir(d):
            print("skip (no data dir):", d); continue   # unmounted deploy target
        ad = os.path.join(d, "audio")
        os.makedirs(ad, exist_ok=True)                  # Quarry passes the data dir; make audio/
        for i, (L, R, rate) in enumerate(tunes):
            secs, nbytes = write_wav(os.path.join(ad, "loadtune%d.wav" % i), L, R, rate)
            if d is outs[0]:
                print("  loadtune%d.wav  %5.1fs  %4dKB  %dHz" % (i, secs, nbytes >> 10, rate))
        print("deployed ->", ad)

if __name__ == "__main__":
    main()
