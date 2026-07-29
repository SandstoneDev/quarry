#!/usr/bin/env python3
"""sa_audio - the source game SFX bank reader + PCM->VAG (Sony 4-bit ADPCM) encoder.

The PSP audio backend uses the hardware mixer `sceSas`, which plays **Sony VAG
ADPCM** voices natively (the same 16-byte/28-sample codec the PS2 SPU2 used).
the source game's PC banks ship as raw **16-bit mono PCM** under `AUDIO/SFX/<PAK>`, indexed
by `AUDIO/CONFIG/*.DAT`. So the port reads the PC config + bank headers, slices each
sample, and transcodes PCM->VAG here (the read-only `gvcslib.vag` decoder validates us).

On-disk formats (verified on real PC 1.0 US bytes, cross-checked vs the reference notes
`AEBankLoader`/`AEBankSlot`):

 BankLkup.dat - AEBankLookup[710], 0xC each:
 u8 PakFileNo; u8 pad[3](=0xCC); u32 FileOffset(BYTES into pak); u32 NumBytes
 PakFiles.dat - AEPakLookup[9], 0x34 each:
 char BaseFilename[12]; u32 FileCopyLSNs[10] (0=FEET 1=GENRL 2=PAIN_A 3=SCRIPT 4..8=SPC)
 AUDIO/SFX/<PAK> bank blob @ (FileOffset, NumBytes):
 AEAudioStream header (0x12C4) = { i16 NumSounds; i16 pad; SoundEntry[400] } + 16-bit PCM body
 SoundEntry (0xC) = { u32 BankOffsetBytes(into body); u32 LoopStartOffset(samples,0xFFFFFFFF=none);
 u16 SampleRateHz; i16 Headroom }
 sample[i] length = SoundEntry[i+1].BankOffsetBytes - SoundEntry[i].BankOffsetBytes
 (last = bodyLen - SoundEntry[i].BankOffsetBytes)

VAG ADPCM frame (16 bytes -> 28 samples) - canonical PSX/PS2/PSP, matches gvcslib.vag decode:
 byte0 = (predictor<<4) | shift ; byte1 = flag ; byte2..15 = 28 nibbles (low first)
 decode: s = sign4(nib)<<12 >> shift ; out = clamp16(s + (h1*F0[p] + h2*F1[p])>>6)
flag: 0 normal, 1 end(one-shot stop), 6 loop-start, 3 loop-end(repeat), 7 silent/priming.

The encoder is closed-loop (DPCM feedback uses the *reconstructed* history so error can't
drift) and brute-forces the 5 predictors x best shift per block, picking min squared error.
Validated by re-decoding with the proven `gvcslib.vag` decoder and checking SNR (see __main__).
"""
import os
import struct

# Sony ADPCM predictor coefficients (numerator over 64) - identical to gvcslib.vag._F0/_F1.
F0 = (0, 60, 115, 98, 122)
F1 = (0, 0, -52, -55, -60)

FRAME = 16
SAMPLES_PER_FRAME = 28
BANK_HDR = 0x12C4          # AEAudioStream header size (4 + 400*0xC)
MAX_SOUNDS = 400
NO_LOOP = 0xFFFFFFFF

# ---- config / bank parsing -------------------------------------------------

class SoundEntry:
    __slots__ = ("off", "loop", "rate", "headroom", "length")
    def __init__(self, off, loop, rate, headroom):
        self.off = off; self.loop = loop; self.rate = rate
        self.headroom = headroom; self.length = 0

class Bank:
    __slots__ = ("bank_id", "pak_no", "file_off", "num_bytes", "sounds", "body", "is_vag")
    def __init__(self, bank_id, pak_no, file_off, num_bytes):
        self.bank_id = bank_id; self.pak_no = pak_no
        self.file_off = file_off; self.num_bytes = num_bytes
        self.sounds = []          # [SoundEntry,...]
        self.body = b""           # sample body after the 0x12C4 header
        self.is_vag = False       # True = PS2 body is native Sony PS-ADPCM (VAG); False = PC 16-bit PCM

def resolve_pak_path(sfx_dir, base):
    """Locate a bank's pak file. PC ships it bare ('GENRL'); the PS2 disc ships
 '<base>01.pak' (+ an '02.pak' byte-duplicate for the DVD seek layout, like
 GTA3_1.IMG). Try the PC name first, then the PS2 forms, then a case-insensitive
 scan. Returns the resolved path (falls back to the bare join so open() raises a
 clear error if truly absent)."""
    for cand in (base, base + ".pak", base + "01.pak", base + "01", base + "02.pak"):
        p = os.path.join(sfx_dir, cand)
        if os.path.isfile(p):
            return p
    up = base.upper()
    try:
        for fn in sorted(os.listdir(sfx_dir)):
            u = fn.upper()
            if u == up or u == up + ".PAK" or u.startswith(up + "0"):
                return os.path.join(sfx_dir, fn)
    except OSError:
        pass
    return os.path.join(sfx_dir, base)

def bank_body_is_vag(body):
    """Sniff a bank body: True = Sony PS-ADPCM (PS2), False = raw 16-bit PCM (PC).
 PS-ADPCM is 16-byte framed; byte[1] of every frame is a loop flag in 0..7 and the
 low nibble of byte[0] is a shift 0..12. Real PCM audio breaks this within a few
 frames (a sample's high byte routinely exceeds 7)."""
    n = min(len(body) // FRAME, 64)
    if n < 4:
        return False
    for f in range(n):
        if body[f * FRAME + 1] > 7:          # flag byte out of range -> PCM
            return False
        if (body[f * FRAME] & 0x0F) > 12:    # ADPCM shift is 0..12
            return False
    return True

def load_banklkup(path):
    """Return [(bank_id, pak_no, file_off, num_bytes), ...] from BankLkup.dat."""
    d = open(path, "rb").read()
    out = []
    for i in range(len(d) // 0xC):
        pak, off, nb = struct.unpack_from("<BxxxII", d, i * 0xC)
        out.append((i, pak, off, nb))
    return out

def load_pakfiles(path):
    """Return [base_name, ...] (index = pak number) from PakFiles.dat."""
    d = open(path, "rb").read()
    return [d[i*0x34:i*0x34+12].split(b"\x00", 1)[0].decode("latin1")
            for i in range(len(d) // 0x34)]

def _vag_sound_len(body, off):
    """Byte length of the VAG sound starting at `off`: scan frames to the SPU end marker
 (flag 1/3/7) inclusive, bounded by the available body. Used for a bank's LAST sound,
 which has no successor offset to bound it (the sounds are packed tight, each ending on
 its end-flag frame, with only inter-BANK sector padding after the final one)."""
    p, n = off, len(body)
    while p + FRAME <= n:
        flag = body[p + 1]
        p += FRAME
        if flag in (1, 3, 7):
            break
    return p - off

def read_bank(cfg_dir, sfx_dir, bank_id, banklkup, pakfiles):
    """Read one bank's header + sample body. Returns a populated Bank."""
    _, pak_no, file_off, num_bytes = banklkup[bank_id]
    pak_path = resolve_pak_path(sfx_dir, pakfiles[pak_no])
    # BankLkup.NumBytes is NOT a reliable body length on the PS2 disc: for several banks it
    # is smaller than the bank's own sample data (bank 3 claims a 12-byte body; bank 5 a
    # NEGATIVE one). A bank's true extent is the span up to the NEXT bank stored in the same
    # pak (banks are packed back-to-back, sector-padded); the last bank runs to end-of-pak.
    # Never read fewer bytes than NumBytes. (On PC NumBytes is correct and this reads the
    # same body plus trailing sector padding, which the slicing below ignores.)
    nxt_off = None
    for (_bid, p, o, _nb) in banklkup:
        if p == pak_no and o > file_off and (nxt_off is None or o < nxt_off):
            nxt_off = o
    with open(pak_path, "rb") as f:
        if nxt_off is not None:
            span = nxt_off - file_off
        else:
            f.seek(0, 2); span = f.tell() - file_off
        span = max(span, num_bytes)
        f.seek(file_off)
        blob = f.read(span)
    b = Bank(bank_id, pak_no, file_off, num_bytes)
    ns, _pad = struct.unpack_from("<hh", blob, 0)
    for i in range(ns):
        off, loop, rate, hr = struct.unpack_from("<IIHh", blob, 4 + i * 0xC)
        b.sounds.append(SoundEntry(off, loop, rate, hr))
    b.body = blob[BANK_HDR:]
    # PS2 disc bodies are already Sony PS-ADPCM (VAG); PC bodies are 16-bit PCM. The header
    # layout is byte-identical between the two - only the sample codec differs.
    b.is_vag = bank_body_is_vag(b.body)
    # Per-sound lengths: interior sounds run to the next sound's offset (header-authoritative,
    # and the sounds are packed tight). The LAST sound has no successor -> VAG: scan to its
    # end-flag frame (precise, drops the inter-bank sector padding); PCM: bound by NumBytes
    # (PC-authoritative body length - the PC path is left byte-for-byte unchanged).
    pcm_body_len = max(0, num_bytes - BANK_HDR)
    for i, s in enumerate(b.sounds):
        if i + 1 < len(b.sounds):
            s.length = max(0, b.sounds[i+1].off - s.off)
        elif b.is_vag:
            s.length = _vag_sound_len(b.body, s.off)
        else:
            s.length = max(0, pcm_body_len - s.off)
    return b

def bank_pcm(bank, i):
    """Raw little-endian 16-bit mono PCM bytes for sound i of a Bank (PC banks)."""
    s = bank.sounds[i]
    return bank.body[s.off: s.off + s.length]

# ---- PS2 native VAG (the one genuine PS2-input addition) -------------------
# On the PS2 disc the SFX bodies are ALREADY Sony PS-ADPCM - the exact codec the PSP
# sceSas/our software mixer decode natively - so the correct path is a straight
# pass-through into the sfx.bin VAG pool (no PCM decode, no re-encode). This is both
# simpler and higher-fidelity than the PC path (which must transcode PCM->VAG).

def bank_vag(bank, i):
    """Raw Sony PS-ADPCM (VAG) frames for sound i of a PS2 bank, ready to drop into the
 sfx.bin VAG blob verbatim. Length is floored to a whole 16-byte frame (trailing DVD
 sector padding on the last sound is dropped; the engine truncates partial frames
 anyway via bytes/VAG_FRAME)."""
    s = bank.sounds[i]
    data = bank.body[s.off: s.off + s.length]
    return data[: len(data) & ~(FRAME - 1)]

def bank_loop_frames(bank, i):
    """PS2 loop point -> sfx.bin loopFrame (VAG frame index; NO_LOOP = one-shot). On PS2
 the second SoundEntry u32 is the SPU loop address as a BYTE offset relative to the
 sound's own upload base (0xFFFFFFFF = no loop); /16 converts it to a frame index,
 matching the engine's loopFrame*VAG_SPF math and the PC encoder's block convention."""
    lp = bank.sounds[i].loop
    return NO_LOOP if lp == NO_LOOP else (lp // FRAME)

def decode_vag(vag):
    """Decode Sony PS-ADPCM (VAG) bytes -> 16-bit mono PCM (LE bytes). Byte-exact match
 of the engine's me_decode_next (the flag byte is ignored). Stdlib only - lets the
 PS2 loading-tune bake produce WAV without pulling any external decoder."""
    out = bytearray()
    h1 = h2 = 0
    for f in range(len(vag) // FRAME):
        fr = vag[f * FRAME:(f + 1) * FRAME]
        shift = fr[0] & 0x0F
        pred = (fr[0] >> 4) & 0x0F
        if pred > 4:
            pred = 4
        f0, f1 = F0[pred], F1[pred]
        for k in range(SAMPLES_PER_FRAME):
            byte = fr[2 + (k >> 1)]
            nib = (byte >> 4) if (k & 1) else (byte & 0x0F)
            s = nib << 12
            if s & 0x8000:
                s -= 0x10000
            s >>= shift
            v = _clamp16(s + ((h1 * f0 + h2 * f1) >> 6))
            out += struct.pack("<h", v)
            h2 = h1; h1 = v
    return bytes(out)

# ---- PCM(16-bit mono) -> VAG ADPCM ----------------------------------------

def _clamp16(v):
    return -32768 if v < -32768 else (32767 if v > 32767 else v)

def _enc_block(x, h1, h2):
    """Encode 28 PCM samples (ints) -> (16 bytes payload, new_h1, new_h2).
 Brute-forces 5 predictors x best shift, picks min squared error (closed loop)."""
    best = None  # (err, predictor, shift, nibbles[28], rh1, rh2)
    for p in range(5):
        f0, f1 = F0[p], F1[p]
        # open-loop residual (original-sample history) -> pick shift that fits 4-bit range
        e1, e2 = h1, h2
        maxres = 0
        for s in x:
            d = s - ((e1 * f0 + e2 * f1) >> 6)
            if d < 0: d = -d
            if d > maxres: maxres = d
            e2, e1 = e1, s
        shift = 0
        while shift < 12 and (0x8000 >> (shift + 1)) > maxres:
            shift += 1
        # closed-loop quantize using reconstructed history -> exact error
        ch1, ch2 = h1, h2
        err = 0
        nibs = []
        for s in x:
            pred = (ch1 * f0 + ch2 * f1) >> 6
            d = s - pred
            q = (d * (1 << shift)) / 4096.0
            nib = int(round(q))
            if nib < -8: nib = -8
            elif nib > 7: nib = 7
            sdec = (nib << 12) >> shift          # arithmetic, matches decoder
            rec = _clamp16(sdec + pred)
            e = s - rec
            err += e * e
            ch2, ch1 = ch1, rec
            nibs.append(nib & 0xF)
        if best is None or err < best[0]:
            best = (err, p, shift, nibs, ch1, ch2)
    _, p, shift, nibs, rh1, rh2 = best
    payload = bytearray(16)
    payload[0] = (p << 4) | shift
    payload[1] = 0
    for i in range(14):
        payload[2 + i] = (nibs[2*i] & 0xF) | ((nibs[2*i + 1] & 0xF) << 4)
    return bytes(payload), rh1, rh2

def encode_vag(pcm, loop_start_samples=NO_LOOP, prime=True):
    """PCM (bytes, 16-bit mono LE) -> VAG ADPCM bytes.

 A leading silent/priming frame (00 07 ..) is prepended (conventional, what retail VAGs
 carry) so the decoder starts from a clean h1=h2=0 block. Loop/end flags are written into
 byte1: one-shot -> last frame flag 1; looping -> loop-start frame flag 6, last frame flag 3.
 Returns (vag_bytes, loop_block_index_or_-1).
 """
    n = len(pcm) // 2
    samples = list(struct.unpack("<%dh" % n, pcm[:n*2])) if n else []
    # pad to a whole 28-sample block
    if len(samples) % SAMPLES_PER_FRAME:
        samples += [0] * (SAMPLES_PER_FRAME - len(samples) % SAMPLES_PER_FRAME)
    out = bytearray()
    if prime:
        out += bytes([0x00, 0x07] + [0] * 14)   # silent priming frame
    h1 = h2 = 0
    nblocks = len(samples) // SAMPLES_PER_FRAME
    loop_blk = -1
    if loop_start_samples != NO_LOOP:
        loop_blk = loop_start_samples // SAMPLES_PER_FRAME
    for bi in range(nblocks):
        blk = samples[bi*SAMPLES_PER_FRAME:(bi+1)*SAMPLES_PER_FRAME]
        payload, h1, h2 = _enc_block(blk, h1, h2)
        payload = bytearray(payload)
        if loop_blk >= 0:
            if bi == loop_blk:
                payload[1] = 6                    # loop start
            if bi == nblocks - 1:
                payload[1] = 3                    # loop end -> repeat to loop start
        else:
            if bi == nblocks - 1:
                payload[1] = 1                    # one-shot end (stop)
        out += payload
    # account for the prepended priming frame in the reported loop block
    if loop_blk >= 0 and prime:
        loop_blk += 1
    return bytes(out), loop_blk


if __name__ == "__main__":
    # Self-test: encode a real footstep PCM -> VAG -> decode with the PROVEN gvcslib decoder
    # -> report SNR. A correct encoder gives high SNR; a convention bug gives garbage/noise.
    import sys, math
    sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
    from gvcslib import vag as gvag

    SA = os.environ.get("SA_ROOT", "") + "/audio"
    CFG, SFX = SA + "/CONFIG", SA + "/SFX"
    bl = load_banklkup(CFG + "/BankLkup.dat")
    pk = load_pakfiles(CFG + "/PakFiles.dat")
    print("paks:", pk)
    for bank_id in (0, 39):                       # FEET_GENERIC, GENRL_COLLISIONS
        b = read_bank(CFG, SFX, bank_id, bl, pk)
        print("bank %d: %d sounds, body %d B" % (bank_id, len(b.sounds), len(b.body)))
        worst = 99
        for i in range(min(5, len(b.sounds))):
            pcm = bank_pcm(b, i)
            if len(pcm) < 64:
                continue
            vag, lb = encode_vag(pcm, b.sounds[i].loop)
            dec = gvag.decode_adpcm(vag)
            orig = list(struct.unpack("<%dh" % (len(pcm)//2), pcm[:len(pcm)//2*2]))
            rec = list(struct.unpack("<%dh" % (len(dec)//2), dec))
            # the priming frame (flag 7) is skipped by the decoder (no output) -> rec aligns directly
            rec = rec[:len(orig)]
            m = min(len(orig), len(rec))
            sig = sum(orig[k]*orig[k] for k in range(m)) or 1
            noi = sum((orig[k]-rec[k])**2 for k in range(m)) or 1
            snr = 10*math.log10(sig/noi)
            ratio = len(vag) / len(pcm)
            print("  snd%2d rate=%d len=%dB -> vag=%dB (%.2fx) SNR=%.1f dB"
                  % (i, b.sounds[i].rate, len(pcm), len(vag), ratio, snr))
            worst = min(worst, snr)
        print("  worst SNR: %.1f dB" % worst)
