#!/usr/bin/env python3
"""sa_ps2_stream - read the PS2 disc's streamed audio (radio, adverts, cutscenes).

The PC build kept these as XOR-encrypted OGG; the PS2 disc stores raw SPU ADPCM, so
none of the PC path applies here. Layout, reversed from the disc and self-validating:

 audio/CONFIG/StrmPaks.dat 16 x char[16] - packId -> <NAME>.PAK
 audio/CONFIG/TrakLkup.dat 12-byte records {u32 packId, u32 offset, u32 size}
 1922 of them, and every offset+size lands inside its
 own pack, which is what confirms the field order.

 <NAME>.PAK at `offset`:
 0x0000 ff ff ff ff, then zeros
 0x1F40 u32 x8: (bytes/32, rate/32, bytes/32, rate/32,
 bytes, rate, bytes, rate) - one descriptor per channel,
 duplicated because these streams are stereo with equal channels
 0x2000 the ADPCM itself, `bytes` per channel, interleaved in 0x2000 blocks;
 the tail up to `size` is padding

Standard SPU ADPCM: 16-byte frames, byte0 = shift | filter<<4, byte1 = flags, then
14 bytes of nibble pairs = 28 samples.

Usage:
 sa_ps2_stream.py list <audio-dir> packs and track counts
 sa_ps2_stream.py info <audio-dir> <trackId> one track's header
 sa_ps2_stream.py wav <audio-dir> <trackId> <out> decode to a 16-bit WAV
"""
import os
import struct
import sys

HDR = 0x2000
# The ADPCM frames do not start at HDR: there are four more bytes of header first.
# Measured, not guessed - a PS-ADPCM frame header is byte0 = shift | filter<<4 with
# filter <= 4 and shift <= 12, and byte1 = flags <= 7, so the frame grid announces
# itself. Testing all 16 possible alignments over 30 elements from CUTSCENE, AMBIENCE,
# CH, TK and ADVERTS gives 100% valid frame headers at +4 and about 10% at +0, at the
# start of the stream and deep inside it alike. Decoding from +0 fed the predictor a
# half-frame of garbage on every frame, which is what made the radio and the cutscene
# voice come out as noise (2.2% of samples pinned to the clamp against 0.00% for the
# known-good VAG bodies in sfx.bin).
DATA = HDR + 4
DESC = 0x1F40
FRAME = 16
SAMPLES_PER_FRAME = 28
INTERLEAVE = 0x2000

# ---- source channel layout ---------------------------------------------------
# An element repeats a fixed period: the left channel's half, the right channel's
# half, then padding. Measured, not assumed - the two halves of one period are the
# same moment in stereo and correlate 0.98 (CUTSCENE), 0.99 (radio) and 0.88
# (AMBIENCE), while halves taken from DIFFERENT periods correlate 0.009. The old
# INTERLEAVE guess of 0x2000 put both halves into each assembled channel, so every
# line of dialogue came out twice in a row.
STREAM_HALF = 0x10000            # bytes of one channel per period
STREAM_PAD = 0x1000              # zero padding that closes each period
STREAM_PERIOD = STREAM_HALF * 2 + STREAM_PAD

# Output interleave: the runtime refills 128 frames per channel at a time, so laying
# the channels out in that unit lets one sequential read serve them all. Keep this in
# step with ADPCM_WINDOW_BYTES in platform_psp/Adpcm.h.
WINDOW = 128 * FRAME


def channel_bytes(raw, ch, bytes_per_ch):
    """Assemble one channel out of an element's payload.

 `raw` starts at DATA (the first ADPCM frame). Returns exactly `bytes_per_ch`
 bytes, or fewer if the element is short.
 """
    out = bytearray()
    period = 0
    while len(out) < bytes_per_ch:
        start = period * STREAM_PERIOD + ch * STREAM_HALF
        if start >= len(raw):
            break
        take = min(STREAM_HALF, bytes_per_ch - len(out))
        chunk = raw[start:start + take]
        if not chunk:
            break
        out += chunk
        period += 1
    return bytes(out)

# SPU ADPCM predictor coefficients, scaled by 64
COEF = ((0, 0), (60, 0), (115, -52), (98, -55), (122, -60))


def load_index(audio_dir):
    cfg = os.path.join(audio_dir, "CONFIG")
    with open(os.path.join(cfg, "StrmPaks.dat"), "rb") as f:
        raw = f.read()
    packs = [raw[i:i + 16].split(b"\0")[0].decode("ascii", "replace")
             for i in range(0, len(raw), 16)]
    with open(os.path.join(cfg, "TrakLkup.dat"), "rb") as f:
        raw = f.read()
    tracks = [struct.unpack_from("<3I", raw, i * 12) for i in range(len(raw) // 12)]
    return packs, tracks


def pack_available(audio_dir, name):
    """Is this pack's .PAK actually on disk?

 The track table describes every pack the disc has, but a caller is often given only
 the one or two packs it asked for. Checking beats catching: a missing pack is normal,
 not an error.
 """
    return os.path.isfile(os.path.join(audio_dir, "STREAMS", name + ".PAK"))


def read_header(audio_dir, packs, tracks, tid):
    pid, off, size = tracks[tid]
    path = os.path.join(audio_dir, "STREAMS", packs[pid] + ".PAK")
    with open(path, "rb") as f:
        f.seek(off)
        head = f.read(HDR)
    if head[:4] != b"\xff\xff\xff\xff":
        raise ValueError("track %d: no stream header at %s+%d" % (tid, packs[pid], off))
    # The descriptor holds two (bytes, rate) pairs per channel. Radio packs fill both
    # - the first scaled down by 32, the second the real figures - while CUTSCENE
    # fills only the first and leaves the rest zero. Pick whichever pair carries a
    # believable sample rate rather than assuming a fixed slot.
    d = struct.unpack_from("<8I", head, DESC)
    nbytes = rate = 0
    for i in (4, 0):
        if 8000 <= d[i + 1] <= 48000 and d[i]:
            nbytes, rate = d[i], d[i + 1]
            break
    return dict(pack=packs[pid], path=path, offset=off, size=size,
                bytes_per_ch=nbytes, rate=rate, channels=2,
                samples=(nbytes // FRAME) * SAMPLES_PER_FRAME)


def subtitle_windows(path):
    """intro1a_subs.bin -> [(startSeconds, endSeconds)] for every line."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"CSUB":
        return []
    n = struct.unpack_from("<H", b, 4)[0]
    p, out = 6, []
    for _ in range(n):
        start, dur, ln = struct.unpack_from("<IIH", b, p)
        p += 10 + ln
        out.append((start / 1000.0, (start + dur) / 1000.0))
    return out


def _coarse_energy(path, off, bytes_per_ch, rate, seconds):
    """Per-second loudness of channel 0, straight off the ADPCM nibbles.

 A frame's amplitude is about mean|nibble| scaled by its own shift, which is all
 that is needed to tell speech from silence - and it avoids decoding 141 elements
 in full just to identify one of them.
 """
    want_frames = int(seconds * rate / SAMPLES_PER_FRAME)
    with open(path, "rb") as f:
        f.seek(off + DATA)
        raw = f.read(min(bytes_per_ch * 2, want_frames * FRAME * 2 + 2 * INTERLEAVE))
    ch0 = bytearray()
    p = 0
    while p < len(raw) and len(ch0) < want_frames * FRAME:
        ch0 += raw[p:p + INTERLEAVE]
        p += 2 * INTERLEAVE
    per_sec = max(1, int(rate / SAMPLES_PER_FRAME))
    energy, acc, cnt = [], 0.0, 0
    for fr in range(len(ch0) // FRAME):
        blk = ch0[fr * FRAME:(fr + 1) * FRAME]
        shift = blk[0] & 0x0F
        if shift > 12:
            shift = 9
        mag = 0
        for byte in blk[2:]:
            for nib in (byte & 0x0F, byte >> 4):
                mag += abs(nib - 16 if nib > 7 else nib)
        acc += (mag / 28.0) * (1 << (12 - shift)) / 4096.0
        cnt += 1
        if cnt >= per_sec:
            energy.append(acc / cnt)
            acc, cnt = 0.0, 0
    return energy


def match_by_subtitles(audio_dir, packs, tracks, pack_name, windows):
    """Which element of `pack_name` is the scene these subtitles belong to?

 Length alone does not identify a cutscene take: the CUTSCENE pack holds 141
 elements whose durations sit close together, and picking the nearest to the
 animation's length gave the wrong scene. The subtitles, however, say exactly when
 someone is speaking - so the right element is the one that is loud inside those
 windows and quiet between them. Returns (tid, score) or (None, 0).
 """
    if not windows or pack_name not in packs:
        return None, 0.0
    pid = packs.index(pack_name)
    span = max(b for _, b in windows)
    best, best_score = None, 0.0
    for tid, (p, off, size) in enumerate(tracks):
        if p != pid:
            continue
        try:
            h = read_header(audio_dir, packs, tracks, tid)
        except (ValueError, OSError):
            continue
        if h["samples"] / float(h["rate"] or 1) < span * 0.9:
            continue                      # too short to hold the whole scene
        energy = _coarse_energy(h["path"], h["offset"], h["bytes_per_ch"], h["rate"], span + 3)
        if not energy:
            continue
        ins = out = 0.0
        ni = no = 0
        for sec, e in enumerate(energy):
            t = sec + 0.5
            if any(a <= t <= b for a, b in windows):
                ins += e; ni += 1
            else:
                out += e; no += 1
        if not ni or not no:
            continue
        sc = (ins / ni) / max(1e-6, out / no)
        if sc > best_score:
            best, best_score = tid, sc
    return best, best_score


def decode_channel(data):
    """SPU ADPCM bytes -> list of int16."""
    out = []
    h1 = h2 = 0
    for p in range(0, len(data) - FRAME + 1, FRAME):
        b0 = data[p]
        shift = b0 & 0x0F
        filt = b0 >> 4
        if filt >= len(COEF):
            filt = 0
        if shift > 12:
            shift = 9                       # retail encoders emit this for silence
        c0, c1 = COEF[filt]
        for i in range(14):
            byte = data[p + 2 + i]
            for nib in (byte & 0x0F, byte >> 4):
                s = nib << 12
                if s & 0x8000:
                    s -= 0x10000
                s >>= shift
                s += (h1 * c0 + h2 * c1) >> 6
                if s > 32767:
                    s = 32767
                elif s < -32768:
                    s = -32768
                out.append(s)
                h2, h1 = h1, s
    return out


def decode(audio_dir, packs, tracks, tid):
    """-> (rate, [left, right]) as int16 lists."""
    h = read_header(audio_dir, packs, tracks, tid)
    n = h["bytes_per_ch"]
    with open(h["path"], "rb") as f:
        f.seek(h["offset"] + DATA)
        raw = f.read(h["size"])
    return h["rate"], [decode_channel(channel_bytes(raw, c, n)) for c in range(2)]


def write_wav(path, rate, chans):
    n = min(len(c) for c in chans)
    inter = bytearray()
    for i in range(n):
        for c in chans:
            v = c[i] & 0xFFFF
            inter += bytes((v & 0xFF, v >> 8))
    nch = len(chans)
    byte_rate = rate * nch * 2
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(inter)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, nch, rate, byte_rate, nch * 2, 16) +
           b"data" + struct.pack("<I", len(inter)))
    with open(path, "wb") as f:
        f.write(hdr + bytes(inter))
    return n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, audio = sys.argv[1], sys.argv[2]
    packs, tracks = load_index(audio)

    if cmd == "list":
        from collections import Counter
        per = Counter(t[0] for t in tracks)
        print("%d tracks over %d packs" % (len(tracks), len(per)))
        for pid in sorted(per):
            print("  %2d %-10s %5d" % (pid, packs[pid], per[pid]))
        return 0

    tid = int(sys.argv[3])
    h = read_header(audio, packs, tracks, tid)
    dur = h["samples"] / float(h["rate"] or 1)
    print("track %d: %s +%d, %d B, %d Hz stereo, %d samples/ch (%.1f s)"
          % (tid, h["pack"], h["offset"], h["size"], h["rate"], h["samples"], dur))
    if cmd == "info":
        return 0
    if cmd != "wav":
        print(__doc__)
        return 2

    rate, chans = decode(audio, packs, tracks, tid)
    n = write_wav(sys.argv[4], rate, chans)
    peak = max(max(abs(v) for v in c) for c in chans)
    rms = (sum(v * v for v in chans[0][:rate * 5]) / max(1, min(len(chans[0]), rate * 5))) ** 0.5
    print("wrote %s: %d frames, peak %d, rms(first 5s) %.0f" % (sys.argv[4], n, peak, rms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
