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

# ⚠ 0x1F84, not 0x2000 - and there is no "+4".
#
# The +4 came from a frame-alignment probe: a PS-ADPCM frame header announces itself
# (byte0 = shift | filter<<4 with filter <= 4 and shift <= 12, byte1 = flags <= 7), and
# testing all 16 alignments gave 100% valid headers at 0x2004. That probe was RIGHT
# about the frame grid and blind to the base, because 0x2004 == 0x1F84 (mod 16): it
# cannot tell a correct base from one 0x80 past it.
#
# Ground truth is on the IOP, not the EE - which is why scanning SLES_525.41 for these
# constants finds nothing. system/IOPAudio.irx copies the header with an explicit length
# of 8068 = 0x1F84 (.text 0x556C) and keeps three such buffers exactly 0x1F84 apart at
# 0xA5B0 / 0xC534 / 0xE4B8. This project's own PC baker has had it right the whole time:
# tools/radio_bake.py:23, HDR = 0x1F84, sizeof(tTrackInfo). The PS2 branch re-derived it
# independently and landed 0x80 off.
HDR = 0x1F84
DATA = HDR
DESC = 0x1F40
NCHAN = 0x1F80                   # u16: how many sub-streams this element carries
FRAME = 16
SAMPLES_PER_FRAME = 28

# ---- source channel layout ---------------------------------------------------
# A period holds ONE block per sub-stream, laid end to end in channel order, each sized
# in proportion to that sub-stream's rate. IOPAudio.irx.text 0x4B50:
#
# block_i = PERIOD * (rate_i / 75) / 660
# ch[i].offset = sum of block_0.. block_{i-1}
#
# PERIOD is 0x21000 - half the 0x42000 stream buffer (.text 0x38DC, halved at 0x5084),
# and the element length is rounded to whole periods by the same code (.text 0x4C20
# divides by 0x21000 and multiplies back).
#
# ★ Radio elements carry FOUR sub-streams, not two: 750, 750, 24000, 24000 Hz. The
# blocks are 0x800, 0x800, 0x10000, 0x10000 and they fill the period EXACTLY
# (320+320+10+10 = 660 rate units). So the stereo audio starts at +0x1000, and the
# 0x1000 an earlier model called "padding that closes each period" is in fact those two
# 750 Hz channels sitting at the FRONT of it.
#
# Reading the audio at +0 instead of +0x1000 put 0xF80 bytes of 750 Hz material into
# every chunk of the left channel, played at 24 kHz - 289 ms sped up 32x, once every
# 0x10000 bytes = 4.78 s. That is the periodic screech; the right channel's matching
# 4.49 s backward jump is the "rewind". It was never a last-period defect: the offset
# is wrong in EVERY period, and the zeros were merely the only part visible to a scan.
#
# CUTSCENE / AMBIENCE / BEATS elements carry only the two 24 kHz channels. 660 is
# hard-coded, so 320+320 leaves 0x1000 of genuine slack at the END of their periods --
# which is how the old "half, half, padding" reading came to look right for them.
STREAM_PERIOD = 0x21000
RATE_UNIT = 75                   # the divisor IOPAudio applies to each rate
RATE_TOTAL = 660                 #...and the denominator it divides the period by


def block_layout(pairs):
    """[(bytes, rate)] -> ([block size per channel], [block offset in the period]).

 Straight from the loop above; the assert is the property that makes the model
 checkable, not decoration: the blocks must fit inside one period.
 """
    sizes = [STREAM_PERIOD * (rate // RATE_UNIT) // RATE_TOTAL for _, rate in pairs]
    offs, acc = [], 0
    for sz in sizes:
        offs.append(acc)
        acc += sz
    if acc > STREAM_PERIOD:
        raise ValueError("blocks total 0x%X, larger than the 0x%X period" % (acc, STREAM_PERIOD))
    return sizes, offs


def channel_bytes(raw, blk_off, blk_size, bytes_per_ch):
    """Assemble one sub-stream out of an element's payload.

 `raw` starts at DATA. `blk_off`/`blk_size` come from block_layout - a channel is
 not "the n-th half of the period", it is its own block at its own offset.
 """
    out = bytearray()
    period = 0
    while len(out) < bytes_per_ch:
        start = period * STREAM_PERIOD + blk_off
        if start >= len(raw):
            break
        take = min(blk_size, bytes_per_ch - len(out))
        chunk = raw[start:start + take]
        if not chunk:
            break
        out += chunk
        period += 1
    return bytes(out)


# Output interleave: the runtime refills 128 frames per channel at a time, so laying
# the channels out in that unit lets one sequential read serve them all. Keep this in
# step with ADPCM_WINDOW_BYTES in platform_psp/Adpcm.h.
WINDOW = 128 * FRAME


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
    # One (bytes, rate) descriptor per SUB-STREAM, and nchan says how many there are.
    # A radio element has four: two at 750 Hz whose byte counts are the audio's / 32,
    # then the 24 kHz stereo pair. An earlier reading called those first two "the same
    # figures scaled by 32" and skipped past them to the pair with a believable rate --
    # right about which pair carries the audio, wrong about what the others were, and
    # blind to the fact that they occupy the front of every period.
    nchan = struct.unpack_from("<H", head, NCHAN)[0]
    if not 1 <= nchan <= 8:
        raise ValueError("track %d: %d sub-streams claimed, expected 1..8" % (tid, nchan))
    pairs = [struct.unpack_from("<2I", head, DESC + i * 8) for i in range(nchan)]
    sizes, offs = block_layout(pairs)

    # The audio is the fastest sub-stream. Radio: indices 2 and 3. Cutscene/ambience:
    # 0 and 1, there being nothing else.
    top = max(r for _b, r in pairs)
    audio = [i for i, (_b, r) in enumerate(pairs) if r == top]
    if not 8000 <= top <= 48000 or not audio:
        raise ValueError("track %d: no sub-stream at a playable rate (rates %s)"
                         % (tid, [r for _b, r in pairs]))
    nbytes = pairs[audio[0]][0]
    return dict(pack=packs[pid], path=path, offset=off, size=size,
                bytes_per_ch=nbytes, rate=top, channels=len(audio),
                nchan=nchan, pairs=pairs,
                blk_off=[offs[i] for i in audio], blk_size=[sizes[i] for i in audio],
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


def _coarse_energy(path, off, bytes_per_ch, rate, seconds, blk_off, blk_size):
    """Per-second loudness of channel 0, straight off the ADPCM nibbles.

 A frame's amplitude is about mean|nibble| scaled by its own shift, which is all
 that is needed to tell speech from silence - and it avoids decoding 141 elements
 in full just to identify one of them.
 """
    want_frames = int(seconds * rate / SAMPLES_PER_FRAME)
    want = want_frames * FRAME
    # Was hand-walking a 0x2000 interleave that does not exist in this format. Go
    # through the real block layout instead, reading whole periods so a block is never
    # cut in half.
    periods = (want + blk_size - 1) // blk_size + 1
    with open(path, "rb") as f:
        f.seek(off + DATA)
        raw = f.read(periods * STREAM_PERIOD)
    ch0 = channel_bytes(raw, blk_off, blk_size, min(want, bytes_per_ch))
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
        energy = _coarse_energy(h["path"], h["offset"], h["bytes_per_ch"], h["rate"], span + 3,
                               h["blk_off"][0], h["blk_size"][0])
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
    return h["rate"], [decode_channel(channel_bytes(raw, h["blk_off"][c], h["blk_size"][c], n))
                       for c in range(len(h["blk_off"]))]


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
