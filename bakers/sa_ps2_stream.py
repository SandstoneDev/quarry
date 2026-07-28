#!/usr/bin/env python3
"""sa_ps2_stream - read the PS2 disc's streamed audio (radio, adverts, cutscenes).

The PC build kept these as XOR-encrypted OGG; the PS2 disc stores raw SPU ADPCM, so
none of the PC path applies here. Layout, reversed from the disc and self-validating:

  audio/CONFIG/StrmPaks.dat   16 x char[16]  - packId -> <NAME>.PAK
  audio/CONFIG/TrakLkup.dat   12-byte records {u32 packId, u32 offset, u32 size}
                              1922 of them, and every offset+size lands inside its
                              own pack, which is what confirms the field order.

  <NAME>.PAK at `offset`:
      0x0000  ff ff ff ff, then zeros
      0x1F40  u32 x8: (bytes/32, rate/32, bytes/32, rate/32,
                       bytes, rate, bytes, rate)  - one descriptor per channel,
              duplicated because these streams are stereo with equal channels
      0x2000  the ADPCM itself, `bytes` per channel, interleaved in 0x2000 blocks;
              the tail up to `size` is padding

Standard SPU ADPCM: 16-byte frames, byte0 = shift | filter<<4, byte1 = flags, then
14 bytes of nibble pairs = 28 samples.

Usage:
  sa_ps2_stream.py list <audio-dir>                    packs and track counts
  sa_ps2_stream.py info <audio-dir> <trackId>          one track's header
  sa_ps2_stream.py wav  <audio-dir> <trackId> <out>    decode to a 16-bit WAV
"""
import os
import struct
import sys

HDR = 0x2000
DESC = 0x1F40
FRAME = 16
SAMPLES_PER_FRAME = 28
INTERLEAVE = 0x2000

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
        f.seek(h["offset"] + HDR)
        raw = f.read(n * 2)
    # channels alternate every INTERLEAVE bytes
    ch = [bytearray(), bytearray()]
    p = 0
    which = 0
    while p < len(raw):
        ch[which] += raw[p:p + INTERLEAVE]
        p += INTERLEAVE
        which ^= 1
    return h["rate"], [decode_channel(bytes(c)) for c in ch]


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
