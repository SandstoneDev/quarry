"""PSP VAG sound banks - the console title ``AUDIO/SETx/SFXn_PSP.RAW`` (Sony 4-bit ADPCM).

derived from the retail files ( RE library/formats/vag-audio.md):
each ``SFXn_PSP.RAW`` is a flat **bank** of standard Sony ``VAGp`` sub-files laid
end to end - there is no separate directory file; every sub-file is self-describing.

Sub-file layout (big-endian header, verified SET0/SFX5_PSP.RAW)
--------------------------------------------------------------
::

 +0x00 char[4] 'VAGp' magic
 +0x04 u32 BE version (0x00000004)
 +0x08 u32 reserved (0)
 +0x0C u32 BE data_size ADPCM byte count that follows the header
 +0x10 u32 BE sample_rate Hz (e.g. 22050, 16000)
 +0x14 12 bytes reserved (0)
 +0x20 char[16] name (e.g. "SFX_CAR_ACCEL.WA")
 +0x30 .. data_size bytes of VAG ADPCM (16-byte frames)

The next sub-file begins at ``offset + 0x30 + data_size`` (header is 0x30, NOT 0x40 -
the 16 bytes at +0x30 are the conventional VAG zero priming frame counted inside
``data_size``). The bank ends when the cursor runs out or the magic stops matching.

VAG ADPCM frame (16 bytes, canonical PSX/PS2/PSP codec)
-------------------------------------------------------
``byte0`` = ``(predictor<<4) | shift``; ``byte1`` = flag (1=end, 7=silent-start, ...);
``byte2..15`` = 14 bytes, each two 4-bit samples (low nibble first) -> 28 PCM samples.
Decode: ``s = sign4(nibble) << 12 >> shift``; then
``out = clamp16(s + (hist1*f0[p] + hist2*f1[p]) >> 6)`` with the standard predictor
coefficient tables. Mono output.

This is a **read-only decoder** (the player path): VAG ADPCM is lossy, so there is no
byte-exact re-encode. The round-trip gate that applies is the **container**:
``rebuild_bank(parse_bank(data), data) == data`` byte-for-byte (re-slicing the bank
reproduces the file), which :func:`rebuild_bank` guarantees and the test suite checks.
"""
import struct

try:  # vectorised decode for the multi-MB banks; pure-python fallback below
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

MAGIC = b"VAGp"
HEADER_SIZE = 0x30          # bytes from 'VAGp' to start of ADPCM data
FRAME = 16                  # ADPCM frame size in bytes
SAMPLES_PER_FRAME = 28

# Sony ADPCM predictor coefficients (numerator over 64).
_F0 = (0, 60, 115, 98, 122)
_F1 = (0, 0, -52, -55, -60)


class VagEntry:
    """One ``VAGp`` sub-file within a bank."""
    __slots__ = ("index", "offset", "data_size", "sample_rate", "name")

    def __init__(self, index, offset, data_size, sample_rate, name):
        self.index = index
        self.offset = offset            # byte offset of 'VAGp' in the bank
        self.data_size = data_size      # ADPCM bytes (starts at offset+0x30)
        self.sample_rate = sample_rate
        self.name = name

    @property
    def data_offset(self):
        return self.offset + HEADER_SIZE

    @property
    def total_size(self):
        return HEADER_SIZE + self.data_size

    @property
    def num_samples(self):
        return (self.data_size // FRAME) * SAMPLES_PER_FRAME

    def duration(self):
        return self.num_samples / self.sample_rate if self.sample_rate else 0.0

    def __repr__(self):
        return ("VagEntry(#%d %r @0x%x data=0x%x rate=%d)"
                % (self.index, self.name, self.offset, self.data_size, self.sample_rate))


def parse_bank(data):
    """Return ``[VagEntry, ...]`` for every contiguous ``VAGp`` sub-file in *data*.

 Walks from offset 0, stepping ``0x30 + data_size`` per entry, stopping when the
 magic no longer matches or the declared size would overrun the buffer.
 """
    data = bytes(data)
    out = []
    o = 0
    n = len(data)
    while o + HEADER_SIZE <= n and data[o:o + 4] == MAGIC:
        # version @+4, data_size @+0xC, sample_rate @+0x10 - all big-endian
        data_size = struct.unpack_from(">I", data, o + 0x0C)[0]
        rate = struct.unpack_from(">I", data, o + 0x10)[0]
        name = data[o + 0x20:o + 0x30].split(b"\x00", 1)[0].decode("latin1", "replace")
        if data_size <= 0 or o + HEADER_SIZE + data_size > n:
            # last/padding entry whose size doesn't fit: take what's left, then stop
            data_size = max(0, n - o - HEADER_SIZE)
            out.append(VagEntry(len(out), o, data_size, rate, name))
            break
        out.append(VagEntry(len(out), o, data_size, rate, name))
        o += HEADER_SIZE + data_size
    return out


def rebuild_bank(entries, data):
    """Re-slice *entries* out of the original *data* and concatenate.

 Byte-exact identity for the container: ``rebuild_bank(parse_bank(d), d) == d``
 for a well-formed bank (this is the round-trip gate for the read-only codec).
 """
    data = bytes(data)
    out = bytearray()
    for e in entries:
        out += data[e.offset:e.offset + e.total_size]
    return bytes(out)


def _clamp16(v):
    return -32768 if v < -32768 else (32767 if v > 32767 else v)


def decode_adpcm(adpcm):
    """Decode raw VAG ADPCM bytes -> mono PCM-16 ``bytes`` (little-endian)."""
    adpcm = bytes(adpcm)
    nframes = len(adpcm) // FRAME
    if _np is not None and nframes:
        return _decode_np(adpcm, nframes)
    return _decode_py(adpcm, nframes)


def _decode_py(adpcm, nframes):  # pragma: no cover - numpy path used in practice
    out = bytearray()
    hist1 = hist2 = 0
    for f in range(nframes):
        base = f * FRAME
        b0 = adpcm[base]
        shift = b0 & 0x0F
        predict = min((b0 >> 4) & 0x0F, len(_F0) - 1)
        flag = adpcm[base + 1]
        if flag == 7:           # silent / playback-stop frame
            continue
        f0, f1 = _F0[predict], _F1[predict]
        for i in range(14):
            byte = adpcm[base + 2 + i]
            for nib in (byte & 0x0F, byte >> 4):
                s = nib << 12
                if s & 0x8000:
                    s -= 0x10000
                s >>= shift
                sample = _clamp16(s + ((hist1 * f0 + hist2 * f1) >> 6))
                hist2, hist1 = hist1, sample
                out += struct.pack("<h", sample)
    return bytes(out)


def _decode_np(adpcm, nframes):
    a = _np.frombuffer(adpcm[:nframes * FRAME], dtype=_np.uint8).reshape(nframes, FRAME)
    shift = (a[:, 0] & 0x0F).astype(_np.int32)
    predict = _np.minimum((a[:, 0] >> 4) & 0x0F, len(_F0) - 1).astype(_np.int32)
    # 14 data bytes -> 28 nibbles per frame (low nibble first, then high)
    payload = a[:, 2:16].astype(_np.int32)            # (nframes, 14)
    lo = payload & 0x0F
    hi = payload >> 4
    nib = _np.empty((nframes, SAMPLES_PER_FRAME), dtype=_np.int32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    s = (nib << 12)
    s = _np.where(s & 0x8000, s - 0x10000, s)          # sign-extend the 16-bit field
    s >>= shift[:, None]                               # arithmetic >> per frame
    f0 = _np.array(_F0, dtype=_np.int32)[predict]
    f1 = _np.array(_F1, dtype=_np.int32)[predict]
    silent = (a[:, 1] == 7)
    # IIR prediction is sequential across all samples; iterate the flat stream.
    flat_s = s.reshape(-1)
    flat_f0 = _np.repeat(f0, SAMPLES_PER_FRAME)
    flat_f1 = _np.repeat(f1, SAMPLES_PER_FRAME)
    flat_silent = _np.repeat(silent, SAMPLES_PER_FRAME)
    out = _np.empty(flat_s.shape[0], dtype=_np.int16)
    h1 = h2 = 0
    for k in range(flat_s.shape[0]):
        if flat_silent[k]:
            out[k] = 0
            continue
        v = int(flat_s[k]) + ((h1 * int(flat_f0[k]) + h2 * int(flat_f1[k])) >> 6)
        v = _clamp16(v)
        out[k] = v
        h2, h1 = h1, v
    if flat_silent.any():
        out = out[~flat_silent]
    return out.tobytes()


def to_wav(pcm, sample_rate, channels=1):
    """Wrap mono PCM-16 *pcm* bytes in a minimal 44-byte WAV (RIFF) container."""
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + len(pcm)), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                             byte_rate, block_align, bits),
        b"data", struct.pack("<I", len(pcm)), pcm,
    ))


def decode_entry_to_wav(data, entry):
    """Decode one :class:`VagEntry` from bank *data* straight to WAV bytes."""
    adpcm = bytes(data)[entry.data_offset:entry.data_offset + entry.data_size]
    return to_wav(decode_adpcm(adpcm), entry.sample_rate)
