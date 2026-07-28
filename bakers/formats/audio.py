"""the source game audio decoder - SFX banks, ADF radio streams, RAW/WAV/MP3/OGG.

Three on-disk families (PC v1.0):
 * AUDIO/SFX/<name> flat concatenation of fixed 0x12C4-header SFX banks
 (u16 NumSounds + 400 x 12B SoundEntry) then raw 16-bit
 signed mono PCM. Banks located via BANKLKUP.DAT, or
 self-discovered by the chaining invariant
 FileOffset + NumBytes + 0x12C4 == next bank.
 * AUDIO/STREAMS/<n> ADF-obfuscated Ogg Vorbis: whole file XORed by a 16-byte
 rolling key indexed by absolute offset; each track's
 'OggS' sits at TrackOffset + 0x1F84 after de-XOR.
 * AUDIO/CONFIG/*.DAT fixed-record index tables (BANKLKUP/TRAKLKUP/PAKFILES/
 STRMPAKS) that locate banks/tracks inside the paks.

RAW/WAV/MP3/OGG are standard containers; we probe + passthrough them via ffmpeg.

 (confirmed; FEET/AA worked examples)
References: , .
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

# ---- fixed layout constants (spec section A/B) -----------------------------
BANK_HEADER_SIZE = 0x12C4          # 4804: u16 NumSounds + u16 pad + 400 * 12
SOUND_ENTRY_SIZE = 12
MAX_SOUNDS = 400
ADF_OGG_OFFSET = 0x1F84            # OGG starts this far into each track region
NO_LOOP = 0xFFFFFFFF

# ADF 16-byte rolling XOR key: KEY[i] = seed[i] ^ i (built once at startup,
#).
ADF_SEED = bytes([0xEA, 0x3B, 0xC6, 0xA2, 0x9E, 0xAD, 0x12, 0xF4,
                  0x40, 0xB9, 0xDD, 0x28, 0x91, 0xE5, 0xF1, 0xFE])
ADF_KEY = bytes(ADF_SEED[i] ^ i for i in range(16))

# tool paths (absolute; the prompt's TOOLS/ffmpeg dir does not exist on disk -
# the binaries live at the repo root /ffmpeg/).
_FFMPEG = ""
_FFPROBE = ""
_SCRATCH = os.path.join(tempfile.gettempdir(), "quarry_audio")

_SOUND_ENTRY = struct.Struct("<IIHh")   # BankOffset, LoopStart, SampleFreq, Headroom
_LKUP_REC = struct.Struct("<BxxxII")    # PakFileNo/StreamPakIndex, pad, off, len


# ==========================================================================
# SFX bank parse
# ==========================================================================

@dataclass
class SfxSound:
    """One SoundEntry + the byte-range of its PCM within the bank data region."""
    index: int
    offset: int                 # BankOffsetBytes (relative to data start)
    size: int                   # length of this sound's PCM in bytes
    sample_rate: int            # SampleFrequency Hz
    loop: Optional[int]         # LoopStartOffset in samples, or None (0xFFFFFFFF)
    headroom: int               # raw i16
    headroom_db: float          # headroom / 100.0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "offset": self.offset,
            "size": self.size,
            "sample_rate": self.sample_rate,
            "loop": self.loop,
            "headroom": self.headroom,
            "headroom_db": self.headroom_db,
        }


def parse_sfx_bank(data: bytes, num_bytes: Optional[int] = None) -> dict:
    """Parse one SFX bank (0x12C4 header + PCM data region).

 `data` must begin at the bank's BankHeader. `num_bytes` is the data-region
 size (from BANKLKUP.NumBytes); when omitted it is inferred from the payload
 length (len(data) - header). Returns {sounds: [{offset,size,sample_rate,
 loop,...}], num_sounds, header_size, num_bytes}.

 Per-sound PCM length = next sound's BankOffset - this one; the last sound
 runs to num_bytes. Bad/out-of-range entries are clamped defensively so one
 corrupt record never kills the bank.
 """
    if len(data) < 4:
        return {"sounds": [], "num_sounds": 0, "header_size": BANK_HEADER_SIZE,
                "num_bytes": 0}

    raw_count = struct.unpack_from("<h", data, 0)[0]
    if raw_count == -1:                 # 0xFFFF: single-sound bank
        count = 1
    elif raw_count < 0:
        count = 0
    else:
        count = min(raw_count, MAX_SOUNDS)

    if num_bytes is None:
        num_bytes = max(len(data) - BANK_HEADER_SIZE, 0)

    # read the directory entries (BankOffset / loop / rate / headroom)
    raw: List[tuple] = []
    for i in range(count):
        eo = 4 + i * SOUND_ENTRY_SIZE
        if eo + SOUND_ENTRY_SIZE > len(data):
            break
        try:
            boff, loop, freq, head = _SOUND_ENTRY.unpack_from(data, eo)
        except struct.error:
            break
        raw.append((boff, loop, freq, head))

    # bound the last sound: prefer num_bytes, else the directory's next slot,
    # else the payload length.
    next_after_last = num_bytes
    if count < MAX_SOUNDS:
        term_eo = 4 + count * SOUND_ENTRY_SIZE
        if term_eo + 4 <= len(data):
            term = struct.unpack_from("<I", data, term_eo)[0]
            if 0 < term <= max(num_bytes, len(data)):
                next_after_last = term

    sounds: List[SfxSound] = []
    for i, (boff, loop, freq, head) in enumerate(raw):
        nxt = raw[i + 1][0] if i + 1 < len(raw) else next_after_last
        # defensive: clamp non-monotonic / out-of-range boundaries
        start = boff if 0 <= boff <= num_bytes else 0
        end = nxt if start <= nxt <= num_bytes else num_bytes
        size = max(end - start, 0)
        sounds.append(SfxSound(
            index=i,
            offset=start,
            size=size,
            sample_rate=freq,
            loop=(None if loop == NO_LOOP else loop),
            headroom=head,
            headroom_db=head / 100.0,
        ))

    return {
        "sounds": [s.as_dict() for s in sounds],
        "num_sounds": len(sounds),
        "header_size": BANK_HEADER_SIZE,
        "num_bytes": num_bytes,
    }


def sfx_sound_pcm(data: bytes, sound: dict) -> bytes:
    """Slice one sound's raw int16 mono PCM out of a bank's bytes."""
    start = BANK_HEADER_SIZE + sound["offset"]
    end = start + sound["size"]
    return data[start:end]


def parse_sfx_pak(data: bytes, banklkup: Optional[List[dict]] = None) -> List[dict]:
    """Split an SFX pak into banks.

 With `banklkup` (records {file_offset, num_bytes} pre-filtered to this pak,
 in pak order) the bank boundaries are authoritative. Without it, banks are
 self-discovered by chaining: nextStart = curStart + 0x12C4 + dataLen, where
 dataLen is derived from the bank's own directory (last sound's terminator).

 Returns a list of bank dicts (each as parse_sfx_bank output) plus a
 'file_offset' key locating the bank in the pak.
 """
    banks: List[dict] = []

    if banklkup:
        for rec in banklkup:
            off = rec["file_offset"]
            nb = rec["num_bytes"]
            slice_ = data[off:off + BANK_HEADER_SIZE + nb]
            bank = parse_sfx_bank(slice_, num_bytes=nb)
            bank["file_offset"] = off
            banks.append(bank)
        return banks

    # self-discovery via chaining. Retail directories DO NOT store the total
    # data size (the Sounds[NumSounds] terminator slot is 0), so the final
    # sound's PCM length is unknowable from the bank alone. We therefore bound
    # each bank's data region by scanning forward for the NEXT valid bank-header
    # signature; the gap between this header and that one is BANK_HEADER_SIZE +
    # this bank's data size. (Authoritative boundaries still come from BANKLKUP;
    # this is the best-effort fallback the spec describes.)
    n = len(data)
    pos = 0
    while pos + BANK_HEADER_SIZE <= n:
        last_off = _bank_min_data_len(data, pos)
        if last_off is None:
            break
        # earliest possible end of this bank's data region
        search_from = pos + BANK_HEADER_SIZE + last_off
        nxt = _next_bank_header(data, search_from)
        if nxt is None:
            data_len = n - (pos + BANK_HEADER_SIZE)     # last bank runs to EOF
            nxt = n
        else:
            data_len = nxt - (pos + BANK_HEADER_SIZE)
        if data_len < 0:
            break
        slice_ = data[pos:pos + BANK_HEADER_SIZE + data_len]
        bank = parse_sfx_bank(slice_, num_bytes=data_len)
        bank["file_offset"] = pos
        banks.append(bank)
        if nxt <= pos:                  # no forward progress -> stop
            break
        pos = nxt
    return banks


def _bank_min_data_len(data: bytes, pos: int) -> Optional[int]:
    """Lower bound on a bank's data size: the last sound's BankOffset.

 The real last-sound PCM extends past this, but it gives a safe floor to
 start scanning for the next bank header from.
 """
    if pos + 4 > len(data):
        return None
    raw_count = struct.unpack_from("<h", data, pos)[0]
    count = 1 if raw_count == -1 else (raw_count if raw_count >= 0 else 0)
    count = min(count, MAX_SOUNDS)
    if count == 0:
        return 0
    last_eo = pos + 4 + (count - 1) * SOUND_ENTRY_SIZE
    if last_eo + 4 > len(data):
        return None
    return struct.unpack_from("<I", data, last_eo)[0]


def _looks_like_bank_header(data: bytes, pos: int) -> bool:
    """Header signature test: NumSounds in 1..400 (or -1) AND Sounds[0]==0.

 Every bank's first sound starts at BankOffset 0, and the directory is sized
 for the declared count - a strong, cheap signature for chaining.
 """
    if pos + BANK_HEADER_SIZE > len(data):
        return False
    raw_count = struct.unpack_from("<h", data, pos)[0]
    if raw_count != -1 and not (1 <= raw_count <= MAX_SOUNDS):
        return False
    # Sounds[0].BankOffset must be 0
    if struct.unpack_from("<I", data, pos + 4)[0] != 0:
        return False
    count = 1 if raw_count == -1 else raw_count
    # all real entries must have a plausible sample rate and monotonic offsets
    prev = -1
    for i in range(count):
        eo = pos + 4 + i * SOUND_ENTRY_SIZE
        boff, _loop, freq, _head = _SOUND_ENTRY.unpack_from(data, eo)
        if boff < prev:                 # offsets must be non-decreasing
            return False
        if not (1000 <= freq <= 96000):  # SA rates sit well inside this band
            return False
        prev = boff
    return True


def _next_bank_header(data: bytes, start: int, limit: int = 1 << 24) -> Optional[int]:
    """Find the next offset >= start whose 4-byte alignment passes the header test.

 Banks abut on small alignment; we step in 2-byte units (PCM is int16) up to
 `limit` bytes ahead. Returns the offset or None if no header found / EOF.
 """
    n = len(data)
    pos = start
    end = min(n, start + limit)
    while pos + BANK_HEADER_SIZE <= n and pos <= end:
        if _looks_like_bank_header(data, pos):
            return pos
        pos += 2
    return None


# ==========================================================================
# ADF stream de-obfuscation
# ==========================================================================

def deobfuscate_adf(data: bytes) -> bytes:
    """De-XOR a whole ADF stream pak: plain[j] = cipher[j] ^ KEY[j & 0xF].

 The XOR index is the absolute file offset (never resets per track), so a
 single full-file pass deobfuscates everything. Re-applying with the same key
 is an involution (round-trips back to the cipher). After this, each track's
 'OggS' lives at TrackOffset + 0x1F84.
 """
    out = bytearray(data)
    key = ADF_KEY
    for j in range(len(out)):
        out[j] ^= key[j & 0x0F]
    return bytes(out)


def extract_adf_track(data: bytes, track_offset: int, track_length: int) -> bytes:
    """Recover one track's clean OGG bytes from an obfuscated stream pak.

 De-XORs only the needed span (track_offset .. track_offset+track_length),
 then returns bytes [track_offset+0x1F84 : track_offset+track_length] - a
 standard Ogg Vorbis file beginning with 'OggS'.
 """
    end = min(track_offset + track_length, len(data))
    start = max(track_offset, 0)
    key = ADF_KEY
    span = bytearray(data[start:end])
    base = start
    for i in range(len(span)):
        span[i] ^= key[(base + i) & 0x0F]
    ogg_start = ADF_OGG_OFFSET
    return bytes(span[ogg_start:])


def find_ogg_offsets(plain: bytes) -> List[int]:
    """Scan already-de-XORed bytes for 'OggS' page starts (track self-discovery).

 Each track region begins 0x1F84 before its first OggS; returns the absolute
 offsets of OggS pages that look like the start of a logical bitstream
 (page type byte 0x02 = beginning-of-stream at +5). Deduplicated, sorted.
 """
    offs: List[int] = []
    pos = plain.find(b"OggS")
    while pos != -1:
        # OggS header_type byte at +5; 0x02 = first page of a logical stream
        if pos + 6 <= len(plain) and (plain[pos + 5] & 0x02):
            offs.append(pos)
        pos = plain.find(b"OggS", pos + 4)
    return offs


# ==========================================================================
# CONFIG index parse (BANKLKUP / TRAKLKUP / PAKFILES / STRMPAKS)
# ==========================================================================

def parse_banklkup(data: bytes) -> List[dict]:
    """BANKLKUP.DAT -> [{pak, file_offset, num_bytes}] (record 0x0C)."""
    out: List[dict] = []
    for pak, off, nb in _LKUP_REC.iter_unpack(data[:len(data) - len(data) % 12]):
        out.append({"pak": pak, "file_offset": off, "num_bytes": nb})
    return out


def parse_traklkup(data: bytes) -> List[dict]:
    """TRAKLKUP.DAT -> [{stream_pak, track_offset, track_length}] (record 0x0C)."""
    out: List[dict] = []
    for sp, off, ln in _LKUP_REC.iter_unpack(data[:len(data) - len(data) % 12]):
        out.append({"stream_pak": sp, "track_offset": off, "track_length": ln})
    return out


def parse_pakfiles(data: bytes) -> List[str]:
    """PAKFILES.DAT -> [BaseFilename] (record 0x34 = name[12] + 10 zero LSNs)."""
    rec = 0x34
    out: List[str] = []
    for i in range(len(data) // rec):
        blk = data[i * rec:i * rec + 12]
        out.append(blk.split(b"\x00", 1)[0].decode("latin-1"))
    return out


def parse_strmpaks(data: bytes) -> List[str]:
    """STRMPAKS.DAT -> [Name] (record 0x10 = NUL-padded name[16])."""
    rec = 0x10
    out: List[str] = []
    for i in range(len(data) // rec):
        blk = data[i * rec:i * rec + rec]
        name = blk.split(b"\x00", 1)[0].decode("latin-1")
        if name:                        # skip blank slots in the table
            out.append(name)
    return out


# ==========================================================================
# WAV wrap / PCM helpers (no external tool)
# ==========================================================================

def pcm_to_wav(pcm: bytes, sample_rate: int = 22050, channels: int = 1,
               bits: int = 16) -> bytes:
    """Wrap raw PCM in a minimal RIFF/WAVE container (PCM, fmt + data chunks)."""
    block_align = channels * (bits // 8)
    byte_rate = sample_rate * block_align
    data_size = len(pcm)
    riff_size = 36 + data_size
    hdr = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
    hdr += b"data" + struct.pack("<I", data_size)
    return hdr + pcm


def extract_sfx_sound_wav(bank_data: bytes, sound_index: int,
                          num_bytes: Optional[int] = None) -> bytes:
    """Decode one SFX sound to a playable WAV (16-bit mono PCM at its rate)."""
    bank = parse_sfx_bank(bank_data, num_bytes=num_bytes)
    if sound_index < 0 or sound_index >= len(bank["sounds"]):
        raise IndexError(f"sound {sound_index} out of range (have {len(bank['sounds'])})")
    s = bank["sounds"][sound_index]
    pcm = sfx_sound_pcm(bank_data, s)
    return pcm_to_wav(pcm, sample_rate=s["sample_rate"], channels=1, bits=16)


# ==========================================================================
# format dispatch / sniffing
# ==========================================================================

def sniff_format(data: bytes, ext: str = "") -> str:
    """Identify an audio payload by magic (+ optional extension hint).

 Returns one of 'ogg', 'wav', 'mp3', 'raw', 'adf', 'unknown'.
 """
    ext = (ext or "").lower().lstrip(".")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    if ext in ("ogg", "wav", "mp3", "raw", "adf"):
        return ext
    return "unknown"


# ==========================================================================
# ffprobe / ffmpeg shell-out
# ==========================================================================

def _to_temp(src, suffix: str) -> tuple:
    """Materialize bytes-or-path to a real temp file in the scratchpad.

 Returns (path, created) where created=True means we own (and must delete) it.
 """
    if isinstance(src, (bytes, bytearray)):
        os.makedirs(_SCRATCH, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=suffix, dir=_SCRATCH)
        with os.fdopen(fd, "wb") as f:
            f.write(src)
        return path, True
    return str(src), False


def probe_audio(src) -> dict:
    """ffprobe a file path or bytes -> normalized {ok, codec, sample_rate,
 channels, duration, format, raw}.

 `src` may be a filesystem path or raw bytes (written to a scratch temp).
 Always JSON-serializable; on failure returns {ok: False, error: ...}.
 """
    if not os.path.exists(_FFPROBE):
        return {"ok": False, "error": "ffprobe not found"}
    path, created = _to_temp(src, ".bin")
    try:
        proc = subprocess.run(
            [_FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"ok": False, "error": "ffprobe failed",
                    "stderr": (proc.stderr or "")[:500]}
        info = json.loads(proc.stdout)
        astream = None
        for st in info.get("streams", []):
            if st.get("codec_type") == "audio":
                astream = st
                break
        if astream is None and info.get("streams"):
            astream = info["streams"][0]
        astream = astream or {}
        fmt = info.get("format", {})
        try:
            dur = float(fmt.get("duration", astream.get("duration", 0.0)) or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        try:
            rate = int(astream.get("sample_rate", 0) or 0)
        except (TypeError, ValueError):
            rate = 0
        return {
            "ok": True,
            "codec": astream.get("codec_name"),
            "sample_rate": rate,
            "channels": int(astream.get("channels", 0) or 0),
            "duration": dur,
            "format": fmt.get("format_name"),
            "bit_rate": int(fmt.get("bit_rate", 0) or 0),
        }
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        return {"ok": False, "error": str(e)}
    finally:
        if created:
            try:
                os.unlink(path)
            except OSError:
                pass


def transcode_to_wav(src, fmt: str, sample_rate: Optional[int] = None,
                     channels: int = 1) -> bytes:
    """Transcode an audio payload to WAV (PCM 16-bit) bytes via ffmpeg.

 `src` is bytes or a path; `fmt` is the source kind ('ogg', 'wav', 'mp3',
 'raw'). For 'raw' (headerless SFX PCM) you MUST pass sample_rate (the
 SoundEntry frequency) - ffmpeg is told s16le/mono. WAV/MP3/OGG are decoded
 normally. Returns the WAV file bytes.
 """
    if not os.path.exists(_FFMPEG):
        raise RuntimeError("ffmpeg not found")
    fmt = (fmt or "").lower().lstrip(".")
    suffix = "." + (fmt if fmt in ("ogg", "wav", "mp3", "raw") else "bin")
    in_path, created = _to_temp(src, suffix)
    os.makedirs(_SCRATCH, exist_ok=True)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=_SCRATCH)
    os.close(out_fd)
    try:
        cmd = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
        if fmt == "raw":
            if not sample_rate:
                raise ValueError("raw PCM requires sample_rate")
            cmd += ["-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels)]
        cmd += ["-i", in_path]
        # always emit canonical PCM 16-bit WAV
        cmd += ["-acodec", "pcm_s16le", "-f", "wav", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {(proc.stderr or '')[:500]}")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if created:
            try:
                os.unlink(in_path)
            except OSError:
                pass
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ==========================================================================
# to_json: JSON-serializable view for the web 'audio' UI
# ==========================================================================

def to_json(parsed) -> dict:
    """Project a parsed-audio result into a JSON-serializable 'audio' view.

 Accepts a bank dict (parse_sfx_bank), a list of banks (parse_sfx_pak), or a
 probe dict (probe_audio). Never embeds PCM/OGG byte blobs - only metadata.
 """
    # list of banks -> a pak view
    if isinstance(parsed, list):
        banks = [to_json(b) for b in parsed]
        return {
            "kind": "sfx_pak",
            "num_banks": len(banks),
            "total_sounds": sum(b.get("num_sounds", 0) for b in banks),
            "banks": banks,
        }

    if isinstance(parsed, dict) and "sounds" in parsed:
        sounds = []
        for s in parsed["sounds"]:
            sounds.append({
                "index": int(s["index"]),
                "offset": int(s["offset"]),
                "size": int(s["size"]),
                "sample_rate": int(s["sample_rate"]),
                "loop": (None if s["loop"] is None else int(s["loop"])),
                "headroom": int(s["headroom"]),
                "headroom_db": float(s["headroom_db"]),
                "duration": (s["size"] / 2 / s["sample_rate"]
                             if s["sample_rate"] else 0.0),
            })
        view = {
            "kind": "sfx_bank",
            "num_sounds": int(parsed.get("num_sounds", len(sounds))),
            "header_size": int(parsed.get("header_size", BANK_HEADER_SIZE)),
            "num_bytes": int(parsed.get("num_bytes", 0)),
            "sounds": sounds,
        }
        if "file_offset" in parsed:
            view["file_offset"] = int(parsed["file_offset"])
        return view

    # probe / generic dict: shallow-coerce to JSON-native scalars
    if isinstance(parsed, dict):
        out = {"kind": "audio_probe"}
        for k, v in parsed.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    return {"kind": "unknown", "value": str(parsed)}
