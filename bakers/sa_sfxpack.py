#!/usr/bin/env python3
"""v2 SFX pack format: sfx_index.bin + sfx_res.bin + sfx_banks.bin.

Pure format layer - no disc access, no deploy policy. audio_bake.py decides WHICH
banks are resident; this module decides only HOW they are laid out, so the layout
can be tested without the game archive.

Sound offsets are relative to their own bank body. A resident body never moves, but
a cell-loaded body lands wherever the arena cell is, and a file-absolute offset
would be meaningless there.
"""
import struct

MAGIC = 0x32584653          # 'SFX2'
VERSION = 4                  # v3 appended the extras block, v4 the surface classes
                             # (sa_audcurve). ⚠ BUMP THIS WHENEVER THE EXTRAS BLOCK
                             # GROWS: the engine keys its parse off this field, and a
                             # stale number ships a file it refuses - 'ARENA NOT
                             # READY', no SFX at all, which is what b827 shipped.
NO_LOOP = 0xFFFFFFFF

WHERE_RESIDENT = 0           # BankRec.where: body lives in sfx_res.bin
WHERE_CELL = 1               # BankRec.where: body lives in sfx_banks.bin, cell-loadable

HDR_FMT = "<8I"
BANK_FMT = "<HBBIIHH"
SOUND_FMT = "<IIIIhH"

HDR = struct.calcsize(HDR_FMT)
BANK_REC = struct.calcsize(BANK_FMT)
SOUND_REC = struct.calcsize(SOUND_FMT)

RESIDENT_LIMIT = 1312 * 1024   # keep in sync with SFX_CELL_RESIDENT's capacity in
                               # src/platform_psp/SfxArena.h. Measured, not guessed:
                               # the resident set is ~1075 KB once every dummy engine
                               # bank joins it, so 1024 KB left no margin at all. The
                               # extra 256 KB comes out of SLOT_SPARE, which has no
                               # consumer in P1; the arena total stays 2048 KB.
                               #
                               # b97x: +32 KB again (1280 -> 1312), same reason, same
                               # source. Radio.c's retune static bed adds bank 59
                               # sounds 1+2 (~24.4 KB) and pushed the measured set to
                               # 1300.2 KB - this project's own audit (running THIS
                               # tool against the disc, not a guess) caught it as
                               # ResidentTooBig below. Took the 32 from
                               # SFX_CELL_SPARE (src/platform_psp/SfxArena.c), which
                               # still has no consumer; the arena total is unchanged,
                               # so pmap.c's world-cache budget is untouched.
CELL_ALIGN = 64              # a cell body starts on a D-cache line (ME flush by range)
RES_ALIGN = 16               # VAG frame


class ResidentTooBig(Exception):
    pass


class DuplicateBankId(Exception):
    pass


class FieldOutOfRange(Exception):
    pass


class Sound(object):
    __slots__ = ("vag", "rate", "loop_frame", "headroom", "off", "bytes")

    def __init__(self, vag, rate, loop_frame=NO_LOOP, headroom=0, off=0, bytes=0):
        self.vag = vag
        self.rate = rate
        self.loop_frame = loop_frame
        self.headroom = headroom
        self.off = off
        self.bytes = bytes


class Bank(object):
    __slots__ = ("bank_id", "resident", "sounds", "where", "data_off", "data_bytes",
                 "first_sound", "num_sounds")

    def __init__(self, bank_id, resident, sounds, where=WHERE_RESIDENT, data_off=0,
                 data_bytes=0, first_sound=0, num_sounds=0):
        self.bank_id = bank_id
        self.resident = resident
        self.sounds = sounds
        self.where = where
        self.data_off = data_off
        self.data_bytes = data_bytes
        self.first_sound = first_sound
        self.num_sounds = num_sounds


class Parsed(object):
    __slots__ = ("banks", "sounds", "resident_bytes", "banks_bytes", "rate_max", "extras")


def _pad(buf, align):
    rem = len(buf) % align
    if rem:
        buf += b"\x00" * (align - rem)
    return buf


def _body(bank_id, sounds):
    """Concatenate a bank's sounds into one relocatable body and stamp the
 bank-relative offsets.

 Validates the two per-sound fields that would otherwise fail silently or
 opaquely: the VAG payload must already be a RES_ALIGN multiple (the mixer
 computes frame count as bytes / RES_ALIGN, so a short tail would drop the
 end of the sound instead of failing at bake time), and headroom must fit
 the wire format's i16.
 """
    body = bytearray()
    for i, s in enumerate(sounds):
        if len(s.vag) % RES_ALIGN:
            raise ValueError("bank %d sound %d: VAG payload is %d bytes, not a "
                             "multiple of %d" % (bank_id, i, len(s.vag), RES_ALIGN))
        if not (-32768 <= s.headroom <= 32767):
            raise FieldOutOfRange("bank %d sound %d: headroom=%d does not fit an "
                                  "i16 (-32768..32767)" % (bank_id, i, s.headroom))
        body = _pad(body, RES_ALIGN)
        s.off = len(body)
        s.bytes = len(s.vag)
        body += s.vag
    return bytes(_pad(body, RES_ALIGN))


def build(banks, extras=b""):
    """-> (index_bytes, resident_bytes, cell_bytes)

 Mutates every Bank and Sound in `banks` in place to record the layout chosen
 for it (where/data_off/data_bytes/first_sound/num_sounds on the Bank,
 off/bytes on each Sound).

 `extras` is the opaque trailing block sa_audcurve.pack_extras() produces - the distance attenuation curve and the event volume table. It is opaque HERE
 on purpose: this module owns where things sit in the file, sa_audcurve owns
 what those two tables mean, and neither has to know the other's business.
 """
    res = bytearray()
    cells = bytearray()
    bank_recs = []
    sound_recs = []
    rate_max = 0
    seen_ids = set()

    for b in banks:
        if b.bank_id in seen_ids:
            raise DuplicateBankId("bank id %d appears more than once - every "
                                  "id-keyed lookup would silently drop one bank's "
                                  "sounds" % b.bank_id)
        seen_ids.add(b.bank_id)
        if not (0 <= b.bank_id <= 0xFFFF):
            raise FieldOutOfRange("bank_id %d does not fit in a u16 (0..65535)" % b.bank_id)

        body = _body(b.bank_id, b.sounds)
        if b.resident:
            res = bytearray(_pad(res, RES_ALIGN))
            off, where = len(res), WHERE_RESIDENT
            res += body
        else:
            cells = bytearray(_pad(cells, CELL_ALIGN))
            off, where = len(cells), WHERE_CELL
            cells += body
        b.where, b.data_off, b.data_bytes = where, off, len(body)
        b.first_sound, b.num_sounds = len(sound_recs), len(b.sounds)
        bank_recs.append(b)
        for s in b.sounds:
            sound_recs.append(s)
            rate_max = max(rate_max, s.rate)

    if len(res) > RESIDENT_LIMIT:
        raise ResidentTooBig("resident set is %d bytes, limit %d - drop banks from the "
                             "resident list or raise RESIDENT_LIMIT together with "
                             "SFX_CELL_RESIDENT's capacity in "
                             "src/platform_psp/SfxArena.h" % (len(res), RESIDENT_LIMIT))

    out = bytearray()
    out += struct.pack(HDR_FMT, MAGIC, VERSION, len(bank_recs), len(sound_recs),
                       len(res), len(cells), rate_max, len(extras))
    for b in bank_recs:
        out += struct.pack(BANK_FMT, b.bank_id, b.where, 0, b.data_off, b.data_bytes,
                           b.first_sound, b.num_sounds)
    for s in sound_recs:
        out += struct.pack(SOUND_FMT, s.off, s.bytes, s.rate, s.loop_frame, s.headroom, 0)
    out += extras
    return bytes(out), bytes(res), bytes(cells)


def parse(index_bytes):
    """Parse an index blob back into a Parsed(banks, sounds, ...).

 Every parsed Sound.vag is empty - the payload lives in the res/cells buffers
 the caller loaded separately, addressed via Sound.off/Sound.bytes.
 """
    magic, version, n_banks, n_sounds, res_b, cells_b, rate_max, extras_b = \
        struct.unpack_from(HDR_FMT, index_bytes, 0)
    if magic != MAGIC:
        raise ValueError("bad magic 0x%08x" % magic)
    if version != VERSION:
        raise ValueError("unsupported version %d" % version)
    p = Parsed()
    p.resident_bytes, p.banks_bytes, p.rate_max = res_b, cells_b, rate_max
    p.banks, p.sounds = [], []
    if extras_b:
        tail = len(index_bytes) - extras_b
        if tail < HDR:
            raise ValueError("index claims %d bytes of extras but is only %d bytes long"
                             % (extras_b, len(index_bytes)))
        p.extras = index_bytes[tail:]
    else:
        p.extras = b""
    o = HDR
    for _ in range(n_banks):
        bank_id, where, _pad0, data_off, data_bytes, first, num = \
            struct.unpack_from(BANK_FMT, index_bytes, o)
        b = Bank(bank_id, where == WHERE_RESIDENT, [], where, data_off, data_bytes, first, num)
        p.banks.append(b)
        o += BANK_REC
    for _ in range(n_sounds):
        off, nbytes, rate, loop, headroom, _pad1 = struct.unpack_from(SOUND_FMT, index_bytes, o)
        p.sounds.append(Sound(b"", rate, loop, headroom, off, nbytes))
        o += SOUND_REC
    return p
