#!/usr/bin/env python3
"""Extract gVehicleAudioSettings[231] from the PS2 executable -> vehaud.bin.

Reversed from SLES_525.41 v2.01: GetVehicleAudioSettings reads
`0x653050 + 0x20 * modelId`, so the record for model 400 sits at 0x656250 and the
stride is 0x20. The PS2 field layout is NOT the PC one - it was recovered by
probing offsets against seven models whose banks/horns/stations were already known.
 section 3.1.

That address is specific to the v2.01 executable, not a property of the game: the
same table sits at 0x655ad0 on v1.03, and other regions/revisions/languages can
relocate it again. read_from_elf() therefore treats TABLE_VA as a hint, validates
it against seven anchor records before trusting it, and falls back to scanning
every PT_LOAD segment for the same seven anchors when the hint does not hold.
"""
import struct

MAGIC = 0x41484556          # 'VEHA'
TABLE_VA = 0x656250         # the v2.01 address - a hint, not a guarantee
STRIDE = 0x20
# 231, not 232 - measured, not assumed. Index 230 (model 630) is the last
# well-formed record (a SPECIAL sentinel: type=10, playerBank=dummyBank=-1,
# hornType=255). Index 231 (model 631) is not a record at all: it decodes as six
# repeats of the u32 0x0056dcd0 followed by 0x0056ddb8, monotonically-increasing
# little-endian pointers into the 0x0056xxxx code region - the start of an
# unrelated, adjacent pointer table. Do not round this back up to 232. This holds
# on both v1.03 and v2.01, at each one's own table address.
COUNT = 231
FIRST_MODEL = 400

# PS2 record field offsets
F_TYPE, F_PBANK, F_DBANK = 0x00, 0x02, 0x04
F_HORN_TYPE, F_HORN_PITCH = 0x0C, 0x10
F_RADIO_STATION, F_RADIO_TYPE = 0x16, 0x17
F_ENGINE_VOL = 0x1C

# Documented `type` enum - see the vehaud.bin format in the plan: 0 CAR 1 BIKE
# 2 BMX 3 BOAT 4 HELI 5 PLANE 8 TRAIN 9 TRAILER 10 SPECIAL. 6 and 7 are gaps in
# the source data, not a typo here: every one of the real 231 records carries a
# value from this set.
VALID_TYPES = frozenset((0, 1, 2, 3, 4, 5, 8, 9, 10))

# The seven models whose banks/horns/stations the reverse pinned down (see
# section 3.1), keyed by record
# index (model - FIRST_MODEL) rather than model id, since that is what a
# candidate table position is checked against: recordIndex -> (type, playerBank,
# dummyBank). Used both to validate TABLE_VA before trusting it and to locate the
# table by scanning when the hint does not hold for a given build.
ANCHORS = {
    0: (0, 99, 98),      # 400 landstalker
    1: (0, 8, 7),         # 401 bravura
    3: (0, 84, 83),       # 403 linerunner
    11: (0, 38, 37),      # 411 infernus
    16: (0, 137, 136),    # 416 ambulance
    61: (1, 125, 124),    # 461 pcj600
    62: (1, 119, 118),    # 462 faggio
}


class FieldOutOfRange(Exception):
    # Same name as sa_sfxpack.FieldOutOfRange, but a distinct, unrelated class --
    # they share no base beyond Exception, so `except sa_sfxpack.FieldOutOfRange`
    # would not catch this one. Catch this module's FieldOutOfRange explicitly.
    pass


class MalformedElf(Exception):
    pass


class TableNotFound(ValueError):
    pass


class AmbiguousTable(ValueError):
    pass


class VehAud(object):
    __slots__ = ("type", "player_bank", "dummy_bank", "horn_type", "horn_pitch",
                 "engine_vol_offset", "radio_station", "radio_type")


def decode_table(table_bytes):
    """table_bytes = COUNT * STRIDE bytes starting at the model-400 record.

 Validates `type` against VALID_TYPES on every record. This is a general guard,
 not a pin on one known-bad value: it is what would have caught the COUNT=232
 bug by itself (the foreign 232nd record decodes to a type outside the enum),
 and it catches a wrong TABLE_VA or a shifted STRIDE the same way.
 """
    if len(table_bytes) < COUNT * STRIDE:
        raise ValueError("table is %d bytes, need %d" % (len(table_bytes), COUNT * STRIDE))
    out = []
    for i in range(COUNT):
        o = i * STRIDE
        r = VehAud()
        r.type, = struct.unpack_from("<h", table_bytes, o + F_TYPE)
        r.player_bank, = struct.unpack_from("<h", table_bytes, o + F_PBANK)
        r.dummy_bank, = struct.unpack_from("<h", table_bytes, o + F_DBANK)
        r.horn_type, = struct.unpack_from("<h", table_bytes, o + F_HORN_TYPE)
        r.horn_pitch, = struct.unpack_from("<f", table_bytes, o + F_HORN_PITCH)
        r.engine_vol_offset, = struct.unpack_from("<f", table_bytes, o + F_ENGINE_VOL)
        r.radio_station, = struct.unpack_from("<b", table_bytes, o + F_RADIO_STATION)
        r.radio_type, = struct.unpack_from("<b", table_bytes, o + F_RADIO_TYPE)
        if r.type not in VALID_TYPES:
            raise FieldOutOfRange(
                "record %d (model %d): type=%d is not one of the documented "
                "values %s" % (i, FIRST_MODEL + i, r.type, sorted(VALID_TYPES)))
        out.append(r)
    return out


def _read_struct(fmt, data, offset, path, what):
    """struct.unpack_from that names the path and the field being read on
 failure, instead of letting a bare struct.error - which names neither - escape from a truncated or corrupt file."""
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as e:
        raise MalformedElf("%s: truncated or corrupt, could not read %s at "
                            "offset 0x%x (%s)" % (path, what, offset, e))


def _elf_segments(data, path):
    """Parse the ELF header and return every PT_LOAD segment as
 (p_offset, p_vaddr, p_filesz)."""
    if data[:4] != b"\x7fELF":
        raise MalformedElf("%s is not an ELF (bad magic)" % path)
    e_phoff, = _read_struct("<I", data, 0x1C, path, "e_phoff")
    e_phentsize, e_phnum = _read_struct("<HH", data, 0x2A, path, "e_phentsize/e_phnum")
    segments = []
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        p_type, p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz = \
            _read_struct("<6I", data, o, path, "program header %d" % i)
        if p_type == 1:
            segments.append((p_offset, p_vaddr, p_filesz))
    return segments


def _anchors_match(data, cand_off):
    """Cheap first filter for a candidate table start at file offset cand_off:
 do all seven anchor records' (type, playerBank, dummyBank) match? This alone
 is already highly discriminating - seven independent int16 triples agreeing
 by chance is vanishingly unlikely - so the more expensive whole-table
 confirmation below only ever runs on genuine candidates in practice.
 """
    for idx, want in ANCHORS.items():
        off = cand_off + idx * STRIDE
        if off + 6 > len(data):
            return False
        if struct.unpack_from("<hhh", data, off) != want:
            return False
    return True


def _confirms_whole_table(data, cand_off):
    """Second filter, run only on positions that already passed the anchor
 check: every one of the COUNT records must decode with an in-enum type
 (reusing decode_table's own guard, so there is exactly one place that
 defines "a valid record"), and the record immediately past the end must
 NOT - that second half is what actually pins the table's length, and it
 holds on both PS2 discs this was verified against (v1.03 and v2.01).
 """
    end = cand_off + COUNT * STRIDE
    if end + STRIDE > len(data):
        return False   # not enough of the file left to check the pinning record
    try:
        decode_table(data[cand_off:end])
    except FieldOutOfRange:
        return False
    t_past, = struct.unpack_from("<h", data, end)
    return t_past not in VALID_TYPES


def _locate_table(data, segments, path):
    """Find the one file offset where the vehicle audio table lives.

 Tries TABLE_VA first (fast path): a hint, not a guarantee, so it is only
 trusted after passing the same anchor and whole-table checks as any scanned
 candidate. If it does not hold - wrong build, or TABLE_VA is not even
 mapped by any PT_LOAD segment - every PT_LOAD segment is scanned at 4-byte
 alignment for a position where all seven anchors match and the whole table
 confirms. Returns (file_offset, table_va).
 """
    table_end = TABLE_VA + COUNT * STRIDE
    for p_offset, p_vaddr, p_filesz in segments:
        if p_vaddr <= TABLE_VA and table_end <= p_vaddr + p_filesz:
            cand_off = p_offset + (TABLE_VA - p_vaddr)
            if _anchors_match(data, cand_off) and _confirms_whole_table(data, cand_off):
                return cand_off, TABLE_VA

    matches = []
    for p_offset, p_vaddr, p_filesz in segments:
        # Clamp to what the file actually contains: a segment's declared p_filesz
        # is untrusted input, and iterating past len(data) would only waste time
        # re-deriving what _anchors_match already rejects one byte at a time.
        avail = min(p_filesz, len(data) - p_offset)
        span = avail - COUNT * STRIDE
        if span < 0:
            continue
        for cand_off in range(p_offset, p_offset + span + 1, 4):
            if _anchors_match(data, cand_off) and _confirms_whole_table(data, cand_off):
                matches.append((cand_off, p_vaddr + (cand_off - p_offset)))

    if not matches:
        raise TableNotFound(
            "%s: no position in any PT_LOAD segment matches all seven anchor "
            "records at stride 0x%x - this executable does not contain a "
            "recognisable vehicle audio table" % (path, STRIDE))
    if len(matches) > 1:
        raise AmbiguousTable(
            "%s: %d distinct positions all match the seven anchors and confirm "
            "as a whole table (at VAs %s) - the signature is not discriminating "
            "for this file, needs a human look"
            % (path, len(matches), ", ".join("0x%08x" % va for _, va in matches)))
    return matches[0]


def read_from_elf(path):
    """Pull the table out of the PS2 ELF using its PT_LOAD mapping."""
    with open(path, "rb") as f:
        data = f.read()
    segments = _elf_segments(data, path)
    cand_off, _table_va = _locate_table(data, segments, path)
    return decode_table(data[cand_off:cand_off + COUNT * STRIDE])


def pack(recs):
    out = bytearray()
    out += struct.pack("<4I", MAGIC, 1, COUNT, FIRST_MODEL)
    for r in recs:
        out += struct.pack("<hhhhffbbH", r.type, r.player_bank, r.dummy_bank,
                           r.horn_type, r.horn_pitch, r.engine_vol_offset,
                           r.radio_station, r.radio_type, 0)
    return bytes(out)
