#!/usr/bin/env python3
"""The two level curves the mixer needs, pulled from the disc.

`CAESound::CalculateVolume` builds a sound's final level in dB out of four terms:

 final = MikeAttenuation(dir) + DistanceAttenuation(|rel| / rollOffFactor)
 + eventVolume - soundHeadroom

Three of those already reach the engine: the direction and the distance are
geometry, and the headroom rides in the sound records of sfx_index.bin. The
remaining two come off the disc and are what this module extracts:

 * the distance attenuation curve - 1280 floats of dB inside the PS2
 executable, indexed by `floor(normalisedDistance * 10)`, where the
 normalised distance is the world distance divided by the sound's
 rollOffFactor. ★ It is FLAT AT 0 dB out to a normalised distance of 4.9
 and only then rolls off; an analytic K/(d+K) stand-in gets that near field
 wrong by 7 dB and is why the world mixed 18 dB under the interface.

 * data/surfaud.dat x data/surfinfo.dat - which of nine audio classes each
 surface belongs to, indexed by the eSurfaceType a COL triangle carries. That is
 what picks the footstep bank: concrete, grass, sand, gravel, wood, metal, tile.

 * AUDIO/CONFIG/EventVol.dat - one signed byte of dB per audio event, indexed
 directly by the event id (`GetDefaultVolume(e) = m_pAudioEventVolumes[e]`).
 The PC and PS2 files agree to within one byte of length, so this is the same
 authored table on both.

Both are verified byte-identical between the v1.03 and v2.01 discs.
"""
import struct

ATTEN_COUNT = 1280           # gSoundDistAttenuationTable; index = floor(normDist * 10)
ATTEN_MAX_NORM_DIST = 128.0  # GetDistanceAttenuation returns -100 dB at or past this
ATTEN_OUT_OF_RANGE_DB = -100.0

# The curve is stored as float dB and shipped as hundredths of a dB in an int16:
# the source values are authored to two decimals (-0.38, -84.29) and the whole
# range fits, so this is lossless for the actual data and halves the table.
ATTEN_DB_SCALE = 100

# Anchor: t[50..59], the first ten attenuating entries, as they appear in both
# discs' executables. Ten consecutive float32 agreeing by accident is not a thing,
# and unlike a hardcoded address this survives a different build or region --
# the same lesson the vehicle audio table taught (sa_vehaud.TABLE_VA).
ATTEN_ANCHOR_INDEX = 50
ATTEN_ANCHOR = bytes.fromhex(
    "5c8fc2be5c8f42bfd7a390bf52b8bebf7b14eebf"
    "a4700dc00ad723c0713d3ac0000050c0b81e65c0")

# Number of event volumes we ship. The file itself holds 45403 of them, but
# everything past AE_END_OF_EVENTS (0x1C45) is per-line ped speech, which this
# port does not produce - carrying it would put 38 KB of -128 in the heap for
# nothing. Every *named* audio event is below the cut, so a producer that asks
# for one it can name always gets a real answer.
EVENTVOL_COUNT = 0x1C46      # AE_END_OF_EVENTS inclusive
EVENTVOL_MUTED = -128        # the file's "this event has no volume" value

# Per-surface audio class, from data/surfaud.dat crossed with data/surfinfo.dat.
# surfaud.dat is keyed by NAME and gives nine booleans; surfinfo.dat is the ORDERED
# list, and its row number is the eSurfaceType a COL triangle's material byte holds.
# Neither file alone is usable: one has the classes, the other has the numbering.
SURF_CONCRETE, SURF_GRASS, SURF_SAND, SURF_GRAVEL = 0, 1, 2, 3
SURF_WOOD, SURF_WATER, SURF_METAL, SURF_TILE = 4, 5, 6, 7
SURF_NONE = 0xFF             # the surface names no class at all -> caller's default

# Column order in surfaud.dat, and the precedence when a row sets more than one.
# WATER is pulled to the front on purpose: it does not pick a different footstep, it
# picks a different EVENT (the original plays a splash from another bank entirely),
# so it has to win over any ground class the row also carries. LGS ("long grass")
# folds into GRASS - the port has one grass bank, not two.
SURFAUD_COLUMNS = ["CON", "GRS", "SND", "GRV", "WOD", "WTR", "MTL", "LGS", "TIL"]
SURFAUD_PRECEDENCE = [("WTR", SURF_WATER), ("GRS", SURF_GRASS), ("LGS", SURF_GRASS),
                      ("SND", SURF_SAND), ("GRV", SURF_GRAVEL), ("WOD", SURF_WOOD),
                      ("MTL", SURF_METAL), ("TIL", SURF_TILE), ("CON", SURF_CONCRETE)]

EXTRAS_MAGIC = 0x56445541    # 'AUDV'


class CurveNotFound(Exception):
    pass


class AmbiguousCurve(Exception):
    pass


class MalformedCurve(Exception):
    pass


def _confirms_atten(vals):
    """A candidate is the real table only if it also has the shape the mixer
 relies on: a flat 0 dB near field, a monotonically falling tail, and an end
 well below -80 dB. The anchor says "these bytes are here"; this says "and
 they are the start of a distance curve"."""
    if len(vals) != ATTEN_COUNT:
        return False
    if any(v != 0.0 for v in vals[:ATTEN_ANCHOR_INDEX]):
        return False
    if any(vals[i + 1] > vals[i] for i in range(ATTEN_COUNT - 1)):
        return False
    return vals[-1] < -80.0


def atten_from_elf(path):
    """Locate and decode the distance attenuation curve in a PS2 executable.

 Returns ATTEN_COUNT floats of dB. Raises rather than guessing: a curve we
 cannot find is a converter that would silently ship a different mix.
 """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        raise MalformedCurve("%s is not an ELF (bad magic)" % path)

    matches = []
    pos = data.find(ATTEN_ANCHOR)
    while pos != -1:
        start = pos - ATTEN_ANCHOR_INDEX * 4
        if start >= 0 and start + ATTEN_COUNT * 4 <= len(data):
            vals = list(struct.unpack_from("<%df" % ATTEN_COUNT, data, start))
            if _confirms_atten(vals):
                matches.append((start, vals))
        pos = data.find(ATTEN_ANCHOR, pos + 1)

    if not matches:
        raise CurveNotFound(
            "%s: no position holds the ten anchor entries of the distance "
            "attenuation curve followed by a well-formed 1280-entry table - "
            "this executable is not a recognisable the source game PS2 build" % path)
    if len(matches) > 1:
        raise AmbiguousCurve(
            "%s: %d positions decode as the distance attenuation curve (file "
            "offsets %s) - the anchor is not discriminating here, needs a "
            "human look" % (path, len(matches),
                            ", ".join("0x%x" % off for off, _ in matches)))
    return matches[0][1]


def eventvol_from_dat(path):
    """Decode AUDIO/CONFIG/EventVol.dat -> EVENTVOL_COUNT signed dB values.

 Stride is one byte: the game indexes this array with the raw event id.
 A file shorter than the cut is padded with EVENTVOL_MUTED rather than
 rejected - a truncated table still plays, it just mutes what it cannot
 describe, and the bake prints how much it padded.
 """
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        raise MalformedCurve("%s is empty" % path)
    vals = [b - 256 if b > 127 else b for b in raw[:EVENTVOL_COUNT]]
    vals += [EVENTVOL_MUTED] * (EVENTVOL_COUNT - len(vals))
    return vals, len(raw)


def surface_classes(surfinfo_path, surfaud_path):
    """-> (classes, n_named, n_classified).

 `classes[i]` is the audio class of eSurfaceType i, which is what a COL triangle's
 material byte names. Surfaces surfinfo.dat lists but surfaud.dat does not describe
 get SURF_NONE rather than a guess: the original leaves them on the generic bank,
 and inventing a class here would be indistinguishable from the disc saying so.
 """
    order = []
    for ln in open(surfinfo_path, encoding="latin-1"):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        order.append(s.split()[0].upper())

    flags = {}
    for ln in open(surfaud_path, encoding="latin-1"):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        parts = s.split()
        if len(parts) < 1 + len(SURFAUD_COLUMNS):
            continue
        try:
            bits = [int(t) for t in parts[1:1 + len(SURFAUD_COLUMNS)]]
        except ValueError:
            continue
        flags[parts[0].upper()] = dict(zip(SURFAUD_COLUMNS, bits))

    classes, classified = [], 0
    for name in order:
        row = flags.get(name)
        cls = SURF_NONE
        if row:
            for col, c in SURFAUD_PRECEDENCE:
                if row.get(col):
                    cls = c
                    break
        if cls != SURF_NONE:
            classified += 1
        classes.append(cls)
    return classes, len(order), classified


def pack_extras(atten, eventvol, surfclass):
    """The trailing 'extras' block of sfx_index.bin v3.

 Self-describing (both counts are in the block) so the engine never has to
 infer a length from the version number alone.
 """
    if len(atten) != ATTEN_COUNT:
        raise MalformedCurve("attenuation curve has %d entries, expected %d"
                             % (len(atten), ATTEN_COUNT))
    if len(eventvol) != EVENTVOL_COUNT:
        raise MalformedCurve("event volume table has %d entries, expected %d"
                             % (len(eventvol), EVENTVOL_COUNT))
    if not surfclass:
        raise MalformedCurve("the surface class table is empty")
    if len(surfclass) > 65535:
        raise MalformedCurve("surface class table has %d entries, max 65535" % len(surfclass))
    out = bytearray(struct.pack("<4I", EXTRAS_MAGIC, ATTEN_COUNT, EVENTVOL_COUNT,
                                len(surfclass)))
    for v in atten:
        q = int(round(v * ATTEN_DB_SCALE))
        if not -32768 <= q <= 32767:
            raise MalformedCurve("attenuation %.2f dB does not fit an int16 of "
                                 "hundredths" % v)
        out += struct.pack("<h", q)
    for v in eventvol:
        if not -128 <= v <= 127:
            raise MalformedCurve("event volume %d dB does not fit a signed byte" % v)
        out += struct.pack("<b", v)
    for v in surfclass:
        if not 0 <= v <= 255:
            raise MalformedCurve("surface class %d does not fit a byte" % v)
        out += struct.pack("<B", v)
    return bytes(out)
