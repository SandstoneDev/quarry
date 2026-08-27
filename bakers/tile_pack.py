#!/usr/bin/env python3
"""Pack a world tile's pieces into one region_<rx>_<ry>.tile archive.

A tile costs up to 14 file opens today; fopen is 33.9 ms on a PSP-3000, so 94% of a
tile load is opening files. About five of those opens fail - .barr exists for no tile
and .anim for six of 184 - and a failed lookup in a 1701-entry directory is not free.

Rules this packer follows, from :
 1. It transcodes, it does not decide. An unrecognised extension is a FAILURE.
 2. No silent losses: the report always carries its denominator.
 3. The byte balance must close: header + entries + padding == file size.
"""
import argparse
import os
import re
import struct
import sys

MAGIC, VERSION, ALIGN = 0x4C495453, 1, 64

# Order fixes the enum in src/platform_psp/TileArcFmt.h - keep the two in step.
# test_tile_pack.py:EnumParity reads the header and fails loudly if they drift.
KINDS = ["pmap", "col", "lod", "dyn", "night", "nightd", "tobj",
         "grass", "sway", "anim", "spin", "road", "barr", "mflags"]
KIND = {k: i for i, k in enumerate(KINDS)}

HDR = struct.Struct("<4I")
ENT = struct.Struct("<3I")

_PIECE_RE = re.compile(r"region_(\d+)_(\d+)\.")
_ARCHIVE_RE = re.compile(r"region_(\d+)_(\d+)\.tile$")


class PackError(Exception):
    pass


def _pad(n):
    return (-n) % ALIGN


def _header(blob):
    """Validate magic/version and return the entry count. Raises PackError for
 anything that isn't a well-formed archive - callers should never see a bare
 struct.error out of a truncated or garbage file."""
    try:
        magic, version, count, _ = HDR.unpack_from(blob, 0)
    except struct.error as e:
        raise PackError("not a .tile (only %d bytes)" % len(blob)) from e
    if magic != MAGIC or version != VERSION:
        raise PackError("not a .tile (magic %08X v%d)" % (magic, version))
    return count


def pack_tile(d, rx, ry, files=None):
    """Write region_<rx>_<ry>.tile from the pieces in `d`. Returns how many pieces
 went in. `files` is an optional pre-filtered directory listing (basenames) - pass it when packing many tiles from one directory so the caller lists `d`
 once instead of once per tile; omitted, this lists `d` itself and still works
 standalone."""
    if files is None:
        files = os.listdir(d)
    pre = "region_%d_%d." % (rx, ry)
    found = {}
    for name in files:
        if not name.startswith(pre) or name.endswith(".tile"):
            continue
        ext = name[len(pre):]
        if ext not in KIND:
            raise PackError("%s: unknown piece '%s' - the packer transcodes, it does "
                            "not decide. Add it to KINDS or remove the file." % (name, ext))
        found[KIND[ext]] = os.path.join(d, name)
    if not found:
        return 0

    kinds = sorted(found)
    off = HDR.size + ENT.size * len(kinds)
    off += _pad(off)
    entries, payload = [], []
    for k in kinds:
        with open(found[k], "rb") as f:
            data = f.read()
        entries.append((k, off, len(data)))
        payload.append(data)
        off += len(data)
        off += _pad(off)

    out = bytearray()
    out += HDR.pack(MAGIC, VERSION, len(kinds), 0)
    for e in entries:
        out += ENT.pack(*e)
    for (_k, o, _s), data in zip(entries, payload):
        out += b"\0" * (o - len(out))
        out += data
    out += b"\0" * _pad(len(out))

    with open(os.path.join(d, "region_%d_%d.tile" % (rx, ry)), "wb") as f:
        f.write(out)
    return len(kinds)


def spans(blob):
    count = _header(blob)
    return [(o, s) for (_k, o, s) in
            (ENT.unpack_from(blob, HDR.size + i * ENT.size) for i in range(count))]


def read_tile(blob):
    """{kind: bytes} for every entry."""
    count = _header(blob)
    out = {}
    for i in range(count):
        k, o, s = ENT.unpack_from(blob, HDR.size + i * ENT.size)
        out[k] = blob[o:o + s]
    return out


def balance(blob):
    """file size - (header + entries + payload + padding). Must be 0."""
    count = _header(blob)
    acc = HDR.size + ENT.size * count
    acc += _pad(acc)
    for i in range(count):
        _k, o, s = ENT.unpack_from(blob, HDR.size + i * ENT.size)
        if o != acc:
            return o - acc
        acc = o + s
        acc += _pad(acc)
    return len(blob) - acc


def _group_by_tile(names):
    """{(rx, ry): [name, ...]} for region_<rx>_<ry>.<ext> source pieces in `names`.
 A bare .tile is an archive, not a piece, and is excluded here the same way
 pack_tile excludes it when scanning for pieces - so the denominator this
 builds is "tiles with live source right now", not "tiles we've ever seen a
 filename for". Grouping once here is what lets main() list a directory a
 single time instead of once per tile."""
    by_tile = {}
    for n in names:
        m = _PIECE_RE.match(n)
        if not m or n.endswith(".tile"):
            continue
        by_tile.setdefault((int(m.group(1)), int(m.group(2))), []).append(n)
    return by_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect-tiles", type=int, default=0,
                    help="additionally require exactly this many tiles packed "
                         "(fails on more or fewer); the packed-vs-found check "
                         "below always runs regardless of this flag")
    a = ap.parse_args()

    all_names = os.listdir(a.dir)
    by_tile = _group_by_tile(all_names)
    tiles = sorted(by_tile)

    packed = pieces = 0
    for rx, ry in tiles:
        n = pack_tile(a.dir, rx, ry, files=by_tile[(rx, ry)])
        if n:
            packed += 1
            pieces += n
    print("tile_pack: packed %d pieces across %d tiles of %d found"
          % (pieces, packed, len(tiles)))

    # Unconditional: the denominator is tiles with live source pieces right now, so
    # this can only be false if pack_tile silently produced nothing for a tile that
    # had source - the exact "green and wrong" shape this module exists to catch.
    # It does not wait on an operator remembering a flag.
    if packed != len(tiles):
        print("FAIL: %d of %d tiles with source pieces did not produce an archive"
              % (len(tiles) - packed, len(tiles)))
        return 1

    if a.expect_tiles and packed != a.expect_tiles:
        print("FAIL: expected %d tiles, packed %d" % (a.expect_tiles, packed))
        return 1

    # Sweep every.tile on disk, not just the ones this run just wrote. Freshly
    # packed ones are good by construction, but a pre-existing archive whose
    # source pieces are gone - so it never entered `tiles` above - still needs
    # checking: it is exactly the "validly-formed old file" that a source-only
    # denominator would otherwise go blind to.
    archives = {(int(m.group(1)), int(m.group(2)))
                for m in (_ARCHIVE_RE.match(n) for n in all_names) if m}
    archives |= set(tiles)
    for rx, ry in sorted(archives):
        p = os.path.join(a.dir, "region_%d_%d.tile" % (rx, ry))
        with open(p, "rb") as f:
            blob = f.read()
        try:
            bal = balance(blob)
        except PackError as e:
            print("FAIL: %d,%d is not a valid archive (%s)" % (rx, ry, e))
            return 1
        if bal != 0:
            print("FAIL: byte balance does not close for %d,%d" % (rx, ry))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
