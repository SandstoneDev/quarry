"""GXT localized text codec for the source console title (PSP).

File layout (verified against ENGLISH/FRENCH/GERMAN/ITALIAN/SPANISH.GXT,
87 tables / 6739 keys each):

 'TABL' (4 bytes magic)
 u32 dir_size (== 12 * ntables)
 TableDirEntry[ntables] {
 char[8] name (NUL-padded)
 u32 file_off
 }

Each sub-table lives at its file_off. The first table's body begins directly
with 'TKEY'; every subsequent table is preceded by an 8-byte name header
(the table name again, NUL-padded). Rule:
 tk = file_off if data[file_off:file_off+4] == b'TKEY'
 tk = file_off + 8 otherwise (skip the duplicate name header)

 'TKEY' (4 bytes magic)
 u32 key_size (== 12 * nkeys)
 KeyEntry[nkeys] {
 u32 data_off (OFFSET FIRST; byte offset relative to TDAT body)
 char[8] name (NUL-padded; keys sorted ASCII-ascending)
 }
 'TDAT' (4 bytes magic)
 u32 dat_size
 <dat_size bytes of UTF-16LE strings, each u16 0x0000 terminated>

Strings are stored contiguously but in a different order than the (sorted)
key list, so the per-key data offsets are preserved verbatim to guarantee a
byte-exact round-trip. Table bodies are aligned to 4 bytes (0 or 2 zero
padding bytes between tables).

decode(data) -> Gxt: an ordered dict {table_name: [(key, string), ...]}
 in key order, carrying the offset metadata for re-encode.
encode(gxt) -> bytes: byte-exact reconstruction of the original file.
"""

from ._io import R, W

MAGIC = b'TABL'


def _find_term(d, start):
    """Return offset of the UTF-16LE NUL terminator (on a 2-byte boundary)."""
    i = start
    n = len(d)
    while i + 1 < n:
        if d[i] == 0 and d[i + 1] == 0:
            return i
        i += 2
    raise ValueError("unterminated UTF-16 string at %d" % start)


def _name8(b):
    """Decode an 8-byte NUL-padded ASCII name to str."""
    return b.split(b'\x00', 1)[0].decode('ascii')


def _pad8(s):
    """Encode a name str to an 8-byte NUL-padded ASCII field."""
    raw = s.encode('ascii')
    if len(raw) > 8:
        raise ValueError("name too long for 8-byte field: %r" % s)
    return raw + b'\x00' * (8 - len(raw))


class Gxt(dict):
    """Ordered {table_name: [(key, string), ...]} with re-encode metadata.

 Public mapping interface: gxt[table_name] -> list of (key, str) tuples in
 the original (ASCII-sorted) key order.

 Internal (per table, parallel to the entry list):
 _offsets[table_name] -> list of data offsets, one per key entry
 _has_header[table_name]-> bool, whether an 8-byte name header precedes it
 """

    def __init__(self):
        super().__init__()
        self._offsets = {}
        self._has_header = {}


def decode(data):
    data = bytes(data)
    r = R(data)
    magic = r.take(4)
    if magic != MAGIC:
        raise ValueError("not a GXT file: magic=%r" % magic)
    dir_size = r.u32()
    ntables = dir_size // 12

    dir_entries = []
    for _ in range(ntables):
        name = _name8(r.take(8))
        file_off = r.u32()
        dir_entries.append((name, file_off))

    gxt = Gxt()
    for name, file_off in dir_entries:
        has_header = data[file_off:file_off + 4] != b'TKEY'
        gxt._has_header[name] = has_header
        tk = file_off + 8 if has_header else file_off

        tr = R(data, tk)
        kmagic = tr.take(4)
        if kmagic != b'TKEY':
            raise ValueError("expected TKEY at %d, got %r" % (tk, kmagic))
        key_size = tr.u32()
        nkeys = key_size // 12

        key_entries = []  # (data_off, key_name)
        for _ in range(nkeys):
            data_off = tr.u32()
            kname = _name8(tr.take(8))
            key_entries.append((data_off, kname))

        dmagic = tr.take(4)
        if dmagic != b'TDAT':
            raise ValueError("expected TDAT at %d, got %r" % (tr.p - 4, dmagic))
        dat_size = tr.u32()
        body = tr.p  # offset of first string byte (TDAT body)

        entries = []
        offsets = []
        for data_off, kname in key_entries:
            start = body + data_off
            end = _find_term(data, start)
            s = data[start:end].decode('utf-16le')
            entries.append((kname, s))
            offsets.append(data_off)

        gxt[name] = entries
        gxt._offsets[name] = offsets
        _ = dat_size  # consumed implicitly via offsets/strings

    return gxt


def encode(gxt, rebuild=False):
    """Serialise a Gxt back to bytes.

 rebuild=False (default): byte-exact - preserves the original per-key data offsets
 (only valid if no string length changed). rebuild=True: recompute all offsets in
 key order, so edited/translated strings of any length produce a valid (not
 byte-exact) GXT the engine reads fine.
 """
    table_bodies = []  # parallel to table order in gxt
    names = list(gxt.keys())

    for name in names:
        entries = gxt[name]
        nkeys = len(entries)

        if rebuild:
            # fresh offsets, storage order == key order
            offsets = []
            cur = 0
            for (_k, s) in entries:
                offsets.append(cur)
                cur += len(s.encode('utf-16le')) + 2
            order = list(range(nkeys))
        else:
            offsets = gxt._offsets[name]
            # storage order = ascending data offset, so offsets land exactly
            order = sorted(range(nkeys), key=lambda i: offsets[i])

        dat = bytearray()
        for i in order:
            if not rebuild and offsets[i] != len(dat):
                raise ValueError(
                    "offset mismatch in table %s key %s: expected %d got %d"
                    % (name, entries[i][0], len(dat), offsets[i]))
            dat += entries[i][1].encode('utf-16le')
            dat += b'\x00\x00'

        kw = W()
        kw.raw(b'TKEY')
        kw.u32(nkeys * 12)
        for (kname, _s), off in zip(entries, offsets):
            kw.u32(off)
            kw.raw(_pad8(kname))
        kw.raw(b'TDAT')
        kw.u32(len(dat))
        kw.raw(bytes(dat))

        body = kw.getvalue()
        if gxt._has_header[name]:
            body = _pad8(name) + body
        table_bodies.append(body)

    # 2) Header + directory, then place table bodies with 4-byte alignment.
    ntables = len(names)
    header_size = 4 + 4 + ntables * 12  # 'TABL' + dir_size + entries

    # First pass: compute each table's file offset (4-byte aligned).
    offsets_in_file = []
    cur = header_size
    for body in table_bodies:
        if cur % 4 != 0:
            cur += (4 - cur % 4) % 4
        offsets_in_file.append(cur)
        cur += len(body)

    w = W()
    w.raw(MAGIC)
    w.u32(ntables * 12)
    for name, foff in zip(names, offsets_in_file):
        w.raw(_pad8(name))
        w.u32(foff)

    for foff, body in zip(offsets_in_file, table_bodies):
        if len(w) < foff:
            w.raw(b'\x00' * (foff - len(w)))
        w.raw(body)

    return w.getvalue()
