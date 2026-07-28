"""the source game GXT (TABL/TKEY/TDAT localized text database) decoder.

SA stores every on-screen string keyed by CRC32(toupper(8-char-name)); the literal
key text is NOT in the file, so tables map a crc-hex string -> decoded text. A master
MAIN GXT is a 'TABL' directory + per-table 'TKEY' (sorted {dataOffset, keyCRC32} pairs)
+ 'TDAT' (packed 1-byte-per-char NUL-terminated string blob).

The key hash is reflected CRC-32 of the uppercased name with the final XOR-out OMITTED
(CKeyArray::ComputeKeyCrc32 0x54f410 returns the raw inverted accumulator) ==
zlib.crc32(upper) ^ 0xFFFFFFFF. The CRC is the standard 0xEDB88320 one.

 (confirmed: CText::Load 0x6cce50, TABL 0x6cc310,
 TKEY 0x6cc110, TDAT 0x6cc270, BinarySearch 0x6cc200, hash 0x54f410)
 ; gvcslib `gxt` codec (VCS variant)
"""
from __future__ import annotations

import struct
import zlib
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# key hashing
# ---------------------------------------------------------------------------

def gxt_key_hash(name: str) -> int:
    """CRC32(toupper(name)) with the final XOR-out omitted (SA TKEY key hash).

 zlib.crc32 applies the standard final ^0xFFFFFFFF; CKeyArray::ComputeKeyCrc32
 returns the raw accumulator, so we re-invert. Result is a u32. Non-ASCII chars
 are upper-cased then encoded latin-1 to stay byte-faithful with toupper().
 """
    upper = name.upper().encode("latin-1", "replace")
    return (zlib.crc32(upper) ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _crc_hex(crc: int) -> str:
    """Canonical key form stored in the returned tables: lower-case '0x' + 8 hex."""
    return "0x%08x" % (crc & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# low-level chunk / record reads (all little-endian, defensively sliced)
# ---------------------------------------------------------------------------

def _u16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0] if off + 2 <= len(buf) else 0


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0] if off + 4 <= len(buf) else 0


def _name8(buf: bytes, off: int) -> str:
    return buf[off:off + 8].split(b"\x00", 1)[0].decode("latin-1")


class Gxt(dict):
    """A parsed GXT file as a real ``{table_name: {crc_hex: display_str}}`` dict.

 Subclasses ``dict`` so callers (and the web view) can treat the parse result as
 the plain mapping the public API promises, while ``version`` / ``bits_per_char``
 ride along as attributes for the encoder path.
 """

    def __init__(self, tables: Optional[Dict[str, Dict[str, str]]] = None,
                 version: int = 0x0004, bits_per_char: int = 0x0008):
        super().__init__(tables or {})
        self.version = version
        self.bits_per_char = bits_per_char


# ---------------------------------------------------------------------------
# TDAT string decode
# ---------------------------------------------------------------------------

def _decode_string(blob: bytes, off: int) -> str:
    """Read one NUL-terminated packed-1-byte string from a TDAT body at `off`.

 SA TDAT is an 8-bit codepage: 0x20-0x7E == ASCII, 0x80+ are SA-font accented
 glyphs (RemapPackedToWide 0x6cc470). We preserve bytes 1:1 and decode as cp1252
 for human-readable display (latin-1 fallback) - round-trip stays byte-exact only
 if raw bytes are re-emitted, which the encoder path must do.
 """
    if off < 0 or off >= len(blob):
        return ""
    end = blob.find(b"\x00", off)
    if end < 0:
        end = len(blob)
    raw = blob[off:end]
    try:
        return raw.decode("cp1252")
    except (UnicodeDecodeError, LookupError):
        return raw.decode("latin-1")


def _parse_tkey_tdat(data: bytes, tkey_hdr_off: int) -> Optional[Tuple[Dict[str, str], int]]:
    """Parse a TKEY chunk + the immediately-following TDAT chunk.

 `tkey_hdr_off` points at the 8-byte TKEY chunk header. Returns (fields, end_off)
 where end_off is the byte just past the TDAT body, or None if the layout is bad.
 """
    if data[tkey_hdr_off:tkey_hdr_off + 4] != b"TKEY":
        return None
    tkey_size = _u32(data, tkey_hdr_off + 4)
    tkey_body = tkey_hdr_off + 8
    tkey_end = tkey_body + tkey_size

    tdat_hdr = tkey_end
    if data[tdat_hdr:tdat_hdr + 4] != b"TDAT":
        return None
    tdat_size = _u32(data, tdat_hdr + 4)
    tdat_body = tdat_hdr + 8
    tdat_end = tdat_body + tdat_size
    blob = data[tdat_body:tdat_end]

    fields: Dict[str, str] = {}
    count = tkey_size // 8
    for i in range(count):
        e = tkey_body + i * 8
        try:
            data_off = _u32(data, e)          # +0x00 dataOffset (TDAT-body-relative)
            crc = _u32(data, e + 4)           # +0x04 keyCRC32 (sort key)
            fields[_crc_hex(crc)] = _decode_string(blob, data_off)
        except Exception:
            # one bad record must not kill the table
            continue
    return fields, tdat_end


# ---------------------------------------------------------------------------
# top-level parse
# ---------------------------------------------------------------------------

def parse_gxt(data: bytes) -> Gxt:
    """Decode a GXT file to a Gxt mapping {table_name: {crc_hex: display_str}}.

 Handles both a master MAIN GXT (TABL directory + many TKEY/TDAT pairs, sub-tables
 prefixed by an 8-char name) and a plain single-table GXT (no TABL, just TKEY+TDAT).
 """
    gxt = Gxt()
    if len(data) < 4:
        return gxt
    gxt.version = _u16(data, 0)
    gxt.bits_per_char = _u16(data, 2)

    off = 4
    # First chunk is TABL (master) or TKEY (plain sub-file).
    tag = data[off:off + 4]

    if tag == b"TABL":
        tabl_size = _u32(data, off + 4)
        tabl_body = off + 8
        n = tabl_size // 12
        for i in range(n):
            e = tabl_body + i * 12
            try:
                name = _name8(data, e)
                file_off = _u32(data, e + 8)
            except Exception:
                continue
            gxt[name] = _parse_subtable_at(data, file_off)
    else:
        # Plain GXT: header -> TKEY -> TDAT, no directory.
        res = _parse_tkey_tdat(data, off)
        gxt["MAIN"] = res[0] if res is not None else {}

    return gxt


def _parse_subtable_at(data: bytes, file_off: int) -> Dict[str, str]:
    """Parse the sub-table whose TABL fileOffset is `file_off`.

 The first/global table (MAIN) sits as a bare TKEY at its offset; every other
 sub-table is prefixed by its 8-char name then TKEY+TDAT. We detect which by the
 4 bytes at file_off (a 'TKEY' tag means bare; otherwise it's the name prefix).
 Returns the crc-hex -> string field map (empty on a malformed offset).
 """
    if file_off < 0 or file_off + 4 > len(data):
        return {}
    try:
        if data[file_off:file_off + 4] == b"TKEY":
            tkey_off = file_off
        else:
            # name-prefixed: skip the 8-char name to reach the TKEY header
            tkey_off = file_off + 8
        res = _parse_tkey_tdat(data, tkey_off)
        return res[0] if res is not None else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# search + export
# ---------------------------------------------------------------------------

def search_gxt(gxt, query: str, *, in_key: bool = True,
               in_value: bool = True) -> List[Tuple[str, str, str]]:
    """Flat case-insensitive search across all tables.

 Returns a list of (table_name, key_hex, value) for entries whose key-hex
 (when `in_key`) or decoded value (when `in_value`) contains `query`.
 Accepts a Gxt or a plain {table: {key: str}} dict.
 """
    q = query.lower()
    hits: List[Tuple[str, str, str]] = []
    for table, entries in gxt.items():
        for key, val in entries.items():
            if (in_value and q in str(val).lower()) or (in_key and q in str(key).lower()):
                hits.append((table, key, val))
    return hits


def to_json(gxt) -> Dict[str, Dict[str, str]]:
    """Serialize to a JSON-ready {table_name: {key_hex: display_str}} dict.

 Every value is a plain str (JSON-serializable), suitable for the SAW text-table
 web view. Accepts a Gxt or an already-plain dict.
 """
    out: Dict[str, Dict[str, str]] = {}
    for table, entries in gxt.items():
        out[str(table)] = {str(k): ("" if v is None else str(v)) for k, v in entries.items()}
    return out
