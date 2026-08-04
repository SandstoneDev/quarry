"""SCM script disassembler / assembler for the source console title (PSP).

MAIN.SCM is the compiled mission/game script consumed by the in-engine SCM
virtual machine. The interpreter (EBOOT FUN_0005e5e8) walks a flat byte stream
("ScriptSpace"):

 op = u16 le @ IP ; IP += 2
 not_flag = op >> 15 # bit 15 = invert the boolean result
 idx = op & 0x7fff # 0x000 .. 0x55a, indexes a 1371-entry table
 handler = COMMAND_TABLE[idx] # @ EBOOT 0x380AB0, 8 bytes/entry

Each handler consumes ``argc`` operands. Operands are self-delimiting: a 1-byte
TYPE tag precedes the value, and the operand-reader (EBOOT FUN_0006189c, with the
per-tag ) advances IP by a fixed amount per tag. The widths
below are the *authoritative* ones recovered from the jumptable handlers
(0x061940..0x061a5c) - total bytes consumed *including* the tag byte:

 tag 0x00 -> 1 (tag only, value 0)
 tag 0x01 -> 1 (tag only, value 0)
 tag 0x02 -> 1 (tag only, value 0)
 tag 0x03 -> 2 (tag + u8)
 tag 0x04 -> 3 (tag + s16)
 tag 0x05 -> 4 (tag + s24)
 tag 0x06 -> 5 (tag + s32) <- GOTO/label targets, imm32
 tag 0x07 -> 2 (tag + s8)
 tag 0x08 -> 3 (tag + s16)
 tag 0x09 -> 5 (tag + s32/float)
 tag 0x0a -> 1 + len(string) + 1 (tag + NUL-terminated string)

Tags >= 0x0b are variable / array references handled by EBOOT FUN_0005da7c; the
width is selected by the tag byte's own value:

 0x0b .. 0x6c -> 1 (single inline var index)
 0x6d .. 0xcc -> 3 (var index + 2 selector bytes)
 0xcd .. 0xe5 -> 2 (var index + 1 selector byte)
 0xe6 .. 0xff -> 4 (var index + 3 selector bytes)

Opcodes 0x37a / 0x37b are the variadic "var" ops: their operand count is not in
the command table; instead the operand stream itself terminates on a tag 0x00
sentinel (END_OF_ARGS). Several opcodes have *no* handler (argc ``None``,
kind ``null``) - these never legitimately appear in the live instruction stream.

FILE LAYOUT (MAIN.SCM)::

 0x00 u32 segPtr0
 0x04 u32 segPtr1
 0x08 entry GOTO (op 0x0002, tag 0x06, u32 code_start) - 7 bytes
 ... global variable space (raw, mostly zero) up to code_start
 code_start .. interleaved code blocks and data tables (jump tables,
 NUL strings, 8-byte segment headers, pointer tables)

Because the live interpreter only ever reaches code by *following* GOTOs and
dispatch tables, large data regions sit between code blocks and are never linearly
decodable. ``disassemble`` therefore performs a resynchronising linear walk:
runs of valid instructions become :class:`Instr` records, everything else is
captured verbatim as :class:`Raw` records. Every byte of the file lands in
exactly one record, and each record carries its exact bytes, so
``assemble(disassemble(b)) == b`` is byte-exact by construction - no region is
faked or dropped.
"""
from __future__ import annotations

import json
import os
import struct

# ---------------------------------------------------------------------------
# command table (opcode -> argc / metadata)
# ---------------------------------------------------------------------------
_DATA = os.path.join(os.path.dirname(__file__), "data", "scm_argcounts.json")

with open(_DATA, "r", encoding="utf-8") as _f:
    _RAW_TABLE = json.load(_f)

# opcode index -> argc (int) or None (no handler / variadic)
ARGC = {int(k): v["argc"] for k, v in _RAW_TABLE.items()}
KIND = {int(k): v.get("kind", "") for k, v in _RAW_TABLE.items()}

MAX_OPCODE = max(ARGC)  # 0x55a

GOTO = 0x0002            # opcode used by the entry jump
VAR_OPS = (0x37a, 0x37b)  # variadic operand ops (END_OF_ARGS terminated)
END_OF_ARGS_TAG = 0x00

# total operand size (incl. tag byte) for fixed-width tags < 0x0b.
# tag 0x0a is the NUL-string tag and is handled separately.
TAG_WIDTH = {
    0x00: 1, 0x01: 1, 0x02: 1, 0x03: 2, 0x04: 3,
    0x05: 4, 0x06: 5, 0x07: 2, 0x08: 3, 0x09: 5,
}
STRING_TAG = 0x0a


def _var_width(tag: int) -> int:
    """Total bytes (incl. tag) for a variable/array-ref operand (tag >= 0x0b)."""
    if tag < 0x6d:
        return 1
    if tag < 0xcd:
        return 3
    if tag < 0xe6:
        return 2
    return 4


# ---------------------------------------------------------------------------
# record types
# ---------------------------------------------------------------------------
class Operand:
    """A single decoded operand: tag byte + the raw value bytes that follow it.

 ``raw`` is exactly the bytes the operand occupies *after* the tag, so the
 full on-disk encoding is ``bytes([tag]) + raw``. ``value`` is a convenience
 decode (int for numeric tags, str for string tag, None otherwise).
 """

    __slots__ = ("tag", "raw", "value")

    def __init__(self, tag: int, raw: bytes):
        self.tag = tag
        self.raw = bytes(raw)
        self.value = _decode_value(tag, self.raw)

    def encode(self) -> bytes:
        return bytes([self.tag]) + self.raw

    def __len__(self) -> int:
        return 1 + len(self.raw)

    def __repr__(self) -> str:
        return f"Operand(tag=0x{self.tag:02x}, value={self.value!r})"


def _decode_value(tag: int, raw: bytes):
    if tag in (0x00, 0x01, 0x02):
        return 0
    if tag == 0x03:                       # u8
        return raw[0]
    if tag == 0x04:                       # s16
        return struct.unpack_from("<h", raw, 0)[0]
    if tag == 0x05:                       # s24
        v = raw[0] | (raw[1] << 8) | (raw[2] << 16)
        if v & 0x800000:
            v -= 0x1000000
        return v
    if tag == 0x06:                       # s32 (label / imm32)
        return struct.unpack_from("<i", raw, 0)[0]
    if tag == 0x07:                       # s8
        return struct.unpack_from("<b", raw, 0)[0]
    if tag == 0x08:                       # s16
        return struct.unpack_from("<h", raw, 0)[0]
    if tag == 0x09:                       # s32 / float bit pattern
        return struct.unpack_from("<i", raw, 0)[0]
    if tag == STRING_TAG:                 # NUL-terminated string (NUL stripped)
        return raw[:-1].split(b"\x00", 1)[0].decode("latin-1")
    return None                           # variable / array ref: leave raw


class Instr:
    """One decoded SCM instruction."""

    __slots__ = ("offset", "opcode", "not_flag", "operands")

    def __init__(self, offset, opcode, not_flag, operands):
        self.offset = offset
        self.opcode = opcode          # idx & 0x7fff (0..0x55a)
        self.not_flag = bool(not_flag)
        self.operands = list(operands)

    @property
    def is_data(self) -> bool:
        return False

    def encode(self) -> bytes:
        word = self.opcode | (0x8000 if self.not_flag else 0)
        out = bytearray(struct.pack("<H", word))
        for op in self.operands:
            out += op.encode()
        return bytes(out)

    def __len__(self) -> int:
        return 2 + sum(len(op) for op in self.operands)

    def __repr__(self) -> str:
        n = "!" if self.not_flag else ""
        return (f"Instr(@0x{self.offset:x} {n}op=0x{self.opcode:03x} "
                f"args={self.operands!r})")


class Raw:
    """A verbatim, non-instruction byte span (var space, jump tables, strings,
 segment headers - anything not part of a linearly-decodable code run)."""

    __slots__ = ("offset", "data")

    def __init__(self, offset, data):
        self.offset = offset
        self.data = bytes(data)

    @property
    def is_data(self) -> bool:
        return True

    def encode(self) -> bytes:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Raw(@0x{self.offset:x} len={len(self.data)})"


# ---------------------------------------------------------------------------
# single-instruction decode
# ---------------------------------------------------------------------------
def _decode_instr(buf: bytes, p: int, end: int):
    """Try to decode one instruction at ``p``.

 Returns (Instr, next_p) on success, or None if the bytes at ``p`` do not form
 a valid instruction.
 """
    if p + 2 > end:
        return None
    word = struct.unpack_from("<H", buf, p)[0]
    not_flag = (word & 0x8000) != 0
    idx = word & 0x7fff
    argc = ARGC.get(idx)
    q = p + 2

    if idx in VAR_OPS:
        # variadic: read operands until an END_OF_ARGS (tag 0x00) sentinel.
        operands = []
        while True:
            if q >= end:
                return None
            tag = buf[q]
            ow = _operand_width(buf, q, end)
            if ow is None:
                return None
            operands.append(Operand(tag, buf[q + 1:q + ow]))
            q += ow
            if tag == END_OF_ARGS_TAG:
                break
        return Instr(p, idx, not_flag, operands), q

    if argc is None:
        # no handler -> not a real instruction
        return None

    operands = []
    for _ in range(argc):
        if q >= end:
            return None
        tag = buf[q]
        ow = _operand_width(buf, q, end)
        if ow is None:
            return None
        operands.append(Operand(tag, buf[q + 1:q + ow]))
        q += ow
    return Instr(p, idx, not_flag, operands), q


def _operand_width(buf: bytes, q: int, end: int):
    """Total bytes (incl. tag) of the operand starting at ``q``; None if invalid."""
    tag = buf[q]
    if tag == STRING_TAG:
        e = q + 1
        while e < end and buf[e] != 0:
            e += 1
        if e >= end:                 # unterminated -> reject
            return None
        return (e + 1) - q
    if tag < 0x0b:
        w = TAG_WIDTH[tag]
    else:
        w = _var_width(tag)
    if q + w > end:
        return None
    return w


def _valid_run(buf: bytes, p: int, end: int, k: int) -> bool:
    """True if at least ``k`` consecutive valid instructions start at ``p``."""
    q = p
    for _ in range(k):
        r = _decode_instr(buf, q, end)
        if r is None:
            return False
        q = r[1]
        if q > end:
            return False
    return True


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
# minimum run length required to *trust* a position as real code while
# resynchronising out of a data region. Larger = fewer false-positive decodes
# of data bytes; round-trip is byte-exact regardless of this value.
_RESYNC_RUN = 4


def code_start(scm_bytes: bytes) -> int:
    """Offset where executable code begins, taken from the entry GOTO target.

 Falls back to 8 (just past the two segment pointers) if the header does not
 start with a GOTO.
 """
    if len(scm_bytes) >= 0x0f:
        word = struct.unpack_from("<H", scm_bytes, 8)[0]
        if (word & 0x7fff) == GOTO and scm_bytes[10] == 0x06:
            return struct.unpack_from("<I", scm_bytes, 11)[0]
    return 8


def disassemble(scm_bytes: bytes):
    """Walk ``scm_bytes`` and return an ordered list of records.

 The list always reconstructs the whole file: a leading :class:`Raw` for the
 header + variable space (decoded structurally only as far as the entry GOTO),
 followed by a resynchronising mix of :class:`Instr` and :class:`Raw` records
 for the code body. ``assemble`` is the exact inverse.
 """
    buf = bytes(scm_bytes)
    n = len(buf)
    records = []

    start = code_start(buf)
    if not (8 <= start <= n):
        start = 8

    # Header + variable space: not part of the instruction stream. Decode the
    # entry GOTO structurally if present so callers can see/edit it, but keep the
    # variable space verbatim.
    if start > 0:
        head_end = 0
        # try to peel off the two segment pointers + entry GOTO as structure
        if start >= 0x0f:
            r = _decode_instr(buf, 8, n)
            if r is not None and r[0].opcode == GOTO:
                records.append(Raw(0, buf[0:8]))       # segPtr0, segPtr1
                records.append(r[0])                   # entry GOTO instr
                head_end = r[1]
        if head_end == 0:
            head_end = 0
        if head_end < start:
            records.append(Raw(head_end, buf[head_end:start]))

    # Code body: resynchronising linear walk.
    p = start
    while p < n:
        r = _decode_instr(buf, p, n)
        if r is not None:
            records.append(r[0])
            p = r[1]
            continue
        # not an instruction: collect raw bytes until we resync onto a run of
        # >= _RESYNC_RUN valid instructions (or run out of bytes).
        raw_start = p
        p += 1
        while p < n and not _valid_run(buf, p, n, _RESYNC_RUN):
            p += 1
        records.append(Raw(raw_start, buf[raw_start:p]))

    return records


def assemble(records) -> bytes:
    """Inverse of :func:`disassemble`: re-emit every record's bytes in order."""
    out = bytearray()
    for rec in records:
        out += rec.encode()
    return bytes(out)


# ---------------------------------------------------------------------------
# convenience helpers
# ---------------------------------------------------------------------------
def instructions(records):
    """Yield only the :class:`Instr` records from a disassembly."""
    return [r for r in records if isinstance(r, Instr)]


def data_regions(records):
    """Yield (offset, length) for every :class:`Raw` record."""
    return [(r.offset, len(r.data)) for r in records if isinstance(r, Raw)]


def coverage(records):
    """Return (instr_bytes, raw_bytes, total) for a disassembly."""
    ib = sum(len(r) for r in records if isinstance(r, Instr))
    rb = sum(len(r) for r in records if isinstance(r, Raw))
    return ib, rb, ib + rb
