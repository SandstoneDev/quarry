"""the source game SCM script-bytecode reader (main.scm + CLEO).

main.scm is the compiled program that drives every mission and most ambient logic.
It is NOT linearly decodable: it opens with a chain of GOTO chunk headers that jump
over data segments (globals image, used-object list, mission table, streamed-script
directory, extra-info), and code is only reached structurally past the chain. The
same flat buffer holds both code and the global-variable image (a global is
addressed as base + byte-offset).

CHUNK CHAIN (each header is 8 bytes, #pragma pack(1)):
 u8[3] InstrGoTo = {0x02,0x00,0x01} (GOTO opcode + imm32 type byte)
 u32 NextChunkOffset (absolute file offset of the NEXT header)
 u8 ChunkIndex (chunk-type id; NOT padding)
Payload spans [header+8 : NextChunkOffset]. The walk follows NextChunkOffset until
the byte at the next offset is no longer a GOTO (0x02) -> that offset is the start of
the resident MAIN code block. Segment indices: 's'(0x73)=GlobalVarSpace, 0=UsedObjects,
1=ScriptFileInfo (mission table), 2=StreamedScriptFileInfo, 3=Unknown, 4=ExtraInfo.

OPCODE STREAM: word = u16 LE; opcode = word & 0x7FFF; NOT-flag = word >> 15.
Operand count is hard-coded per handler and NOT on disk, but operands are
self-delimiting: each begins with a 1-byte TYPE TAG then a fixed-width payload, so a
generic disassembler advances IP correctly by greedily consuming operands while the
next byte is a valid tag (0x00..0x13) and treating the first non-tag byte as the next
opcode word. This resyncs exactly on every missionScriptOffsets[n] for retail main.scm.

SA OPERAND TAGS: 0x00 end-of-args, 0x01 imm32, 0x02 global, 0x03 local, 0x04 imm8,
0x05 imm16, 0x06 float32 (IEEE-754), 0x07/0x08 number arrays (6B ArrayAccess),
0x09 static 8-byte string, 0x0A/0x0B short-string var, 0x0C/0x0D short-string arrays,
0x0E pascal string (u8 len + bytes), 0x0F static 16-byte string, 0x10/0x11 long-string
var, 0x12/0x13 long-string arrays.

CLEO (.cm/.cs/.csa/.csi) is the SAME opcode stream WITHOUT the 5-segment header, so
``disassemble(body, 0)`` decodes a CLEO script directly (no parse_scm needed).

PC v1.0 NOTE: the retail UsedObjects record is a bare 24-byte name (no trailing id
dword); the 28-byte stride in the spec came from a PS2 mod. The stride is derived from
the chunk payload size so both layouts decode.

 (confirmed chunk chain + operand tag table)
Loaders: CTheScripts::Init 0x46e440, CRunningScript::Process 0x46f670,
 CollectParameters 0x469790, ReadObjectNamesFromScript 0x476850.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Chunk header
_GOTO_SIG = b"\x02\x00\x01"          # GOTO opcode (0x0002) + imm32 type byte (0x01)
_HDR_LEN = 8                          # u8[3] sig + u32 next + u8 index

# Chunk indices
_CHUNK_GLOBALS = 0x73                 # 's'
_CHUNK_USED_OBJECTS = 0
_CHUNK_SCRIPT_INFO = 1
_CHUNK_STREAMED = 2
_CHUNK_UNKNOWN3 = 3
_CHUNK_EXTRA_INFO = 4

# Operand type tags -> (json type name, payload byte width). Width None = variable.
_TAG_END = 0x00
_TAG_INFO = {
    0x00: ("end", 0),
    0x01: ("imm32", 4),
    0x02: ("global", 2),
    0x03: ("local", 2),
    0x04: ("imm8", 1),
    0x05: ("imm16", 2),
    0x06: ("float", 4),
    0x07: ("garray", 6),     # GLOBAL_NUMBER_ARRAY (ArrayAccess)
    0x08: ("larray", 6),     # LOCAL_NUMBER_ARRAY
    0x09: ("str8", 8),       # STATIC_SHORT_STRING
    0x0A: ("gstr_var", 2),   # GLOBAL_SHORT_STRING_VAR
    0x0B: ("lstr_var", 2),   # LOCAL_SHORT_STRING_VAR
    0x0C: ("gstr_arr", 6),
    0x0D: ("lstr_arr", 6),
    0x0E: ("pstr", None),    # STATIC_PASCAL_STRING: u8 len + bytes
    0x0F: ("str16", 16),     # STATIC_LONG_STRING
    0x10: ("glstr_var", 2),  # GLOBAL_LONG_STRING_VAR
    0x11: ("llstr_var", 2),  # LOCAL_LONG_STRING_VAR
    0x12: ("glstr_arr", 6),
    0x13: ("llstr_arr", 6),
}
_VALID_TAGS = frozenset(_TAG_INFO)


def _cstr(buf: bytes, off: int, n: int) -> str:
    return buf[off:off + n].split(b"\x00", 1)[0].decode("latin-1")


# --------------------------------------------------------------------------- #
# Chunk chain
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    index: int
    header_offset: int
    payload_start: int
    payload_end: int

    @property
    def payload_size(self) -> int:
        return max(0, self.payload_end - self.payload_start)


def _walk_chunks(data: bytes) -> List[Chunk]:
    """Follow the GOTO chunk chain from offset 0.

 Stops once the byte at the next offset is no longer a GOTO (0x02) or the offset
 leaves the buffer; that next offset is the start of the resident MAIN code.
 """
    chunks: List[Chunk] = []
    hop = 0
    n = len(data)
    for _ in range(8):  # retail has 6; cap to avoid runaway on junk
        if hop + _HDR_LEN > n or data[hop:hop + 3] != _GOTO_SIG:
            break
        nxt = struct.unpack_from("<I", data, hop + 3)[0]
        idx = data[hop + 7]
        end = nxt if (hop + _HDR_LEN) <= nxt <= n else n
        chunks.append(Chunk(idx, hop, hop + _HDR_LEN, end))
        if not (hop + _HDR_LEN) <= nxt < n:
            break
        hop = nxt
        if data[hop:hop + 3] != _GOTO_SIG:  # reached the code block
            break
    return chunks


def _code_start(data: bytes, chunks: List[Chunk]) -> int:
    """File offset where the resident MAIN code begins (after the last header)."""
    if not chunks:
        return 0
    return chunks[-1].payload_end


# --------------------------------------------------------------------------- #
# Segment decoders (each defensively wrapped at the call site)
# --------------------------------------------------------------------------- #

def _decode_used_objects(data: bytes, ch: Chunk) -> List[str]:
    """UsedObjects (idx 0): u32 count then `count` name records.

 Retail PC uses a bare 24-byte name record (no id dword); some mods use 28. The
 stride is derived from the chunk payload so both decode; a record begins with a
 NUL-terminated ASCII name (first record is an all-zero placeholder in retail).
 """
    p = ch.payload_start
    if p + 4 > len(data):
        return []
    count = struct.unpack_from("<I", data, p)[0]
    p += 4
    if count <= 0 or count > 1_000_000:
        return []
    avail = ch.payload_end - p
    stride = avail // count if count else 0
    if stride < 20:                      # not enough room for a 20-char name field
        stride = max(stride, 20)
    names: List[str] = []
    for i in range(count):
        off = p + i * stride
        if off + 20 > len(data):
            break
        try:
            names.append(_cstr(data, off, min(stride, 24)))
        except Exception:
            names.append("")
    return names


def _decode_script_info(data: bytes, ch: Chunk) -> Dict:
    """ScriptFileInfo (idx 1): the mission table.

 Header: u32 mainScriptSize, u32 largestMissionScriptSize, u16 numExclusiveMissions,
 u16 highestLocalVarUsed, u32 <count field>, then the mission offsets array. The
 array length is derived from the chunk payload size (== numExclusiveMissions for
 retail) so we never trust the trailing dword blindly. offsets[0] == mainScriptSize.
 """
    p = ch.payload_start
    info = {"main_size": 0, "largest_mission_size": 0, "num_exclusive_missions": 0,
            "highest_local_var": 0, "count": 0, "offsets": []}
    if p + 16 > len(data):
        return info
    main_size, largest = struct.unpack_from("<II", data, p)
    n_excl, hi_local = struct.unpack_from("<HH", data, p + 8)
    raw_count = struct.unpack_from("<I", data, p + 12)[0]
    arr_start = p + 16
    # Derive offset count from payload bytes; this is the authoritative length.
    derived = max(0, (ch.payload_end - arr_start) // 4)
    count = derived if derived else min(raw_count, 0xFFFF)
    offsets: List[int] = []
    for i in range(count):
        o = arr_start + i * 4
        if o + 4 > len(data):
            break
        offsets.append(struct.unpack_from("<I", data, o)[0])
    info.update(main_size=main_size, largest_mission_size=largest,
                num_exclusive_missions=n_excl, highest_local_var=hi_local,
                count=len(offsets), offsets=offsets)
    return info


def _decode_streamed(data: bytes, ch: Chunk) -> List[Dict]:
    """StreamedScriptFileInfo (idx 2): u32 largest, u32 count, then 28-byte records
 {char name[20]; u32 fileOffset; u32 size}."""
    p = ch.payload_start
    if p + 8 > len(data):
        return []
    _largest, count = struct.unpack_from("<II", data, p)
    p += 8
    if count <= 0 or count > 1_000_000:
        return []
    recs: List[Dict] = []
    for i in range(count):
        off = p + i * 28
        if off + 28 > len(data):
            break
        try:
            name = _cstr(data, off, 20)
            file_off, size = struct.unpack_from("<II", data, off + 20)
            recs.append({"name": name, "offset": file_off, "size": size})
        except Exception:
            continue
    return recs


def _decode_extra_info(data: bytes, ch: Chunk) -> Dict:
    """ExtraInfo (idx 4): u32 globalVarSpaceSize, u32 buildNumber."""
    p = ch.payload_start
    if p + 8 > len(data):
        return {"global_var_space_size": 0, "build_number": 0}
    gss, build = struct.unpack_from("<II", data, p)
    return {"global_var_space_size": gss, "build_number": build}


# --------------------------------------------------------------------------- #
# Public: parse_scm
# --------------------------------------------------------------------------- #

def parse_scm(data: bytes) -> Dict:
    """Parse a main.scm header into a JSON-serializable dict.

 Returns:
 {
 globals_size, globals_offset, main_offset,
 models: [str, ...], # UsedObjects names
 missions: {count, offsets:[int], main_size, largest_mission_size,
 num_exclusive_missions, highest_local_var},
 streamed_scripts: [{name, offset, size}, ...],
 global_var_space_size, build_number,
 chunks: [{index, header_offset, payload_start, payload_end}, ...],
 }

 Every value is a list/dict/str/int - directly handed to the web UI.
 """
    chunks = _walk_chunks(data)
    by_idx: Dict[int, Chunk] = {}
    for c in chunks:
        by_idx.setdefault(c.index, c)

    # GlobalVarSpace is the FIRST chunk's payload (its index byte is 's' at hdr0+7).
    globals_offset = chunks[0].payload_start if chunks else _HDR_LEN
    globals_end = chunks[0].payload_end if chunks else _HDR_LEN
    globals_size = max(0, globals_end - globals_offset)

    models: List[str] = []
    if _CHUNK_USED_OBJECTS in by_idx:
        try:
            models = _decode_used_objects(data, by_idx[_CHUNK_USED_OBJECTS])
        except Exception:
            models = []

    missions = {"count": 0, "offsets": [], "main_size": 0, "largest_mission_size": 0,
                "num_exclusive_missions": 0, "highest_local_var": 0}
    if _CHUNK_SCRIPT_INFO in by_idx:
        try:
            missions = _decode_script_info(data, by_idx[_CHUNK_SCRIPT_INFO])
        except Exception:
            pass

    streamed: List[Dict] = []
    if _CHUNK_STREAMED in by_idx:
        try:
            streamed = _decode_streamed(data, by_idx[_CHUNK_STREAMED])
        except Exception:
            streamed = []

    extra = {"global_var_space_size": 0, "build_number": 0}
    if _CHUNK_EXTRA_INFO in by_idx:
        try:
            extra = _decode_extra_info(data, by_idx[_CHUNK_EXTRA_INFO])
        except Exception:
            pass

    return {
        "globals_size": globals_size,
        "globals_offset": globals_offset,
        "main_offset": _code_start(data, chunks),
        "models": models,
        "missions": missions,
        "streamed_scripts": streamed,
        "global_var_space_size": extra["global_var_space_size"],
        "build_number": extra["build_number"],
        "chunks": [
            {"index": c.index, "header_offset": c.header_offset,
             "payload_start": c.payload_start, "payload_end": c.payload_end}
            for c in chunks
        ],
    }


# --------------------------------------------------------------------------- #
# Public: disassemble
# --------------------------------------------------------------------------- #

def _decode_operand(data: bytes, ip: int) -> Optional[Dict]:
    """Decode ONE operand at ip. Returns (node, new_ip) packed in node['_next'],
 or None if the byte is not a valid type tag (i.e. the next opcode word)."""
    tag = data[ip]
    if tag not in _VALID_TAGS:
        return None
    tname, width = _TAG_INFO[tag]
    start = ip + 1
    node = {"type": tname}

    if tag == _TAG_END:
        node["_next"] = start
        return node

    if tag == 0x01:                                   # imm32
        if start + 4 > len(data):
            return None
        node["value"] = struct.unpack_from("<i", data, start)[0]
        node["_next"] = start + 4
    elif tag == 0x04:                                 # imm8 (sign-extended)
        if start + 1 > len(data):
            return None
        node["value"] = struct.unpack_from("<b", data, start)[0]
        node["_next"] = start + 1
    elif tag == 0x05:                                 # imm16 (sign-extended)
        if start + 2 > len(data):
            return None
        node["value"] = struct.unpack_from("<h", data, start)[0]
        node["_next"] = start + 2
    elif tag == 0x06:                                 # float32
        if start + 4 > len(data):
            return None
        f = struct.unpack_from("<f", data, start)[0]
        # NaN/inf would break a strict (allow_nan=False) JSON encoder -> coerce to None
        node["value"] = f if (f == f and f not in (float("inf"), float("-inf"))) else None
        node["_next"] = start + 4
    elif tag in (0x02, 0x03, 0x0A, 0x0B, 0x10, 0x11):  # var refs: u16
        if start + 2 > len(data):
            return None
        node["value"] = struct.unpack_from("<H", data, start)[0]
        node["_next"] = start + 2
    elif tag in (0x07, 0x08, 0x0C, 0x0D, 0x12, 0x13):  # ArrayAccess (6 bytes)
        if start + 6 > len(data):
            return None
        base, idx_var = struct.unpack_from("<HH", data, start)
        arr_size = data[start + 4]
        bits = data[start + 5]
        node["array"] = {
            "base": base, "index_var": idx_var, "size": arr_size,
            "element_type": bits & 0x7F, "index_is_global": bool(bits & 0x80),
        }
        node["_next"] = start + 6
    elif tag == 0x09:                                 # static 8-byte string
        if start + 8 > len(data):
            return None
        node["value"] = _cstr(data, start, 8)
        node["_next"] = start + 8
    elif tag == 0x0F:                                 # static 16-byte string
        if start + 16 > len(data):
            return None
        node["value"] = _cstr(data, start, 16)
        node["_next"] = start + 16
    elif tag == 0x0E:                                 # pascal: u8 len + bytes
        if start + 1 > len(data):
            return None
        slen = data[start]
        if start + 1 + slen > len(data):
            return None
        node["value"] = _cstr(data, start + 1, slen)
        node["_next"] = start + 1 + slen
    else:
        # known-width fallback (shouldn't hit; widths covered above)
        if width is None or start + width > len(data):
            return None
        node["_next"] = start + width
    return node


def disassemble(data: bytes, start: int, limit: int = 2000,
                end: Optional[int] = None) -> List[Dict]:
    """Disassemble the opcode stream beginning at ``start``.

 Returns a list of {offset, opcode, not_flag, args, size}. ``opcode`` is numeric
 (word & 0x7FFF); ``not_flag`` is bit15. Operands are decoded by their SA type tag
 and consumed greedily: the walker reads operands while the next byte is a valid
 tag (0x00..0x13) and stops at the first non-tag byte, which is the next opcode.
 This advances IP correctly and resyncs exactly on every mission boundary, so when
 a code region's ``end`` is given the final instruction ends exactly on it.

 ``end`` bounds the region (defaults to len(data)); decoding stops before reading an
 opcode at or beyond it. Works for a CLEO body too (``start=0``) since CLEO is the
 header-less opcode stream. Stops at ``limit`` instructions or when it cannot advance.
 """
    insns: List[Dict] = []
    ip = start
    n = len(data) if end is None else min(end, len(data))
    while ip + 2 <= n and len(insns) < limit:
        word = struct.unpack_from("<H", data, ip)[0]
        opcode = word & 0x7FFF
        not_flag = bool(word >> 15)
        cur = ip + 2
        args: List[Dict] = []
        while cur < n:
            node = _decode_operand(data, cur)
            if node is None:
                break
            nxt = node.pop("_next")
            if nxt <= cur:                # guard against zero-width stalls
                cur = nxt
                if node["type"] == "end":
                    break
                break
            cur = nxt
            is_end = node["type"] == "end"
            args.append(node)
            if is_end:                    # variadic terminator ends this instruction
                break
        insns.append({
            "offset": ip,
            "opcode": opcode,
            "not_flag": not_flag,
            "args": args,
            "size": cur - ip,
        })
        if cur <= ip:                     # never make progress -> bail
            break
        ip = cur
    return insns


# --------------------------------------------------------------------------- #
# Public: to_json
# --------------------------------------------------------------------------- #

def to_json(data: bytes, disasm_limit: int = 2000) -> Dict:
    """Full JSON view for the web UI: header dict + a disassembly slice of MAIN.

 Everything returned is JSON-serializable (lists/dicts/str/int/float/bool). The
 ``disassembly`` slice covers the start of the resident MAIN code (capped by
 ``disasm_limit`` instructions for a responsive preview).
 """
    head = parse_scm(data)
    head["disassembly"] = disassemble(data, head["main_offset"], limit=disasm_limit)
    return head
