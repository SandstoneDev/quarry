"""the source game IPL placement decoder (text sections + binary 'bnry' lumps).

An IPL places pre-defined map objects into the world: each placement is one
(modelId, position, quaternion, flagsArea, lod) INST record plus auxiliary
scene data (cull zones, garages, interior enter/exits, pickups, stunt jumps,
timecycle boxes, ambient-audio zones, occluders, car generators). Two physical
encodings, one instance semantics:

 * TEXT IPL - ASCII, '<keyword> .. end' bracketed sections; commas AND
 whitespace both separate fields; '#' starts a comment. (DATA/MAPS/*.ipl)
 * BINARY IPL - FourCC 'bnry', a 0x4C header (6 u32 counts then 6 (offset,size)
 u32 pairs) + a packed 0x28-byte INST array + an optional second (car-gen)
 section. These are the ~164 *_stream*.ipl lumps streamed from gta3.img.

Per-record INST layout is byte-identical across both variants. This module
auto-detects text vs binary by the leading 'bnry' magic and emits a normalized
instance list (positions + per-instance model_id) usable by the SAW table +
scatter preview - fixing the garbage preview of binary stream IPLs.

 (confirmed, with a countryn_stream0 worked
example). Deeper note: 
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# ---- binary layout constants (little-endian) ------------------------------
_MAGIC = b"bnry"
_HDR_LEN = 0x4C            # 6 u32 counts (0x04..0x18) + 6 (offset,size) pairs (0x1C..0x4B)
_INST_STRIDE = 0x28        # 40 bytes per INST record
_AUX_STRIDE = 0x30         # 48 bytes per car-generator (opaque) record
_INST = struct.Struct("<7f3i")   # posXYZ, quat xyzw, modelId, flagsArea, lodId

# flagsArea high-bit flag names (low byte is the area/interior code)
_FLAG_NAMES = {
    0x100: "redundantStream",
    0x200: "dontStream",
    0x400: "underwater",
    0x800: "tunnel",
    0x1000: "tunnelTransition",
}

# Recognized text section keywords. 'mult' is a no-op LOD-multiplier marker but
# is still a valid section opener; 'zone' is legacy (absent in SA map IPLs but
# accepted). Unknown keywords are ignored by the loader.
_SECTIONS = frozenset({
    "inst", "cull", "zone", "grge", "enex", "pick", "jump",
    "tcyc", "auzo", "mult", "occl", "cars", "path",
})


# --------------------------------------------------------------------------- #
# write-back helpers
# --------------------------------------------------------------------------- #
def _fnum(v) -> str:
    """Format a number for text output. repr(float) is the shortest decimal that
 round-trips to the same IEEE-754 double, so re-parsing reproduces the value
 exactly; whole-number floats keep a trailing '.0' for readability."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f) and abs(f) < 1e16:
        return f"{int(f)}.0"
    return repr(f)


def _tok_str(t) -> str:
    """Re-emit a coerced aux token. Ints/floats use _fnum; strings pass through
 verbatim (the re-parse coerces the bare token back to the same value)."""
    if isinstance(t, bool):
        return str(int(t))
    if isinstance(t, (int, float)):
        return _fnum(t)
    return str(t)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Inst:
    """One world placement. `pos` is verbatim; `rot` is the RAW stored quaternion
 (SA stores the CONJUGATE of the world orientation - see display_quat)."""
    model_id: int
    pos: List[float]
    rot: List[float]                 # quaternion (x, y, z, w)
    interior: int = 0                # text 'interior' column (text only)
    area: int = 0                    # flagsArea low byte (binary)
    flags: int = 0                   # flagsArea high bits (binary)
    lod: int = -1                    # LOD-parent index, -1 = none
    name: Optional[str] = None       # model name (text only; binary resolves via IDE)

    def to_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "pos": [float(c) for c in self.pos],
            "rot": [float(c) for c in self.rot],
            "interior": int(self.interior),
            "area": int(self.area),
            "flags": int(self.flags),
            "lod": int(self.lod),
        }
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class Section:
    """A non-inst text section: a list of raw-token rows (lossless passthrough).

 Each row is a list of tokens where numeric tokens are coerced to int/float
 and the rest stay str (so quoted names like "POLICE1" survive)."""
    keyword: str
    rows: List[List[Union[int, float, str]]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def parse_ipl(data: bytes) -> Dict:
    """Parse an IPL (text or binary). Returns a dict:

 { "type": "text"|"binary",
 "inst": [ {model_id, name?, pos:[x,y,z], rot:[x,y,z,w], interior, area,
 flags, lod}, ... ],
 ...text aux sections (cull/zone/grge/enex/pick/jump/tcyc/auzo/occl/cars/
 path/mult) as {section: [ {tokens:[...]}, ... ]} (text only)...,
 ...binary header fields (num_inst/num_aux/offset_inst/offset_aux/
 aux_records) ... (binary only) }
 """
    parsed = _parse_binary(data) if data[:4] == _MAGIC else _parse_text(data)
    # Public API exposes each instance as a plain dict (spec: inst:[{...}]); the
    # Inst dataclass is the internal builder. Materialize here so callers can
    # index r["pos"], r["model_id"], etc. directly.
    parsed["inst"] = [r.to_dict() if isinstance(r, Inst) else r for r in parsed["inst"]]
    return parsed


def to_json(parsed: Dict) -> Dict:
    """Return a fully JSON-serializable view of `parsed` (lists/dicts/str/int/
 float/bool only). The opaque binary aux blob is hex-encoded so no raw bytes
 leak to the UI."""
    out: Dict = {"type": parsed["type"], "inst": [_inst_json(r) for r in parsed["inst"]]}
    if parsed["type"] == "binary":
        out["num_inst"] = int(parsed["num_inst"])
        out["num_aux"] = int(parsed["num_aux"])
        out["offset_inst"] = int(parsed["offset_inst"])
        out["offset_aux"] = int(parsed["offset_aux"])
        aux = parsed.get("aux_records") or b""
        out["aux_records_hex"] = bytes(aux).hex()
        out["aux_records_len"] = len(aux)
    else:
        for kw in parsed.get("sections", []):
            out[kw] = [{"tokens": list(row["tokens"])} for row in parsed[kw]]
    return out


def write_ipl(parsed: Dict) -> bytes:
    """Serialize a parsed IPL (the dict from parse_ipl) back to bytes.

 * type == "text" -> rebuild the ASCII IPL and return UTF-8 bytes.
 * type == "binary" -> rebuild the 'bnry' lump (0x4C header + 0x28 INST array +
 verbatim car-gen aux blob). Byte-identical to the on-disk lump for the whole
 retail gta3.img corpus (see write_ipl_binary for the exact record layout).
 """
    if parsed.get("type") == "binary":
        return write_ipl_binary(parsed)
    return write_ipl_text(parsed).encode("utf-8")


def write_ipl_text(parsed: Dict) -> str:
    """Rebuild the text form of an IPL (inverse of _parse_text / _parse_inst_row).

 inst rows: ``id, name, interior, x, y, z, rx, ry, rz, rw, lod`` (11 fields), then
 each passthrough aux section (cull/grge/enex/.../mult) re-joined from its coerced
 token list, in the order they first appeared (`parsed["sections"]`). Token values
 round-trip exactly; quotes the engine stripped from names are not re-added (the
 re-parse coerces the bare token back to the same string)."""
    out: List[str] = []
    out.append("inst\n")
    for r in parsed.get("inst", []):
        d = r.to_dict() if isinstance(r, Inst) else r
        name = d.get("name")
        if name is None:
            name = "dummy"
        px, py, pz = d["pos"]
        rx, ry, rz, rw = d["rot"]
        out.append(
            f"{d['model_id']}, {name}, {d.get('interior', 0)}, "
            f"{_fnum(px)}, {_fnum(py)}, {_fnum(pz)}, "
            f"{_fnum(rx)}, {_fnum(ry)}, {_fnum(rz)}, {_fnum(rw)}, "
            f"{d.get('lod', -1)}\n"
        )
    out.append("end\n")

    for kw in parsed.get("sections", []):
        rows = parsed.get(kw, [])
        out.append(f"{kw}\n")
        for row in rows:
            toks = row["tokens"] if isinstance(row, dict) else row
            out.append(", ".join(_tok_str(t) for t in toks) + "\n")
        out.append("end\n")
    return "".join(out)


def write_ipl_binary(parsed: Dict) -> bytes:
    """Rebuild a 'bnry' lump. Exact byte layout emitted (little-endian):

 0x00 char[4] magic 'bnry'
 0x04 u32[6] counts = [numInst, 0, 0, 0, numCars(aux), 0]
 0x1C u32[12] six (offset,size) pairs; only:
 pair0 = (0x4C, 0) -> INST array
 pair4 = (0x4C + numInst*0x28, 0) -> car-gen array
 (written only when numCars > 0)
 every other pair is (0, 0). Sizes are always 0, matching retail.
 0x4C INST[numInst], stride 0x28 each:
 f32 posX, posY, posZ
 f32 rotX, rotY, rotZ, rotW (raw stored quaternion)
 s32 modelId
 s32 flagsArea = (flags & ~0xFF) | (area & 0xFF)
 s32 lod
 car-gen blob (numCars * 0x30 bytes) appended verbatim from aux_records.

 This reproduces the on-disk header exactly (the retail lumps store the aux offset
 in pair slot 4, not slot 5; parse_ipl recomputes it, so the round trip is stable).
 """
    insts = parsed.get("inst", [])
    num_inst = len(insts)
    aux = bytes(parsed.get("aux_records") or b"")
    # num_aux: prefer the recorded header count, else derive from the blob length.
    num_aux = int(parsed.get("num_aux", len(aux) // _AUX_STRIDE))

    offset_inst = _HDR_LEN
    offset_aux = offset_inst + num_inst * _INST_STRIDE

    counts = [num_inst, 0, 0, 0, num_aux, 0]
    pairs = [0] * 12
    pairs[0] = offset_inst            # pair slot 0 offset -> INST array
    if num_aux > 0:
        pairs[8] = offset_aux         # pair slot 4 offset -> car-generator array

    buf = bytearray()
    buf += _MAGIC
    buf += struct.pack("<6I", *counts)
    buf += struct.pack("<12I", *pairs)

    for r in insts:
        d = r.to_dict() if isinstance(r, Inst) else r
        px, py, pz = d["pos"]
        rx, ry, rz, rw = d["rot"]
        flags_area = (int(d.get("flags", 0)) & ~0xFF) | (int(d.get("area", 0)) & 0xFF)
        buf += _INST.pack(
            float(px), float(py), float(pz),
            float(rx), float(ry), float(rz), float(rw),
            int(d["model_id"]), flags_area, int(d.get("lod", -1)),
        )

    buf += aux
    return bytes(buf)


def display_quat(rot, flags: int = 0):
    """Replicate CFileLoader::CreateEntityFromInstance (0x549f10) rotation: SA
 stores the CONJUGATE of the world orientation. Returns the display quaternion
 (x, y, z, w) for rendering/preview.

 Fast path (yaw-only, |x|<=0.05 and |y|<=0.05 and not the 0x200 full path):
 rebuild from heading = (z>=0 ? -2 : +2)*acos(w) about Z. Otherwise negate the
 xyz vector part (keep w). A naive quat->matrix from the RAW quat is mirrored.
 """
    qx, qy, qz, qw = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
    full_path_flag = bool(flags & 0x200) and qx != 0.0 and qy != 0.0
    if abs(qx) <= 0.05 and abs(qy) <= 0.05 and not full_path_flag:
        w = max(-1.0, min(1.0, qw))
        angle = math.acos(w)
        heading = (-2.0 * angle) if qz >= 0.0 else (2.0 * angle)
        half = heading * 0.5
        return [0.0, 0.0, math.sin(half), math.cos(half)]
    return [-qx, -qy, -qz, qw]


# --------------------------------------------------------------------------- #
# binary path
# --------------------------------------------------------------------------- #
def _parse_binary(data: bytes) -> Dict:
    if len(data) < _HDR_LEN:
        raise ValueError(f"binary IPL too short for 0x4C header ({len(data)} bytes)")

    counts = struct.unpack_from("<6I", data, 0x04)   # numInst, unk1..3, numAux2, unk4
    pairs = struct.unpack_from("<12I", data, 0x1C)   # (offset,size) x6
    num_inst = counts[0]
    num_aux = counts[4]
    offset_inst = pairs[0]
    offset_aux = pairs[10]

    # The loader reads numInst/numAux as s16 (always <= 0x7FFF). On disk
    # offsetInst is ALWAYS 0x4C; tolerate a stray 0 (some crafted lumps) by
    # falling back to the spec-fixed location.
    if offset_inst == 0:
        offset_inst = _HDR_LEN

    insts: List[Inst] = []
    base = offset_inst
    for i in range(num_inst):
        off = base + i * _INST_STRIDE
        if off + _INST_STRIDE > len(data):
            break  # truncated tail: keep what decoded, don't kill the file
        try:
            insts.append(_decode_bin_record(data, off))
        except Exception:
            # one malformed record must not sink the lump
            continue

    # If offsetAux looks bogus (0 with aux present), recompute the invariant.
    if num_aux > 0 and offset_aux == 0:
        offset_aux = offset_inst + num_inst * _INST_STRIDE

    aux_records = b""
    if num_aux > 0:
        end = offset_aux + num_aux * _AUX_STRIDE
        aux_records = data[offset_aux:end]   # opaque car-gen blob, kept verbatim

    return {
        "type": "binary",
        "inst": insts,
        "num_inst": num_inst,
        "num_aux": num_aux,
        "offset_inst": offset_inst,
        "offset_aux": offset_aux,
        "aux_records": aux_records,
    }


def _decode_bin_record(buf: bytes, off: int) -> Inst:
    (px, py, pz, rx, ry, rz, rw, model_id, flags_area, lod) = _INST.unpack_from(buf, off)
    return Inst(
        model_id=model_id,
        pos=[px, py, pz],
        rot=[rx, ry, rz, rw],
        interior=flags_area & 0xFF,          # binary folds interior into the area byte
        area=flags_area & 0xFF,
        flags=flags_area & ~0xFF,
        lod=lod,
        name=None,
    )


# --------------------------------------------------------------------------- #
# text path
# --------------------------------------------------------------------------- #
def _tokenize_line(line: str) -> List[str]:
    """Tokenizer per CFileLoader::LoadLine (0x548eb0): replace every control char
 (<0x20) AND every ',' with a space, then split on whitespace runs. Quotes are
 literal characters (enex names ship quoted)."""
    out_chars = []
    for ch in line:
        if ch == "," or ord(ch) < 0x20:
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    return "".join(out_chars).split()


def _coerce(tok: str) -> Union[int, float, str]:
    """Coerce a token to int or float when it is purely numeric, else keep str
 (with surrounding double-quotes stripped, as the engine does for names)."""
    s = tok
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    # int?
    try:
        return int(s)
    except ValueError:
        pass
    # float?
    try:
        return float(s)
    except ValueError:
        return s


def _parse_text(data: bytes) -> Dict:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    result: Dict = {"type": "text", "inst": []}
    sections: List[str] = []          # ordered non-inst section keywords seen
    aux: Dict[str, List[dict]] = {}
    current: Optional[str] = None

    for raw in text.splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped[0] == "#":
            continue
        toks = _tokenize_line(raw)
        if not toks:
            continue
        head = toks[0].lower()

        if current is None:
            # expecting a section keyword
            if head in _SECTIONS:
                current = head
            # any other bare token outside a section is ignored
            continue

        if head == "end":
            current = None
            continue

        if current == "inst":
            inst = _parse_inst_row(toks)
            if inst is not None:
                result["inst"].append(inst)
        else:
            row = {"tokens": [_coerce(t) for t in toks]}
            if current not in aux:
                aux[current] = []
                sections.append(current)
            aux[current].append(row)

    for kw, rows in aux.items():
        result[kw] = rows
    result["sections"] = sections
    return result


def _parse_inst_row(toks: List[str]) -> Optional[Inst]:
    """inst row: id, modelName, interior, posX,posY,posZ, rotX,rotY,rotZ,rotW, lod
 (11 tokens). Defensive: a malformed row is skipped, not fatal."""
    if len(toks) < 11:
        return None
    try:
        model_id = int(toks[0])
        name = toks[1]
        interior = int(toks[2])
        pos = [float(toks[3]), float(toks[4]), float(toks[5])]
        rot = [float(toks[6]), float(toks[7]), float(toks[8]), float(toks[9])]
        lod = int(toks[10])
    except (ValueError, IndexError):
        return None
    return Inst(
        model_id=model_id,
        pos=pos,
        rot=rot,
        interior=interior,
        area=interior & 0xFF,
        flags=interior & ~0xFF,
        lod=lod,
        name=name,
    )


# --------------------------------------------------------------------------- #
# json helpers
# --------------------------------------------------------------------------- #
def _inst_json(r) -> dict:
    if isinstance(r, Inst):
        return r.to_dict()
    # already a dict (defensive)
    return r


def flag_names(flags: int) -> List[str]:
    """Decode flagsArea high bits to a list of flag names (UI convenience)."""
    return [name for bit, name in _FLAG_NAMES.items() if flags & bit]
