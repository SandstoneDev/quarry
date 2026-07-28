"""the source game IDE item-definition decoder (ASCII sections).

Maps a streaming model id to its DFF/TXD names, draw distance(s), flags, and the
per-category metadata read by CFileLoader::LoadObjectTypes (0x5d4d60): objs/tobj
(CSimpleModelInfo / CTimeModelInfo), anim/hier (CClumpModelInfo), cars
(CVehicleModelInfo), peds (CPedModelInfo), weap, txdp, 2dfx, path.

The shared tokenizer CFileLoader::LoadLine (0x548eb0) turns every comma AND every
control byte (<0x20) into a space, then strips leading whitespace - so commas and
runs of whitespace are equivalent field separators and field parsing is just
str.split(). objs/tobj rows auto-detect a single-mesh short form vs an explicit
"meshCount + N LOD draw distances" long form (the 4.0 drawDist threshold gate).

 (confirmed per-section byte/token layout)
Deeper: (section/row tables)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union

# Section keywords -> internal section id (LoadObjectTypes 0x5d4d60 dispatch).
KEYWORDS = {
    "objs": 1, "tobj": 3, "weap": 4, "hier": 5, "anim": 6,
    "cars": 7, "peds": 8, "path": 9, "2dfx": 10, "txdp": 0xB,
}

# objs/tobj PASS A drawDist threshold (the literal 4.0 constant in 0x5d0210).
_DRAW_DIST_MIN = 4.0

# cars `type` -> vehicle-class enum (+0x3c).
VEHICLE_TYPE_ENUM = {
    "car": 0, "mtruck": 1, "quad": 2, "heli": 3, "plane": 4, "boat": 5,
    "train": 6, "f_heli": 3, "f_plane": 8, "bike": 9, "bmx": 10, "trailer": 0xB,
}
# cars `class` -> spawn-group enum (+0x4d).
VEHICLE_CLASS_ENUM = {
    "normal": 0, "poorfamily": 1, "richfamily": 2, "executive": 3, "worker": 4,
    "big": 5, "taxi": 6, "moped": 7, "motorbike": 8, "leisureboat": 9,
    "workerboat": 10, "bicycle": 0xB, "ignore": 0xFF,
}

# IDE flags word bits (applied by 0x5d00d0 / anim's 0x5d0070 into modelinfo+0x12).
_FLAG_BITS = [
    (0x00000001, "wet_road"),
    (0x00000004, "draw_additive"),
    (0x00000200, "no_zwrite"),
    (0x00000400, "dont_cull"),
    (0x00000800, "is_door"),
    (0x00001000, "alt_alloc"),
    (0x00002000, "group_2000"),
    (0x00004000, "group_4000"),
    (0x00008000, "breakable"),
    (0x00080000, "group_80000"),
    (0x00100000, "interior_object"),
    (0x00200000, "no_backface_cull"),
    (0x00400000, "group_400000"),
]


# --------------------------- low-level helpers ---------------------------

def _normalize_line(line: str) -> str:
    """Reproduce LoadLine 0x548eb0: every char < 0x20 and every ',' -> space, then
 lstrip. Returns the whitespace-canonical line (commas already collapsed)."""
    out = []
    for ch in line:
        o = ord(ch)
        if o < 0x20 or ch == ",":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out).lstrip()


def _to_int(tok: str, base: int = 10, default: int = 0) -> int:
    """sscanf %d / %x semantics: read the leading integer, tolerate junk tails."""
    try:
        return int(tok, base)
    except (ValueError, TypeError):
        # sscanf would consume a leading sign+digits prefix; emulate loosely.
        s = tok.strip()
        sign = 1
        if s[:1] in "+-":
            sign = -1 if s[0] == "-" else 1
            s = s[1:]
        digits = "0123456789abcdefABCDEF" if base == 16 else "0123456789"
        acc = ""
        for c in s:
            if c in digits:
                acc += c
            else:
                break
        if acc:
            try:
                return sign * int(acc, base)
            except ValueError:
                pass
        return default


def _to_float(tok: str, default: float = 0.0) -> float:
    try:
        return float(tok)
    except (ValueError, TypeError):
        return default


def decode_flags(flags: int) -> Dict[str, bool]:
    """Decode the raw IDE flags int into named booleans (raw value kept verbatim
 by the caller; bits decoded lazily here)."""
    return {name: bool(flags & bit) for bit, name in _FLAG_BITS}


# --------------------------- section row dataclasses ---------------------------

@dataclass
class ObjsRow:
    id: int
    dff: str
    txd: str
    mesh_count: int
    draw_dist: float
    lod_dists: List[float]
    flags: int
    flags_decoded: Dict[str, bool]
    section: str = "objs"


@dataclass
class TobjRow:
    id: int
    dff: str
    txd: str
    mesh_count: int
    draw_dist: float
    lod_dists: List[float]
    flags: int
    flags_decoded: Dict[str, bool]
    time_on: int
    time_off: int
    section: str = "tobj"


@dataclass
class AnimRow:
    id: int
    dff: str
    txd: str
    anim: str
    draw_dist: float
    flags: int
    flags_decoded: Dict[str, bool]
    has_anim: bool
    section: str = "anim"


@dataclass
class WeapRow:
    id: int
    dff: str
    txd: str
    anim: str
    mesh_count: int
    draw_dist: float
    section: str = "weap"


@dataclass
class HierRow:
    id: int
    dff: str
    txd: str
    anim: Optional[str]
    draw_dist: Optional[float]
    section: str = "hier"


@dataclass
class CarsRow:
    id: int
    dff: str
    txd: str
    type: str
    type_id: int
    handling: str
    game_name: str
    anim_group: str
    vehicle_class: str
    vehicle_class_id: int
    frequency: int
    level: int
    comp_rules: int
    wheel_id: int
    wheel_scale: float
    wheel_scale_rear: Optional[float]
    upgrades: Optional[int]
    section: str = "cars"


@dataclass
class PedsRow:
    id: int
    dff: str
    txd: str
    ped_type: str
    stat_name: str
    anim_group: str
    cars_can_drive: int
    flags2: int
    anim_file: str
    radio1: int
    radio2: int
    voices: List[str]
    section: str = "peds"


@dataclass
class TxdpRow:
    child: str
    parent: str
    id: None = None
    section: str = "txdp"


@dataclass
class FxRow:
    id: int
    x: float
    y: float
    z: float
    effect_type: int
    tail: List[str]
    section: str = "2dfx"


@dataclass
class RawRow:
    """Fallback for path rows / unparseable lines: keep the raw tokens."""
    tokens: List[str]
    id: Optional[int] = None
    section: str = "path"


# --------------------------- per-section parsers ---------------------------

def _parse_objs(tok: List[str]) -> ObjsRow:
    rid = _to_int(tok[0])
    dff = tok[1] if len(tok) > 1 else ""
    txd = tok[2] if len(tok) > 2 else ""
    # PASS A: id name txd drawDist flags (exactly 5 tokens AND drawDist >= 4.0)
    pass_a = False
    if len(tok) == 5:
        d = _to_float(tok[3])
        if d >= _DRAW_DIST_MIN:
            draw, flags, lods, mc = d, _to_int(tok[4]), [d], 1
            pass_a = True
    if not pass_a:
        # PASS B: token[3] = meshCount, then meshCount LOD dists, then flags.
        mc = _to_int(tok[3]) if len(tok) > 3 else 1
        if mc not in (1, 2, 3):
            mc = 1
        lods = [_to_float(tok[4 + i]) for i in range(mc) if 4 + i < len(tok)]
        if not lods:
            lods = [0.0]
        fi = 4 + mc
        flags = _to_int(tok[fi]) if fi < len(tok) else 0
        draw = lods[-1]
    return ObjsRow(rid, dff, txd, mc, draw, lods, flags, decode_flags(flags))


def _parse_tobj(tok: List[str]) -> TobjRow:
    rid = _to_int(tok[0])
    dff = tok[1] if len(tok) > 1 else ""
    txd = tok[2] if len(tok) > 2 else ""
    # PASS A short form: id name txd drawDist flags timeOn timeOff (7 tokens, dist>=4)
    pass_a = False
    if len(tok) == 7:
        d = _to_float(tok[3])
        if d >= _DRAW_DIST_MIN:
            draw, flags = d, _to_int(tok[4])
            lods, mc = [d], 1
            time_on, time_off = _to_int(tok[5]), _to_int(tok[6])
            pass_a = True
    if not pass_a:
        # PASS B: meshCount + N LOD dists + flags + timeOn + timeOff
        mc = _to_int(tok[3]) if len(tok) > 3 else 1
        if mc not in (1, 2, 3):
            mc = 1
        lods = [_to_float(tok[4 + i]) for i in range(mc) if 4 + i < len(tok)]
        if not lods:
            lods = [0.0]
        fi = 4 + mc
        flags = _to_int(tok[fi]) if fi < len(tok) else 0
        time_on = _to_int(tok[fi + 1]) if fi + 1 < len(tok) else 0
        time_off = _to_int(tok[fi + 2]) if fi + 2 < len(tok) else 0
        draw = lods[-1]
    return TobjRow(rid, dff, txd, mc, draw, lods, flags, decode_flags(flags),
                   time_on, time_off)


def _parse_anim(tok: List[str]) -> AnimRow:
    rid = _to_int(tok[0])
    dff = tok[1] if len(tok) > 1 else ""
    txd = tok[2] if len(tok) > 2 else ""
    anim = tok[3] if len(tok) > 3 else "null"
    draw = _to_float(tok[4]) if len(tok) > 4 else 0.0
    flags = _to_int(tok[5]) if len(tok) > 5 else 0
    has_anim = anim.lower() != "null"
    return AnimRow(rid, dff, txd, anim, draw, flags, decode_flags(flags), has_anim)


def _parse_weap(tok: List[str]) -> WeapRow:
    rid = _to_int(tok[0])
    dff = tok[1] if len(tok) > 1 else ""
    txd = tok[2] if len(tok) > 2 else ""
    anim = tok[3] if len(tok) > 3 else "null"
    mc = _to_int(tok[4]) if len(tok) > 4 else 1
    draw = _to_float(tok[5]) if len(tok) > 5 else 0.0
    return WeapRow(rid, dff, txd, anim, mc, draw)


def _parse_hier(tok: List[str]) -> HierRow:
    rid = _to_int(tok[0])
    dff = tok[1] if len(tok) > 1 else ""
    txd = tok[2] if len(tok) > 2 else ""
    # SA hier rows commonly carry a trailing anim token + drawDist; both optional.
    anim = tok[3] if len(tok) > 3 else None
    draw = _to_float(tok[4]) if len(tok) > 4 else None
    return HierRow(rid, dff, txd, anim, draw)


def _parse_cars(tok: List[str]) -> CarsRow:
    # %d %s %s %s %s %s %s %s %d %d %x %d %f %f %d - trailing 1-2 tokens optional.
    rid = _to_int(tok[0])

    def s(i):
        return tok[i] if i < len(tok) else ""

    dff, txd, vtype = s(1), s(2), s(3)
    handling, game_name_raw, anim_group, vclass = s(4), s(5), s(6), s(7)
    game_name = game_name_raw.replace("_", " ")
    frequency = _to_int(tok[8]) if len(tok) > 8 else 0
    level = _to_int(tok[9]) if len(tok) > 9 else 0
    comp_rules = _to_int(tok[10], 16) if len(tok) > 10 else 0
    wheel_id = _to_int(tok[11]) if len(tok) > 11 else -1
    wheel_scale = _to_float(tok[12]) if len(tok) > 12 else 1.0
    wheel_scale_rear = _to_float(tok[13]) if len(tok) > 13 else None
    upgrades = _to_int(tok[14]) if len(tok) > 14 else None
    return CarsRow(
        rid, dff, txd, vtype, VEHICLE_TYPE_ENUM.get(vtype.lower(), -1),
        handling, game_name, anim_group, vclass,
        VEHICLE_CLASS_ENUM.get(vclass.lower(), -1),
        frequency, level, comp_rules, wheel_id, wheel_scale,
        wheel_scale_rear, upgrades,
    )


def _parse_peds(tok: List[str]) -> PedsRow:
    # %d %s %s %s %s %s %x %x %s %d %d %s %s %s
    rid = _to_int(tok[0])

    def s(i):
        return tok[i] if i < len(tok) else ""

    dff, txd, ped_type, stat_name, anim_group = s(1), s(2), s(3), s(4), s(5)
    cars_can_drive = _to_int(tok[6], 16) if len(tok) > 6 else 0
    flags2 = _to_int(tok[7], 16) if len(tok) > 7 else 0
    anim_file = s(8)
    radio1 = _to_int(tok[9]) if len(tok) > 9 else 0
    radio2 = _to_int(tok[10]) if len(tok) > 10 else 0
    voices = [t for t in tok[11:] if t]
    return PedsRow(rid, dff, txd, ped_type, stat_name, anim_group,
                   cars_can_drive, flags2, anim_file, radio1, radio2, voices)


def _parse_txdp(tok: List[str]) -> TxdpRow:
    child = tok[0] if len(tok) > 0 else ""
    parent = tok[1] if len(tok) > 1 else ""
    return TxdpRow(child, parent)


def _parse_2dfx(tok: List[str]) -> FxRow:
    rid = _to_int(tok[0])
    x = _to_float(tok[1]) if len(tok) > 1 else 0.0
    y = _to_float(tok[2]) if len(tok) > 2 else 0.0
    z = _to_float(tok[3]) if len(tok) > 3 else 0.0
    effect_type = _to_int(tok[4]) if len(tok) > 4 else 0
    tail = list(tok[5:])  # subtype grammar decoded lazily; raw tail preserved
    return FxRow(rid, x, y, z, effect_type, tail)


def _parse_path(tok: List[str]) -> RawRow:
    return RawRow(list(tok), _to_int(tok[0]) if tok else None, "path")


_SECTION_PARSERS = {
    "objs": _parse_objs,
    "tobj": _parse_tobj,
    "anim": _parse_anim,
    "weap": _parse_weap,
    "hier": _parse_hier,
    "cars": _parse_cars,
    "peds": _parse_peds,
    "txdp": _parse_txdp,
    "2dfx": _parse_2dfx,
    "path": _parse_path,
}


# --------------------------- public API ---------------------------

def parse_ide(text_or_bytes: Union[str, bytes]) -> Dict[str, List[dict]]:
    """Parse an IDE file into {section_name: [row_dict, ...]}.

 Accepts str or bytes (decoded latin-1, errors='replace'). Each row is a plain
 dict of named fields (see the per-section dataclasses). Unknown sections and
 malformed rows are tolerated: a single bad record is recorded as an _error
 entry rather than killing the whole file.
 """
    if isinstance(text_or_bytes, (bytes, bytearray)):
        text = bytes(text_or_bytes).decode("latin-1", errors="replace")
    else:
        text = text_or_bytes

    sections: Dict[str, List[dict]] = {}
    current: Optional[str] = None

    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        norm = _normalize_line(line)
        if not norm:                       # empty / all-space / all-control
            continue
        if norm[0] == "#":                 # comment
            continue
        if norm.rstrip() == "end":         # close section
            current = None
            continue

        tokens = norm.split()
        if not tokens:
            continue

        if current is None:
            kw = tokens[0].lower()
            if kw in KEYWORDS:
                current = kw
                sections.setdefault(current, [])
            # else: stray row outside a section -> ignored (engine ignores it)
            continue

        # inside a section: decode the row defensively
        parser = _SECTION_PARSERS.get(current)
        try:
            row = parser(tokens)
            sections[current].append(asdict(row))
        except Exception as e:  # one bad record must not kill the file
            sections[current].append({
                "section": current, "_error": str(e), "tokens": list(tokens),
            })

    return sections


# --------------------------- write-back (inverse of parse_ide) ---------------------------

def _fnum(v: float) -> str:
    """Format a float compactly without trailing-zero noise but lossless.

 `repr` already yields the shortest decimal string that round-trips to the same
 IEEE-754 double (e.g. 0.768 -> '0.768', 299.0 -> '299.0'). We only special-case
 whole-number floats expressed in exponent form to keep them human-friendly.
 """
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f) and abs(f) < 1e16:
        return f"{int(f)}.0"
    return repr(f)


def _row_objs(r: dict) -> str:
    # Inverse of _parse_objs. Emit the short form when single-mesh (re-parses to
    # the same row); otherwise the explicit meshCount + N LOD-dist long form.
    rid, dff, txd = r["id"], r["dff"], r["txd"]
    mc = r.get("mesh_count", 1)
    lods = r.get("lod_dists") or [r.get("draw_dist", 0.0)]
    flags = r.get("flags", 0)
    if mc == 1 and len(lods) == 1:
        # short form: id, dff, txd, drawDist, flags
        return f"{rid}, {dff}, {txd}, {_fnum(lods[0])}, {flags}"
    parts = [str(rid), dff, txd, str(mc)] + [_fnum(d) for d in lods] + [str(flags)]
    return ", ".join(parts)


def _row_tobj(r: dict) -> str:
    rid, dff, txd = r["id"], r["dff"], r["txd"]
    mc = r.get("mesh_count", 1)
    lods = r.get("lod_dists") or [r.get("draw_dist", 0.0)]
    flags = r.get("flags", 0)
    time_on, time_off = r.get("time_on", 0), r.get("time_off", 0)
    if mc == 1 and len(lods) == 1:
        return f"{rid}, {dff}, {txd}, {_fnum(lods[0])}, {flags}, {time_on}, {time_off}"
    parts = ([str(rid), dff, txd, str(mc)] + [_fnum(d) for d in lods]
             + [str(flags), str(time_on), str(time_off)])
    return ", ".join(parts)


def _row_anim(r: dict) -> str:
    # id, dff, txd, anim, drawDist, flags
    return (f"{r['id']}, {r['dff']}, {r['txd']}, {r.get('anim', 'null')}, "
            f"{_fnum(r.get('draw_dist', 0.0))}, {r.get('flags', 0)}")


def _row_weap(r: dict) -> str:
    # id, dff, txd, anim, meshCount, drawDist
    return (f"{r['id']}, {r['dff']}, {r['txd']}, {r.get('anim', 'null')}, "
            f"{r.get('mesh_count', 1)}, {_fnum(r.get('draw_dist', 0.0))}")


def _row_hier(r: dict) -> str:
    # id, dff, txd [, anim [, drawDist]] - both trailing fields optional.
    parts = [str(r["id"]), r["dff"], r["txd"]]
    if r.get("anim") is not None:
        parts.append(r["anim"])
        if r.get("draw_dist") is not None:
            parts.append(_fnum(r["draw_dist"]))
    return ", ".join(parts)


def _row_cars(r: dict) -> str:
    # id dff txd type handling gameName animGroup class freq level compRules(hex)
    # wheelId wheelScale [wheelScaleRear] [upgrades]
    # gameName: parser applied '_'->' '; restore by ' '->'_' so it re-parses equal.
    game_name = str(r.get("game_name", "")).replace(" ", "_")
    parts = [
        str(r["id"]), r["dff"], r["txd"], r.get("type", ""), r.get("handling", ""),
        game_name, r.get("anim_group", ""), r.get("vehicle_class", ""),
        str(r.get("frequency", 0)), str(r.get("level", 0)),
        format(r.get("comp_rules", 0) & 0xFFFFFFFF, "x"),
        str(r.get("wheel_id", -1)), _fnum(r.get("wheel_scale", 1.0)),
    ]
    if r.get("wheel_scale_rear") is not None:
        parts.append(_fnum(r["wheel_scale_rear"]))
    if r.get("upgrades") is not None:
        parts.append(str(r["upgrades"]))
    return ", ".join(parts)


def _row_peds(r: dict) -> str:
    # id dff txd pedType statName animGroup carsCanDrive(hex) flags2(hex) animFile
    # radio1 radio2 voice...
    parts = [
        str(r["id"]), r["dff"], r["txd"], r.get("ped_type", ""),
        r.get("stat_name", ""), r.get("anim_group", ""),
        format(r.get("cars_can_drive", 0) & 0xFFFFFFFF, "x"),
        format(r.get("flags2", 0) & 0xFFFFFFFF, "x"),
        r.get("anim_file", ""), str(r.get("radio1", 0)), str(r.get("radio2", 0)),
    ]
    parts.extend(r.get("voices", []))
    return ", ".join(parts)


def _row_txdp(r: dict) -> str:
    return f"{r.get('child', '')}, {r.get('parent', '')}"


def _row_2dfx(r: dict) -> str:
    parts = [str(r["id"]), _fnum(r.get("x", 0.0)), _fnum(r.get("y", 0.0)),
             _fnum(r.get("z", 0.0)), str(r.get("effect_type", 0))]
    parts.extend(str(t) for t in r.get("tail", []))
    return ", ".join(parts)


def _row_path(r: dict) -> str:
    return ", ".join(str(t) for t in r.get("tokens", []))


_SECTION_WRITERS = {
    "objs": _row_objs,
    "tobj": _row_tobj,
    "anim": _row_anim,
    "weap": _row_weap,
    "hier": _row_hier,
    "cars": _row_cars,
    "peds": _row_peds,
    "txdp": _row_txdp,
    "2dfx": _row_2dfx,
    "path": _row_path,
}


def write_ide(parsed: Dict[str, List[dict]]) -> str:
    """Serialize a parsed IDE (the dict returned by parse_ide) back to SA IDE text.

 Each section is emitted as ``<section>\\n`` <rows...> ``end\\n`` and is the exact
 inverse of the matching ``_parse_<section>``: parse_ide(write_ide(x)) reproduces
 the same typed rows. Comments / blank lines are not preserved (the engine drops
 them). Rows that failed to parse (recorded with an ``_error`` key) fall back to
 re-emitting their raw token list, so even a malformed source survives the round
 trip.

 Lossy-but-self-consistent fields (round-trip to the same *parsed* value, not the
 original bytes): cars.game_name ('_'<->' '), objs/tobj single-mesh rows always
 re-emit in the short form, and hex fields (peds.cars_can_drive/flags2,
 cars.comp_rules) re-emit lowercase without a leading sign.
 """
    out: List[str] = []
    for section, rows in parsed.items():
        writer = _SECTION_WRITERS.get(section)
        out.append(f"{section}\n")
        for r in rows:
            if writer is None or "_error" in r:
                # unknown section or a record that didn't parse: re-emit raw tokens
                out.append(", ".join(str(t) for t in r.get("tokens", [])) + "\n")
            else:
                out.append(writer(r) + "\n")
        out.append("end\n")
    return "".join(out)


def to_json(parsed: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Return a JSON-serializable view of a parsed IDE.

 parse_ide already yields plain dicts (lists/dicts/str/int/float/bool), so this
 is a defensive deep-copy that coerces any stray non-native leaf to a string - the web server hands the result straight to the UI table view.
 """
    def _coerce(o):
        if isinstance(o, dict):
            return {str(k): _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_coerce(v) for v in o]
        if isinstance(o, bool) or o is None:
            return o
        if isinstance(o, (int, float, str)):
            return o
        return str(o)

    return {sect: [_coerce(r) for r in rows] for sect, rows in parsed.items()}
