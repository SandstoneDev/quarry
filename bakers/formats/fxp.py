"""the source game effects.fxp particle-FX blueprint bank (plain ASCII) decoder.

effects.fxp is a data-driven particle system bank: ~82 named FX systems
(FxSystemBP_c), each holding one or more emitter primitives (FxEmitterBP_c),
each carrying FX_INFO_* blocks (EMRATE/EMLIFE/EMSPEED/EMDIR/SIZE/COLOUR/FORCE...)
whose values are keyframed curves (FX_INTERP_DATA -> FX_KEYFLOAT_DATA TIME/VAL
pairs) over normalized 0..1 time. It is line-based ASCII (NOT binary); a per-
system integer version (109 in retail) gates optional fields.

The parse is marker-driven and resilient (per the spec decode-plan): a cursor
reads physical lines, each system is bounded by the next 'FX_SYSTEM_DATA:'
marker, and info subfields are discovered by a count-free peek loop rather than
a fixed per-type table - so rarer info types decode without a hard-coded
schema. Each system is decoded inside a try/except so one malformed record
cannot kill the whole file.

 (confirmed line/field layout of
FxSystemBP_c::LoadData 0x5d68d0; CULLDIST*256.0 -> u16 @+0x18 at 0x5d6a91)
Loaders: CFxManager::LoadFxp 0x5d86f0, ParseFxSystemBlock 0x5d82e0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Tags that terminate an FX_INFO subfield peek-loop (i.e. mark the start of the
# next info / emitter trailer / prim / system). Matched on the line's first token.
_INFO_BOUNDARY = {
    "FX_PRIM_EMITTER_DATA:", "FX_PRIM_BASE_DATA:", "FX_PRIM_DATA:",
    "FX_SYSTEM_DATA:", "NUM_INFOS:",
    "LODSTART:", "LODEND:", "OMITTEXTURES:", "TXDNAME:",
}

# Emitter-trailer keys consumed opportunistically after the info list.
_EMITTER_TRAILER = {"LODSTART:", "LODEND:", "OMITTEXTURES:", "TXDNAME:"}


@dataclass
class Curve:
    """One FX_INTERP_DATA interpolator: LOOPED flag + (time, val) keyframes."""
    looped: int = 0
    keys: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class Info:
    """One FX_INFO_*_DATA block (e.g. EMSPEED, FORCE, COLOUR)."""
    name: str                                    # short tag, e.g. "EMSPEED"
    code: str                                    # full tag, e.g. "FX_INFO_EMSPEED_DATA:"
    timemode: Optional[int] = None               # TIMEMODEPRT byte, if present
    curves: Dict[str, Curve] = field(default_factory=dict)  # subfield -> Curve


@dataclass
class Emitter:
    """One FX_PRIM_EMITTER_DATA / FX_PRIM_BASE_DATA primitive."""
    name: str = ""
    matrix: List[float] = field(default_factory=list)        # 12 floats (3x3 + translate)
    textures: List[Optional[str]] = field(default_factory=list)  # 4 entries, NULL -> None
    alpha_on: int = 0
    src_blend: int = 0
    dst_blend: int = 0
    num_infos: int = 0
    lod_start: Optional[float] = None
    lod_end: Optional[float] = None
    omit_textures: Optional[int] = None
    txd_name: Optional[str] = None
    infos: List[Info] = field(default_factory=list)


@dataclass
class System:
    """One FX_SYSTEM_DATA block (FxSystemBP_c, 0x24 bytes)."""
    name: str = ""
    version: int = 0
    filename: Optional[str] = None
    length: float = 0.0
    loop_interval_min: float = 0.0
    loop_length: float = 0.0
    playmode: int = 0
    culldist: float = 0.0
    culldist_u16: int = 0
    bounding_sphere: Optional[List[float]] = None   # (cx, cy, cz, radius) or None
    emitters: List[Emitter] = field(default_factory=list)


@dataclass
class Fxp:
    version: int                                  # version of the first system (retail 109)
    systems: List[System] = field(default_factory=list)


# --------------------------- line cursor ----------------------------

class _Cursor:
    """Physical-line reader matching CFileMgr::ReadLine semantics.

 Blank lines are real line slots (positional separators), so nextline()
 returns the next physical line verbatim (CR/LF stripped). peek() looks at
 the upcoming physical line without consuming it.
 """

    def __init__(self, text: str):
        # strip CR; keep blank lines as empty strings (they are line slots)
        norm = text.replace("\r\n", "\n").replace("\r", "\n")
        self.lines = norm.split("\n")
        self.i = 0
        self.n = len(self.lines)

    def eof(self) -> bool:
        return self.i >= self.n

    def nextline(self) -> str:
        if self.i >= self.n:
            return ""
        s = self.lines[self.i]
        self.i += 1
        return s

    def peek(self) -> Optional[str]:
        if self.i >= self.n:
            return None
        return self.lines[self.i]

    def peek_tag(self) -> Optional[str]:
        ln = self.peek()
        if ln is None:
            return None
        toks = ln.split()
        return toks[0] if toks else ""


# --------------------------- token helpers --------------------------

def _toks(line: str) -> List[str]:
    return line.split()


def _floats(line: str, n: int) -> List[float]:
    """First token = tag (ignored); next n tokens parsed as floats (0.0 if short)."""
    parts = line.split()[1:]
    out: List[float] = []
    for k in range(n):
        try:
            out.append(float(parts[k]))
        except (IndexError, ValueError):
            out.append(0.0)
    return out


def _all_floats(line: str) -> List[float]:
    out: List[float] = []
    for t in line.split()[1:]:
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _one_int(line: str, default: int = 0) -> int:
    parts = line.split()[1:]
    for t in parts:
        try:
            return int(float(t))
        except ValueError:
            continue
    return default


def _one_float(line: str, default: float = 0.0) -> float:
    parts = line.split()[1:]
    for t in parts:
        try:
            return float(t)
        except ValueError:
            continue
    return default


def _one_str(line: str, default: Optional[str] = None) -> Optional[str]:
    parts = line.split()[1:]
    if not parts:
        return default
    v = parts[0]
    return None if v.upper() == "NULL" else v


def _info_shortname(tag: str) -> str:
    """FX_INFO_EMSPEED_DATA: -> EMSPEED. Tolerates a leading '?' sentinel."""
    t = tag.lstrip("?")
    if t.startswith("FX_INFO_"):
        t = t[len("FX_INFO_"):]
    if t.endswith("_DATA:"):
        t = t[:-len("_DATA:")]
    elif t.endswith(":"):
        t = t[:-1]
    return t


def _find_next_system(cur: _Cursor) -> bool:
    """Advance the cursor to just past the next 'FX_SYSTEM_DATA:' line.

 Returns True if positioned at the version line of a system; False at EOF.
 """
    while not cur.eof():
        toks = _toks(cur.nextline())
        if toks and toks[0] == "FX_SYSTEM_DATA:":
            return True
    return False


# --------------------------- block parsers --------------------------

def _parse_curve(cur: _Cursor) -> Curve:
    """Parse one FX_INTERP_DATA block (assumes the FX_INTERP_DATA: line is next)."""
    c = Curve()
    # consume the FX_INTERP_DATA: marker
    if cur.peek_tag() == "FX_INTERP_DATA:":
        cur.nextline()
    # LOOPED:
    if cur.peek_tag() == "LOOPED:":
        c.looped = _one_int(cur.nextline())
    # NUM_KEYS:
    num_keys = 0
    if cur.peek_tag() == "NUM_KEYS:":
        num_keys = _one_int(cur.nextline())
    # read exactly num_keys FX_KEYFLOAT_DATA blocks (defensive: stop if structure breaks)
    for _ in range(max(num_keys, 0)):
        if cur.peek_tag() != "FX_KEYFLOAT_DATA:":
            break
        cur.nextline()                     # FX_KEYFLOAT_DATA:
        t = _one_float(cur.nextline()) if cur.peek_tag() == "TIME:" else 0.0
        v = _one_float(cur.nextline()) if cur.peek_tag() == "VAL:" else 0.0
        c.keys.append((t, v))
    return c


def _parse_info(cur: _Cursor) -> Info:
    """Parse one FX_INFO_*_DATA block via a count-free subfield peek loop."""
    tag = _toks(cur.nextline())[0]
    info = Info(name=_info_shortname(tag), code=tag.lstrip("?"))

    # optional TIMEMODEPRT byte (high-class infos carry it; detect positionally)
    if cur.peek_tag() == "TIMEMODEPRT:":
        info.timemode = _one_int(cur.nextline())

    # subfields: each is a bare "<NAME>:" tag line immediately followed by an
    # FX_INTERP_DATA block. Loop until we hit a boundary marker (next info /
    # emitter trailer / prim / system) or a structural surprise.
    while True:
        tag2 = cur.peek_tag()
        if tag2 is None:                       # EOF
            break
        if tag2 == "":                         # blank separator line: consume + continue
            cur.nextline()
            continue
        if tag2 in _INFO_BOUNDARY or tag2.startswith("FX_INFO_") or \
                tag2.startswith("?FX_INFO_"):
            break
        if tag2 == "FX_INTERP_DATA:":
            # subfield tag was missing/odd; attach under a synthetic key
            key = f"_field{len(info.curves)}"
            info.curves[key] = _parse_curve(cur)
            continue
        # a normal subfield: "<NAME>:" then an FX_INTERP_DATA block must follow
        subfield = tag2.rstrip(":")
        cur.nextline()                          # consume the subfield tag line
        # skip any blank lines between the subfield tag and its curve
        while cur.peek_tag() == "":
            cur.nextline()
        if cur.peek_tag() == "FX_INTERP_DATA:":
            info.curves[subfield] = _parse_curve(cur)
        else:
            # no curve followed -> not actually a subfield; stop to stay in sync
            # (record an empty curve so the field isn't silently lost)
            info.curves[subfield] = Curve()
            break
    return info


def _parse_emitter(cur: _Cursor) -> Emitter:
    """Parse one emitter: FX_PRIM_BASE_DATA + NUM_INFOS infos + trailer.

 Assumes the FX_PRIM_EMITTER_DATA: marker has already been consumed.
 """
    em = Emitter()
    # find FX_PRIM_BASE_DATA: (skip the blank separator line)
    while not cur.eof():
        tag = cur.peek_tag()
        if tag == "FX_PRIM_BASE_DATA:":
            cur.nextline()
            break
        if tag in ("FX_PRIM_EMITTER_DATA:", "FX_SYSTEM_DATA:"):
            return em                           # malformed: bail without consuming marker
        cur.nextline()

    # base data (order-sensitive, but tolerate missing lines)
    if cur.peek_tag() == "NAME:":
        em.name = _one_str(cur.nextline(), "") or ""
    if cur.peek_tag() == "MATRIX:":
        em.matrix = _floats(cur.nextline(), 12)
    for key in ("TEXTURE:", "TEXTURE2:", "TEXTURE3:", "TEXTURE4:"):
        if cur.peek_tag() == key:
            em.textures.append(_one_str(cur.nextline()))
        else:
            em.textures.append(None)
    if cur.peek_tag() == "ALPHAON:":
        em.alpha_on = _one_int(cur.nextline())
    if cur.peek_tag() == "SRCBLENDID:":
        em.src_blend = _one_int(cur.nextline())
    if cur.peek_tag() == "DSTBLENDID:":
        em.dst_blend = _one_int(cur.nextline())

    # NUM_INFOS (skip blank separator first)
    while cur.peek_tag() == "":
        cur.nextline()
    if cur.peek_tag() == "NUM_INFOS:":
        em.num_infos = _one_int(cur.nextline())

    # info list - driven by NUM_INFOS but resilient to over/under-count
    parsed = 0
    while parsed < em.num_infos:
        # skip blanks
        while cur.peek_tag() == "":
            cur.nextline()
        tag = cur.peek_tag()
        if tag is None:
            break
        if tag.lstrip("?").startswith("FX_INFO_"):
            em.infos.append(_parse_info(cur))
            parsed += 1
        elif tag in _EMITTER_TRAILER or tag in ("FX_SYSTEM_DATA:", "FX_PRIM_EMITTER_DATA:"):
            break                               # ran out of infos early
        else:
            cur.nextline()                       # unknown filler, skip

    # emitter trailer (LODSTART/LODEND/OMITTEXTURES/TXDNAME), opportunistic
    while True:
        while cur.peek_tag() == "":
            cur.nextline()
        tag = cur.peek_tag()
        if tag == "LODSTART:":
            em.lod_start = _one_float(cur.nextline())
        elif tag == "LODEND:":
            em.lod_end = _one_float(cur.nextline())
        elif tag == "OMITTEXTURES:":
            em.omit_textures = _one_int(cur.nextline())
        elif tag == "TXDNAME:":
            v = _one_str(cur.nextline())
            em.txd_name = None if (v is None or v.upper() == "NOTXDSET") else v
        else:
            break
    return em


def _parse_system(cur: _Cursor) -> System:
    """Parse one FX_SYSTEM_DATA block. Cursor is positioned at the version line.

 Replicates FxSystemBP_c::LoadData 0x5d68d0 field order with version gates,
 but is marker-resilient (bounded by the next FX_SYSTEM_DATA:).
 """
    sysd = System()
    # cursor is positioned at the version line (a bare decimal int, e.g. "109")
    ver_line = cur.nextline()
    try:
        sysd.version = int(ver_line.split()[0])
    except (IndexError, ValueError):
        sysd.version = _one_int(ver_line)
    ver = sysd.version

    # 1. blank separator (discarded)
    if cur.peek_tag() == "":
        cur.nextline()
    # 2. if version>100: FILENAME line
    if ver > 100 and cur.peek_tag() == "FILENAME:":
        sysd.filename = _one_str(cur.nextline())
    # 3. NAME
    if cur.peek_tag() == "NAME:":
        sysd.name = _one_str(cur.nextline(), "") or ""
    # 4. LENGTH (play length)
    if cur.peek_tag() == "LENGTH:":
        sysd.length = _one_float(cur.nextline())
    # 5. version>=106: LOOPINTERVALMIN + 2nd LENGTH (loop length); else 0,0
    if ver >= 106:
        if cur.peek_tag() == "LOOPINTERVALMIN:":
            sysd.loop_interval_min = _one_float(cur.nextline())
        if cur.peek_tag() == "LENGTH:":
            sysd.loop_length = _one_float(cur.nextline())
    # 6. PLAYMODE
    if cur.peek_tag() == "PLAYMODE:":
        sysd.playmode = _one_int(cur.nextline()) & 0xFF
    # 7. CULLDIST -> raw float + u16(value*256)
    if cur.peek_tag() == "CULLDIST:":
        sysd.culldist = _one_float(cur.nextline())
        sysd.culldist_u16 = int(sysd.culldist * 256.0) & 0xFFFF
    # 8. version>103: BOUNDINGSPHERE cx cy cz radius
    if ver > 103 and cur.peek_tag() == "BOUNDINGSPHERE:":
        sysd.bounding_sphere = _floats(cur.nextline(), 4)
    # 9. NUM_PRIMS
    num_prims = 0
    if cur.peek_tag() == "NUM_PRIMS:":
        num_prims = _one_int(cur.nextline())

    # 10. prim pass: read num_prims markers; emitter prims get fully parsed.
    parsed = 0
    while parsed < num_prims:
        while cur.peek_tag() == "":
            cur.nextline()
        tag = cur.peek_tag()
        if tag is None or tag == "FX_SYSTEM_DATA:":
            break
        if tag == "FX_PRIM_EMITTER_DATA:":
            cur.nextline()                       # consume the marker
            sysd.emitters.append(_parse_emitter(cur))
            parsed += 1
        elif tag.startswith("FX_PRIM_"):
            cur.nextline()                       # non-emitter prim marker: skip (none in retail)
            parsed += 1
        else:
            cur.nextline()                       # filler

    # 11/12. version>=108 / >=109 generic system-tail reads are, in retail v109,
    # already consumed as the emitter trailer (LODSTART/.../TXDNAME). Any stray
    # trailer lines before the next FX_SYSTEM_DATA: are skipped by _find_next_system.
    return sysd


# ------------------------------ public API --------------------------

def parse_fxp(text) -> dict:
    """Parse effects.fxp ASCII text into a JSON-serializable dict tree.

 Returns::

 {"version": int,
 "systems": [
 {"name": str, "version": int, "length": float, "playmode": int,
 "culldist": float, "culldist_u16": int, "bounding_sphere": [..]|None,
 "emitters": [
 {"props": {"name", "matrix":[12 floats], "textures":[4],
 "texture".."texture4", "alpha_on", "src_blend",
 "dst_blend", "num_infos",
 "infos": [{"name","code","timemode",
 "curves": {sub: {"looped","keys":[[t,v]...]}}}]}}
 ]}
 ]}

 The top-level version is the version of the first system (retail 109).
 Accepts str or bytes. Every value is list/dict/str/int/float/None.
 """
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("latin-1")

    cur = _Cursor(text)
    systems: List[System] = []
    top_version = 0

    # Skip a leading FX_PROJECT_DATA header line if present; then loop systems.
    while _find_next_system(cur):
        try:
            sysd = _parse_system(cur)
        except Exception as e:  # one bad system must not kill the bank
            sysd = System(name=f"<error:{e}>")
        if top_version == 0 and sysd.version:
            top_version = sysd.version
        systems.append(sysd)

    return {
        "version": top_version,
        "systems": [_system_json(s) for s in systems],
    }


def _curve_json(c: Curve) -> dict:
    return {
        "looped": int(c.looped),
        "keys": [[float(t), float(v)] for (t, v) in c.keys],
    }


def _info_json(inf: Info) -> dict:
    return {
        "name": inf.name,
        "code": inf.code,
        "timemode": (None if inf.timemode is None else int(inf.timemode)),
        "curves": {k: _curve_json(v) for k, v in inf.curves.items()},
    }


def _emitter_json(em: Emitter) -> dict:
    textures = list(em.textures) + [None] * (4 - len(em.textures))
    props = {
        "name": em.name,
        "matrix": [float(x) for x in em.matrix],
        "textures": [t for t in textures[:4]],
        "texture": textures[0] if len(textures) > 0 else None,
        "texture2": textures[1] if len(textures) > 1 else None,
        "texture3": textures[2] if len(textures) > 2 else None,
        "texture4": textures[3] if len(textures) > 3 else None,
        "alpha_on": int(em.alpha_on),
        "src_blend": int(em.src_blend),
        "dst_blend": int(em.dst_blend),
        "num_infos": int(em.num_infos),
        "lod_start": (None if em.lod_start is None else float(em.lod_start)),
        "lod_end": (None if em.lod_end is None else float(em.lod_end)),
        "omit_textures": (None if em.omit_textures is None else int(em.omit_textures)),
        "txd_name": em.txd_name,
        "infos": [_info_json(i) for i in em.infos],
    }
    return {"props": props}


def _system_json(s: System) -> dict:
    return {
        "name": s.name,
        "version": int(s.version),
        "filename": s.filename,
        "length": float(s.length),
        "loop_interval_min": float(s.loop_interval_min),
        "loop_length": float(s.loop_length),
        "playmode": int(s.playmode),
        "culldist": float(s.culldist),
        "culldist_u16": int(s.culldist_u16),
        "bounding_sphere": (None if s.bounding_sphere is None
                            else [float(x) for x in s.bounding_sphere]),
        "emitters": [_emitter_json(e) for e in s.emitters],
    }


def to_json(doc) -> dict:
    """Convert a parse_fxp result (or Fxp/System list) into a JSON-safe dict.

 `parse_fxp` already returns the dict form, so to_json is idempotent on it:
 every value is list/dict/str/int/float/bool/None - the SAW web server hands
 this straight to the UI list view.
 """
    if isinstance(doc, dict):
        # already the dict tree produced by parse_fxp -> pass through, but
        # coerce the version to int and keep systems as-is (already JSON-safe).
        return {
            "version": int(doc.get("version", 0) or 0),
            "systems": [
                s if isinstance(s, dict) else _system_json(s)
                for s in doc.get("systems", [])
            ],
        }
    # Fxp dataclass (or anything with.version/.systems of System dataclasses)
    return {
        "version": int(doc.version),
        "systems": [
            s if isinstance(s, dict) else _system_json(s)
            for s in doc.systems
        ],
    }
