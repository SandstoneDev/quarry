"""the source game pedstats.dat decoder (ASCII, one STAT_* row per ped personality).

data/pedstats.dat is CPedStats::Initialise's table: per-ped-type behavioural
tuning read row-by-row (the rows MUST stay in enum order in-engine; the STAT_*
token is the human-readable key). '#' lines are comments, blank lines skipped.

The file self-documents its columns in the leading '#' legend (A:..K:):

 A name STAT_* personality key (the dict key here)
 B flee_distance float
 C heading_change_rate float (degrees)
 D fear 0-100 (100 = scared of everything)
 E temper 0-100 (100 = bad tempered)
 F lawfulness 0-100 (100 = boy scout)
 G sexiness 0-100
 H attack_strength float multiplier to dealt damage
 I defend_weakness float multiplier to received damage
 J shooting_rate 0-100
 K decision_maker flags / default-decision-maker selector

Stock SA rows carry no leading index, but some variants prepend a numeric index
before the STAT name - the parser tolerates both (it locates the STAT_* token).
The full raw token list is kept in ``cols`` so the table view can show every
column even where a name is uncertain.

Public API:
 parse_pedstats(text) -> dict[str, PedStat] (keyed by STAT name)
 to_json(d) -> dict (JSON-serializable, table view)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# small forgiving coercers (atoi/atof never crash a row)
def _f(tok: Optional[str], default: float = 0.0) -> float:
    try:
        return float(tok)
    except (ValueError, TypeError):
        return default


def _i(tok: Optional[str], default: int = 0) -> int:
    try:
        return int(tok)
    except (ValueError, TypeError):
        try:
            return int(float(tok))
        except (ValueError, TypeError):
            return default


@dataclass
class PedStat:
    """One pedstats.dat row. ``cols`` is the full raw token list starting at the
 STAT name (any leading index is dropped); the named fields decode the
 documented SA columns (A..K)."""
    name: str
    cols: List[str] = field(default_factory=list)
    flee_distance: float = 0.0
    heading_change_rate: float = 0.0
    fear: int = 0
    temper: int = 0
    lawfulness: int = 0
    sexiness: int = 0
    attack_strength: float = 0.0
    defend_weakness: float = 0.0
    shooting_rate: int = 0
    flags: int = 0


def _decode(cols: List[str]) -> PedStat:
    # cols[0] = STAT name, then B..K.
    g = lambda i: cols[i] if i < len(cols) else None  # noqa: E731
    return PedStat(
        name=cols[0],
        cols=list(cols),
        flee_distance=_f(g(1)),
        heading_change_rate=_f(g(2)),
        fear=_i(g(3)),
        temper=_i(g(4)),
        lawfulness=_i(g(5)),
        sexiness=_i(g(6)),
        attack_strength=_f(g(7)),
        defend_weakness=_f(g(8)),
        shooting_rate=_i(g(9)),
        flags=_i(g(10)),
    )


def parse_pedstats(text: str) -> Dict[str, PedStat]:
    """Parse pedstats.dat into ``{STAT_NAME: PedStat}``.

 Tolerates an optional leading integer index before the STAT name. '#'
 comments and blank lines are skipped. A malformed row never aborts the file.
 """
    out: Dict[str, PedStat] = {}
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        s = line.lstrip("﻿").lstrip()   # tolerate a UTF-8 BOM on the first row
        if not s:
            continue
        if s.startswith("#"):
            continue
        toks = s.split()
        if not toks:
            continue
        # drop an optional leading numeric index so cols[0] is the STAT name
        if toks[0].lstrip("+-").isdigit() and len(toks) > 1:
            toks = toks[1:]
        if not toks[0].startswith("STAT"):
            continue                # not a stat row
        try:
            rec = _decode(toks)
        except Exception:
            continue
        out[rec.name] = rec
    return out


def to_json(d: Dict[str, PedStat]) -> Dict[str, dict]:
    """Render the {name: PedStat} map as JSON-serializable dicts for the table view."""
    return {
        name: {
            "name": s.name,
            "flee_distance": s.flee_distance,
            "heading_change_rate": s.heading_change_rate,
            "fear": s.fear,
            "temper": s.temper,
            "lawfulness": s.lawfulness,
            "sexiness": s.sexiness,
            "attack_strength": s.attack_strength,
            "defend_weakness": s.defend_weakness,
            "shooting_rate": s.shooting_rate,
            "flags": s.flags,
            "cols": list(s.cols),
        }
        for name, s in d.items()
    }
