"""the source game weapon.dat decoder (ASCII, per-row marker selects the column layout).

data/weapon.dat is CWeaponInfo::Initialise's tuning table: one row per weapon
(and weapon skill-level), parsed by CWeaponInfo::LoadWeaponData. Every data line
is prefixed by a one-character MARKER that selects which sscanf format runs:

 0xA3 ('£' latin-1) MELEE weapons (11 tokens)
 '$' GUN weapons (25 short / 29 with projectile tail)
 '%' per-anim-group AIM-OFFSET rows - NOT weapons, skipped

The file self-documents its columns in the leading '#' legend block (A:/B:/C,D:…).
'#' lines are comments; the file ends at the literal token "ENDWEAPONDATA".

Both weapon layouts share a leading prefix (name, fireType, targetRange,
weaponRange, modelId1, modelId2, weaponslot); past the slot the two diverge:

 MELEE col8.. : baseCombo(str) numCombos(int) flags(hex) stealthAnimGroup(str)
 GUN col8.. : animGroup(str) ammoClip(int) damage(int) fireOffset(x,y,z) ...
 skill reqStat accuracy moveSpeed anim1{s,e,f} anim2{s,e,f}
 breakoutTime flags(hex) [speed radius lifespan spread]

To keep one flat, UI-friendly record we map the well-known columns into named
fields (dispatched by marker) and ALWAYS keep the full raw token list in ``cols``
so the table view can show every column even where a name is uncertain.

Public API:
 parse_weapon_dat(text) -> list[Weapon]
 to_json(weapons) -> list[dict] (JSON-serializable, web table view)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Row markers. 0xA3 decodes to '£' under latin-1 (how the file is read).
_MELEE_MARKER = "\xa3"   # '£'
_GUN_MARKER = "$"
_AIM_MARKER = "%"        # aim-offset table rows (per anim group) - not weapons
_TERMINATOR = "ENDWEAPONDATA"

KNOWN_FIRE_TYPES = {
    "MELEE", "INSTANT_HIT", "PROJECTILE", "AREA_EFFECT", "CAMERA", "USE",
}


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


def _hx(tok: Optional[str], default: int = 0) -> int:
    try:
        return int(tok, 16)
    except (ValueError, TypeError):
        return default


@dataclass
class Weapon:
    """One weapon.dat row. ``cols`` is the full raw token list (column order
 preserved); the named fields are a best-effort decode of the documented SA
 columns and differ slightly between melee and gun rows (see module docstring).
 """
    name: str
    fire_type: str
    marker: str                                  # '£' melee | '$' gun
    cols: List[str] = field(default_factory=list)

    # shared prefix (both layouts)
    target_range: float = 0.0
    weapon_range: float = 0.0
    model_id1: int = -1
    model_id2: int = -1
    weapon_slot: int = 0

    # past the slot the meaning diverges by marker; we surface a common superset.
    anim_group: str = ""        # melee: baseCombo | gun: AssocGroupId
    clip_size: int = 0          # melee: numCombos | gun: ammoClip
    damage: int = 0             # gun only (melee has no damage column)
    flags: int = 0              # hex flags word (melee col L / gun col 'a')
    fire_offset: Optional[List[float]] = None    # gun only (x,y,z)


def _decode_melee(toks: List[str], marker: str) -> Weapon:
    # A name | B fireType | C,D ranges | E,F modelIds | I slot |
    # J baseCombo | K numCombos | L flags(hex) | M stealthAnimGroup
    g = lambda i: toks[i] if i < len(toks) else None  # noqa: E731
    return Weapon(
        name=toks[0],
        fire_type=g(1) or "",
        marker=marker,
        cols=list(toks),
        target_range=_f(g(2)),
        weapon_range=_f(g(3)),
        model_id1=_i(g(4), -1),
        model_id2=_i(g(5), -1),
        weapon_slot=_i(g(6)),
        anim_group=g(7) or "",      # baseCombo
        clip_size=_i(g(8)),         # numCombos
        flags=_hx(g(9)),
    )


def _decode_gun(toks: List[str], marker: str) -> Weapon:
    # A name | B fireType | C,D ranges | E,F modelIds | I slot |
    # J animGroup | K ammoClip | L damage | M,N,O fireOffset |... | a flags(hex)
    # | [b speed c radius d lifespan e spread] (29-token long form)
    g = lambda i: toks[i] if i < len(toks) else None  # noqa: E731
    w = Weapon(
        name=toks[0],
        fire_type=g(1) or "",
        marker=marker,
        cols=list(toks),
        target_range=_f(g(2)),
        weapon_range=_f(g(3)),
        model_id1=_i(g(4), -1),
        model_id2=_i(g(5), -1),
        weapon_slot=_i(g(6)),
        anim_group=g(7) or "",      # AssocGroupId
        clip_size=_i(g(8)),         # ammoClip
        damage=_i(g(9)),
    )
    if len(toks) > 12:
        w.fire_offset = [_f(g(10)), _f(g(11)), _f(g(12))]
    # The flags column 'a' sits just before the optional 4-float projectile tail.
    # Short gun rows = 25 tokens (flags is the last token); long = 29 (flags at
    # index 24, then speed/radius/lifespan/spread).
    if len(toks) >= 29:
        w.flags = _hx(g(24))
    elif len(toks) >= 25:
        w.flags = _hx(g(len(toks) - 1))
    return w


def parse_weapon_dat(text: str) -> List[Weapon]:
    """Parse weapon.dat into a flat list of Weapon rows.

 The '%' aim-offset rows are skipped (they are per-anim-group aiming tables,
 not weapons). Parsing stops at the ENDWEAPONDATA terminator. One malformed
 row never aborts the whole file.
 """
    out: List[Weapon] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        s = line.lstrip("﻿")   # tolerate a UTF-8 BOM on the first data line
        s = s.lstrip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith(_TERMINATOR):
            break
        marker = s[0]
        if marker not in (_MELEE_MARKER, _GUN_MARKER, _AIM_MARKER):
            continue                # stray / unknown line
        if marker == _AIM_MARKER:
            continue                # aim-offset table, not a weapon
        toks = s[1:].split()
        if not toks:
            continue
        try:
            if marker == _MELEE_MARKER:
                out.append(_decode_melee(toks, "£"))
            else:
                out.append(_decode_gun(toks, "$"))
        except Exception:
            continue
    return out


def to_json(weapons: List[Weapon]) -> List[dict]:
    """Render the weapon list as JSON-serializable dicts for the SAW table view."""
    rows: List[dict] = []
    for w in weapons:
        d = {
            "name": w.name,
            "fire_type": w.fire_type,
            "marker": w.marker,
            "target_range": w.target_range,
            "weapon_range": w.weapon_range,
            "model_id1": w.model_id1,
            "model_id2": w.model_id2,
            "weapon_slot": w.weapon_slot,
            "anim_group": w.anim_group,
            "clip_size": w.clip_size,
            "damage": w.damage,
            "flags": w.flags,
            "cols": list(w.cols),
        }
        if w.fire_offset is not None:
            d["fire_offset"] = list(w.fire_offset)
        rows.append(d)
    return rows
