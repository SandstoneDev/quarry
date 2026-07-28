"""the source game item-definition (.IDE) parser - objs/tobj/anim sections (read-only).

The three sections that define a PLACEABLE world model do NOT share a column
layout, so a def carries its `section` plus RESOLVED attributes.  Callers must
read `flags` / `draw_dist` / `anim_block` / `time_on` / `time_off` and never
index `fields` themselves - a tobj row ends with timeOff (not flags) and an
anim row carries an extra animBlock column before the draw distance:

    objs   id, dff, txd, [meshCount,] drawDist x meshCount, flags
    tobj   <the objs columns>, timeOn, timeOff
    anim   id, dff, txd, animBlock, drawDist, flags

`fields` stays the raw CSV split for anything that still wants it.
"""
import os

# sections we keep (one def per row); the rest are peds/vehicles/txd parents.
MODEL_SECTIONS = ("objs", "tobj", "anim")
_SECTIONS = ("objs", "tobj", "anim", "peds", "cars", "weap", "hier", "txdp", "2dfx")

DEFAULT_DRAW_DIST = 300.0


def _as_int(s, default=0):
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return default


def _as_float(s, default=DEFAULT_DRAW_DIST):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _objs_tail(parts, first=3):
    """Resolve the objs-style tail `[meshCount,] drawDist x meshCount, flags`
    that starts at column `first`.  Returns (draw_dist, flags).

    The 5-column short form omits meshCount (one mesh, one distance); the long
    form spells it out (SA ships exactly one: `320, airtrain_vlo, generic, 1,
    2000, 0`).  We keep the FIRST (nearest) distance, like the engine's LOD."""
    n = len(parts)
    flags = _as_int(parts[-1], 0)
    dd = _as_float(parts[first]) if n > first else DEFAULT_DRAW_DIST
    if n - first >= 3:
        mc = _as_int(parts[first], -1)
        if 1 <= mc <= 4 and n == first + mc + 2:
            dd = _as_float(parts[first + 1])
    return dd, flags


class ObjDef:
    __slots__ = ("id", "dff", "txd", "fields", "section",
                 "flags", "draw_dist", "anim_block", "time_on", "time_off")

    def __init__(self, id, dff, txd, fields, section="objs"):
        self.id, self.dff, self.txd, self.fields = id, dff, txd, fields
        self.section = section
        self.anim_block = ""
        self.time_on = self.time_off = None
        if section == "anim":
            # id, dff, txd, animBlock, drawDist, flags
            self.anim_block = fields[3] if len(fields) > 3 else ""
            self.draw_dist = _as_float(fields[4]) if len(fields) > 4 else DEFAULT_DRAW_DIST
            self.flags = _as_int(fields[5], 0) if len(fields) > 5 else 0
        elif section == "tobj":
            # the objs columns, then the hour window - fields[-1] is timeOff.
            self.time_on = _as_int(fields[-2], 0) if len(fields) > 5 else 0
            self.time_off = _as_int(fields[-1], 0) if len(fields) > 5 else 0
            self.draw_dist, self.flags = _objs_tail(fields[:-2])
        else:
            self.draw_dist, self.flags = _objs_tail(fields)

    def __repr__(self):
        return "ObjDef(%d, %r, %r, %s)" % (self.id, self.dff, self.txd, self.section)


def parse_ide(path):
    """Return {model_id: ObjDef} from the `objs`/`tobj`/`anim` sections of one .IDE."""
    out = {}
    section = None
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if section is None:
                if low in _SECTIONS:
                    section = low
                continue
            if low == "end":
                section = None
                continue
            if section not in MODEL_SECTIONS:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                mid = int(parts[0])
            except ValueError:
                continue
            out[mid] = ObjDef(mid, parts[1], parts[2], parts, section)
    return out


def parse_maps(data_dir):
    """Parse DEFAULT.IDE + every DATA/MAPS/**/*.IDE into one {model_id: ObjDef}."""
    out = {}
    default = os.path.join(data_dir, "DEFAULT.IDE")
    if os.path.isfile(default):
        out.update(parse_ide(default))
    maps = os.path.join(data_dir, "MAPS")
    for root, _d, files in os.walk(maps):
        for fn in files:
            if fn.lower().endswith(".ide"):
                out.update(parse_ide(os.path.join(root, fn)))
    return out
