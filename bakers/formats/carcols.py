"""the source game CARCOLS.DAT - vehicle colour palette + per-vehicle paint combos.

The engine paints a car by replacing materials whose colour is a sentinel "marker"
with the chosen palette colour (then the texture MODULATES against it).
Confirmed from the source game: ,
palette , .

File layout (ASCII, '#' comments, commas == spaces, sections end at `end`):
 col # the palette - one (R,G,B) per line, POSITIONAL (line index = colour id)
 R,G,B
 ...
 end
 car # 2-colour vehicles: <name> p1 s1 p2 s2 ... (combos of primary,secondary)
 taxi, 6,1
 ...
 end
 car4 # 4-colour vehicles: <name> p s t q p s t q ...
 ...
 end

Marker material colours (RGB) → paint slot, from SetEditableMaterials immediates:
 (60,255,0)=primary (255,0,175)=secondary (0,255,255)=tertiary (255,0,255)=quaternary
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# material RGB marker -> slot index (primary/secondary/tertiary/quaternary)
PAINT_MARKERS: Dict[Tuple[int, int, int], int] = {
    (60, 255, 0): 0,
    (255, 0, 175): 1,
    (0, 255, 255): 2,
    (255, 0, 255): 3,
}

# vehicle LIGHT state markers (2nd sentinel group in):
# front-left / front-right / rear-left / rear-right lamp materials. The engine drives
# these at runtime (day = neutral texture, night = emissive); a static viewer must
# NOT tint by them (else one headlight renders orange, the other turquoise).
LIGHT_MARKERS: Dict[Tuple[int, int, int], int] = {
    (255, 175, 0): 0,
    (0, 255, 200): 1,
    (185, 255, 0): 2,
    (255, 60, 0): 3,
}


def parse_carcols(text: str) -> Dict:
    """Parse CARCOLS.DAT → {'palette': {id:(r,g,b)}, 'cars': {name:[combo,...]}}.

 A combo is a tuple of palette ids (len 2 for `car`, len 4 for `car4`)."""
    palette: Dict[int, Tuple[int, int, int]] = {}
    cars: Dict[str, List[Tuple[int, ...]]] = {}
    section: Optional[str] = None
    col_idx = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        low = line.lower()
        if low in ("col", "car", "car4"):
            section = low
            continue
        if low == "end":
            section = None
            continue
        toks = line.split()
        if section == "col":
            try:
                palette[col_idx] = (int(toks[0]), int(toks[1]), int(toks[2]))
            except (ValueError, IndexError):
                pass
            col_idx += 1
        elif section in ("car", "car4"):
            name = toks[0].lower()
            ids: List[int] = []
            for t in toks[1:]:
                try:
                    ids.append(int(t))
                except ValueError:
                    pass
            step = 2 if section == "car" else 4
            combos = [tuple(ids[i:i + step]) for i in range(0, len(ids) - step + 1, step)]
            if combos:
                cars.setdefault(name, []).extend(combos)
    return {"palette": palette, "cars": cars}


def combo_count(parsed: Dict, name: str) -> int:
    return len(parsed.get("cars", {}).get(name.lower(), []))


def resolve_colors(parsed: Dict, name: str, combo: int = 0) -> Optional[List[Tuple[int, int, int]]]:
    """Palette RGBs for [primary, secondary, (tertiary, quaternary)] of a vehicle's
 chosen colour combo, or None if the vehicle is not in carcols."""
    combos = parsed.get("cars", {}).get(name.lower())
    if not combos:
        return None
    ids = combos[combo % len(combos)]
    pal = parsed.get("palette", {})
    return [pal.get(cid, (255, 255, 255)) for cid in ids]


def to_json(parsed: Dict) -> Dict:
    return {
        "num_colours": len(parsed.get("palette", {})),
        "num_vehicles": len(parsed.get("cars", {})),
        "palette": [list(parsed["palette"][i]) for i in sorted(parsed.get("palette", {}))],
    }
