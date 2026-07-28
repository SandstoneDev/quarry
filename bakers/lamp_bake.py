#!/usr/bin/env python3
"""Bake the real the source game street-lamp light positions into lamps.bin so the engine can
register point lights + coronas at night (replaces the two demo lamps).

Pipeline (gvcslib READ-ONLY): IDE objs -> the lamp-post model ids (name starts with
lamppost / streetlamp / mlamppost / lampost); IPL placements (text maps + binary
*_stream*.ipl in gta3.img) -> instances of those ids; each lamp's bulb sits ~5.5m
above the base. De-duplicated by rounded position (a lamp + its LOD proxy overlap).

Out: assets_build/lamps.bin = u32 count + per-lamp { f32 x,y,z, r,g,b, radius } (28B).
  python tools/lamp_bake.py
"""
import os, struct, sys, math
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_ide, sa_ipl, sa_img

GAME   = ""
DATA   = os.path.join(GAME, "data")
GTA3   = os.path.join(GAME, "models", "gta3.img")
OUT    = ""

BULB_Z   = 5.5                       # bulb height above the instance base
LAMP_COL = (1.0, 0.58, 0.24)         # warm sodium-vapour orange
LAMP_RAD = 14.0
NAME_PREFIXES = ("lamppost", "streetlamp", "mlamppost", "lampost")
# keep to Los Santos so we don't ship LV/SF decorative posts of the same name family
LS_BBOX = (-400.0, 3200.0, -3000.0, -600.0)   # x0,x1,y0,y1


def main():
    defs = sa_ide.parse_maps(DATA)
    lamp_ids = {mid for mid, d in defs.items()
                if d.dff.lower().startswith(NAME_PREFIXES)}
    print("lamp model ids:", sorted(lamp_ids),
          "->", sorted({defs[m].dff for m in lamp_ids}))

    insts = []
    # text IPLs
    maps = os.path.join(DATA, "maps")
    for root, _d, files in os.walk(maps):
        for fn in files:
            if fn.lower().endswith(".ipl"):
                try: insts += sa_ipl.parse_text_ipl(os.path.join(root, fn))
                except Exception: pass
    # binary IPLs in gta3.img
    img = sa_img.SaImg(GTA3)
    for nm in img.names():
        if nm.lower().endswith(".ipl"):
            try:
                blob = img.extract(nm)
                if blob[:4] == b"bnry":
                    insts += sa_ipl.parse_binary_ipl(blob)
            except Exception: pass
    print("total instances scanned:", len(insts))

    seen = set()
    lamps = []
    x0, x1, y0, y1 = LS_BBOX
    for it in insts:
        if it.model_id not in lamp_ids:
            continue
        x, y, z = it.pos
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        key = (round(x, 1), round(y, 1), round(z, 1))
        if key in seen:
            continue
        seen.add(key)
        lamps.append((x, y, z + BULB_Z))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<I", len(lamps)))
        for (x, y, z) in lamps:
            f.write(struct.pack("<7f", x, y, z,
                                LAMP_COL[0], LAMP_COL[1], LAMP_COL[2], LAMP_RAD))
    print("wrote %s  (%d LS street lamps)" % (OUT, len(lamps)))
    # quick Ganton-area count
    g = sum(1 for (x, y, z) in lamps if 2250 <= x <= 2800 and -2000 <= y <= -1400)
    print("  of which in the Ganton bbox:", g)


if __name__ == "__main__":
    main()
