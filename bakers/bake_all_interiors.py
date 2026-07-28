#!/usr/bin/env python3
"""bake_all_interiors - bake interior_<name>.pmap/.col for every enterable door.

Reads the enex pairs (enex_bake.parse_enex_lines), and for each interior-side
enex bakes the geometry pocket around its OWN entrance position (that's where
the interior door + room sit) with its area code. Dedup by name (a pair shares
one name -> one interior). Skips pockets with no geometry (savehouses whose
interior geometry we don't stream yet, empty areas).
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enex_bake
import interior_bake

def main():
    # OUTPUT dir via argv: the Quarry converter passes its own <data>/interiors.
    # Interior geometry comes from SA_ROOT (the extracted disc); enex pairs are
    # parsed from SA_ROOT's text IPLs (enex_bake honours SA_ROOT too).
    if len(sys.argv) > 1:
        interior_bake.OUT_DIR = sys.argv[1]
    os.makedirs(interior_bake.OUT_DIR, exist_ok=True)

    rows = enex_bake.parse_enex_lines()
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    done = set(); ok = 0; empty = 0; total_kb = 0
    for name in sorted(by_name):
        sides = by_name[name]
        if len(sides) < 2 or name in done:
            continue
        # interior side = the one with interior != 0
        inner = next((s for s in sides if s["interior"] != 0), None)
        if inner is None:
            continue
        done.add(name)
        area = inner["interior"]
        # bake the pocket around the interior door's OWN position
        centre = (inner["x"], inner["y"], inner["z"])
        res = interior_bake.bake_interior(name, area, centre, radius=48.0)
        if res is None:
            empty += 1
        else:
            ok += 1; total_kb += res[0] + res[1]
    print(f"\nbaked {ok} interiors, {empty} empty/skipped, {total_kb/1024:.1f}MB total")

if __name__ == "__main__":
    main()
