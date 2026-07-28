#!/usr/bin/env python3
"""pmap_dd_bump - raise the per-model draw distance of ALPHA models (foliage,
wires, fences = they carry an amode 1/2 submesh) to a floor, so they stay drawn
from an elevated / far camera instead of culling at their short SA draw_dist
(bushes ~50-150u -> invisible from a noclip fly-over).

Only alpha models are bumped - opaque walls/ground keep their SA distance (they
have LOD proxies; foliage does not, so it just pops out without a floor).

Usage: pmap_dd_bump.py <region_dir-or-file> [floor=250]
"""
import glob
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene


def bump_file(path, floor):
    sc = psp_scene.read_scene(open(path, "rb").read())
    n = 0
    for md in sc.models:
        is_alpha = any(0 <= sm.texture < len(sc.textures)
                       and ((sc.textures[sm.texture].num_levels >> 8) & 3) != 0
                       for sm in md.submeshes)
        if is_alpha and md.draw_dist < floor:
            md.draw_dist = float(floor)
            n += 1
    out = psp_scene.write_scene(sc.models, sc.textures, sc.instances, sc.grid)
    open(path, "wb").write(out)
    print("  %s: bumped %d alpha models -> dd>=%d" % (os.path.basename(path), n, floor))
    return n


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: pmap_dd_bump.py <region_dir-or-file> [floor]"); return 1
    target = argv[0]
    floor = int(argv[1]) if len(argv) > 1 else 250
    files = (sorted(glob.glob(os.path.join(target, "region_*.pmap")))
             if os.path.isdir(target) else [target])
    tot = 0
    for f in files:
        tot += bump_file(f, floor)
    print("DONE: %d alpha models bumped across %d files" % (tot, len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
