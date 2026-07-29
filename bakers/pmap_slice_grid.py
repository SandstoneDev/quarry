#!/usr/bin/env python3
"""Slice an ALREADY-tessellated whole-map .pmap into regional tiles.

The --grid baker (sa_export_pmap.py) slices the *raw* scene; its region files
are therefore NOT tessellated and hit the PSP guard-band hole bug on camera
rotate. This driver instead reads a finished (tessellated + tex-capped)
whole-map .pmap back into scene objects and re-runs the same regional slicer, so
every region tile inherits the tessellated geometry verbatim.

Usage: python pmap_slice_grid.py <in_tess.pmap> <out_dir> [tile_units=900]
"""
import os
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)

from gvcslib import psp_scene
from gvcslib.work.sa_export_pmap import build_grid_pmaps


def main():
    if len(sys.argv) < 3:
        print("usage: pmap_slice_grid.py <in_tess.pmap> <out_dir> [tile_units=900]")
        return 1
    src = sys.argv[1]
    out_dir = sys.argv[2]
    tile = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0

    print("reading", src)
    data = open(src, "rb").read()
    scene = psp_scene.read_scene(data)
    print("scene: models=%d textures=%d instances=%d grid=%dx%d cell=%.0f"
          % (len(scene.models), len(scene.textures), len(scene.instances),
             scene.grid.cells_x, scene.grid.cells_y, scene.grid.cell_size))

    build_grid_pmaps(scene.models, scene.textures, scene.instances,
                     out_dir, tile, scene.grid.cell_size)
    print("slice complete ->", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
