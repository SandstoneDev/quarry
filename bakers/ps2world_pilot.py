#!/usr/bin/env python3
"""ps2world_pilot: bake a world chunkset straight off the disc.


Drives the battle exporter (gvcslib work/sa_export_pmap.py: IPL+IDE+TXD+pack+
write_scene+build_grid_pmaps) with OUR ps2dff decoder monkey-patched over
gvcslib.sa_dff.decode - correct day colours (gvcslib's own decoder packs the
night set as the only colour) and a DMA-walk that survives every layout.

Usage:
 ps2world_pilot.py <extractedIsoRoot> <outDir> [x0 y0 x1 y1]

Default bbox = the Grove Street tile 12_2 of the engine's 14x14 450u grid
(origin -2994,-2938): x 2406..2856, y -2038..-1588. The output chunkset
(region_0_0.pmap + regions.bin) drops into data/world/<name>/ with
chunkset.txt pointing at it.
"""
import os
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ps2dff                                      # our decoder (day+night)
import sa_spin                                     # IFP clip ->.spin descriptor
from gvcslib import sa_dff
from gvcslib.work import sa_export_pmap
from ps2_uv_tess import cap_uv_span                # caps each triangle's UV extent


def _decode_night(blob):
    """decode_sa with the NIGHT colour set in .colors - the whole exporter then
 packs a night-lit twin of every region; ps2night_sidecar.py lifts the 5551
 colours out of it into region_*.night files."""
    m = ps2dff.decode_sa(blob)
    for mesh in m.meshes:
        mesh.colors = mesh.colors_night
    return m


# monkey-patch: every sa_dff.decode call in the exporter goes through us
NIGHT = "--night" in sys.argv
if NIGHT:
    sys.argv.remove("--night")
sa_dff.decode = _decode_night if NIGHT else ps2dff.decode_sa


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "all":
        bbox = None                                 # whole map
    elif len(sys.argv) >= 7:
        bbox = tuple(float(x) for x in sys.argv[3:7])
    else:
        bbox = (2406.0, -2038.0, 2856.0, -1588.0)   # Grove tile 12_2
    os.makedirs(out, exist_ok=True)

    # Global texture fallback: SA models reference textures living in a parent /
    # generic TXD (e.g. the wires in des_wires.txd), which a model's own-TXD
    # lookup misses -> untextured white surfaces. Build a name->TXD index once
    # (cached), decode-on-demand with a small cache.
    from tex_index import build_index
    from gvcslib.sa_txd import decode as _txd_decode
    from gvcslib.sa_img import SaImg
    _cache = os.path.join(os.path.dirname(out.rstrip("/\\")), "texindex.pkl")
    _idx = build_index(root + "/MODELS/GTA3.IMG", _cache)
    _img = SaImg(root + "/MODELS/GTA3.IMG")
    _txd_cache = {}

    _miss = set()

    def tex_fallback(name_lower):
        tn = _idx.get(name_lower)
        if tn is None:
            if name_lower not in _miss:        # DIAG: texture in NO txd -> renders white (wires)
                _miss.add(name_lower)
                print("TEXMISS", repr(name_lower))
            return None
        d = _txd_cache.get(tn)
        if d is None:
            try:
                d = _txd_decode(_img.extract(tn))
            except Exception:
                d = {}
            _txd_cache[tn] = d
        return d.get(name_lower) or next((v for k, v in d.items()
                                          if k.lower() == name_lower), None)

    # Call the exporter internals so we can (1) UV-split before slicing (fixes
    # the stretched sidewalk), (2) ship NATIVE T8 <=128px textures (sharp), and
    # (3) resolve parent/generic-TXD textures via tex_fallback. The dff_decode
    # monkey-patch carries our day+night colour stream; build_pmap's dff_decode
    # default binds the ORIGINAL decoder at import, so pass ours EXPLICITLY.
    # spin_resolver reads each IDE `anim` model's IFP clip out of gta3.img and
    # reduces it to {axis, mode, rate, amplitude} -> the.spin sidecar (the
    # animation data itself never leaves the user's disc).
    sm, st, si = sa_export_pmap.build_pmap(
        root, bbox, cell_size=400.0, tex_fmt="T8", tex_max=128,
        verbose=True, return_scene=True,
        dff_decode=(_decode_night if NIGHT else ps2dff.decode_sa),  # NIGHT: night colour stream (was always day ->.night==day, windows never lit)
        tex_fallback=tex_fallback,
        spin_resolver=sa_spin.make_resolver(_img))
    # Cap every triangle's UV extent BEFORE the grid slice. The GE reads 16-bit
    # texcoords unsigned over a 16-tile window, so a triangle that tiles a
    # texture ~14 times renders smeared, and a submesh wider than the window
    # cannot be shifted into it at all (pmap_uv_unsign then falls back to a
    # per-vertex wrap, which is not affine across the triangle -> mis-applied
    # texture). The PC-derived world never showed this because map_export/geom.py
    # applies the same cap; the PS2 path bypasses geom.py entirely.
    cap_uv_span(sm)
    sa_export_pmap.build_grid_pmaps(sm, st, si, out, 450.0, 400.0)
    print("grid export complete ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
