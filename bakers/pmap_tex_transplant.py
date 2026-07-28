#!/usr/bin/env python3
"""pmap_tex_transplant.py - swap ROAD textures in production tiles for 128px ones.

Why this direction: the only surviving whole-map sources predate two exporter
fixes (is_lod-from-binary-IPL, +351 instances), so any set built FROM them loses
the Grove bridge (its detail/LOD flags are inverted: dd 450 -> 150). The
production tiles are the only world with correct instances/flags - so instead
of rebuilding the world around 128px textures, transplant the 128px texels INTO
the production tiles:

  for each production tile (v3):
      decode to scene objects (per-model/per-texture LZ4 inflate)
      find wet_road models (position match, road_sidecar_bake logic)
      pair each with the same-position model in the DONOR tile (the tx128road
        bake, whose road textures are 128px)
      submesh k <-> submesh k: replace the production texture's texels/clut/
        dims with the donor's 128px version (dedup: replace once per texture)
      write v2 -> lz4 back to v3

Everything else in the tile - instances, flags, draw distances, models,
non-road textures - stays production bit-for-bat. Sidecars (.col/.lod/.road/
.dyn) are copied from production unchanged (instances are untouched).

Usage: python pmap_tex_transplant.py <prod_dir> <donor128_dir> <out_dir>
"""
import glob
import os
import shutil
import struct
import subprocess
import sys

import lz4.block

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from pmap_graft import DonorTile, VERT, IDX
from road_sidecar_bake import wetroad_positions


def tile_to_scene(path):
    """Full v2/v3 tile -> psp_scene objects (v3 pools inflated)."""
    dt = DonorTile(path)
    textures = []
    for i in range(dt.texture_count):
        w, h, fmt, _tf, _tb, bw, _cf, ce, nl = dt.texture(i)
        tb, cb = dt.texture_pools(i)
        textures.append(psp_scene.Texture(
            width=w, height=h, format=fmt, texel_bytes=tb, buffer_width=bw,
            clut_bytes=cb, clut_entries=ce, num_levels=nl))
    models = []
    for i in range(dt.model_count):
        fs, scnt, mscale, mcenter, br, dd = dt.model(i)
        v0, _v1, i0, _i1 = dt.model_span(i)
        vb, ib = dt.model_pools(i)
        subs = []
        for s in range(fs, fs + scnt):
            tex, vf, vc, if_, ic = dt.submesh(s)
            subs.append(psp_scene.Submesh(
                texture=tex,
                vertex_bytes=vb[(vf - v0) * VERT:(vf - v0 + vc) * VERT],
                index_bytes=ib[(if_ - i0) * IDX:(if_ - i0 + ic) * IDX]))
        models.append(psp_scene.Model(
            submeshes=subs, scale=mscale, center=mcenter,
            bound_radius=br, draw_dist=dd))
    instances = []
    for (mi, pos, quat, iscale, interior, cell) in dt.instances():
        instances.append(psp_scene.Instance(
            model=mi, pos=pos, quat=tuple(q / 32767.0 for q in quat),
            scale=iscale, interior=interior, cell=cell))
    g = struct.unpack_from("<2f f 2I I 4x", dt.data, dt.grid_off)
    grid = psp_scene.Grid(cell_size=g[2], min_x=g[0], min_y=g[1],
                          cells_x=g[3], cells_y=g[4])
    return psp_scene.Scene(models=models, textures=textures,
                           instances=instances, grid=grid)


def road_models(scene, road_pos):
    out = set()
    for i in scene.instances:
        k = (round(i.pos[0], 1), round(i.pos[1], 1), round(i.pos[2], 1))
        if k in road_pos:
            out.add(i.model)
    return out


def transplant_tile(prod_path, donor_path, out_path, road_pos):
    prod = tile_to_scene(prod_path)
    donor = tile_to_scene(donor_path)

    # same-position model pairing (detail models only need it)
    donor_by_pos = {}
    for i in donor.instances:
        donor_by_pos[(round(i.pos[0], 1), round(i.pos[1], 1),
                      round(i.pos[2], 1))] = i.model

    replaced = set()
    n_swap = 0
    for pm in road_models(prod, road_pos):
        # find one prod instance of this model to locate the donor twin
        dmi = None
        for i in prod.instances:
            if i.model == pm:
                k = (round(i.pos[0], 1), round(i.pos[1], 1), round(i.pos[2], 1))
                dmi = donor_by_pos.get(k)
                if dmi is not None:
                    break
        if dmi is None:
            continue
        psubs = prod.models[pm].submeshes
        dsubs = donor.models[dmi].submeshes
        if len(psubs) != len(dsubs):
            continue                        # structure differs: skip, stay at 64
        for ps, ds in zip(psubs, dsubs):
            if ps.texture < 0 or ds.texture < 0 or ps.texture in replaced:
                continue
            dt_ = donor.textures[ds.texture]
            if max(dt_.width, dt_.height) <= 64:
                continue                    # donor not larger: nothing to gain
            pt = prod.textures[ps.texture]
            pt.width = dt_.width; pt.height = dt_.height
            pt.format = dt_.format; pt.buffer_width = dt_.buffer_width
            pt.texel_bytes = dt_.texel_bytes
            pt.clut_bytes = dt_.clut_bytes; pt.clut_entries = dt_.clut_entries
            pt.num_levels = dt_.num_levels
            replaced.add(ps.texture)
            n_swap += 1

    out = psp_scene.write_scene(prod.models, prod.textures,
                                prod.instances, prod.grid)
    open(out_path, "wb").write(out)
    return n_swap


def main():
    if len(sys.argv) < 4:
        print("usage: pmap_tex_transplant.py <prod_dir> <donor128_dir> <out_dir>")
        return 1
    prod_dir, donor_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    road_pos, n_wet, _ = wetroad_positions()
    print(f"wet_road: {n_wet} models, {len(road_pos)} placements")

    total = 0
    for pp in sorted(glob.glob(os.path.join(prod_dir, "region_*_*.pmap"))):
        nm = os.path.basename(pp)
        dp = os.path.join(donor_dir, nm)
        op = os.path.join(out_dir, nm)
        if os.path.exists(dp):
            n = transplant_tile(pp, dp, op, road_pos)
        else:
            shutil.copy2(pp, op)
            n = 0
        if n:
            print("%-22s %d road textures -> 128" % (nm, n))
            total += n
        # sidecars: production verbatim (instances untouched)
        base = os.path.splitext(pp)[0]
        for ext in (".col", ".lod", ".road", ".dyn"):
            if os.path.exists(base + ext):
                shutil.copy2(base + ext, os.path.splitext(op)[0] + ext)
    # manifest + loose files
    for nm in ("regions.bin", "lod.bin", "water.bin"):
        src = os.path.join(prod_dir, nm)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, nm))
    print("TRANSPLANT TOTAL: %d textures upgraded" % total)

    py = sys.executable
    r = subprocess.run([py, os.path.join(TOOLS, "pmap_lz4.py"), "--dir", out_dir])
    if r.returncode != 0:
        raise SystemExit("lz4 step failed")
    print("TRANSPLANT COMPLETE:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
