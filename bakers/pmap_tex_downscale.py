#!/usr/bin/env python3
"""Downscale the textures inside already-sliced region_<rx>_<ry>.pmap tiles in place.

The streaming cache (~5.8MB on real HW) THRASHES because the visible working set's
textures don't fit -> the bg thread freads constantly (60-160ms/frame), stealing the
main loop's CPU. Halving the texture cap (128 -> 64) shrinks each texture ~4x (area),
so far more fit resident -> the thrash stops. This re-encodes per tile (no whole-map
re-bake): decode each swizzled T8 plane to RGBA, bilinear-downscale, re-author T8.

Instance order / positions are untouched, so region_*.lod stay valid (re-run
lod_bake_regions afterwards anyway to be safe).

Usage: python pmap_tex_downscale.py <region_dir> [maxdim=64] [--road-tier <px>]

--road-tier <px> (b397, the tx128 road tier): textures used by wet_road MODELS
(the same position-match as road_sidecar_bake) cap at <px> instead of maxdim --
sharp asphalt/lane markings while everything else stays at the 64 budget.
"""
import sys, os, glob

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene, psp_tex
from PIL import Image

MAXDIM = 64


def downscale_tex(t, maxdim=None):
    if maxdim is None:
        maxdim = MAXDIM
    if t.format != psp_tex.GU_PSM_T8:          # only T8 planes (skip T4/RGBA)
        return False
    if max(t.width, t.height) <= maxdim:
        return False
    alpha_mode = (t.num_levels >> 8) & 0xFF     # PRESERVE: high byte = blend/cull class
    d = {"width": t.width, "height": t.height, "gu_pixfmt": t.format,
         "texel_bytes": t.texel_bytes, "clut_bytes": t.clut_bytes}
    rgba = psp_tex.decode_psp_texture(d)         # -> RGBA8888 level 0
    img = Image.frombytes("RGBA", (t.width, t.height), rgba)
    w2, h2 = t.width, t.height
    while max(w2, h2) > maxdim:                   # min 16: T8 swizzle needs stride%16==0
        w2 = max(16, w2 // 2)
        h2 = max(16, h2 // 2)
    img = img.resize((w2, h2), Image.BILINEAR)
    a = psp_tex.author_psp_texture(img.tobytes(), w2, h2, fmt="T8", mipmaps=False)
    t.width = a["width"]; t.height = a["height"]; t.format = a["gu_pixfmt"]
    t.texel_bytes = a["texel_bytes"]
    t.buffer_width = a["width"]                  # T8 stride == width
    t.clut_bytes = a["clut_bytes"]; t.clut_entries = a["clut_entries"]
    t.num_levels = 1 | (alpha_mode << 8)         # drop mips, keep alpha class
    return True


def main():
    argv = [a for a in sys.argv[1:]]
    road_tier = 0
    if "--road-tier" in argv:
        i = argv.index("--road-tier")
        road_tier = int(argv[i + 1])
        del argv[i:i + 2]
    keep_stretched = 0          # keep UNDER-TILED ground/canal textures sharp (0..1 UV over big area)
    if "--keep-stretched" in argv:
        i = argv.index("--keep-stretched")
        keep_stretched = int(argv[i + 1])
        del argv[i:i + 2]
    if not argv:
        print("usage: pmap_tex_downscale.py <region_dir> [maxdim] [--road-tier <px>]"); return 1
    region_dir = argv[0]
    global MAXDIM
    if len(argv) > 1:
        MAXDIM = int(argv[1])
    road_pos = None
    if road_tier:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from road_sidecar_bake import wetroad_positions, region_road_models
        road_pos, n_wet, _ = wetroad_positions()
        print(f"road tier {road_tier}px: {n_wet} wet_road models, {len(road_pos)} placements")
    tiles = sorted(glob.glob(os.path.join(region_dir, "region_*.pmap")))
    if not tiles:
        print("no region_*.pmap in", region_dir); return 1
    tot_before = tot_after = 0
    for path in tiles:
        before = os.path.getsize(path)
        road_tex = set()
        sc = psp_scene.read_scene(open(path, "rb").read())
        if road_pos is not None:
            from road_sidecar_bake import region_road_models
            for mi in region_road_models(path, road_pos):
                for sm in sc.models[mi].submeshes:
                    if sm.texture >= 0:
                        road_tex.add(sm.texture)
        stretched_tex = set()
        if keep_stretched:
            import numpy as np
            for md in sc.models:
                for sm in md.submeshes:
                    if sm.texture < 0:
                        continue
                    v = np.frombuffer(sm.vertex_bytes, np.uint8).reshape(-1, 12)
                    uv = v[:, 0:4].view("<i2").astype(np.float32) / 4096.0
                    pos = v[:, 6:12].view("<i2").astype(np.float32) * md.scale
                    uspan = max(uv[:, 0].max() - uv[:, 0].min(), uv[:, 1].max() - uv[:, 1].min())
                    wsz = max(pos[:, 0].max() - pos[:, 0].min(), pos[:, 1].max() - pos[:, 1].min())
                    # world-units per texture tile; > 40u/tile = under-tiled ground/canal
                    # (0..1 UV stretched over a big flat piece) -> keep it sharp, not 64px-blurry
                    if wsz > 40.0 and uspan > 0.01 and (wsz / uspan) > 40.0:
                        stretched_tex.add(sm.texture)
        n = nroad = nstr = 0
        for ti, t in enumerate(sc.textures):
            cap = road_tier if ti in road_tex else MAXDIM
            if keep_stretched and ti in stretched_tex and cap < keep_stretched:
                cap = keep_stretched
                nstr += 1
            # ALPHA art (amode 1/2 = foliage/wires/cracks) carries THIN detail that
            # a 64px cap averages BELOW the alpha-test threshold -> the whole thing
            # vanishes (sparse bushes gone, thin wires gone). Keep it at 128. Opaque
            # (amode 0 = walls/ground) still drops to 64 to hold the cache budget.
            if ((t.num_levels >> 8) & 3) != 0 and cap < 128:
                cap = 128
            if downscale_tex(t, cap):
                n += 1
                if ti in road_tex:
                    nroad += 1
        out = psp_scene.write_scene(sc.models, sc.textures, sc.instances, sc.grid)
        open(path, "wb").write(out)
        after = len(out)
        tot_before += before; tot_after += after
        extra = (" road-tex=%d @%d" % (len(road_tex), road_tier)) if road_pos is not None else ""
        print("%-22s %d tex down%s  %.1f -> %.1f MB" %
              (os.path.basename(path), n, extra, before / 1e6, after / 1e6))
    print("TOTAL %.1f -> %.1f MB (%.0f%%)" %
          (tot_before / 1e6, tot_after / 1e6, 100.0 * tot_after / tot_before))
    return 0


if __name__ == "__main__":
    sys.exit(main())
