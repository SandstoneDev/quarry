#!/usr/bin/env python3
"""pack - float submeshes -> psp_scene.Model (s16 prims) + deduped textures.

psp_mesh.pack_model does the s16 position/UV quantization - SAFE now because
geom.py guarantees UV span <= 15.5 tiles (s16 range is +-8 after the *4096
fixed-point, and rebased spans stay inside after the [0,1) rebase).
Textures: 64px max, T4 when the downscaled image has <=16 unique colours,
else T8; mips on; deduped by texture name across the whole export.
`_make_texture` (gvcslib work/sa_export_pmap, import-only) packs the authored
dict into psp_scene.Texture with the runtime conventions (buffer_width,
alpha_mode folded into num_levels byte 1).
"""
import os
import re
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_mesh, psp_tex, psp_scene
from gvcslib.sa_dff import SaModel, SaMesh
from gvcslib.work.sa_export_pmap import _make_texture

TEX_MAX = 64
# Road-128 REVERTED (build 250 HW test): bumping 52 road/ground textures to 128 added
# ~+606KB/tile - 6x the UV-tessellation's +~100KB - and the streaming cache (only
# ~1.9MB of a ~4.3MB heap after peds/audio/COL) overflowed: notload spiked to 385,
# strmq to 1023, "unloaded patches even where standing" on HW. Textures are the wrong
# lever on this memory budget. Keep everything at 64; road sharpness needs a memory win
# first (T4 mostly N/A here, or a bigger cache). The UV-stripe fix stays (cheap geom).
TEX_MAX_ROAD = 64
_ROAD_RE = re.compile(
    r"road|tarmac|pave|cross|sidew|hiway|highway|offroad|_path|kerb|curb|asphalt",
    re.IGNORECASE)
# PC-DFF prelit is authored darker than the shipped world (the old export decoded
# PS2 SAR-Mod colours whose 128=1.0 convention doubled them; the whole timecycle
# is tuned against THAT brightness). Empirical match: new/old ground ratio was
# ~0.73 at noon -> boost 1.35 with clamp.
PRELIT_BOOST = 1.35


def _downscale(rgba, w, h, cap=TEX_MAX):
    if max(w, h) <= cap:
        return rgba, w, h
    from PIL import Image
    im = Image.frombytes("RGBA", (w, h), bytes(rgba))
    nw = max(1, w * cap // max(w, h))
    nh = max(1, h * cap // max(w, h))
    im = im.resize((nw, nh), Image.LANCZOS)
    return im.tobytes(), nw, nh


class TexPool:
    """Dedup by (txd, texture) name pair; author once. index() returns the
 psp_scene texture index. The namespace is PER-TXD - SA reuses generic
 names ('grass', 'wall1') across txds with different pixels; a global
 name key handed the grass a road-arrow decal (build-198 pilot, GE debugger
 showed a 64x32 arrow texture bound to the lawn)."""
    def __init__(self):
        self.list = []                     # psp_scene.Texture
        self.byname = {}
        self.missing = []

    def index(self, name, txd, txd_name=""):
        nm = (name or "").lower()
        if not nm:
            return -1
        key = (txd_name.lower(), nm)
        if key in self.byname:
            return self.byname[key]
        idx = -1
        entry = txd.get(nm) if txd else None
        if entry is not None:
            w, h, rgba = entry
            cap = TEX_MAX_ROAD if _ROAD_RE.search(nm) else TEX_MAX
            if os.environ.get("TEX_DIAG"):
                import numpy as _np
                nc = len(_np.unique(_np.frombuffer(bytes(rgba), _np.uint32)))
                sys.stderr.write("TEX %-24s %dx%d colors=%d cap=%d txd=%s\n"
                                 % (nm, w, h, nc, cap, txd_name))
            rgba, w, h = _downscale(rgba, w, h, cap)
            # T8, NO MIPS - matching the shipped world exactly: the old
            # pipeline (pmap_tex_downscale) forced num_levels=1 everywhere, so
            # the world renderer's multi-level upload had NEVER run before
            # these tiles - and the first mipped world tiles striped in-game
            # (build-199 solo-tile experiment). T4 + mips return only after a
            # dedicated GE roundtrip test.
            fmt = "T8"
            # GE-TEST (TEX_T4=1, opt-in): the build-199 stripe was T4+MIPS; T4 WITHOUT
            # mips is untested. Author any texture that is already <=16 colours post-
            # downscale as T4 (half the texel bytes + a 16-entry CLUT vs 256) so the
            # renderer's GU_PSM_T4 path (PMAP_FMT_T4) gets a real workout. If these
            # render clean on HW, T4 unlocks the "same memory, 2x resolution" win.
            if os.environ.get("TEX_T4"):
                import numpy as _np
                if len(_np.unique(_np.frombuffer(bytes(rgba), _np.uint32))) <= 16:
                    fmt = "T4"
            try:
                t = psp_tex.author_psp_texture(rgba, w, h, fmt=fmt, mipmaps=False)
                # author classifies alpha only inside its [96..200) mid-band;
                # SA shadow/decal layers sit OUTSIDE it (subtle 200-249 or deep
                # 32-95 alphas) and came out amode=0 -> drawn OPAQUE: the fence/
                # house shadow strips rendered as solid dark tiling bands (THE
                # over >2% of pixels forces blend (or cutout if mostly holes).
                if t.get("alpha_mode", 0) == 0:
                    import numpy as _np
                    al = _np.frombuffer(bytes(rgba), _np.uint8)[3::4]
                    trans = int((al < 32).sum())
                    mid = int(((al >= 32) & (al < 250)).sum())
                    if trans + mid > al.size * 0.02:
                        t["alpha_mode"] = 1 if trans > mid else 2
                self.list.append(_make_texture(t))
                idx = len(self.list) - 1
            except Exception as e:
                self.missing.append("%s: %s" % (key, e))
        else:
            self.missing.append(key)
        self.byname[key] = idx
        return idx


def pack_processed(parts, texpool, txd, dd, txd_name=""):
    """geom.process_geometry output -> psp_scene.Model. Returns None if empty."""
    model = SaModel()
    texidx = []
    for part in parts:
        remap = {}
        me = SaMesh(material_index=len(texidx))
        me.positions = []
        me.uv = []
        me.colors = []
        me.triangles = []
        for tri in part["tris"]:
            t = []
            for (p, uv, col) in tri:
                key = (round(p[0], 3), round(p[1], 3), round(p[2], 3),
                       round(uv[0], 4), round(uv[1], 4), col)
                li = remap.get(key)
                if li is None:
                    li = len(me.positions)
                    remap[key] = li
                    me.positions.append(p)
                    me.uv.append(uv)
                    r, g, b, a = col
                    r = min(255, int(r * PRELIT_BOOST))
                    g = min(255, int(g * PRELIT_BOOST))
                    b = min(255, int(b * PRELIT_BOOST))
                    me.colors.append((r << 24) | (g << 16) | (b << 8) | a)
                t.append(li)
            me.triangles.append(tuple(t))
        if not me.triangles:
            continue
        model.meshes.append(me)
        texidx.append(texpool.index(part["mat"].texture_name, txd, txd_name))
    if not model.meshes:
        return None
    packed = psp_mesh.pack_model(model)
    subs = []
    for prim in packed["prims"]:
        ti = texidx[prim["material_index"]]
        subs.append(psp_scene.Submesh(
            texture=ti if ti >= 0 else -1,
            vertex_bytes=prim["vertex_bytes"],
            index_bytes=prim["index_bytes"]))
    cx, cy, cz = packed["center"]
    br = 0.0
    for mesh in model.meshes:
        for (x, y, z) in mesh.positions:
            d2 = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
            if d2 > br:
                br = d2
    return psp_scene.Model(submeshes=subs, scale=packed["scale"],
                           center=packed["center"], bound_radius=br ** 0.5,
                           draw_dist=float(dd))
