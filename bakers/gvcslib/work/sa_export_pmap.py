#!/usr/bin/env python3
"""Drive the decoded-the source game -> PSP v2 `.pmap` back-end for a region (or --all).

Pipeline (sibling modules, none edited):
    sa_ipl   placements (model_id, pos, rot-quat xyzw, interior)
    sa_ide   model_id -> ObjDef(.dff, .txd, section, flags, draw_dist, ...)
             - objs AND tobj AND anim; read the RESOLVED attributes, the three
             sections do not share a column layout
    sa_dff   .dff blob -> SaModel{meshes[], materials[{texture_name,color}]}
    sa_txd   .txd blob -> {name: (w, h, rgba8888_bytes)}
    psp_mesh.pack_model        SaModel -> compact GE int16 prims (per material)
    psp_tex.author_psp_texture RGBA8888 -> swizzled T8/T4 plane + linear CLUT
    psp_scene.write_scene      welds vertex/index/texel/clut POOLS + model/
                               submesh/texture/instance tables + XY zone grid
                               into one streamable little-endian v2 `.pmap`
                               the C engine (work/psp_engine) draws verbatim.

Each psp_mesh prim becomes one Submesh referencing the texture authored for that
material (deduplicated by texture name across the whole export).  Coordinates
stay in the source game native space (Z-up, XY ground) - the viewer is Z-up.

IDE `anim` rows are CAnimatedBuilding clumps (a revolving LV sign, a windmill).
With a `spin_resolver` callback the export splits each rotating atomic into its
own model + instance, parked at the clump frame's world position, and tags it
with the IFP clip reduced to {axis, mode, rate, amplitude} - which
build_grid_pmaps writes as the per-region `.spin` sidecar, so no animation data
ever ships.

IDE `tobj` rows are time-of-day models (neon, lit-window overlays, the _dy/_nt
swap pairs) that exist only inside an hour window.  Their geometry bakes like any
other model; the window itself rides the per-region `.tobj` sidecar, keyed by the
INSTANCE index - the engine hides a listed instance outside its hours.

Run:
    cd gvcslib && PYTHONPATH=. python gvcslib/work/sa_export_pmap.py        # LA bbox
    cd gvcslib && PYTHONPATH=. python gvcslib/work/sa_export_pmap.py --all  # whole map
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time

from gvcslib import sa_ipl, sa_ide, sa_dff, sa_txd, psp_mesh, psp_tex, psp_scene
from gvcslib import sa_dff_pc, sa_txd_d3d9
from gvcslib.sa_img import SaImg

ROOT = ""
ROOT_PC = ""

# Default region: the downtown-LA bbox (proven ~1.5 MB vertical slice).
DEFAULT_BBOX = (1300.0, -1900.0, 1900.0, -1400.0)


def _img_name(stem, ext):
    s = stem.lower()
    return s if s.endswith(ext) else s + ext


def _pow2_ok(w, h):
    """psp_tex swizzle needs byte-stride % 16 == 0 and h % 8 == 0."""
    return w >= 16 and h >= 8 and (w % 16 == 0) and (h % 8 == 0)


def _downscale(rgba, w, h, maxdim):
    """Downscale an RGBA image so max(w,h) <= maxdim, halving (keeps pow2).
    Returns (w2, h2, rgba2).  No-op if already small enough or maxdim<=0."""
    if maxdim <= 0 or (w <= maxdim and h <= maxdim):
        return w, h, rgba
    from PIL import Image
    w2, h2 = w, h
    while w2 > maxdim or h2 > maxdim:
        w2 = max(16, w2 // 2)
        h2 = max(8, h2 // 2)
    img = Image.frombytes("RGBA", (w, h), bytes(rgba)).resize(
        (w2, h2), Image.BILINEAR)
    return w2, h2, img.tobytes()


def _dilate_rgb(rgba, w, h, passes=4):
    """Flood opaque RGB into transparent texels (alpha<128). PS2 stores a black
    RGB under transparent alpha; PSP bilinear filtering blends that black into the
    leaf/wire EDGE -> a dark fringe / dirty speckle on foliage. Bleeding the
    neighbour's opaque colour outward means the filter samples leaf-green either
    side of the cutout edge instead of leaf+black. Alpha is UNCHANGED (the cutout
    shape stays exact); only the invisible RGB under transparent texels is filled."""
    import numpy as np
    a = np.frombuffer(rgba, np.uint8).reshape(h, w, 4).copy()
    op = a[:, :, 3] >= 128
    if op.all() or not op.any():
        return rgba                       # fully opaque or fully empty: nothing to bleed
    rgb = a[:, :, :3]
    for _ in range(passes):
        if op.all():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sop = np.roll(np.roll(op, dy, 0), dx, 1)
            srgb = np.roll(np.roll(rgb, dy, 0), dx, 1)
            fill = (~op) & sop
            rgb[fill] = srgb[fill]
            op = op | fill
    return a.tobytes()


def _make_texture(tex: dict) -> psp_scene.Texture:
    """Wrap an author_psp_texture() dict into a psp_scene.Texture (v2).

    texel_bytes may hold a whole mip chain, so derive the LEVEL-0 buffer width
    from the pixel width (not from the byte length)."""
    w = tex["width"]
    if tex["gu_pixfmt"] == psp_tex.GU_PSM_T8:
        buffer_width = w                     # 1 byte/texel, stride == w
    else:                                    # T4: stride = max(w//2,16) bytes
        buffer_width = max(w, 32)            # texels = stride * 2
    # pack alpha_mode (byte 1) alongside mip count (byte 0) into num_levels.
    nl = tex.get("num_levels", 1) | (tex.get("alpha_mode", 0) << 8)
    return psp_scene.Texture(
        width=w, height=tex["height"], format=tex["gu_pixfmt"],
        texel_bytes=tex["texel_bytes"], buffer_width=buffer_width,
        clut_bytes=tex["clut_bytes"], clut_entries=tex["clut_entries"],
        num_levels=nl,
    )


def _decal_edge_fade(rgba, w, h, margin=0.08):
    """Fade a decal texture's alpha to 0 in the outer `margin` ring so the coplanar
    decal QUAD leaves no hard rectangular edge on the surface it overlays: without
    this the mesh border of the semi-transparent plane is visible in game. Content
    near the centre is untouched; only the outermost ~`margin` fraction of the
    texture ramps the alpha down to 0 at the very edge. Returns new RGBA bytes
    (same length)."""
    import numpy as np
    a = np.frombuffer(bytes(rgba), np.uint8).reshape(h, w, 4).astype(np.float32)
    mx = max(1.0, w * margin)
    my = max(1.0, h * margin)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = np.clip(np.minimum(xx, w - 1 - xx) / mx, 0.0, 1.0)
    fy = np.clip(np.minimum(yy, h - 1 - yy) / my, 0.0, 1.0)
    f = np.minimum(fx, fy)
    f = f * f * (3.0 - 2.0 * f)                      # smoothstep 0..1
    a[..., 3] *= f
    return a.astype(np.uint8).tobytes()


def _rot_is_identity(r, eps=1e-4):
    """3x3 row-major identity within float-stream noise (a stock DFF stores
    1.0 as 0.99999994 often enough that an exact compare rejects half the
    LV signs)."""
    return all(abs(r[k] - (1.0 if k in (0, 4, 8) else 0.0)) <= eps
               for k in range(9))


class _SubModel:
    """A slice of a decoded SaModel: the subset of meshes that hangs off one
    clump frame, sharing the parent model's material table.  psp_mesh.pack_model
    and the submesh loop only ever touch .meshes / .materials."""
    __slots__ = ("meshes", "materials")

    def __init__(self, meshes, materials):
        self.meshes, self.materials = meshes, materials


def _bake_clump_frames(model):
    """Move every remaining mesh from its clump-frame space into MODEL space.

    ONLY for IDE `anim` models: CFileLoader keeps a CClumpModelInfo's frame tree,
    so a static atomic really does sit at its frame's offset.  An ATOMIC model
    (objs/tobj) must NOT get this - SetRelatedModelInfoCB hands it a brand-new
    identity frame, so its authored offset is not part of the in-game model."""
    for mesh in model.meshes:
        ltm = getattr(mesh, "frame_ltm", None)
        if not ltm:
            continue
        r, t = ltm
        mesh.positions = [(x * r[0] + y * r[3] + z * r[6] + t[0],
                           x * r[1] + y * r[4] + z * r[7] + t[1],
                           x * r[2] + y * r[5] + z * r[8] + t[2])
                          for (x, y, z) in mesh.positions]


def _inst_rot(quat):
    """Row-major 3x3 with v_world = R @ v_local for an IPL instance quaternion.

    MIRRORS the engine's build_inst_rot (src/game_sa/Renderer.c) exactly, fast
    path included: SA's CFileLoader::CreateEntityFromInstance stores the
    CONJUGATE, and near-vertical objects take a yaw-only branch that drops the
    small qx/qy.  A split animated atomic is placed at pos + R*framePos, so if
    this disagreed with the engine the sign would drift off its pole."""
    qx, qy, qz, qw = quat
    if abs(qx) <= 0.05 and abs(qy) <= 0.05:
        n = qz * qz + qw * qw
        if n > 1e-8:
            iv = 1.0 / math.sqrt(n)
            qz *= iv; qw *= iv
        ch = qw * qw - qz * qz
        sh = -2.0 * qw * qz
        return (ch, -sh, 0.0, sh, ch, 0.0, 0.0, 0.0, 1.0)
    qx, qy, qz = -qx, -qy, -qz
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n > 1e-8:
        iv = 1.0 / math.sqrt(n)
        qx *= iv; qy *= iv; qz *= iv; qw *= iv
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy))


def _split_animated(model, spins, verbose=False, tag=""):
    """Pull the rotating atomics out of a CAnimatedBuilding clump.

    Removes them from `model.meshes` (so the caller can bake the frame matrix
    into everything that stays) and returns [(spin, frame_offset, meshes)].

    `spins` = {frame_name_lower: (axis, mode, rate, amp)} from the model's IFP
    clip.  An animated atomic only earns its own model+instance when a per-
    INSTANCE matrix tweak can actually reproduce it - the same shape as the
    existing .sway hook, no per-submesh matrix switching:

      * nothing else animated in its parent chain, and nothing animated below it
        (a nested rotator needs two matrices; the pumpjacks are that, phase 2);
      * its local-to-model rotation is identity, so the model keeps frame-local
        vertices, the .spin axis stays the clip's own axis and the instance
        offset is just the frame translation.

    Everything that fails stays in the static model with its frame matrix baked
    - position-correct, simply not moving."""
    frames = getattr(model, "frames", None)
    if not frames or not spins:
        return []
    anim_idx = {i for i, f in enumerate(frames) if f.name.lower() in spins}
    if not anim_idx:
        return []

    def chain(i):
        seen = 0
        p = frames[i].parent
        while 0 <= p < len(frames) and seen < len(frames):
            yield p
            p = frames[p].parent
            seen += 1

    nested = set()
    for i in anim_idx:
        for p in chain(i):
            if p in anim_idx:
                nested.add(i)        # animated under an animated frame
                nested.add(p)        # ... and the frame it hangs off
    eligible = {}
    for i in sorted(anim_idx):
        f = frames[i]
        why = None
        if i in nested:
            why = "nested rotator"
        elif not _rot_is_identity(f.ltm_rot):
            why = "frame is rotated"
        if why:
            if verbose:
                print("    spin skip %s.%s: %s" % (tag, f.name, why))
            continue
        eligible[i] = (spins[f.name.lower()], f.ltm_pos)

    if not eligible:
        return []
    groups = {}
    static = []
    for mesh in model.meshes:
        fi = getattr(mesh, "frame_index", -1)
        if fi in eligible:
            groups.setdefault(fi, []).append(mesh)
        else:
            static.append(mesh)
    model.meshes = static
    return [(eligible[fi][0], eligible[fi][1], groups[fi])
            for fi in sorted(groups)]


def build_pmap(root, bbox, *, no_interior=True, cell_size=400.0,
               tex_fmt="T8", tex_max=0, verbose=True,
               dff_decode=sa_dff.decode, txd_decode=sa_txd.decode,
               return_scene=False, tex_fallback=None, decal_names=None,
               spin_resolver=None):
    # decal_names: set of lower-case texture names used by DRAW_LAST (IDE flag
    # 0x4) models - wires, cracks, graffiti, ground-holes. SA draws these in the
    # trailing blended pass; force alpha_mode 2 (translucent) so our engine routes
    # them to the sorted z-write-OFF + two-sided pass. Per-texture alpha detection
    # alone misclassifies them opaque -> white ribbons / z-fighting cracks.
    _decal = decal_names or set()
    # tex_fallback(name_lower) -> (w,h,rgba) or None: resolves a texture that is
    # NOT in a model's own TXD from the global index (SA parent/generic TXDs,
    # e.g. wires in des_wires.txd) instead of leaving the surface untextured.
    im = SaImg(root + "/MODELS/GTA3.IMG")
    ide = sa_ide.parse_maps(root + "/DATA")
    insts = sa_ipl.load_all(root + "/DATA", im)
    if verbose:
        print("placements loaded:", len(insts))

    if bbox is None:
        def keep(i):
            return not (no_interior and i.interior not in (0, -1, 13))
    else:
        x0, y0, x1, y1 = bbox
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        def keep(i):
            if no_interior and i.interior not in (0, -1, 13):
                return False
            x, y, _ = i.pos
            return xmin <= x <= xmax and ymin <= y <= ymax

    sel = [i for i in insts if keep(i)]
    want = sorted({i.model_id for i in sel})
    if verbose:
        print("instances in selection:", len(sel),
              "| unique models:", len(want))

    have = set(n.lower() for n in im.names())

    scene_models = []                 # psp_scene.Model
    scene_textures = []               # psp_scene.Texture
    tex_id_of = {}                    # texture_name(lower) -> texture index
    model_index_of = {}              # model_id -> index into scene_models
    lod_ids = set()                  # model_ids that are LOD proxies

    def author_named(name, txd, txd_name, tmax=None):
        """Author (or fetch cached) a texture by (TXD, material name); return
        its index, or -1 if it can't be resolved/authored.

        Keyed by (txd, name) NOT name alone: SA reuses the same texture name in
        different TXDs for DIFFERENT images (~35% of names collide), so a
        name-only cache puts the wrong texture on ~a third of surfaces.

        tmax overrides the global tex_max for this texture (ground/road surfaces
        tile ~1x so they want a higher cap than walls/props); the cache key
        includes it so the same image can exist at two resolutions."""
        if tmax is None:
            tmax = tex_max
        nm = (name or "").strip().lower()
        if not nm:
            return -1
        key = ((txd_name or "").lower(), nm, tmax)
        if key in tex_id_of:
            return tex_id_of[key]
        entry = txd.get(name) or txd.get(nm)
        if entry is None:
            for k, v in txd.items():
                if k.lower() == nm:
                    entry = v
                    break
        if entry is None and tex_fallback is not None:
            entry = tex_fallback(nm)          # global index: parent/generic TXDs
        if entry is None:
            tex_id_of[key] = -1
            return -1
        w, h, rgba = entry
        if not _pow2_ok(w, h) or len(rgba) != w * h * 4:
            tex_id_of[key] = -1
            return -1
        if tmax:
            w, h, rgba = _downscale(rgba, w, h, tmax)
            if not _pow2_ok(w, h):
                tex_id_of[key] = -1
                return -1
        # native-ish format: T4 when the image uses <=16 unique colours (the PS2
        # world majority) -> ~half the texel bytes -> lighter tiles -> streaming
        # keeps up (no pop-in / white surfaces). T8 otherwise. Bounded scan.
        use_fmt = tex_fmt
        if tex_fmt != "T4":
            seen = set()
            t4 = True
            for i in range(0, len(rgba), 4):
                seen.add(rgba[i:i + 4])
                if len(seen) > 16:
                    t4 = False
                    break
            if t4:
                use_fmt = "T4"
        # amode 3 = translucent ground/wall DECAL (forward-biased blend sub-pass):
        # cracks, potholes, dirt/oil stains, gang graffiti. Distinguish from a HARD
        # cutout (wire/fence/foliage) by the OPAQUE FRACTION - a cutout keeps a solid
        # strand (a>=240 over a real % of texels), a decal's ink is ~0..230. Route
        # translucent (opaque<10%) pure-dark-ink OR DRAW_LAST textures to amode 3;
        # "crack" by name is a fallback. The `nm in _decal` (IDE DRAW_LAST 0x4) / pure-
        # dark gate keeps this off glass/foliage/effects -> no pink junk-quad regression.
        # DECIDED BEFORE authoring so the alpha can be EDGE-FADED: a coplanar decal quad
        # whose ink/background reaches the texture border otherwise shows its rectangular
        # MESH edge on the surface below, which is visible in game.
        alphas = rgba[3::4]
        opaque_frac = (sum(x >= 240 for x in alphas) / len(alphas)) if alphas else 1.0
        # A decal is SPARSE ink over a surface that still has to show through. opaque_frac
        # only counts a>=240, so a broadly SEMI-opaque sheet - a baked shadow, a dirt
        # overlay - scores 0% opaque and used to pass this gate. On the decal path it then
        # gets the forward depth bias and the alpha test that keeps only its darkest core,
        # which reads as a black patch on the very wall it was meant to shade. Measured on a
        # full-map bake: 21 of 152 amode-3 textures covered more than 40% of their own area
        # at a>=64, up to 88%, including 64x64 masks that are pure black at alpha 114 across
        # 82% of the image and sit on house-sized models. Require sparseness explicitly.
        covered_frac = (sum(x >= 64 for x in alphas) / len(alphas)) if alphas else 1.0
        rgbset = set()
        for i in range(0, len(rgba), 4):
            rgbset.add(bytes(rgba[i:i + 3]))
            if len(rgbset) > 2:
                break
        pure_dark = len(rgbset) <= 2
        is_decal_tex = ("crack" in nm) or (opaque_frac < 0.10 and covered_frac < 0.40
                                           and (pure_dark or nm in _decal))
        if is_decal_tex:
            rgba = _decal_edge_fade(rgba, w, h)     # fade alpha->0 at the quad border
        try:
            tex = psp_tex.author_psp_texture(rgba, w, h, fmt=use_fmt, mipmaps=True)
        except Exception:
            # T4 author can reject odd sizes -> fall back to T8
            try:
                tex = psp_tex.author_psp_texture(rgba, w, h, fmt=tex_fmt, mipmaps=True)
            except Exception:
                tex_id_of[key] = -1
                return -1
        if is_decal_tex:
            tex["alpha_mode"] = 3
        tid = len(scene_textures)
        scene_textures.append(_make_texture(tex))
        tex_id_of[key] = tid
        return tid

    def emit_model(sub, d, txd, is_lod, is_decal, spin=None):
        """Pack one model (or one split-off piece of a clump) and append it to
        the scene tables.  Returns its index, or None when it packs to nothing."""
        try:
            packed = psp_mesh.pack_model(sub)
        except Exception:
            return None                  # bad geometry (NaN UV/vertex) -> skip model
        if not packed["prims"]:
            return None

        (mnx, mny, mnz), (mxx, mxy, mxz) = psp_mesh.model_aabb(sub)
        xext = mxx - mnx; yext = mxy - mny; zext = mxz - mnz
        mscale = packed["scale"]

        submeshes = []
        for prim in packed["prims"]:
            mi = prim["material_index"]
            name = ""
            if 0 <= mi < len(sub.materials):
                name = sub.materials[mi].get("texture_name") or ""
            if is_decal and name:         # DRAW_LAST model -> its textures render blended
                _decal.add(name.strip().lower())
            # A big PLANAR surface whose texture maps ~1x (road, skate ramp,
            # river embankment, lot) needs more texels than a tiled wall: at a
            # 128px cap it reads as "stretched/blurry".  Detect it PER-SUBMESH
            # and orientation-INDEPENDENTLY (ramps are sloped, so the old "flat"
            # test missed them): the 2nd-largest extent is big (it's a plane,
            # not a pole) AND the texture repeats < ~2.5x.  Tiled walls keep 128.
            gtmax = tex_max
            if tex_max:
                vb = prim["vertex_bytes"]; nv = len(vb) // 12
                if nv >= 6:
                    us = []; vs = []; xs = []; ys = []; zs = []
                    for i in range(nv):
                        u, v = struct.unpack_from('<hh', vb, i * 12)
                        x, y, z = struct.unpack_from('<hhh', vb, i * 12 + 6)
                        us.append(u); vs.append(v); xs.append(x); ys.append(y); zs.append(z)
                    exts = sorted((max(xs) - min(xs), max(ys) - min(ys),
                                   max(zs) - min(zs)))
                    planar = exts[1] * mscale        # 2nd-largest = surface size
                    tiles = max(max(us) - min(us), max(vs) - min(vs)) / 32768.0 * 8.0
                    # UPPER BOUND 40m: huge meshes (channel floor/walls 80-141m, LOD
                    # proxies) gain almost nothing from 256 (still ~2px/m) but cost
                    # RAM -> they bloated the working set and crashed real HW by
                    # heap fragmentation.  Only MEDIUM 1x surfaces (roads, ramps,
                    # lots) get the upgrade, where 256 actually sharpens them.
                    if 6.0 < planar < 40.0 and tiles < 2.5:
                        gtmax = min(256, tex_max * 2)
            tex_index = author_named(name, txd, d.txd, tmax=gtmax)
            _uvs = sub.materials[mi].get("uvscroll") if 0 <= mi < len(sub.materials) else None
            submeshes.append(psp_scene.Submesh(
                texture=tex_index,
                vertex_bytes=prim["vertex_bytes"],
                index_bytes=prim["index_bytes"],
                uvscroll=_uvs,
            ))
        if not submeshes:
            return None

        radius = 0.5 * math.sqrt(xext**2 + yext**2 + zext**2)
        # wind-sway class (per-model): SA bIsTree(0x2000)/bIsPalm(0x4000) IDE flags,
        # OR'd with a name heuristic (stock flags are inconsistent). Palm sways harder.
        # LOD proxies never sway. sway_min_z = base pivot = AABB min-Z relative to the
        # pack center (the matrix-shear pivots there so the trunk base stays planted).
        _dfl = d.dff.lower()
        if (d.flags & 0x4000) or "palm" in _dfl:
            sway_class = 2
        elif (d.flags & 0x2000) or "tree" in _dfl:
            sway_class = 1
        else:
            sway_class = 0
        if is_lod or spin:
            sway_class = 0               # proxies and rotators never sway
        # time-of-day window (IDE `tobj`): the model is only in the world between
        # timeOn and timeOff (on > off wraps midnight). Bit 7 of the ON byte carries
        # the IDE ADDITIVE flag (8) - only those go to the engine's additive last
        # pass; a plain tobj (a _dy/_nt swap, a floodbeam) renders normally, and
        # additiving all of them makes the dark-prelit night twins invisible.
        # LOD proxies keep their window too: the _dy/_nt district pairs are tobj
        # both sides, so an ungated pair would draw on top of each other all day.
        tobj = None
        if d.section == "tobj" and d.time_on is not None:
            tobj = ((int(d.time_on) & 0x7F) | (0x80 if d.flags & 8 else 0),
                    int(d.time_off) & 0xFF)
        idx = len(scene_models)
        scene_models.append(psp_scene.Model(
            submeshes=submeshes, scale=packed["scale"],
            center=packed["center"], bound_radius=radius, draw_dist=d.draw_dist,
            sway_class=sway_class, sway_min_z=(mnz - packed["center"][2]),
            spin=spin, tobj=tobj,
        ))
        return idx

    t0 = time.time()
    ok = fail = 0
    n_spin = 0
    anim_split = {}                  # model_id -> [(model index, frame offset)]
    for n, mid in enumerate(want):
        d = ide.get(mid)
        if not d:
            fail += 1; continue
        # SA LOD proxies (dff name "LOD...") are crude low-poly district meshes.
        # Keep them and TAG so the engine draws them only at distance (real LOD)
        # instead of z-fighting the detailed city up close.
        is_lod = d.dff.lower().startswith("lod")
        is_decal = bool(d.flags & 0x4)    # IDE DRAW_LAST
        dn = _img_name(d.dff, ".dff"); tn = _img_name(d.txd, ".txd")
        if dn not in have:
            fail += 1; continue
        try:
            model = dff_decode(im.extract(dn))
        except Exception:
            fail += 1; continue
        if not any(me.triangles for me in model.meshes):
            fail += 1; continue

        try:
            txd = txd_decode(im.extract(tn)) if tn in have else {}
        except Exception:
            txd = {}

        # IDE `anim` = a CClumpModelInfo whose clump keeps its frame tree (unlike
        # an atomic model, whose frame SA replaces with an identity one), so the
        # frame matrices are real here: split the rotating atomics off into their
        # own model+instance and bake the matrix into whatever stays behind.
        if d.section == "anim":
            parts = []
            if spin_resolver is not None:
                parts = _split_animated(
                    model, spin_resolver(d.anim_block, d.dff), verbose, d.dff)
            _bake_clump_frames(model)
            for spin, off, meshes in parts:
                sub = _SubModel(meshes, model.materials)
                si = emit_model(sub, d, txd, is_lod, is_decal, spin=spin)
                if si is not None:
                    anim_split.setdefault(mid, []).append((si, off))
                    n_spin += 1
                    ok += 1
            if not any(me.triangles for me in model.meshes):
                continue                 # every atomic was animated: no static part

        idx = emit_model(model, d, txd, is_lod, is_decal)
        if idx is None:
            if mid not in anim_split:
                fail += 1
            continue
        model_index_of[mid] = idx
        if is_lod:
            lod_ids.add(mid)
        ok += 1
        if verbose and (n + 1) % 500 == 0:
            print("  decoded %d/%d (ok %d fail %d, tex %d) %.1fs"
                  % (n + 1, len(want), ok, fail, len(scene_textures),
                     time.time() - t0))

    if verbose:
        print("models ok:", ok, "fail:", fail,
              "| textures:", len(scene_textures),
              "| animated split-offs:", n_spin)

    # ---- spatial grid over the placed instances (XY ground plane) ----
    placed = [i for i in sel
              if i.model_id in model_index_of or i.model_id in anim_split]
    if placed:
        gx0 = min(i.pos[0] for i in placed)
        gy0 = min(i.pos[1] for i in placed)
        gx1 = max(i.pos[0] for i in placed)
        gy1 = max(i.pos[1] for i in placed)
    else:
        gx0 = gy0 = -1.0; gx1 = gy1 = 1.0
    cols = max(1, int((gx1 - gx0) // cell_size) + 1)
    rows = max(1, int((gy1 - gy0) // cell_size) + 1)
    grid = psp_scene.Grid(cell_size=cell_size, min_x=gx0, min_y=gy0,
                          cells_x=cols, cells_y=rows)

    scene_instances = []
    n_lod = n_spin_inst = 0
    for i in placed:
        is_lod = i.model_id in lod_ids
        if is_lod:
            n_lod += 1
        if i.model_id in model_index_of:
            scene_instances.append(psp_scene.Instance(
                model=model_index_of[i.model_id],
                pos=i.pos, quat=i.rot, scale=1.0,
                interior=(1 if is_lod else 0),   # 1 = LOD proxy, 0 = detail
            ))
        # a rotating atomic rides its own instance, parked at the clump frame's
        # world position (pos + R*framePos) with the SAME rotation - so its model
        # vertices stay frame-local and the engine's .spin turn about the model
        # origin is exactly the turn the IFP describes.
        for midx, off in anim_split.get(i.model_id, ()):
            r = _inst_rot(i.rot)
            scene_instances.append(psp_scene.Instance(
                model=midx, quat=i.rot, scale=1.0, interior=0,
                pos=(i.pos[0] + r[0] * off[0] + r[1] * off[1] + r[2] * off[2],
                     i.pos[1] + r[3] * off[0] + r[4] * off[1] + r[5] * off[2],
                     i.pos[2] + r[6] * off[0] + r[7] * off[1] + r[8] * off[2]),
            ))
            n_spin_inst += 1

    if verbose:
        print("instances placed:", len(scene_instances),
              "(lod %d / detail %d)" % (n_lod, len(scene_instances) - n_lod),
              "| animated %d" % n_spin_inst,
              "| grid %dx%d (%d cells) cell_size %.0f"
              % (cols, rows, cols * rows, cell_size))

    # --grid mode wants the raw (global) scene to slice into regional tiles itself.
    if return_scene:
        return scene_models, scene_textures, scene_instances

    data = psp_scene.write_scene(scene_models, scene_textures,
                                 scene_instances, grid)
    return data, {
        "selected": len(sel), "models": len(scene_models),
        "textures": len(scene_textures), "instances": len(scene_instances),
        "cells": cols * rows,
    }


REGION_MAGIC = b"PRGN"


def build_grid_pmaps(scene_models, scene_textures, scene_instances,
                     out_dir, region_size, cell_size, verbose=True):
    """Slice the global scene into square region tiles of `region_size` world
    units; write one `region_<rx>_<ry>.pmap` per non-empty tile + a `regions.bin`
    manifest the engine reads to map a camera position -> region tile.

    Each region .pmap holds ONLY the instances whose XY centre is in the tile,
    the models they reference and the textures those models reference, all
    re-indexed to dense LOCAL tables.  Texel bytes are duplicated across tiles on
    purpose: a tiny resident prefix per region is the whole point (frees the
    ~24MB user heap for the streaming cache); disk cost is irrelevant."""
    os.makedirs(out_dir, exist_ok=True)
    xs = [i.pos[0] for i in scene_instances]
    ys = [i.pos[1] for i in scene_instances]
    ox, oy = min(xs), min(ys)
    mx, my = max(xs), max(ys)
    nx = max(1, int((mx - ox) // region_size) + 1)
    ny = max(1, int((my - oy) // region_size) + 1)

    def tile_of(x, y):
        rx = int((x - ox) // region_size)
        ry = int((y - oy) // region_size)
        rx = 0 if rx < 0 else (nx - 1 if rx >= nx else rx)
        ry = 0 if ry < 0 else (ny - 1 if ry >= ny else ry)
        return rx, ry

    buckets = {}
    for inst in scene_instances:
        buckets.setdefault(tile_of(inst.pos[0], inst.pos[1]), []).append(inst)

    counts = {}
    for (rx, ry), insts in sorted(buckets.items()):
        used_models = sorted({inst.model for inst in insts})
        mlocal = {g: l for l, g in enumerate(used_models)}
        used_tex = set()
        for g in used_models:
            for sm in scene_models[g].submeshes:
                if sm.texture >= 0:
                    used_tex.add(sm.texture)
        used_tex = sorted(used_tex)
        tlocal = {g: l for l, g in enumerate(used_tex)}

        local_textures = [scene_textures[g] for g in used_tex]
        local_models = []
        for g in used_models:
            gm = scene_models[g]
            local_models.append(psp_scene.Model(
                submeshes=[psp_scene.Submesh(
                    texture=(tlocal[sm.texture] if sm.texture >= 0 else -1),
                    vertex_bytes=sm.vertex_bytes, index_bytes=sm.index_bytes,
                    uvscroll=sm.uvscroll)
                    for sm in gm.submeshes],
                scale=gm.scale, center=gm.center,
                bound_radius=gm.bound_radius, draw_dist=gm.draw_dist,
                sway_class=gm.sway_class, sway_min_z=gm.sway_min_z,
                spin=gm.spin, tobj=gm.tobj))
        local_insts = [psp_scene.Instance(
            model=mlocal[inst.model], pos=inst.pos, quat=inst.quat,
            scale=inst.scale, interior=inst.interior) for inst in insts]

        gx0 = min(i.pos[0] for i in local_insts)
        gy0 = min(i.pos[1] for i in local_insts)
        gx1 = max(i.pos[0] for i in local_insts)
        gy1 = max(i.pos[1] for i in local_insts)
        cols = max(1, int((gx1 - gx0) // cell_size) + 1)
        rows = max(1, int((gy1 - gy0) // cell_size) + 1)
        grid = psp_scene.Grid(cell_size=cell_size, min_x=gx0, min_y=gy0,
                              cells_x=cols, cells_y=rows)

        data = psp_scene.write_scene(local_models, local_textures,
                                     local_insts, grid)
        with open(os.path.join(out_dir, "region_%d_%d.pmap" % (rx, ry)), "wb") as f:
            f.write(data)
        # .sway sidecar (per-region, keyed by LOCAL model index): the engine matrix-
        # shears wind-sway trees/palms. {sway_class i32 (0/1/2), sway_min_z f32 (base
        # pivot)} per local model. Only emitted when the tile has any sway model, so
        # most tiles have no file and the engine simply skips the shear there.
        if any(m.sway_class for m in local_models):
            sw = bytearray(b"SWAY")
            sw += struct.pack("<I", len(local_models))
            for m in local_models:
                sw += struct.pack("<if", int(m.sway_class), float(m.sway_min_z))
            with open(os.path.join(out_dir, "region_%d_%d.sway" % (rx, ry)), "wb") as f:
                f.write(sw)
        # .anim sidecar (animated-texture UV-scroll, 'UVSC' + count + count x {u32
        # global_submesh_index, f32 du_dt, f32 dv_dt}). The global submesh index is
        # model-major (write_scene order) == the engine's w->submeshes[i]. Sparse:
        # only animated submeshes; the file exists only where a tile has any (LV/LS
        # neon + waterfalls; Grove has none). Engine skips a tile with no .anim.
        anim_recs = []
        gsi = 0
        for lm in local_models:
            for sm in lm.submeshes:
                if sm.uvscroll:
                    anim_recs.append((gsi, sm.uvscroll[0], sm.uvscroll[1]))
                gsi += 1
        # .spin sidecar (per-region, keyed by LOCAL model index like .sway): the SA
        # CAnimatedBuilding rotators (LV/SF/LS revolving signs, windmills, the A51
        # radar) that build_pmap split off their clump. 'SPIN' + u32 model_count +
        # count x {u8 axis 0=X/1=Y/2=Z, u8 mode 0=spin/1=swing, u16 pad, f32
        # rate_deg_per_sec, f32 amplitude_deg} = 12 B/record - the natural C
        # layout, so the engine can fread straight into a PmapSpin[] like it does
        # for .sway (the two u8s alone would leave the floats 2-byte aligned, which
        # the PSP cannot load). phase = rate*t; spin -> angle = phase (amplitude
        # unused), swing -> angle = amplitude*sin(phase). No pivot: the split model's
        # vertices are frame-local, so the turn is about its own origin. Only written
        # where a tile actually has a rotator, so most tiles cost nothing.
        if anim_recs:
            av = bytearray(b"UVSC")
            av += struct.pack("<I", len(anim_recs))
            for gi, du, dv in anim_recs:
                av += struct.pack("<Iff", gi, float(du), float(dv))
            with open(os.path.join(out_dir, "region_%d_%d.anim" % (rx, ry)), "wb") as f:
                f.write(av)
        if any(m.spin for m in local_models):
            sp = bytearray(b"SPIN")
            sp += struct.pack("<I", len(local_models))
            for m in local_models:
                ax, mode, rate, amp = m.spin or (0, 0, 0.0, 0.0)
                sp += struct.pack("<BBHff", int(ax) & 3, int(mode) & 1, 0,
                                  float(rate), float(amp))
            with open(os.path.join(out_dir, "region_%d_%d.spin" % (rx, ry)), "wb") as f:
                f.write(sp)
        # .tobj sidecar (SA IDE time-of-day models: neon, lit windows, _dy/_nt
        # swaps). 'TOBJ' + u32 count + count x {u16 instance, u8 on, u8 off} = the
        # engine's PmapTobj[], fread straight in. Keyed by the INSTANCE index, NOT
        # the local model index like .sway/.spin - the same model can stand in the
        # tile several times and the render collect hides per instance. write_scene
        # has just STAMPED inst.cell and lays the instances out sorted by it, so
        # walk that same stable permutation to number them the way the file does.
        # Sparse like the others: no tobj model in the tile -> no file at all.
        tobj_recs = []
        for fi, k in enumerate(sorted(range(len(local_insts)),
                                      key=lambda j: local_insts[j].cell)):
            tv = local_models[local_insts[k].model].tobj
            if tv:
                tobj_recs.append((fi, tv[0], tv[1]))
        if tobj_recs:
            tb = bytearray(b"TOBJ")
            tb += struct.pack("<I", len(tobj_recs))
            for fi, on, off in tobj_recs:
                tb += struct.pack("<HBB", fi, on & 0xFF, off & 0xFF)
            with open(os.path.join(out_dir, "region_%d_%d.tobj" % (rx, ry)), "wb") as f:
                f.write(tb)
        counts[(rx, ry)] = len(local_insts)
        if verbose:
            print("  region %d,%d: inst=%d models=%d tex=%d -> %.1fMB"
                  % (rx, ry, len(local_insts), len(local_models),
                     len(local_textures), len(data) / (1024.0 * 1024.0)))

    man = bytearray()
    man += REGION_MAGIC
    man += struct.pack("<I", 1)                  # version
    man += struct.pack("<ff", ox, oy)            # world min corner of tile (0,0)
    man += struct.pack("<f", float(region_size))
    man += struct.pack("<II", nx, ny)
    man += struct.pack("<f", float(cell_size))
    for ry in range(ny):
        for rx in range(nx):
            man += struct.pack("<I", counts.get((rx, ry), 0))
    with open(os.path.join(out_dir, "regions.bin"), "wb") as f:
        f.write(man)
    if verbose:
        print("wrote regions.bin: %dx%d tiles, %d non-empty, origin (%.0f,%.0f) tile %.0f"
              % (nx, ny, len(counts), ox, oy, region_size))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("X0", "Y0", "X1", "Y1"), default=None)
    ap.add_argument("--all", action="store_true",
                    help="export the whole SA map (no bbox)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--pc", action="store_true",
                    help="use PC (D3D9) assets: generic-float DFF + DXT/raw TXD")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "input.pmap"))
    ap.add_argument("--cell-size", type=float, default=400.0)
    ap.add_argument("--tex", choices=("T8", "T4"), default="T8")
    ap.add_argument("--tex-max", type=int, default=0,
                    help="downscale textures so max(w,h) <= N (0 = no cap)")
    ap.add_argument("--grid", type=float, default=0.0,
                    help="regional chunk mode: square tile size in world units "
                         "(0 = single .pmap). Writes region_<rx>_<ry>.pmap + regions.bin")
    ap.add_argument("--grid-out", default=None,
                    help="output dir for region_*.pmap + regions.bin (default: dir of --out)")
    a = ap.parse_args(argv)

    root = a.root or (ROOT_PC if a.pc else ROOT)
    dff_decode = sa_dff_pc.decode if a.pc else sa_dff.decode
    txd_decode = sa_txd_d3d9.decode if a.pc else sa_txd.decode

    if a.all:
        bbox = None
    elif a.bbox:
        bbox = tuple(a.bbox)
    else:
        bbox = DEFAULT_BBOX

    if a.grid > 0:
        out_dir = a.grid_out or os.path.dirname(a.out) or "."
        sm, st, si = build_pmap(root, bbox, cell_size=a.cell_size,
                                tex_fmt=a.tex, tex_max=a.tex_max, verbose=True,
                                dff_decode=dff_decode, txd_decode=txd_decode,
                                return_scene=True)
        build_grid_pmaps(sm, st, si, out_dir, a.grid, a.cell_size)
        print("grid export complete ->", out_dir)
        return

    data, counts = build_pmap(root, bbox, cell_size=a.cell_size,
                              tex_fmt=a.tex, tex_max=a.tex_max, verbose=True,
                              dff_decode=dff_decode, txd_decode=txd_decode)
    with open(a.out, "wb") as f:
        f.write(data)
    print("wrote %s : %d bytes (%.2f MB)"
          % (a.out, len(data), len(data) / (1024.0 * 1024.0)))

    # ---- read back + verify the pools/tables resolve ----
    scene = psp_scene.read_scene(data)
    assert len(scene.models) == counts["models"]
    assert len(scene.textures) == counts["textures"]
    assert len(scene.instances) == counts["instances"]
    assert scene.grid.cell_count == counts["cells"]
    # every instance references a real model index
    for inst in scene.instances:
        assert 0 <= inst.model < len(scene.models), inst.model
    # every submesh slice resolves; tally geometry
    nsub = nvert = ntri = 0
    for m in scene.models:
        for sm in m.submeshes:
            assert len(sm.vertex_bytes) == sm.vertex_count * psp_scene.VERTEX_SIZE
            assert len(sm.index_bytes) == sm.index_count * psp_scene.INDEX_SIZE
            assert sm.texture == -1 or 0 <= sm.texture < len(scene.textures)
            nsub += 1; nvert += sm.vertex_count; ntri += sm.index_count // 3
    # every texture decodes back to w*h*4 RGBA via the engine-ref path
    for t in scene.textures:
        tx = {"width": t.width, "height": t.height,
              "gu_pixfmt": t.format, "texel_bytes": t.texel_bytes,
              "clut_bytes": t.clut_bytes, "clut_entries": t.clut_entries}
        rgba = psp_tex.decode_psp_texture(tx)
        assert len(rgba) == t.width * t.height * 4

    print("READ-BACK OK: models=%d submeshes=%d textures=%d instances=%d "
          "cells=%d (verts=%d tris=%d)"
          % (len(scene.models), nsub, len(scene.textures),
             len(scene.instances), scene.grid.cell_count, nvert, ntri))
    return 0


if __name__ == "__main__":
    sys.exit(main())
