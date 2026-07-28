#!/usr/bin/env python3
"""pmap_uv_split.py - split submeshes whose UV tiling exceeds the s16 range.

!!! DEPRECATED (b572): built on the FALSE "signed +-8" model (the GE window is
really UNSIGNED [0,16) tiles). Splitting logic lives in map_export/geom.py
(uv_split + min-floor _rebase); for deployed pmaps use tools/pmap_uv_unsign.py.

THE striped-road fix. UVs are stored s16 = uv*4096 -> representable range is
[-8, +8) texture tiles. SA roads/asphalt legitimately tile 10-24x across one
submesh; those verts CLAMP at +-8 and the UV gradient collapses -> the smeared
"striped" texture. pmap_uv_recenter fixes submeshes whose OFFSET is out of range
but cannot help when the SPAN itself is > 16 tiles.

This transform runs at the SCENE level (before build_grid_pmaps slices anything):
for each submesh whose UV span exceeds SPAN_MAX tiles, bucket its triangles by
UV supercell (BUCKET tiles), re-emit each bucket as its own submesh with a
whole-tile offset subtracted (GU_REPEAT-identical). Every piece then fits the
s16 range -> no clamp -> no stripes. Geometry/appearance byte-identical; costs a
few duplicated verts + extra submeshes on the offending (road) models only.

Usage (as a module):   from pmap_uv_split import split_scene_models
                       split_scene_models(scene.models)   # in place
"""
import struct

UV_ONE   = 4096          # s16 units per 1.0 uv (gvcslib UV_FIXED_ONE)
SPAN_MAX = 15.5          # tiles: split any submesh spanning more than this
BUCKET   = 12            # tiles per supercell (12 + tri extent << 16 range)
VSTRIDE  = 12            # u,v(s16) color(u16) x,y,z(s16)


def _split_submesh(sm_cls, sm):
    """Return a list of range-safe submeshes replacing `sm` (or [sm] if fine)."""
    nv = len(sm.vertex_bytes) // VSTRIDE
    ni = len(sm.index_bytes) // 2
    if nv == 0 or ni < 3:
        return [sm]
    verts = list(struct.iter_unpack("<hhHhhh", sm.vertex_bytes))
    us = [v[0] for v in verts]; vs = [v[1] for v in verts]
    span_u = (max(us) - min(us)) / UV_ONE
    span_v = (max(vs) - min(vs)) / UV_ONE
    if span_u <= SPAN_MAX and span_v <= SPAN_MAX:
        return [sm]                                   # fits after recentring

    idx = struct.unpack("<%dH" % ni, sm.index_bytes)
    buckets = {}                                      # (bu,bv) -> [tri indices]
    for t in range(ni // 3):
        a, b, c = idx[t*3], idx[t*3+1], idx[t*3+2]
        mu = (verts[a][0] + verts[b][0] + verts[c][0]) // 3
        mv = (verts[a][1] + verts[b][1] + verts[c][1]) // 3
        key = (mu // (BUCKET * UV_ONE), mv // (BUCKET * UV_ONE))
        buckets.setdefault(key, []).append(t)

    out = []
    for key, tris in sorted(buckets.items()):
        remap = {}
        bverts = []
        bidx = []
        # whole-tile offset that recentres this bucket near 0
        su = sv = 0; cnt = 0
        for t in tris:
            for k in (idx[t*3], idx[t*3+1], idx[t*3+2]):
                su += verts[k][0]; sv += verts[k][1]; cnt += 1
        ou = int(round((su // cnt) / UV_ONE)) * UV_ONE
        ov = int(round((sv // cnt) / UV_ONE)) * UV_ONE
        for t in tris:
            for k in (idx[t*3], idx[t*3+1], idx[t*3+2]):
                j = remap.get(k)
                if j is None:
                    j = len(bverts); remap[k] = j
                    u, v, col, x, y, z = verts[k]
                    nu = u - ou; nv2 = v - ov
                    if nu >  32767: nu =  32767
                    if nu < -32768: nu = -32768
                    if nv2 >  32767: nv2 =  32767
                    if nv2 < -32768: nv2 = -32768
                    bverts.append((nu, nv2, col, x, y, z))
                bidx.append(j)
        vb = b"".join(struct.pack("<hhHhhh", *v) for v in bverts)
        ib = struct.pack("<%dH" % len(bidx), *bidx)
        # carry every build-time per-submesh attribute onto the pieces: uvscroll
        # (animated-texture UV rate -> the .anim sidecar) belongs to the MATERIAL,
        # so each piece of a split scrolling sign must keep scrolling. Dropping it
        # silently killed the animation on any over-tiled animated submesh.
        out.append(sm_cls(texture=sm.texture, vertex_bytes=vb, index_bytes=ib,
                          uvscroll=sm.uvscroll))
    return out


def split_scene_models(models, verbose=True):
    """In-place: replace over-tiled submeshes in every model. Returns stats."""
    n_split = n_pieces = 0
    for m in models:
        new = []
        changed = False
        for sm in m.submeshes:
            pieces = _split_submesh(type(sm), sm)
            if len(pieces) > 1:
                n_split += 1; n_pieces += len(pieces); changed = True
            new.extend(pieces)
        if changed:
            m.submeshes = new
    if verbose:
        print("uv_split: %d over-tiled submeshes -> %d range-safe pieces" % (n_split, n_pieces))
    return n_split, n_pieces


if __name__ == "__main__":
    print(__doc__)
