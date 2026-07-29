#!/usr/bin/env python3
"""ps2_uv_tess - cap every triangle's UV extent, then make each submesh fit the
GE's texture-coordinate window. Replaces the deprecated pmap_uv_split in the PS2
world bake.

WHY. The GE decodes a 16-bit vertex texcoord as UNSIGNED u1.15, so with the
engine's global sceGuTexScale(8,8) the sampling window is [0,16) tiles, wrapping
mod 16 (Sony GE-UM 6.1/6.5; research/striped_textures_rootcause_and_fix.md).
Two things go wrong when a single triangle tiles a texture many times over:

 * a triangle spanning ~14 tiles repeats the texture 14x across itself at 16-bit
 precision, which reads as a stretched/smeared surface; and
 * a submesh whose span exceeds the window cannot be shifted into it at all, so
 pmap_uv_unsign falls back to per-vertex fract-wrap - not affine across the
 triangle, so the texture ends up mis-applied rather than merely blurry.

The PC-derived world never showed this because map_export/geom.py caps every
triangle edge at UV_EDGE_MAX tiles before packing. The PS2 path bypasses geom.py
(geometry arrives pre-instanced via ps2dff/sa_export_pmap) and used to rely on
pmap_uv_split, which only looks at whole-submesh spans and lets a 14-tile
triangle through. Measured on the same 8 regions: PC set max span 4.00 tiles with
zero triangles above it, PS2 set 13.96 with 8.95% above 4 tiles.

WHAT THIS DOES, per submesh:
 1. tessellate - crack-free 1->4 subdivision, repeated until no triangle's UV
 extent exceeds UV_EDGE_MAX tiles. A per-submesh edge-midpoint
 cache gives an edge shared by two triangles a single shared
 midpoint, so no T-junctions appear.
 2. bucket + rebase - partition the triangles by the UV cell of their centroid
 and shift each part by a whole number of tiles (invariant under
 GU_REPEAT, so appearance-neutral). Bucket span is at most
 UV_CELL + UV_EDGE_MAX = 12 tiles, comfortably inside the window.

Note there is no corrupt-UV sanitize step here, unlike geom.py's float-space
pipeline. By the time geometry reaches us the UVs are already packed s16, so the
widest span expressible is 65535 raw = exactly 16 tiles, and the absurd spans
geom._sanitize_uv exists to catch (one shipped model tiles ~2.7e7 times) cannot
survive the packing. Two passes therefore take any input from 16 tiles to 4. The
work bound below is kept as a guard, not as an expected path.

Post-condition: every triangle's UV extent is <= UV_EDGE_MAX tiles and every
submesh sits inside [0, 16) tiles, so the later pmap_uv_unsign pass finds nothing
to fix.

Usage (as a module): from ps2_uv_tess import cap_uv_span
 stats = cap_uv_span(scene_models)
"""
import struct

UV_ONE = 4096              # raw units per 1.0 uv (gvcslib UV_FIXED_ONE)
VSTRIDE = 12               # u,v (16-bit) colour (u16) x,y,z (s16)
# The GE reads a 16-bit texcoord UNSIGNED, so the UV words are handled as raw
# u16 throughout: a rebased value can legitimately be up to 12 tiles (49152),
# which no signed 16-bit view can hold. Position stays genuinely signed.
VFMT = "<HHHhhh"
U16_WRAP = 65536

UV_EDGE_MAX = 4.0          # tiles: cap on any triangle's UV extent (matches geom.py)
UV_CELL = 8.0              # tiles: bucket size; CELL + EDGE_MAX = 12 < 16 window

# A subdivision pass quadruples the triangle count, so bound the work. Two passes
# suffice for any s16 input (16 tiles -> 8 -> 4); reaching either bound means
# something upstream is wrong, and such a submesh is fract-wrapped instead.
MAX_PASSES = 6
MAX_TRIS = 200000
U16_MAX = 65535


def _avg5551(a, b):
    """Midpoint of two 5551 vertex colours, channel by channel, alpha from a."""
    r = ((a & 0x1F) + (b & 0x1F)) >> 1
    g = (((a >> 5) & 0x1F) + ((b >> 5) & 0x1F)) >> 1
    bl = (((a >> 10) & 0x1F) + ((b >> 10) & 0x1F)) >> 1
    return r | (g << 5) | (bl << 10) | (a & 0x8000)


def _mid(va, vb):
    """Midpoint vertex of an edge: linear in UV and position, 5551-aware colour."""
    return ((va[0] + vb[0]) // 2,
            (va[1] + vb[1]) // 2,
            _avg5551(va[2], vb[2]),
            (va[3] + vb[3]) // 2,
            (va[4] + vb[4]) // 2,
            (va[5] + vb[5]) // 2)


def _tri_uv_extent(verts, a, b, c):
    """Max of the U and V extents of one triangle, in raw units."""
    us = (verts[a][0], verts[b][0], verts[c][0])
    vs = (verts[a][1], verts[b][1], verts[c][1])
    return max(max(us) - min(us), max(vs) - min(vs))


def _fract_wrap(verts):
    """Per-vertex wrap into one tile. Identical under GU_REPEAT (uv and uv+N
 sample the same texel), and the only sane answer for corrupt source UVs."""
    return [(v[0] % UV_ONE, v[1] % UV_ONE, v[2], v[3], v[4], v[5]) for v in verts]


def _to_signed(w):
    return w - U16_WRAP if w > 32767 else w


def _pick_domain(verts, tris):
    """Decide whether the UV words were authored signed (centred on zero) or
 already sit in the GE's unsigned window, and return verts in that domain.

 The author's domain is the one where triangles are locally small: read in the
 wrong domain, a triangle straddling zero appears to span nearly the whole
 16-tile window. Same rule as pmap_uv_unsign, and it is what makes a second
 pass over our own output a no-op."""
    if not tris:
        return verts
    signed = [(_to_signed(v[0]), _to_signed(v[1]), v[2], v[3], v[4], v[5])
              for v in verts]
    span_u = max((_tri_uv_extent(verts, *t) for t in tris), default=0)
    span_s = max((_tri_uv_extent(signed, *t) for t in tris), default=0)
    return signed if span_s < span_u else verts


def _tessellate(verts, tris, limit):
    """Subdivide 1->4 until every triangle's UV extent is <= limit raw units.

 verts is grown in place; a shared edge-midpoint cache keyed on the vertex
 pair keeps the mesh crack-free. Returns (tris, ok): ok is False when the
 work bound was hit, which tells the caller to fract-wrap instead."""
    for _ in range(MAX_PASSES):
        if all(_tri_uv_extent(verts, a, b, c) <= limit for a, b, c in tris):
            return tris, True
        if len(tris) * 4 > MAX_TRIS:
            return tris, False
        midcache = {}

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            m = midcache.get(key)
            if m is None:
                m = len(verts)
                verts.append(_mid(verts[i], verts[j]))
                midcache[key] = m
            return m

        out = []
        for a, b, c in tris:
            if _tri_uv_extent(verts, a, b, c) <= limit:
                out.append((a, b, c))
                continue
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        tris = out
    return tris, all(_tri_uv_extent(verts, a, b, c) <= limit for a, b, c in tris)


def _bucket_and_rebase(sm_cls, sm, verts, tris):
    """Partition by UV cell, shift each part by whole tiles into [0, span].

 A whole-tile shift is invariant under GU_REPEAT, so this is appearance-
 neutral; min-floor (rather than centring on the mean) is what puts the result
 in the GE's UNSIGNED window instead of straddling zero."""
    cell = int(UV_CELL * UV_ONE)
    buckets = {}
    for a, b, c in tris:
        mu = (verts[a][0] + verts[b][0] + verts[c][0]) // 3
        mv = (verts[a][1] + verts[b][1] + verts[c][1]) // 3
        buckets.setdefault((mu // cell, mv // cell), []).append((a, b, c))

    out = []
    for _, btris in sorted(buckets.items()):
        remap = {}
        bverts = []
        bidx = []
        for tri in btris:
            for k in tri:
                j = remap.get(k)
                if j is None:
                    j = len(bverts)
                    remap[k] = j
                    bverts.append(verts[k])
                bidx.append(j)
        # min-floor: subtract whole tiles so the piece starts inside tile 0
        ou = (min(v[0] for v in bverts) // UV_ONE) * UV_ONE
        ov = (min(v[1] for v in bverts) // UV_ONE) * UV_ONE
        bverts = [(v[0] - ou, v[1] - ov, v[2], v[3], v[4], v[5]) for v in bverts]

        # post-condition: min-floor put every UV inside the GE's unsigned window,
        # so the u16 pack below is exact. Bucket span is at most CELL+EDGE_MAX.
        worst_uv = max(max(v[0], v[1]) for v in bverts)
        if worst_uv >= U16_WRAP or min(min(v[0], v[1]) for v in bverts) < 0:
            raise SystemExit("ps2_uv_tess: rebased UV %d outside [0,65536) "
                             "(texture %r)" % (worst_uv, sm.texture))
        if len(bverts) > U16_MAX:
            # indices are u16; a bucket this big cannot be addressed. Real region
            # submeshes are orders of magnitude smaller, so this means something
            # upstream is wrong - say so loudly rather than emit a broken piece.
            raise SystemExit("ps2_uv_tess: bucket of %d vertices exceeds the u16 "
                             "index range (texture %r)" % (len(bverts), sm.texture))
        vb = b"".join(struct.pack(VFMT, *v) for v in bverts)
        ib = struct.pack("<%dH" % len(bidx), *bidx)
        # carry every per-submesh attribute onto the pieces: uvscroll (the
        # animated-texture UV rate feeding the.anim sidecar) belongs to the
        # MATERIAL, so each piece of a split scrolling sign must keep scrolling.
        out.append(sm_cls(texture=sm.texture, vertex_bytes=vb, index_bytes=ib,
                          uvscroll=sm.uvscroll))
    return out


def _process_submesh(sm_cls, sm, limit, stats):
    """Return the list of submeshes replacing `sm` (possibly just [sm])."""
    nv = len(sm.vertex_bytes) // VSTRIDE
    ni = len(sm.index_bytes) // 2
    if nv == 0 or ni < 3:
        return [sm]

    verts = list(struct.iter_unpack(VFMT, sm.vertex_bytes))
    idx = struct.unpack("<%dH" % ni, sm.index_bytes)
    tris = [(idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]) for t in range(ni // 3)]
    if any(max(t) >= nv for t in tris):
        return [sm]                                   # malformed: leave it alone
    verts = _pick_domain(verts, tris)

    stats["tris_before"] += len(tris)
    n_tris_in = len(tris)

    tris, ok = _tessellate(verts, tris, limit)
    if not ok:
        # the work bound fired: the input is pathological, so wrap per vertex
        # rather than keep subdividing. Identical under GU_REPEAT.
        verts = _fract_wrap(verts)
        stats["wrapped"] += 1

    worst = max((_tri_uv_extent(verts, *t) for t in tris), default=0)
    stats["max_span_after"] = max(stats["max_span_after"], worst / float(UV_ONE))
    stats["tris_after"] += len(tris)

    pieces = _bucket_and_rebase(sm_cls, sm, verts, tris)
    if len(pieces) > 1 or len(tris) != n_tris_in:
        stats["subdivided"] += 1
    return pieces


def cap_uv_span(models, uv_edge_max=UV_EDGE_MAX, verbose=True):
    """In place: cap every triangle's UV extent across every model. Returns stats."""
    limit = int(uv_edge_max * UV_ONE)
    stats = dict(tris_before=0, tris_after=0, subdivided=0, wrapped=0,
                 max_span_after=0.0, pieces=0)
    for m in models:
        new = []
        for sm in m.submeshes:
            pieces = _process_submesh(type(sm), sm, limit, stats)
            new.extend(pieces)
        stats["pieces"] += len(new)
        m.submeshes = new
    if verbose:
        grow = (100.0 * stats["tris_after"] / stats["tris_before"] - 100.0) \
            if stats["tris_before"] else 0.0
        print("uv-tess: %d submeshes reworked, tris %d -> %d (%+.1f%%), "
              "%d fract-wrapped, max span now %.2f tiles (cap %.1f)"
              % (stats["subdivided"], stats["tris_before"], stats["tris_after"],
                 grow, stats["wrapped"], stats["max_span_after"], uv_edge_max))
    return stats


if __name__ == "__main__":
    print(__doc__)
