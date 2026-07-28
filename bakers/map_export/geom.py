#!/usr/bin/env python3
"""geom - float-space mesh building: SAW Geometry -> welded/tessellated/
UV-split submeshes. NOTHING here is quantized; s16 packing happens in pack.py.

Rules (spec):
  sanitize:   corrupt-UV submeshes (span > UV_GARBAGE tiles, e.g. easykerb's ~2.7e7)
              are per-vertex fract-wrapped - GU_REPEAT-identical, kills the blow-up
  tessellate: no triangle edge longer than 24 world units (GE guard-band) AND no
              edge spanning more than UV_EDGE_MAX texture tiles (the striped-road fix:
              caps every triangle's UV extent so a high-tiling short-world-edge quad
              can't stay a single >8-tile triangle that the s16 UV packing clamps)
  uv_split:   a submesh whose UV span exceeds UV_SPAN_MAX tiles is partitioned by the
              UV cell (UV_CELL) of each triangle's centroid; every part is rebased by
              an INTEGER offset (tiling-safe). CELL + UV_EDGE_MAX kept < 15 so the
              rebased span never reaches the +-8-tile s16 clamp -> no stripes.
  recenter:   every submesh min UV moved into [0,1) by an integer shift
"""
import math
import os
import sys

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from formats.dff import _destrip          # SAW

TESS_EDGE = 24.0
# UV packing: raw = uv*4096, and the GE reads the 16 bits as UNSIGNED u1.15 --
# with the engine's TexScale(8,8) the window is [0,16) tiles, wrap mod 16
# (Sony GE-UM 6.1/6.5; research/striped_textures_rootcause_and_fix.md).
# The striped-road fix has TWO stages:
#  1. UV_EDGE_MAX: tessellate caps every triangle's UV extent (max-min over any pair
#     is an edge, so all-edges<=E => triangle span<=E). Kills single high-tiling
#     triangles that world-length tessellation left whole.
#  2. UV_SPAN_MAX / UV_CELL: uv_split buckets a submesh; bucket span ~= CELL +
#     triangle_span (~12.6 worst), then _rebase MIN-FLOOR shifts it into
#     [0, span+1) - always positive, inside the 16-tile window, and no
#     triangle can cross the window seam.
UV_EDGE_MAX = 4.0
UV_SPAN_MAX = 8.0
UV_CELL = 8.0
# UV span above this = CORRUPT source UV, not legitimate tiling (e.g. easykerb ships a
# ~27,000,000-tile span from a garbage float). Splitting it explodes geometry until the
# guard fires -> clamp -> stripe. Nothing real tiles 64x, so per-vertex fract-wrap these
# (GU_REPEAT samples the same texel for uv and uv+N) -> span < 1, no stripe, no blow-up.
UV_GARBAGE = 64.0


def geometry_submeshes(geo):
    """SAW Geometry -> [{'mat': Material, 'tris': [((pos,uv,col) x3)]}] in LOCAL
    space. Vertex colour = prelit (day) if present, else material colour."""
    uvs = geo.uvs[0] if geo.uvs else [(0.0, 0.0)] * geo.num_vertices
    pre = geo.prelit_colors
    out = []
    for sp in geo.splits:
        mi = sp["mat_index"]
        mat = geo.materials[mi] if 0 <= mi < len(geo.materials) else None
        if mat is None:
            continue
        idx = _destrip(sp["indices"]) if sp["strip"] else list(sp["indices"])
        tris = []
        for i in range(0, len(idx) - 2, 3):
            t = []
            ok = True
            for gi in (idx[i], idx[i+1], idx[i+2]):
                if gi >= geo.num_vertices or gi >= len(geo.vertices):
                    ok = False
                    break
                p = geo.vertices[gi]
                if p[0] != p[0] or p[1] != p[1] or p[2] != p[2]:   # NaN vert
                    ok = False
                    break
                u, v = uvs[gi] if gi < len(uvs) else (0.0, 0.0)
                if u != u or v != v:
                    u = v = 0.0
                if pre is not None and gi < len(pre):
                    r, g, b, a = pre[gi]
                else:
                    r, g, b, a = mat.color
                t.append(((p[0], p[1], p[2]), (u, v), (r, g, b, a)))
            if ok:
                tris.append(tuple(t))
        if tris:
            out.append({"mat": mat, "tris": tris})
    return out


def tessellate(tris):
    """Split triangles until every edge satisfies BOTH: world length <= TESS_EDGE
    (guard-band culling) AND UV span <= UV_EDGE_MAX tiles. The UV limit is the
    striped-road fix: a high-tiling quad with a short world edge (a wall/road
    tiling 16x over a few units) never tripped the world-length test, so it
    stayed a single triangle spanning >16 UV tiles -> the s16*4096 packing
    (+-8 tile range) clamped it -> stripes. Splitting by UV span too caps every
    triangle's UV extent so uv_split() can then bucket the submesh clamp-free.
    Queue-based; safety cap prevents runaway on pathological tiling."""
    def mid(a, b):
        return (
            tuple((a[0][k] + b[0][k]) * 0.5 for k in range(3)),
            tuple((a[1][k] + b[1][k]) * 0.5 for k in range(2)),
            tuple(int((a[2][k] + b[2][k]) * 0.5) for k in range(4)),
        )
    def score(a, b):
        """max(world-length, UV-span) as a fraction of its limit; >1 => split."""
        dx = a[0][0]-b[0][0]; dy = a[0][1]-b[0][1]; dz = a[0][2]-b[0][2]
        wlen = (dx*dx + dy*dy + dz*dz) ** 0.5
        uv = max(abs(a[1][0]-b[1][0]), abs(a[1][1]-b[1][1]))
        return max(wlen / TESS_EDGE, uv / UV_EDGE_MAX)
    out = []
    queue = list(tris)
    guard = max(64, len(tris) * 1024)             # cap: 1024x growth per submesh
    while queue:
        if len(out) + len(queue) > guard:
            out.extend(queue)                     # give up gracefully, keep data
            break
        a, b, c = queue.pop()
        s = [score(a, b), score(b, c), score(c, a)]
        m = max(range(3), key=lambda i: s[i])
        if s[m] <= 1.0:
            out.append((a, b, c))
            continue
        if m == 0:
            p = mid(a, b); queue.append((a, p, c)); queue.append((p, b, c))
        elif m == 1:
            p = mid(b, c); queue.append((a, b, p)); queue.append((a, p, c))
        else:
            p = mid(c, a); queue.append((a, b, p)); queue.append((p, b, c))
    return out


def _rebase(tris):
    """Integer-shift UVs so the span STARTS in [0,1) tile. The GE reads 16-bit
    UVs as UNSIGNED u1.15 (no signed path in hardware) - with the engine's
    TexScale(8,8) the window is [0,16) tiles, wrapping mod 16. Negative values
    alias to +16 tiles; a triangle whose UV range crosses 0 then interpolates
    the LONG way round the window (~16-span reversed repeats) = the striped
    roads / stretched interiors bug. Min-floor keeps every value in
    [0, span+1) which fits the window for any span <= 15 tiles (uv_split caps
    real spans at ~12.6). NEVER centre spans around 0."""
    us = [p[1][0] for t in tris for p in t]
    vs = [p[1][1] for t in tris for p in t]
    mu = math.floor(min(us))
    mv = math.floor(min(vs))
    if mu == 0 and mv == 0:
        return tris
    return [tuple((p[0], (p[1][0] - mu, p[1][1] - mv), p[2]) for p in t)
            for t in tris]


def uv_split(tris):
    """[[tris]] partitioned so each part's UV span <= UV_SPAN_MAX; each part
    rebased by integer floor of its min UV (tiling-safe)."""
    us = [p[1][0] for t in tris for p in t]
    vs = [p[1][1] for t in tris for p in t]
    if not us:
        return [tris]
    if max(us) - min(us) <= UV_SPAN_MAX and max(vs) - min(vs) <= UV_SPAN_MAX:
        return [_rebase(tris)]
    buckets = {}
    for t in tris:
        cu = sum(p[1][0] for p in t) / 3.0
        cv = sum(p[1][1] for p in t) / 3.0
        buckets.setdefault((math.floor(cu / UV_CELL), math.floor(cv / UV_CELL)),
                           []).append(t)
    return [_rebase(ts) for ts in buckets.values()]


def _sanitize_uv(tris, tex="?"):
    """Corrupt-UV guard: if a submesh's UV span is absurd (> UV_GARBAGE tiles) the
    source floats are garbage (NaN/inf/huge). Per-vertex fract-wrap so every UV lands
    in [0,1) - identical under GU_REPEAT, span < 1, no clamp, no tessellation blow-up."""
    us = [p[1][0] for t in tris for p in t]
    vs = [p[1][1] for t in tris for p in t]
    if not us:
        return tris
    finite = all(math.isfinite(x) for x in us) and all(math.isfinite(x) for x in vs)
    if finite and max(us) - min(us) <= UV_GARBAGE and max(vs) - min(vs) <= UV_GARBAGE:
        return tris
    if os.environ.get("UV_DIAG"):
        sys.stderr.write("  UV_DIAG corrupt tex=%s ntri=%d -> fract-wrapped\n"
                         % (tex, len(tris)))
    def fr(x):
        return 0.0 if not math.isfinite(x) else x - math.floor(x)
    return [tuple((p[0], (fr(p[1][0]), fr(p[1][1])), p[2]) for p in t) for t in tris]


def process_geometry(geo):
    """Full float pipeline for one SAW Geometry:
    -> [{'mat': Material, 'tris': [...] }] tessellated + UV-safe."""
    out = []
    for sub in geometry_submeshes(geo):
        tris = _sanitize_uv(sub["tris"],
                            getattr(sub["mat"], "texture_name", "?"))
        tris = tessellate(tris)
        for part in uv_split(tris):
            out.append({"mat": sub["mat"], "tris": part})
    return out


if __name__ == "__main__":
    # a 40x40-tile textured triangle (long world edges + big UV span)
    big = [(((0, 0, 0), (0, 0), (255, 255, 255, 255)),
            ((100, 0, 0), (40, 0), (255, 255, 255, 255)),
            ((0, 100, 0), (0, 40), (255, 255, 255, 255)))]
    t = tessellate(big)
    worst_w = worst_uv = 0.0
    for (a, b, c) in t:
        for p, q in ((a, b), (b, c), (c, a)):
            worst_w = max(worst_w, math.dist(p[0], q[0]))
            worst_uv = max(worst_uv, abs(p[1][0]-q[1][0]), abs(p[1][1]-q[1][1]))
    print("tess: tris %d worst world-edge %.1f (<=%.0f)  worst uv-edge %.2f (<=%.1f)"
          % (len(t), worst_w, TESS_EDGE, worst_uv, UV_EDGE_MAX))
    parts = uv_split(t)
    lo, hi = 1e9, -1e9
    for ts in parts:
        for tr in ts:
            for p in tr:
                lo = min(lo, p[1][0], p[1][1])
                hi = max(hi, p[1][0], p[1][1])
    print("uv_split: parts %d  rebased UV range [%.2f, %.2f] "
          "(need [0, <16): GE u16 window)" % (len(parts), lo, hi))
    # corrupt-UV guard: a ~2.7e7-tile span (easykerb) must fract-wrap to span < 1
    junk = [(((0, 0, 0), (0.1, 0.0), (255, 255, 255, 255)),
             ((1, 0, 0), (2.7e7 + 0.6, 0.0), (255, 255, 255, 255)),
             ((0, 1, 0), (0.1, 3.2), (255, 255, 255, 255)))]
    sj = _sanitize_uv(junk)
    ju = [p[1][0] for tr in sj for p in tr] + [p[1][1] for tr in sj for p in tr]
    jspan = max(ju) - min(ju)
    print("sanitize: 2.7e7 corrupt span -> %.2f (<1 expected)" % jspan)
    assert worst_w <= TESS_EDGE + 1e-3
    assert worst_uv <= UV_EDGE_MAX + 1e-3
    assert lo >= 0.0 and hi < 15.75, (lo, hi)   # GE unsigned window, seam-free
    assert jspan < 1.0
    print("geom OK")
