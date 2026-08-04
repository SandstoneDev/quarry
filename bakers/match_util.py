#!/usr/bin/env python3
"""match_util - position->DFF-name matching that survives co-located LODs.

Root cause found 2026-07-21 (night-windows forensics): an SA LOD proxy
instance sits at IDENTICAL world coordinates as its HI model instance, and
text-IPL LOD rows enter the lookup grid before binary-IPL HI rows - so a
first-hit position match resolves EVERY LOD-paired model to its lod* DFF.
That poisoned the .night/.nightd bakes (lodganghous night set is uniformly
dull -> authored 255-white windows lost) and can poison any name-driven
texture repair the same way.

Fix: collect ALL candidates within 0.5u and pick by DFF vertex count - the deployed model's vertex pool is always >= its true source (tessellation
only adds verts), so the best candidate is the LARGEST DFF that still fits
under the deployed count (with a small tolerance). A deployed LOD model
(few hundred verts) correctly rejects the HI DFF (thousands) and keeps the
lod* name; a deployed HI model picks the HI DFF over the lod*.
"""


def build_grid(insts):
    grid = {}
    for i in insts:
        x, y, z = i["pos"]
        grid.setdefault((round(x), round(y)), []).append((x, y, z, i["name"]))
    return grid


def match_all(grid, x, y, z, tol=0.5, ztol=None):
    """every source instance name within tol (dedup, order preserved)."""
    if ztol is None:
        ztol = tol
    out = []
    for cx in (round(x) - 1, round(x), round(x) + 1):
        for cy in (round(y) - 1, round(y), round(y) + 1):
            for (sx, sy, sz, nm) in grid.get((cx, cy), ()):
                if abs(sx - x) <= tol and abs(sy - y) <= tol and abs(sz - z) <= ztol:
                    if nm not in out:
                        out.append(nm)
    return out


def pick_by_verts(cands, prod_verts, dff_vert_count):
    """cands: candidate names; prod_verts: deployed model vertex total;
 dff_vert_count(name)->int|None. Largest DFF that fits under the deployed
 count (x1.05+8 tolerance); if none fits, the smallest DFF overall."""
    best, best_v = None, -1
    small, small_v = None, 1 << 30
    for nm in cands:
        v = dff_vert_count(nm)
        if not v:
            continue
        if v <= prod_verts * 1.05 + 8 and v > best_v:
            best, best_v = nm, v
        if v < small_v:
            small, small_v = nm, v
    return best if best is not None else small
