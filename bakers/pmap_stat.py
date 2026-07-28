#!/usr/bin/env python3
"""pmap_stat - quantify a region_*.pmap tile: model/submesh/instance counts,
texture format mix (T4 vs T8 vs other) and total resident texel+clut bytes,
geometry bytes, and how many submeshes render UNTEXTURED (tex_id < 0 -> white
on GE). Use to compare a rebaked tile against the battle reference tile so the
streaming weight (resident texture MB) and the white-surface count are visible.

Reads a v2 (raw) pmap; run pmap_lz4_decompress.py on a v3 tile first.

Usage: python pmap_stat.py <region.pmap> [more.pmap ...]
"""
import os
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_scene

GU_PSM_T4 = 4
GU_PSM_T8 = 5


def stat(path):
    sc = psp_scene.read_scene(open(path, "rb").read())
    n_t4 = n_t8 = n_other = 0
    texel_bytes = clut_bytes = 0
    for t in sc.textures:
        if t.format == GU_PSM_T4:
            n_t4 += 1
        elif t.format == GU_PSM_T8:
            n_t8 += 1
        else:
            n_other += 1
        texel_bytes += len(t.texel_bytes)
        clut_bytes += len(t.clut_bytes)
    # submeshes: count untextured (tex_id < 0) and total tris
    sm_total = sm_white = tris_total = tris_white = 0
    for m in sc.models:
        for sm in m.submeshes:
            sm_total += 1
            # psp_scene.Submesh stores u16 (GU_INDEX_16BIT) LOCAL indices in `index_bytes`
            # (a triangle LIST - index_count is a mult of 3). tris = u16-count/3 = bytes/2/3.
            ntri = len(sm.index_bytes) // 2 // 3 if hasattr(sm, "index_bytes") else 0
            tris_total += ntri
            tid = getattr(sm, "tex_id", getattr(sm, "texture", -1))
            if tid is None or tid < 0:
                sm_white += 1
                tris_white += ntri
    tex_mb = (texel_bytes + clut_bytes) / (1024.0 * 1024.0)
    print("=== %s (%.2f MB file)" % (os.path.basename(path),
                                     os.path.getsize(path) / 1048576.0))
    print("  models=%d submeshes=%d instances=%d tris=%d"
          % (len(sc.models), sm_total, len(sc.instances), tris_total))
    print("  textures=%d  T4=%d T8=%d other=%d  RESIDENT tex=%.2f MB"
          % (len(sc.textures), n_t4, n_t8, n_other, tex_mb))
    print("  UNTEXTURED submeshes=%d (%.1f%%)  their tris=%d  <- white on GE"
          % (sm_white, 100.0 * sm_white / max(1, sm_total), tris_white))
    return tex_mb, sm_white


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            stat(p)
        except Exception as e:
            print("!! %s: %s" % (p, e))
