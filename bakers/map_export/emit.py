#!/usr/bin/env python3
"""emit - global scene lists -> region tiles + regions.bin v2 + post passes.

Reuses gvcslib work/sa_export_pmap.build_grid_pmaps (CALL only, gvcslib is
read-only) for slicing + regions.bin, then bumps the manifest version 1 -> 2:
the runtime reads v2 as "draw_dist is honest SA data" (DD_SCALE 1.0).
Post: col_bake regions, lod_bake_regions, pmap_lz4 --dir (existing tools).
"""
import glob
import os
import struct
import subprocess
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib.work.sa_export_pmap import build_grid_pmaps

TOOLS = ""


def emit_regions(models, textures, instances, out_dir, tile=450.0, cell=400.0):
    counts = build_grid_pmaps(models, textures, instances, out_dir, tile, cell)
    man = os.path.join(out_dir, "regions.bin")
    with open(man, "r+b") as f:              # version 1 -> 2 (honest-dd marker)
        f.seek(4)
        f.write(struct.pack("<I", 2))
    return counts


def budget_report(out_dir, idx_cap=92 * 1024):
    """Index-area (= header+tables before the pools) per tile vs the runtime
    idx pool cap. psp_scene v2 header word 12 = vertex pool offset."""
    worst = []
    for p in sorted(glob.glob(os.path.join(out_dir, "region_*.pmap"))):
        with open(p, "rb") as f:
            hdr = struct.unpack("<20I", f.read(80))
        idx_bytes = hdr[12]
        worst.append((idx_bytes, os.path.basename(p)))
    worst.sort(reverse=True)
    over = sum(1 for b, _ in worst if b > idx_cap)
    print("index-area top5 (cap %dKB, over-cap tiles: %d):"
          % (idx_cap // 1024, over))
    for b, n in worst[:5]:
        flag = "  !! OVER CAP" if b > idx_cap else ""
        print("  %-22s %6.1fKB%s" % (n, b / 1024.0, flag))
    return worst


def write_lod_files(out_dir, scene_instances, links):
    """Per-tile region_X_Y.lod straight from the exporter's OWN links (global
    scene indices), replacing the lod_bake_regions key-matching pass. That pass
    matched pos+quat+is_lod keys against SA IPLs re-parsed with different LOD
    rules - co-located detail/proxy pairs (the Grove bridge) collided and the
    DETAIL inherited the proxy's inbound refs, so the renderer's mutual
    exclusion skipped the bridge standalone (build-197 pilot bug).

    Both sides of the match here come from the same writer: the on-disk
    instance record (write_scene cell-sorts them) is looked up by the exact
    pos/quat/interior bytes we handed psp_scene."""
    from gvcslib.psp_scene import _q15
    import glob

    def skey(inst):
        return (struct.pack("<3f", *inst.pos)
                + struct.pack("<4h", _q15(inst.quat[0]), _q15(inst.quat[1]),
                              _q15(inst.quat[2]), _q15(inst.quat[3]))
                + bytes((1 if inst.interior else 0,)))

    tiles = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "region_*.pmap"))):
        with open(p, "rb") as f:
            hdr = struct.unpack("<20I", f.read(80))
            ic, ioff = hdr[9], hdr[10]
            f.seek(ioff)
            idata = f.read(ic * 36)
        keys = []
        for k in range(ic):
            o = k * 36
            interior = struct.unpack_from("<i", idata, o + 28)[0]
            keys.append(idata[o + 4:o + 24] + bytes((1 if interior else 0,)))
        tiles[os.path.basename(p)] = keys

    key_loc = {}
    for name, keys in tiles.items():
        for k, kb in enumerate(keys):
            key_loc.setdefault(kb, (name, k))

    sloc = [key_loc.get(skey(si)) for si in scene_instances]
    out_links = {name: [-1] * len(keys) for name, keys in tiles.items()}
    linked = cross = unresolved = 0
    for i, si in enumerate(scene_instances):
        t = links[i] if i < len(links) else -1
        if t < 0:
            continue
        a = sloc[i]
        b = sloc[t] if 0 <= t < len(sloc) else None
        if not a or not b:
            unresolved += 1
            continue
        if a[0] != b[0]:
            cross += 1
            continue                       # proxy in a neighbour tile: detail
                                           # just ends at its band (map edge rule)
        out_links[a[0]][a[1]] = b[1]
        linked += 1
    for name, lk in out_links.items():
        lp = os.path.join(out_dir, name.replace(".pmap", ".lod"))
        with open(lp, "wb") as f:
            f.write(b"PLOD")
            f.write(struct.pack("<II", 1, len(lk)))
            f.write(struct.pack("<%di" % len(lk), *lk))
    print("  .lod (exporter-owned): %d tiles, links %d, cross-tile %d, unresolved %d"
          % (len(out_links), linked, cross, unresolved))


def post_passes(out_dir):
    py = sys.executable
    for cmd in (
        [py, os.path.join(TOOLS, "col_bake.py"), "regions", out_dir],
        [py, os.path.join(TOOLS, "pmap_lz4.py"), "--dir", out_dir],
    ):
        print(">", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit("post pass failed: %s" % cmd)
