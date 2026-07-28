#!/usr/bin/env python3
"""map_audit - exhaustive health check of a FULL-MAP bake (184 regions).

Catches the failure modes that show up in-game as "holes in the map":
  * a grid cell regions.bin calls non-empty but that has no .pmap        -> void tile
  * a .pmap that fails to parse / has zero instances                     -> void tile
  * submeshes with tex_id < 0 (render UNTEXTURED = white on GE)          -> white surfaces
  * a missing .col                                                       -> fall through world
  * a tile whose resident texture MB is a big outlier                    -> cache thrash, and
    thrash is what makes houses/grass vanish (see the storm-panic case)
  * versus a BASELINE bake: any region that LOST instances/models        -> regression

Usage:
  python map_audit.py <bake_dir> [--baseline <installed_dir>] [--json <out.json>]
  python map_audit.py <bake_dir> --quiet      # anomalies only, no per-region table
"""
import json
import os
import struct
import sys
import tempfile

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import psp_scene                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pmap_lz4_decompress                                       # noqa: E402

SIDECARS = ("col", "lod", "grass", "night", "nightd", "sway", "anim", "spin",
            "tobj", "dyn", "road")
# a tile heavier than this in resident texture bytes risks evicting its own models
TEX_MB_WARN = 6.0


def read_regions_bin(d):
    """PRGN manifest (build_grid_pmaps): magic|ver|ox,oy|tile|nx,ny|cell then nx*ny
    u32 instance counts, row-major (0 = empty cell).
    -> (ox, oy, tile, nx, ny, {(rx,ry): count}) or None."""
    p = os.path.join(d, "regions.bin")
    if not os.path.exists(p):
        return None
    b = open(p, "rb").read()
    if len(b) < 32 or b[:4] != b"PRGN":
        return None
    ox, oy = struct.unpack_from("<ff", b, 8)
    tile = struct.unpack_from("<f", b, 16)[0]
    nx, ny = struct.unpack_from("<II", b, 20)
    counts, off = {}, 32
    for ry in range(ny):
        for rx in range(nx):
            if off + 4 > len(b):
                break
            c = struct.unpack_from("<I", b, off)[0]
            off += 4
            if c:
                counts[(rx, ry)] = c
    return ox, oy, tile, nx, ny, counts


def scan_pmap(path, tmpdir):
    """Parse a region tile (inflating v3 -> v2 first). -> dict of counts, or {'error':...}."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        ver = struct.unpack_from("<I", head, 4)[0] if len(head) >= 8 else 0
        use = path
        if ver == 3:
            use = os.path.join(tmpdir, os.path.basename(path) + ".v2")
            pmap_lz4_decompress.decompress(path, use)
        sc = psp_scene.read_scene(open(use, "rb").read())
        if use != path:
            try:
                os.remove(use)
            except OSError:
                pass
    except Exception as e:                                       # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}

    n_sub = n_white = 0
    for m in sc.models:
        for s in m.submeshes:
            n_sub += 1
            if getattr(s, "texture", -1) < 0:
                n_white += 1
    tex_bytes = 0
    for t in sc.textures:
        tex_bytes += len(t.texel_bytes or b"")
        tex_bytes += len(t.clut_bytes or b"")
    return {
        "version": ver,
        "instances": len(sc.instances),
        "models": len(sc.models),
        "submeshes": n_sub,
        "white_submeshes": n_white,
        "textures": len(sc.textures),
        "tex_mb": round(tex_bytes / 1048576.0, 2),
        "file_mb": round(os.path.getsize(path) / 1048576.0, 2),
    }


def audit(d, tmpdir):
    out = {}
    names = sorted(n for n in os.listdir(d)
                   if n.startswith("region_") and n.endswith(".pmap"))
    for i, n in enumerate(names):
        key = n[:-5]                                             # region_X_Y
        r = scan_pmap(os.path.join(d, n), tmpdir)
        r["sidecars"] = [e for e in SIDECARS
                         if os.path.exists(os.path.join(d, key + "." + e))
                         and os.path.getsize(os.path.join(d, key + "." + e)) > 0]
        out[key] = r
        if (i + 1) % 20 == 0:
            print("  ... %d/%d" % (i + 1, len(names)), flush=True)
    return out


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    quiet = "--quiet" in argv
    if quiet:
        argv.remove("--quiet")
    jout = None
    if "--json" in argv:
        k = argv.index("--json")
        jout = argv[k + 1]
        del argv[k:k + 2]
    base_dir = None
    if "--baseline" in argv:
        k = argv.index("--baseline")
        base_dir = argv[k + 1]
        del argv[k:k + 2]
    d = argv[0]

    tmpdir = tempfile.mkdtemp(prefix="mapaudit_")
    print("== audit %s" % d, flush=True)
    cur = audit(d, tmpdir)
    base = None
    if base_dir:
        print("== baseline %s" % base_dir, flush=True)
        base = audit(base_dir, tmpdir)

    grid = read_regions_bin(d)
    anomalies = []

    # coverage: every non-empty grid cell must have a tile
    if grid:
        ox, oy, tile, cols, rows, cells = grid
        print("grid: %dx%d tile %.0f origin (%.0f,%.0f) - %d non-empty cells, %d instances"
              % (cols, rows, tile, ox, oy, len(cells), sum(cells.values())))
        for (cx, cy), want in sorted(cells.items()):
            key = "region_%d_%d" % (cx, cy)
            if key not in cur:
                anomalies.append(("VOID", key,
                                  "regions.bin says %d instances but no .pmap" % want))
            elif "error" not in cur[key] and cur[key]["instances"] != want:
                anomalies.append(("COUNT", key, "regions.bin %d != pmap %d instances"
                                  % (want, cur[key]["instances"])))

    tot = {"instances": 0, "models": 0, "submeshes": 0, "white_submeshes": 0}
    for key in sorted(cur):
        r = cur[key]
        if "error" in r:
            anomalies.append(("PARSE", key, r["error"]))
            continue
        for k in tot:
            tot[k] += r.get(k, 0)
        if r["instances"] == 0:
            anomalies.append(("EMPTY", key, "0 instances"))
        if r["white_submeshes"]:
            anomalies.append(("WHITE", key, "%d untextured submeshes (of %d)"
                              % (r["white_submeshes"], r["submeshes"])))
        if "col" not in r["sidecars"]:
            anomalies.append(("NOCOL", key, "no collision -> fall through world"))
        if r["tex_mb"] > TEX_MB_WARN:
            anomalies.append(("HEAVY", key, "%.2f MB resident textures (cache thrash risk)"
                              % r["tex_mb"]))
        if base and key in base and "error" not in base[key]:
            b = base[key]
            for field, tol in (("instances", 0.95), ("models", 0.95)):
                if b[field] and r[field] < b[field] * tol:
                    anomalies.append(("LOST", key, "%s %d -> %d (-%.0f%%)"
                                      % (field, b[field], r[field],
                                         100.0 * (1 - r[field] / float(b[field])))))
            if r["white_submeshes"] > b.get("white_submeshes", 0):
                anomalies.append(("NEWWHITE", key, "white submeshes %d -> %d"
                                  % (b.get("white_submeshes", 0), r["white_submeshes"])))
    if base:
        for key in sorted(base):
            if key not in cur:
                anomalies.append(("GONE", key, "present in baseline, missing in new bake"))

    if not quiet:
        print("\n%-14s %7s %6s %7s %6s %7s %7s  %s"
              % ("region", "inst", "mdl", "submsh", "white", "texMB", "fileMB", "sidecars"))
        for key in sorted(cur):
            r = cur[key]
            if "error" in r:
                print("%-14s  PARSE ERROR: %s" % (key, r["error"]))
                continue
            print("%-14s %7d %6d %7d %6d %7.2f %7.2f  %s"
                  % (key, r["instances"], r["models"], r["submeshes"],
                     r["white_submeshes"], r["tex_mb"], r["file_mb"],
                     ",".join(r["sidecars"])))

    sc_hist = {}
    for r in cur.values():
        for e in r.get("sidecars", []):
            sc_hist[e] = sc_hist.get(e, 0) + 1
    print("\n== TOTALS  regions=%d  instances=%d  models=%d  submeshes=%d  white=%d"
          % (len(cur), tot["instances"], tot["models"], tot["submeshes"],
             tot["white_submeshes"]))
    print("== sidecar coverage: %s"
          % "  ".join("%s=%d" % (k, sc_hist[k]) for k in sorted(sc_hist)))

    if anomalies:
        print("\n== ANOMALIES (%d)" % len(anomalies))
        by_kind = {}
        for kind, key, msg in anomalies:
            by_kind.setdefault(kind, []).append((key, msg))
        for kind in sorted(by_kind):
            items = by_kind[kind]
            print("  [%s] %d" % (kind, len(items)))
            for key, msg in items[:25]:
                print("     %-14s %s" % (key, msg))
            if len(items) > 25:
                print("     ... +%d more" % (len(items) - 25))
    else:
        print("\n== NO ANOMALIES")

    if jout:
        json.dump({"regions": cur, "anomalies": anomalies, "totals": tot},
                  open(jout, "w"), indent=1)
        print("json -> %s" % jout)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
