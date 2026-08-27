#!/usr/bin/env python3
"""stream_io_model.py - static IO instrument for the region streamer.

Reads a baked chunkset (region_<rx>_<ry>.pmap, v2/v3/v4) and reports the exact
disk workload the PSP background streamer faces, WITHOUT running anything:

 * resident prefix (the index pmap_load reads synchronously on a tile swap),
 * per-tile streamed blob count + size distribution,
 * FILE-OFFSET LOCALITY: for every zone-grid cell, the set of model/texture
 blobs its instances need, and how far apart those blobs sit in the file.
 This is what decides whether the b411/b717 cluster pull (48 blobs within
 +-384KB, 256KB stdio vbuf) turns into sequential card reads or degenerates
 into one Memory-Stick seek per blob.
 * a transfer-time model at a measured card rate + seek cost.

Nothing here is a fix. It measures the workload so the on-device numbers have
something to be compared against.

Usage:
 python tools/stream_io_model.py <chunkset_dir> [--rate 9.0] [--seek 0.008]
 [--cells] [--tile RX,RY] [--json out.json]
"""
import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

HDR = "<24I"                 # 96 bytes, v4; v2/v3 have a shorter header, same prefix start
HDR_SZ = struct.calcsize(HDR)
F = ("magic version file_size model_count model_off submesh_count submesh_off "
     "texture_count texture_off instance_count instance_off grid_off "
     "vertex_off vertex_bytes index_off index_bytes texel_off texel_bytes "
     "clut_off clut_bytes comp_flag comp_model_off comp_tex_off uvrange_off").split()

PMAP_MAGIC = 0x50414D50

# Matches src/platform_psp/pmap.h's own PMAP_VERSION_STRIPPED (5): a world-store
# stage 2a stripped tile (tools/world_store_build.py strip_tile). comp_model/
# comp_tex hold GLOBAL ids there, not local byte offsets, and there is no blob
# region left for this instrument to measure at all - reading one as an ordinary
# tile used to just silently report it as "no compressed blobs" (self.comp = False,
# since comp_flag=2 != 1) instead of refusing it by name; found in the same review
# that found src/platform_psp/IoBench.c doing the analogous thing in C. A version
# is only a gate where something checks it - this file has its own separate copy
# of every.pmap constant (by design, see the module docstring) and so needed its
# own separate fix, same as IoBench.c did.
PMAP_VERSION_STRIPPED = 5


class Region(object):
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path)
        with open(path, "rb") as fp:
            head = fp.read(HDR_SZ)
            if len(head) < HDR_SZ:
                raise ValueError("short file")
            h = dict(zip(F, struct.unpack(HDR, head)))
            if h["magic"] != PMAP_MAGIC:
                raise ValueError("bad magic")
            if h["version"] == PMAP_VERSION_STRIPPED:
                raise ValueError(
                    "this is a STRIPPED tile (version=%d), not a self-contained .pmap - "
                    "it belongs to the world store (tools/world_store_build.py) and this "
                    "instrument cannot measure a blob layout that no longer exists in the "
                    "file without a companion world.idx/world.dat" % h["version"])
            self.h = h
            self.prefix = h["vertex_off"]          # bytes pmap_load reads synchronously
            fp.seek(0)
            self.buf = fp.read(self.prefix)        # the whole resident index
        self.comp = h["version"] >= 3 and h["comp_flag"] == 1

    # ---- blob tables -----------------------------------------------------
    def blobs(self):
        """[(kind, idx, off, csize)] for every non-empty streamed blob."""
        if not self.comp:
            return []
        out = []
        for kind, cnt, off in ((0, self.h["model_count"], self.h["comp_model_off"]),
                               (1, self.h["texture_count"], self.h["comp_tex_off"])):
            for i in range(cnt):
                o, c = struct.unpack_from("<II", self.buf, off + i * 8)
                if c:
                    out.append((kind, i, o, c))
        return out

    def comp_at(self, kind, idx):
        base = self.h["comp_model_off"] if kind == 0 else self.h["comp_tex_off"]
        return struct.unpack_from("<II", self.buf, base + idx * 8)

    # ---- geometry tables -------------------------------------------------
    def models(self):
        off, n = self.h["model_off"], self.h["model_count"]
        return [struct.unpack_from("<IIffffff", self.buf, off + i * 32) for i in range(n)]

    def submeshes(self):
        off, n = self.h["submesh_off"], self.h["submesh_count"]
        return [struct.unpack_from("<iIIII", self.buf, off + i * 20) for i in range(n)]

    def instances(self):
        off, n = self.h["instance_off"], self.h["instance_count"]
        return [struct.unpack_from("<IfffhhhhfIi", self.buf, off + i * 36) for i in range(n)]

    def grid(self):
        g = self.h["grid_off"]
        min_x, min_y, cell, cx, cy, npool, _pad = struct.unpack_from("<fffIIII", self.buf, g)
        cells = cx * cy
        coff_at = g + 28
        cidx_at = coff_at + (cells + 1) * 4
        coff = struct.unpack_from("<%di" % (cells + 1), self.buf, coff_at)
        cidx = struct.unpack_from("<%dH" % npool, self.buf, cidx_at)
        return dict(min_x=min_x, min_y=min_y, cell=cell, cx=cx, cy=cy,
                    coff=coff, cidx=cidx)


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f%s" % (n, u)
        n /= 1024.0
    return "?"


def pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100.0))]


def cell_locality(r, seek_s, rate_bps, window=256 * 1024):
    """Per zone-grid cell: the blob set its instances need and its file spread.

 A 'run' is a maximal group of blobs whose gaps are all below `window`, i.e.
 what ONE sequential stdio buffer pass can cover. runs == 1 means the cell's
 working set is one contiguous card read; runs == nblob means every blob is
 its own seek.
 """
    if not r.comp:
        return [], {}
    g = r.grid()
    subs = r.submeshes()
    mdl = r.models()
    inst = r.instances()
    mtex = []
    for m in mdl:
        first, cnt = m[0], m[1]
        t = set()
        for s in range(first, first + cnt):
            ti = subs[s][0]
            if ti >= 0:
                t.add(ti)
        mtex.append(t)

    rows = []
    for c in range(g["cx"] * g["cy"]):
        lo, hi = g["coff"][c], g["coff"][c + 1]
        if hi <= lo:
            continue
        models, texs = set(), set()
        for k in range(lo, hi):
            ii = g["cidx"][k]
            if ii >= len(inst):
                continue
            m = inst[ii][0]
            if m < len(mdl):
                models.add(m)
                texs |= mtex[m]
        spans = []
        for m in models:
            o, cz = r.comp_at(0, m)
            if cz:
                spans.append((o, cz))
        for t in texs:
            o, cz = r.comp_at(1, t)
            if cz:
                spans.append((o, cz))
        if not spans:
            continue
        spans.sort()
        nbytes = sum(cz for _, cz in spans)
        fspan = spans[-1][0] + spans[-1][1] - spans[0][0]
        runs = 1
        cur_end = spans[0][0] + spans[0][1]
        for o, cz in spans[1:]:
            if o - cur_end >= window:
                runs += 1
            cur_end = max(cur_end, o + cz)
        rows.append(dict(cell=c, nblob=len(spans), bytes=nbytes,
                         span=fspan, runs=runs,
                         t_ideal=nbytes / rate_bps + runs * seek_s,
                         t_random=nbytes / rate_bps + len(spans) * seek_s))
    if not rows:
        return [], {}
    summ = dict(
        cells=len(rows),
        blob_p50=pct([x["nblob"] for x in rows], 50),
        blob_p95=pct([x["nblob"] for x in rows], 95),
        bytes_p50=pct([x["bytes"] for x in rows], 50),
        bytes_p95=pct([x["bytes"] for x in rows], 95),
        runs_p50=pct([x["runs"] for x in rows], 50),
        runs_p95=pct([x["runs"] for x in rows], 95),
        span_p95=pct([x["span"] for x in rows], 95),
        t_ideal_p95=pct([x["t_ideal"] for x in rows], 95),
        t_random_p95=pct([x["t_random"] for x in rows], 95),
    )
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rate", type=float, default=9.0, help="MiB/s sequential card rate")
    ap.add_argument("--seek", type=float, default=0.008, help="seconds per random seek")
    ap.add_argument("--cells", action="store_true", help="per-cell locality pass (slow)")
    ap.add_argument("--tile", default=None, help="only this tile, e.g. 11,2")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rate_bps = a.rate * 1024 * 1024
    paths = sorted(f for f in os.listdir(a.dir) if re.match(r"region_\d+_\d+\.pmap$", f))
    if a.tile:
        rx, ry = a.tile.split(",")
        paths = [p for p in paths if p == "region_%s_%s.pmap" % (rx.strip(), ry.strip())]
    if not paths:
        sys.exit("no region_*.pmap in %s" % a.dir)

    regs, bad = [], []
    for p in paths:
        try:
            regs.append(Region(os.path.join(a.dir, p)))
        except Exception as e:                       # noqa: BLE001
            bad.append((p, str(e)))

    tot_file = sum(r.size for r in regs)
    tot_prefix = sum(r.prefix for r in regs)
    print("=" * 78)
    print("CHUNKSET %s" % os.path.abspath(a.dir))
    print("  regions parsed %d  (unreadable %d)" % (len(regs), len(bad)))
    for p, e in bad:
        print("    BAD %s: %s" % (p, e))
    print("  total on disk        %s" % human(tot_file))
    print("  resident prefix sum  %s   (%.1f%% of the set)"
          % (human(tot_prefix), 100.0 * tot_prefix / max(1, tot_file)))
    print("  pmap versions        %s" % dict(Counter(r.h["version"] for r in regs)))
    print()

    rows = []
    for r in regs:
        bl = r.blobs()
        sizes = [c for _, _, _, c in bl]
        rows.append(dict(
            name=r.name, file=r.size, prefix=r.prefix,
            models=r.h["model_count"], texs=r.h["texture_count"],
            insts=r.h["instance_count"], nblob=len(bl),
            cbytes=sum(sizes),
            avg=(sum(sizes) // len(sizes)) if sizes else 0,
            p50=pct(sizes, 50), p95=pct(sizes, 95), mx=max(sizes) if sizes else 0,
        ))
    rows.sort(key=lambda x: -x["file"])

    print("TOP 12 TILES BY FILE SIZE")
    print("  %-20s %9s %9s %6s %6s %7s %7s %8s %8s"
          % ("tile", "file", "prefix", "mdl", "tex", "inst", "blobs", "avgblob", "maxblob"))
    for x in rows[:12]:
        print("  %-20s %9s %9s %6d %6d %7d %7d %8s %8s"
              % (x["name"], human(x["file"]), human(x["prefix"]), x["models"],
                 x["texs"], x["insts"], x["nblob"], human(x["avg"]), human(x["mx"])))
    print()

    allsz = [x["avg"] for x in rows]
    nb = [x["nblob"] for x in rows]
    pf = [x["prefix"] for x in rows]
    print("ACROSS ALL %d TILES" % len(rows))
    print("  blobs/tile      p50 %d   p95 %d   max %d" % (pct(nb, 50), pct(nb, 95), max(nb)))
    print("  avg blob size   p50 %s   p95 %s" % (human(pct(allsz, 50)), human(pct(allsz, 95))))
    print("  resident prefix p50 %s   p95 %s   max %s"
          % (human(pct(pf, 50)), human(pct(pf, 95)), human(max(pf))))
    print()

    print("WHOLE-TILE FILL COST  (card %.1f MiB/s sequential, seek %.0f ms)"
          % (a.rate, a.seek * 1000))
    print("  %-20s %9s %7s %10s %11s %10s"
          % ("tile", "cbytes", "blobs", "seq", "clustered", "per-blob"))
    for x in rows[:12]:
        seq = x["cbytes"] / rate_bps + a.seek
        clus = x["cbytes"] / rate_bps + ((x["nblob"] + 47) // 48) * a.seek
        rnd = x["cbytes"] / rate_bps + x["nblob"] * a.seek
        print("  %-20s %9s %7d %9.2fs %10.2fs %9.2fs"
              % (x["name"], human(x["cbytes"]), x["nblob"], seq, clus, rnd))
    print()
    print("  the prefix read is SYNCHRONOUS on the main thread at a tile swap:")
    for x in rows[:5]:
        print("        %-20s prefix %8s -> %.0f ms of blocking fread"
              % (x["name"], human(x["prefix"]), 1000.0 * x["prefix"] / rate_bps))
    print()

    out = dict(dir=os.path.abspath(a.dir), rate_mibs=a.rate, seek_s=a.seek, tiles=rows)

    if a.cells:
        print("PER-CELL BLOB LOCALITY  (a run = blobs one %s sequential window covers)"
              % human(256 * 1024))
        print("  %-20s %6s %7s %7s %8s %7s %10s %10s"
              % ("tile", "cells", "blob50", "blob95", "byte95", "runs95", "t_ideal95", "t_rand95"))
        cellout = {}
        for r in regs:
            crows, s = cell_locality(r, a.seek, rate_bps)
            if not s:
                continue
            cellout[r.name] = s
            print("  %-20s %6d %7d %7d %8s %7d %9.2fs %9.2fs"
                  % (r.name, s["cells"], s["blob_p50"], s["blob_p95"],
                     human(s["bytes_p95"]), s["runs_p95"],
                     s["t_ideal_p95"], s["t_random_p95"]))
        out["cells"] = cellout
        print()

    if a.json:
        with open(a.json, "w") as fp:
            json.dump(out, fp, indent=1)
        print("wrote %s" % a.json)


if __name__ == "__main__":
    main()
