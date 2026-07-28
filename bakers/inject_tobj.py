#!/usr/bin/env python3
"""inject_tobj - add SA time-objects (IDE tobj: neon / lit-window overlays)
into the deployed regions, with a region_X_Y.tobj hour-window sidecar (b576).

OBSOLETE for anything baked by the current exporter. sa_export_pmap now keeps the
IDE tobj section (the models bake like any other) and build_grid_pmaps writes the
region_X_Y.tobj sidecar itself, so running this over such a bake would graft a
SECOND copy of every neon on top of the one already there. It refuses to start
when the chunks dir already carries .tobj sidecars; it stays only for the legacy
prod tiles that predate that exporter fix.

The old exporter could not resolve tobj models, so production has none of the
~76 neon/night-glow placements. The fresh map-export-v2 bake HAS them baked as
plain models. This tool imports each tobj instance (model + textures) from the
fresh bake into the matching prod tile.

STABLE-INDEX GUARANTEE: everything is APPENDED - models, submeshes, pools,
textures, instances - and the file is re-emitted with the ORIGINAL order
preserved (the grid's cell -> instance mapping is an indirection pool, so
instances need not be cell-sorted). Existing instance indices (region .lod /
.dyn sidecars) and vertex indices (.nightd) stay valid.

Usage: inject_tobj.py <chunks_dir> <fresh_raw_dir> [--dry]
"""
import glob
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
from pmap_tex_transplant import tile_to_scene
from pmap_tex_t4from128 import load_v2

HDR = struct.Struct('<20I')
MODEL = struct.Struct('<2I6f')
SUB = struct.Struct('<i4I')
TEX = struct.Struct('<HH7I')
INST = struct.Struct('<I3f4hfii')
GRIDH = struct.Struct('<2f f 2I I i')     # min_x min_y cell cells_x cells_y count pad

FRESH_OX, FRESH_OY, TILE = 471.0, -2745.0, 450.0


def _align16(b):
    while len(b) % 16:
        b.append(0)


def emit_v2(hdr_proto, models, subs, texs, insts, grid_raw_params, cell_lists,
            vpool, ipool, tpool, cpool):
    """Stable-order v2 writer mirroring psp_scene layout."""
    (min_x, min_y, cell, cx, cy) = grid_raw_params
    cells = cx * cy
    inst_index = []
    cell_off = []
    for c in range(cells):
        cell_off.append(len(inst_index))
        inst_index += cell_lists[c]
    cell_off.append(len(inst_index))

    out = bytearray(80)
    def put(table, rec):
        off = len(out)
        for r in rec:
            out.extend(table.pack(*r))
        _align16(out)
        return off
    model_off = put(MODEL, models)
    sub_off = put(SUB, subs)
    tex_off = put(TEX, texs)
    inst_off = put(INST, insts)
    grid_off = len(out)
    out.extend(GRIDH.pack(min_x, min_y, cell, cx, cy, len(inst_index), 0))
    for v in cell_off:
        out.extend(struct.pack('<i', v))
    for v in inst_index:
        out.extend(struct.pack('<H', v))
    _align16(out)
    vertex_off = len(out); out.extend(vpool); _align16(out)
    index_off = len(out); out.extend(ipool); _align16(out)
    texel_off = len(out); out.extend(tpool); _align16(out)
    clut_off = len(out); out.extend(cpool)
    HDR.pack_into(out, 0, 0x50414D50, 2, len(out),
                  len(models), model_off, len(subs), sub_off,
                  len(texs), tex_off, len(insts), inst_off, grid_off,
                  vertex_off, len(vpool), index_off, len(ipool),
                  texel_off, len(tpool), clut_off, len(cpool))
    return bytes(out)


def main():
    argv = sys.argv[1:]
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    chunks_dir, fresh_dir = argv

    # The exporter emits region_X_Y.tobj itself now, next to the tobj geometry it
    # bakes as normal models - grafting on top of that DUPLICATES every neon
    # (doubled instances, doubled models/textures, doubled sidecar entries). The
    # sidecar is the marker: if any tile has one, this bake is already served.
    served = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".tobj"))
    if served:
        sys.exit("inject_tobj: REFUSING - %s already carries tobj data (%d .tobj "
                 "sidecars, e.g. %s).\nsa_export_pmap.build_grid_pmaps writes them "
                 "at bake time; injecting here would duplicate the geometry. This "
                 "tool is only for legacy tiles baked before that exporter fix."
                 % (chunks_dir, len(served), served[0]))

    print("loading source (tobj set)...", flush=True)
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    insts_src = sa_source.load_instances(defs, img)
    tobj_time = {}
    for mid, d in defs.items():
        if "time_on" in d:
            # bit7 of the ON byte = IDE ADDITIVE flag (8): the runtime routes
            # those into the additive last pass; plain tobj (_nt swaps, beams)
            # render normally (b586 - b584 wrongly additived ALL tobj).
            add = 0x80 if (int(d.get("flags", 0)) & 8) else 0
            tobj_time[d["dff"]] = ((int(d["time_on"]) & 0x7F) | add,
                                   int(d["time_off"]) & 0xFF)
    tobj_src = [i for i in insts_src if i["name"] in tobj_time]
    print(f"tobj defs={len(tobj_time)} placements={len(tobj_src)}")

    # One pass over ALL fresh tiles: harvest each tobj model's payload by NAME
    # (fresh grid naming differs from prod - never guess tiles). Payload =
    # plain bytes so no scene stays resident.
    print("harvesting tobj payloads from the fresh bake...", flush=True)
    payload = {}          # name -> model payload
    inst_geo = {}         # (round pos) -> (name, quat, scale)
    src_at = {}
    for s in tobj_src:
        k = (round(s["pos"][0]*2), round(s["pos"][1]*2), round(s["pos"][2]*2))
        src_at[k] = s["name"]
    for fp in sorted(glob.glob(os.path.join(fresh_dir, "*.pmap"))):
        try:
            sc = tile_to_scene(fp)
        except Exception:
            continue
        for finst in sc.instances:
            k = (round(finst.pos[0]*2), round(finst.pos[1]*2),
                 round(finst.pos[2]*2))
            nm = src_at.get(k)
            if nm is None:
                continue
            inst_geo[k] = (nm, finst.quat, finst.scale)
            if nm in payload:
                continue
            fm = sc.models[finst.model]
            subs_p = []
            texmap = {}
            for sm in fm.submeshes:
                ti = sm.texture
                if ti >= 0 and ti not in texmap:
                    ft = sc.textures[ti]
                    texmap[ti] = (ft.width, ft.height, ft.format,
                                  bytes(ft.texel_bytes), ft.buffer_width,
                                  bytes(ft.clut_bytes), ft.clut_entries,
                                  ft.num_levels)
                subs_p.append((ti, bytes(sm.vertex_bytes), bytes(sm.index_bytes)))
            payload[nm] = dict(scale=fm.scale, center=fm.center,
                               radius=fm.bound_radius, dd=fm.draw_dist,
                               subs=subs_p, textures=texmap)
    print(f"harvested: {len(payload)} tobj models, {len(inst_geo)} placements",
          flush=True)

    total = files = 0
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.endswith(".pmap"):
            continue
        path = os.path.join(chunks_dir, fn)
        blob, ver = load_v2(path)
        h = HDR.unpack_from(blob, 0)
        g = GRIDH.unpack_from(blob, h[11])
        min_x, min_y, cellsz, cx, cy = g[0], g[1], g[2], g[3], g[4]
        x1, y1 = min_x + cx * cellsz, min_y + cy * cellsz
        want = [s for s in tobj_src
                if min_x <= s["pos"][0] < x1 and min_y <= s["pos"][1] < y1
                and not s["is_lod"]]
        if not want:
            continue

        models = [list(MODEL.unpack_from(blob, h[4] + 32*i)) for i in range(h[3])]
        subs = [list(SUB.unpack_from(blob, h[6] + 20*i)) for i in range(h[5])]
        texs = [list(TEX.unpack_from(blob, h[8] + 32*i)) for i in range(h[7])]
        insts = [list(INST.unpack_from(blob, h[10] + 36*i)) for i in range(h[9])]
        vpool = bytearray(blob[h[12]:h[12]+h[13]])
        ipool = bytearray(blob[h[14]:h[14]+h[15]])
        tpool = bytearray(blob[h[16]:h[16]+h[17]])
        cpool = bytearray(blob[h[18]:h[18]+h[19]])
        cells = cx * cy
        cell_off = struct.unpack_from('<%di' % (cells+1), blob,
                                      h[11] + GRIDH.size)
        idx_off = h[11] + GRIDH.size + 4*(cells+1)
        inst_index = struct.unpack_from('<%dH' % g[5], blob, idx_off)
        cell_lists = [list(inst_index[cell_off[c]:cell_off[c+1]])
                      for c in range(cells)]

        # skip if an instance already sits at a tobj pos (idempotency)
        have = {(round(i[1]*2), round(i[2]*2), round(i[3]*2)) for i in insts}

        added = []
        tex_remap = {}
        model_remap = {}
        for s in want:
            px, py, pz = s["pos"]
            k = (round(px*2), round(py*2), round(pz*2))
            if k in have:
                continue
            geo = inst_geo.get(k)
            if geo is None or s["name"] not in payload:
                continue
            nm_, fq, fscale = geo
            pl = payload[s["name"]]
            if s["name"] not in model_remap:
                first = len(subs)
                for (ti, vb, ib) in pl["subs"]:
                    if ti < 0:
                        nt = -1
                    else:
                        rk = (s["name"], ti)
                        if rk not in tex_remap:
                            (tw, th, tf, tb, tbw, cb, ce, nlev) = pl["textures"][ti]
                            _align16(tpool)
                            tfirst = len(tpool); tpool += tb
                            _align16(cpool)
                            cfirst = len(cpool); cpool += cb
                            texs.append([tw, th, tf, tfirst, len(tb),
                                         tbw, cfirst, ce, nlev])
                            tex_remap[rk] = len(texs) - 1
                        nt = tex_remap[rk]
                    vfirst = len(vpool) // 12
                    ifirst = len(ipool) // 2
                    vpool += vb
                    ipool += ib
                    subs.append([nt, vfirst, len(vb)//12, ifirst, len(ib)//2])
                models.append([first, len(pl["subs"]), pl["scale"],
                               pl["center"][0], pl["center"][1], pl["center"][2],
                               pl["radius"], pl["dd"]])
                model_remap[s["name"]] = len(models) - 1
            mi = model_remap[s["name"]]
            cellx = int((px - min_x) // cellsz); celly = int((py - min_y) // cellsz)
            if cellx < 0: cellx = 0
            if celly < 0: celly = 0
            if cellx >= cx: cellx = cx - 1
            if celly >= cy: celly = cy - 1
            cell = celly * cx + cellx
            inst_i = len(insts)
            q = [int(round(c * 32767)) for c in fq]
            insts.append([mi, px, py, pz, q[0], q[1], q[2], q[3],
                          fscale, 0, cell])
            cell_lists[cell].append(inst_i)
            on, off = tobj_time[s["name"]]
            added.append((inst_i, on, off, s["name"]))

        if not added:
            continue
        files += 1; total += len(added)
        print(f"  {fn}: +{len(added)} tobj "
              f"{[(a[3], a[1], a[2]) for a in added[:4]]}", flush=True)
        if dry:
            continue
        out = emit_v2(h, models, subs, texs, insts,
                      (min_x, min_y, cellsz, cx, cy), cell_lists,
                      vpool, ipool, tpool, cpool)
        if ver == 3:
            tmp = tempfile.mktemp(suffix='.pmap')
            open(tmp, 'wb').write(out)
            subprocess.check_call([sys.executable,
                                   os.path.join(TOOLS, 'pmap_lz4.py'),
                                   tmp, path], stdout=subprocess.DEVNULL)
            os.remove(tmp)
        else:
            open(path, 'wb').write(out)
        with open(path[:-5] + ".tobj", "wb") as f:
            f.write(struct.pack('<II', 0x4A424F54, len(added)))
            for inst_i, on, off, _nm in added:
                f.write(struct.pack('<HBB', inst_i, on, off))
    print(f"DONE: {total} tobj instances into {files} regions")


if __name__ == "__main__":
    main()
