#!/usr/bin/env python3
"""ped_path_bake - SA ped path nodes (NODES*.DAT) -> per-region pedpath tiles.

Reads every DATA/PATHS/NODES%d.DAT via SAW parse_nodes, keeps the PED nodes
(they follow the vehicle nodes in each file), links them globally by
(area_id, node_id), then slices into the SAME region grid as the target
chunk set (origin/tile size from its regions.bin).

pedpath_<rx>_<ry>.bin layout (LE):
 'PPTH' u16 nnodes u16 nlinks
 nnodes * { f32 x,y,z; u8 width8 (units*8, cap 255); u8 spawnProb (0-15);
 u8 flags (bit0 water, bit1 switchedOff, bit2 deadEnd);
 u8 nlinks; u16 firstLink; u16 pad }
 nlinks * u32 ref: (rx<<24)|(ry<<16)|localIdx - target tile + local index
Runtime resolves cross-tile refs against loaded tiles (miss = link ignored).

Usage: python ped_path_bake.py <chunks_dir_with_regions.bin> [out_dir=same]
"""
import glob
import os
import struct
import sys

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from formats.dat import parse_nodes

ROOT_PC = ""


def main():
    chunks = sys.argv[1] if len(sys.argv) > 1 else \
        ""
    out = sys.argv[2] if len(sys.argv) > 2 else chunks

    man = open(os.path.join(chunks, "regions.bin"), "rb").read()
    _magic, _ver, ox, oy, tile, nx, ny, _cell = struct.unpack_from("<2I3f2If", man, 0)
    print("grid: origin(%.0f,%.0f) tile %.0f %dx%d" % (ox, oy, tile, nx, ny))

    # ---- load all ped nodes globally ----
    by_addr = {}          # (area, node_id) -> dict
    order = []            # insertion order of keys
    for path in sorted(glob.glob(os.path.join(ROOT_PC, "data/Paths/NODES*.DAT")) +
                       glob.glob(os.path.join(ROOT_PC, "data/paths/nodes*.dat"))):
        data = open(path, "rb").read()
        nd = parse_nodes(data)
        for n in nd.nodes[nd.num_vehicle_nodes:]:          # PED nodes only
            links = []
            for li in range(n.num_links):
                k = n.base_link_id + li
                if k < len(nd.links):
                    a = nd.links[k]
                    if not a.is_null:
                        links.append((a.area_id, a.node_id))
            key = (n.area_id, n.node_id)
            if key in by_addr:
                continue
            by_addr[key] = {
                "pos": (n.x, n.y, n.z),
                "width": n.path_width,
                "prob": n.spawn_probability,
                "water": n.water_node,
                "off": n.is_switched_off,
                "dead": n.on_dead_end,
                "links": links,
            }
            order.append(key)
    print("ped nodes: %d" % len(by_addr))

    # ---- assign to tiles ----
    def tile_of(x, y):
        rx = int((x - ox) // tile)
        ry = int((y - oy) // tile)
        if rx < 0 or ry < 0 or rx >= nx or ry >= ny:
            return None
        return (rx, ry)

    tiles = {}            # (rx,ry) -> [keys]
    loc = {}              # key -> (rx,ry,local_idx)
    for key in order:
        t = tile_of(by_addr[key]["pos"][0], by_addr[key]["pos"][1])
        if t is None:
            continue
        lst = tiles.setdefault(t, [])
        loc[key] = (t[0], t[1], len(lst))
        lst.append(key)

    # ---- emit ----
    worst_nodes = worst_links = 0
    for (rx, ry), keys in sorted(tiles.items()):
        nodes_blob = bytearray()
        links_blob = bytearray()
        nlinks_total = 0
        for key in keys:
            nd = by_addr[key]
            first = nlinks_total
            cnt = 0
            for lk in nd["links"]:
                tgt = loc.get(lk)
                if tgt is None:
                    continue
                links_blob += struct.pack("<I", (tgt[0] << 24) | (tgt[1] << 16) | tgt[2])
                cnt += 1
            nlinks_total += cnt
            w8 = int(min(255, max(0, nd["width"] * 8)))
            flags = (1 if nd["water"] else 0) | (2 if nd["off"] else 0) | (4 if nd["dead"] else 0)
            x, y, z = nd["pos"]
            nodes_blob += struct.pack("<3f4BHH", x, y, z, w8, nd["prob"], flags, cnt,
                                      first, 0)
        buf = b"PPTH" + struct.pack("<HH", len(keys), nlinks_total) + nodes_blob + links_blob
        open(os.path.join(out, "pedpath_%d_%d.bin" % (rx, ry)), "wb").write(buf)
        worst_nodes = max(worst_nodes, len(keys))
        worst_links = max(worst_links, nlinks_total)
    print("tiles written: %d, worst nodes/tile %d, worst links/tile %d"
          % (len(tiles), worst_nodes, worst_links))
    print("worst tile bytes: %d" % (8 + worst_nodes * 20 + worst_links * 4))


if __name__ == "__main__":
    main()
