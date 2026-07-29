#!/usr/bin/env python3
"""Bake per-region LOD files: region_<rx>_<ry>.lod next to each region_<rx>_<ry>.pmap.

The whole-map lod.bin (lod_bake.py) keys instances by GLOBAL index, which is useless
once the map is sliced into region tiles (each tile re-numbers its instances 0..N).
This bakes one .lod PER tile, keyed by that tile's LOCAL instance index, so the
runtime LOD chain (detail XOR proxy mutual exclusion) works in chunk/region mode --
killing the detail+proxy overdraw the global stretch fallback causes.

A detail and its LOD proxy sit at the same XY, so they almost always land in the SAME
tile -> the link resolves locally; a proxy in a neighbour tile gets -1 (the detail
just vanishes far out, as on the real map edge).

Reuses lod_bake.load_ipl_links (IPL `lod` field -> key->key map) verbatim; only the
pmap-key step is per-tile. Same PLOD layout as lod.bin.

Usage: python lod_bake_regions.py <region_dir> (dir holding region_*.pmap)
"""
import os, sys, glob, struct
import lod_bake   # sibling: load_ipl_links, ROOTS, ipl_key, _qpack


def load_tile_keys(path):
    """instance keys (pos f32 + quat s16 + is_lod byte) for one region .pmap tile,
 in on-disk order, plus key->local-index (first wins on dup)."""
    with open(path, 'rb') as f:
        v = struct.unpack_from('<20I', f.read(80), 0)
        ic, ioff = v[9], v[10]
        f.seek(ioff); idata = f.read(ic * 36)
    keys = []; key_to_idx = {}
    for k in range(ic):
        o = k * 36
        interior = struct.unpack_from('<i', idata, o + 28)[0]
        key = idata[o + 4:o + 24] + bytes((1 if interior else 0,))   # pos+quat+is_lod
        keys.append(key)
        if key not in key_to_idx:
            key_to_idx[key] = k
    return ic, keys, key_to_idx


def load_tile_instances(path):
    """(ic, [(pos_xyz, is_lod)]) for one region tile, in on-disk order."""
    with open(path, 'rb') as f:
        v = struct.unpack_from('<20I', f.read(80), 0)
        ic, ioff = v[9], v[10]
        f.seek(ioff); idata = f.read(ic * 36)
    out = []
    for k in range(ic):
        o = k * 36
        px, py, pz = struct.unpack_from('<3f', idata, o + 4)
        interior = struct.unpack_from('<i', idata, o + 28)[0]
        out.append(((px, py, pz), 1 if interior else 0))
    return ic, out


def main():
    if len(sys.argv) < 2:
        print("usage: lod_bake_regions.py <region_dir>")
        return 1
    region_dir = sys.argv[1]
    tiles = sorted(glob.glob(os.path.join(region_dir, "region_*.pmap")))
    if not tiles:
        print("no region_*.pmap in", region_dir); return 1
    print("tiles:", len(tiles))

    # POSITION-BASED LOD linking (robust). SA places a detail's LOD proxy at the
    # SAME world position as the detail; the pmap stores each instance's exact
    # f32 position. So for every detail (is_lod=0) we link to the co-located
    # proxy (is_lod=1) by position - no IPL re-parse, no key-matching.
    #
    # This replaces the old ipl-`lod`-index reconstruction, which was broken:
    # the binary-IPL lod field (lae2_stream5: riverbridge2 lod=197 in a 147-inst
    # file) is a GLOBAL building-pool index, not per-file, so re-parsing IPLs
    # and matching keys produced garbage links -> the invisible Grove bridge.
    QUANT = 8.0    # 1/8u position bucket (details+proxies share exact pos; this
                   # just tolerates f32 noise without merging distinct spots)

    def pkey(p):
        return (round(p[0] * QUANT), round(p[1] * QUANT), round(p[2] * QUANT))

    total_inst = total_links = 0
    for path in tiles:
        ic, insts = load_tile_instances(path)
        # bucket proxies by position
        prox_at = {}
        for k, (pos, is_lod) in enumerate(insts):
            if is_lod:
                prox_at.setdefault(pkey(pos), []).append(k)
        lod_idx = [-1] * ic
        for k, (pos, is_lod) in enumerate(insts):
            if is_lod:
                continue                              # proxies don't link
            cand = prox_at.get(pkey(pos))
            if cand:
                # co-located proxy (first; SA is 1 proxy per detail spot)
                lod_idx[k] = cand[0]
                total_links += 1
        out = path[:-5] + ".lod"                      # region_x_y.pmap ->.lod
        with open(out, 'wb') as o:
            o.write(struct.pack('<3I', 0x444F4C50, 1, ic))   # 'PLOD', version 1, count
            o.write(struct.pack('<%di' % ic, *lod_idx))
        total_inst += ic
    print("wrote %d .lod files: %d instances, %d LOD links (position-matched)"
          % (len(tiles), total_inst, total_links))
    return 0


if __name__ == "__main__":
    sys.exit(main())
