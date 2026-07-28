#!/usr/bin/env python3
"""dyn_sidecar_bake - per-region dynamic-prop sidecars for the PSP runtime.

The .pmap instance records carry only a TILE-LOCAL model index (no SA model id),
so the runtime can't look props up in dynobj.bin directly. This tool re-reads
every region_X_Y.pmap, matches instances BY POSITION against the SA IPL
placements of dynamic models (object.dat names from dyn_names.txt), and writes

  region_X_Y.dyn:  'DYNS' u16 count u16 pad, then {u16 instIdx, u16 entryIdx}[]

entryIdx indexes the dynobj.bin entry table. Positions are unique per prop
(the same rounding key col_bake uses to dedupe), so the match is exact.

Usage: python dyn_sidecar_bake.py [chunks_dir]
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynobj_bake import parse_object_dat, load_ide_id2name

SA = os.environ.get("SA_ROOT", "")   # extract root; the chain sets SA_ROOT (like col_bake/grass)
MAPS = SA + "/data/maps"
CHUNKS = ""


def dyn_entry_table():
    """Rebuild the SAME entry list/order dynobj_bake.py wrote, returning
    name -> entryIdx. Must mirror dynobj_bake.main() exactly."""
    import sa_col
    rows = parse_object_dat()
    dyn = {n: r for n, r in rows.items()
           if not (r["mass"] >= 99998.0 and r["cdeff"] == 0)}
    id2name = load_ide_id2name()
    name2ids = {}
    for mid, nm in id2name.items():
        name2ids.setdefault(nm, []).append(mid)
    img = sa_col.ImgArchive(sa_col.IMG)
    idx, _ = sa_col.build_index(img)
    entries = []
    entry_of = {}
    name_entry = {}
    for name, r in sorted(dyn.items()):
        if name not in name2ids:
            continue
        cm = idx.get(name)
        col_r, col_h = 0.3, 2.0
        if cm is not None:
            try:
                (mnx, mny, mnz), (mxx, mxy, mxz) = cm.bmin, cm.bmax
                col_r = max(0.12, min(1.6, max(mxx - mnx, mxy - mny) * 0.5))
                col_h = max(0.3, min(12.0, mxz - mnz))
            except (AttributeError, TypeError):
                pass
        flags = (1 if r["bspk"] else 0) | (2 if r["expl"] else 0)
        key = (r["mass"], r["turn"], r["elast"], r["uproot"], r["cdmult"],
               r["smash"], r["bvel"], r["bvr"], col_r, col_h,
               r["cdeff"], r["spc"], r["bgun"], flags)
        ei = entry_of.get(key)
        if ei is None:
            ei = len(entries)
            entry_of[key] = ei
            entries.append(key)
        name_entry[name] = ei
    return name_entry


def sa_dyn_positions(name_entry):
    """{(rx, ry, rz rounded 0.1): entryIdx} for every IPL placement of a dynamic
    model (text IPLs by name + binary stream IPLs by id->name)."""
    id2name = load_ide_id2name()
    import sa_col
    img = sa_col.ImgArchive(sa_col.IMG)

    pos_map = {}

    def key(px, py, pz):
        return (round(px, 1), round(py, 1), round(pz, 1))

    # text IPLs
    ipls = glob.glob(MAPS + "/**/*.ipl", recursive=True) + \
           glob.glob(MAPS + "/**/*.IPL", recursive=True)
    for ipl in sorted(set(os.path.normcase(p) for p in ipls)):
        sec = None
        for line in open(ipl, "r", errors="replace"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low == "inst":
                sec = "inst"; continue
            if low == "end":
                sec = None; continue
            if sec != "inst":
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 11:
                continue
            nm = parts[1].lower()
            ei = name_entry.get(nm)
            if ei is None:
                continue
            try:
                px, py, pz = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            pos_map[key(px, py, pz)] = ei
    # binary stream IPLs
    for e in img.names(".ipl"):
        blob = img.read(e)
        if blob[:4] != b"bnry":
            continue
        numInst = struct.unpack_from("<I", blob, 4)[0]
        offInst = struct.unpack_from("<I", blob, 4 + 6 * 4)[0]
        for i in range(numInst):
            q = offInst + i * 40
            px, py, pz = struct.unpack_from("<3f", blob, q)
            mid = struct.unpack_from("<i", blob, q + 28)[0]
            nm = id2name.get(mid, "").lower()
            ei = name_entry.get(nm)
            if ei is not None:
                pos_map[key(px, py, pz)] = ei
    return pos_map


def pmap_instances(path):
    """Yield (instIdx, x, y, z) from a v2/v3 .pmap (header layout in pmap.h)."""
    blob = open(path, "rb").read()
    # header: magic,ver,size, model_cnt,model_off, submesh_cnt,submesh_off,
    #         texture_cnt,texture_off, instance_cnt(36), instance_off(40)
    inst_count = struct.unpack_from("<I", blob, 36)[0]
    inst_off = struct.unpack_from("<I", blob, 40)[0]
    for i in range(inst_count):
        o = inst_off + i * 36                    # sizeof(PmapInstance) = 36
        x, y, z = struct.unpack_from("<3f", blob, o + 4)
        yield i, x, y, z


def main():
    chunks = sys.argv[1] if len(sys.argv) > 1 else CHUNKS
    name_entry = dyn_entry_table()
    print(f"dyn entry names: {len(name_entry)}")
    pos_map = sa_dyn_positions(name_entry)
    print(f"SA dyn placements: {len(pos_map)}")

    total = files = 0
    for pmap in sorted(glob.glob(os.path.join(chunks, "region_*_*.pmap"))):
        base = os.path.splitext(pmap)[0]
        matches = []
        for i, x, y, z in pmap_instances(pmap):
            ei = pos_map.get((round(x, 1), round(y, 1), round(z, 1)))
            if ei is not None:
                matches.append((i, ei))
        out = base + ".dyn"
        if not matches:
            if os.path.exists(out):
                os.remove(out)
            continue
        buf = b"DYNS" + struct.pack("<HH", len(matches), 0)
        for i, ei in matches:
            buf += struct.pack("<HH", i, ei)
        open(out, "wb").write(buf)
        total += len(matches); files += 1
    print(f"wrote {files} region .dyn sidecars, {total} dynamic instances")


if __name__ == "__main__":
    main()
