#!/usr/bin/env python3
"""mflags_sidecar_bake - per-region .mflags sidecars carrying 4 real SA IDE render flags.

Direct sibling of road_sidecar_bake.py. The .pmap stores only a tile-local model index
(no SA id), so this tool re-reads each region_X_Y.pmap, matches its instances BY POSITION
against the SA IPL placements, looks up each matched model's IDE `Flags` int, decodes the
4 render bits below, and writes them keyed by the tile-local MODEL index:

 TWOSIDED IDE 0x200000 DISABLE_BACKFACE_CULLING -> .mflags bit0
 ADDITIVE IDE 0x8 ADDITIVE -> .mflags bit1 (objs section ONLY --
 tobj-section additive is already carried by region_X_Y.tobj
 bit7; load_defs tags every tobj row with a 'time_on' key, so a
 def WITHOUT 'time_on' == objs section)
 DRAWLAST IDE 0x4 DRAW_LAST -> .mflags bit2
 NOZWRITE IDE 0x40 NO_ZBUFFER_WRITE -> .mflags bit3

Output region_X_Y.mflags (same framing as region_X_Y.road):

 'MFLG' u16 modelCount u16 version(=1), then modelCount x { u16 localModelIdx, u8 flags }

sparse - only models with >=1 of the 4 bits are emitted. A missing .mflags file is a
runtime no-op (Streaming.load_region_mflags), so every region without one behaves exactly
as before. The runtime ORs PMAP_MODELFLAG_{TWOSIDED,ADDITIVE,DRAWLAST,NOZWRITE} into that
model's per-model flag word and the renderer applies each (ide_flags_pipeline_audit.md).

Usage: python tools/mflags_sidecar_bake.py [chunks_dir] [--verify]
 --verify = dry run: report the counts, write nothing, delete nothing.
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_export"))
import sa_source

CHUNKS = ""

# SA IDE Flags int bits (PC/PS2 text.ide layout - ide_flags_pipeline_audit.md §0).
IDE_TWOSIDED = 0x200000
IDE_ADDITIVE = 0x8
IDE_DRAWLAST = 0x4
IDE_NOZWRITE = 0x40

#.mflags payload byte bits (match Streaming.load_region_mflags bit0..3).
F_TWOSIDED = 1
F_ADDITIVE = 2
F_DRAWLAST = 4
F_NOZWRITE = 8


def model_flag_bits(d):
    """Decode one def's IDE Flags int -> our 4-bit .mflags payload byte. ADDITIVE is
 honored ONLY for objs-section models (a tobj row carries 'time_on'); tobj additive
 is the existing region_X_Y.tobj bit7 path and must not be duplicated here."""
    fl = d.get("flags", 0)
    is_tobj = "time_on" in d
    b = 0
    if fl & IDE_TWOSIDED:
        b |= F_TWOSIDED
    if (fl & IDE_ADDITIVE) and not is_tobj:
        b |= F_ADDITIVE
    if fl & IDE_DRAWLAST:
        b |= F_DRAWLAST
    if fl & IDE_NOZWRITE:
        b |= F_NOZWRITE
    return b


def flagged_positions():
    """({(x,y,z rounded 0.1): OR of flag bits}, n_flagged_models, n_instances, counts).
 sa_source handles both text + binary IPLs (same source road_sidecar_bake matches on)."""
    defs = sa_source.load_defs()
    img = sa_source.open_img()
    inst = sa_source.load_instances(defs, img)

    mbits = {}
    for mid, d in defs.items():
        b = model_flag_bits(d)
        if b:
            mbits[mid] = b

    pos = {}
    for i in inst:
        b = mbits.get(i["model_id"])
        if not b:
            continue
        x, y, z = i["pos"]
        k = (round(x, 1), round(y, 1), round(z, 1))
        pos[k] = pos.get(k, 0) | b                       # union on a shared position key

    counts = {"tw": 0, "add": 0, "dl": 0, "nz": 0}       # unique flagged MODELS, per bit
    for b in mbits.values():
        if b & F_TWOSIDED:
            counts["tw"] += 1
        if b & F_ADDITIVE:
            counts["add"] += 1
        if b & F_DRAWLAST:
            counts["dl"] += 1
        if b & F_NOZWRITE:
            counts["nz"] += 1
    return pos, len(mbits), len(inst), counts


def region_model_flags(path, flag_pos):
    """{tile-local model idx: OR of flag bits} for this region's instances whose position
 matches a flagged SA placement. Identical .pmap walk to road_sidecar_bake."""
    blob = open(path, "rb").read()
    inst_count = struct.unpack_from("<I", blob, 36)[0]
    inst_off = struct.unpack_from("<I", blob, 40)[0]
    out = {}
    for i in range(inst_count):
        o = inst_off + i * 36                             # sizeof(PmapInstance) = 36
        mi = struct.unpack_from("<I", blob, o)[0]         # local model index at record+0
        x, y, z = struct.unpack_from("<3f", blob, o + 4)  # pos at record+4
        b = flag_pos.get((round(x, 1), round(y, 1), round(z, 1)))
        if b:
            out[mi] = out.get(mi, 0) | b
    return out


def _per_bit(mf):
    tw = sum(1 for b in mf.values() if b & F_TWOSIDED)
    ad = sum(1 for b in mf.values() if b & F_ADDITIVE)
    dl = sum(1 for b in mf.values() if b & F_DRAWLAST)
    nz = sum(1 for b in mf.values() if b & F_NOZWRITE)
    return tw, ad, dl, nz


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verify = "--verify" in sys.argv
    chunks = args[0] if args else CHUNKS

    flag_pos, n_mdl, n_inst, counts = flagged_positions()
    print("flagged IDE models: %d (tw=%d add=%d dl=%d nz=%d) | SA instances scanned: %d "
          "| flagged placement keys: %d"
          % (n_mdl, counts["tw"], counts["add"], counts["dl"], counts["nz"],
             n_inst, len(flag_pos)))

    total_models = files = 0
    reg_tw = reg_add = reg_dl = reg_nz = 0
    for pmap in sorted(glob.glob(os.path.join(chunks, "region_*_*.pmap"))):
        base = os.path.splitext(pmap)[0]
        out = base + ".mflags"
        mf = region_model_flags(pmap, flag_pos)
        if not mf:
            if not verify and os.path.exists(out):
                os.remove(out)                            # stale sidecar -> drop (like.road)
            continue
        if not verify:
            buf = b"MFLG" + struct.pack("<HH", len(mf), 1)   # magic + u16 count + u16 version
            for mi in sorted(mf):
                buf += struct.pack("<HB", mi, mf[mi])        # u16 localIdx + u8 flags
            open(out, "wb").write(buf)
        files += 1
        total_models += len(mf)
        tw, ad, dl, nz = _per_bit(mf)
        reg_tw += tw
        reg_add += ad
        reg_dl += dl
        reg_nz += nz
        print("  %s.mflags: %d models (tw=%d add=%d dl=%d nz=%d)"
              % (os.path.basename(base), len(mf), tw, ad, dl, nz))

    verb = "would write" if verify else "wrote"
    print("mflags: %s %d regions, %d models flagged: tw=%d add=%d dl=%d nz=%d"
          % (verb, files, total_models, reg_tw, reg_add, reg_dl, reg_nz))


if __name__ == "__main__":
    main()
