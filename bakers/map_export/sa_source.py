#!/usr/bin/env python3
"""sa_source - enumerate + decode the source game map data via the SAW parsers.

Outputs plain dicts/lists:
 defs {model_id: {"dff","txd","dd","flags"}} (objs + tobj)
 instances [{"model_id","name","pos","quat","interior","is_lod"}]
 meshes come later (geom.py) - this module only lists WHICH dffs are needed.

IPL sources: every text .ipl from gta.dat + every binary *stream*.ipl inside
gta3.img (SA streams most of the world placements as binary IPL; their
model_id resolves via the IDE id map). is_lod = instance is referenced by
another instance's lod index, or its model name starts with "lod".
"""
import os
import sys

SAW = os.environ.get("SAW_ROOT", "")
if SAW not in sys.path:
    sys.path.insert(0, SAW)
from core.imgarchive import ImgArchive
from formats.ide import parse_ide
from formats.ipl import parse_ipl

# SA_ROOT env override: the Quarry converter points this at the user's extracted
# PS2 disc (world/grass/mflags/interior bakers all reach the map data through
# here). Defaults to the PC dev tree so the local dev loop is unchanged. The IDE
# tables + text/binary IPL placements are platform-independent game data, so the
# same parsers read the PS2 extract as-is.
ROOT_PC = os.environ.get("SA_ROOT", "")


def _gta_dat_files():
    """(ide_paths, ipl_paths) listed by data/gta.dat (DATA\\MAPS lines)."""
    ides, ipls = [], []
    for line in open(os.path.join(ROOT_PC, "data/gta.dat"), errors="replace"):
        s = line.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("ide "):
            ides.append(os.path.join(ROOT_PC, s[4:].strip().replace("\\", "/")))
        elif low.startswith("ipl "):
            ipls.append(os.path.join(ROOT_PC, s[4:].strip().replace("\\", "/")))
    return ides, ipls


def load_defs():
    """{model_id: {dff, txd, dd, flags}} from every gta.dat IDE (objs+tobj).
 dd = lod_dists[0] (SA multi-mesh models keep the first)."""
    defs = {}
    ides, _ = _gta_dat_files()
    for path in ides:
        if not os.path.exists(path):
            continue
        r = parse_ide(open(path, errors="replace").read())
        for sec in ("objs", "tobj"):
            for o in r.get(sec, []):
                dd = o.get("lod_dists") or [o.get("draw_dist", 300.0)]
                defs[o["id"]] = {
                    "dff": o["dff"].lower(),
                    "txd": o["txd"].lower(),
                    "dd": float(dd[0]),
                    "flags": o.get("flags", 0),
                }
                if sec == "tobj":       # b576: hour window (neon 20..6 etc.)
                    defs[o["id"]]["time_on"] = o.get("time_on", 20)
                    defs[o["id"]]["time_off"] = o.get("time_off", 6)
    return defs


def load_instances(defs, img):
    """All world placements: text IPLs (gta.dat) + binary stream IPLs (gta3.img).
 Returns [{model_id,name,pos,quat,interior,is_lod,lod_ref}] where lod_ref is
 the GLOBAL index (into this returned list) of the instance's LOD proxy, or
 -1. Link rules (SA):
 text ipl: inst.lod = row index in the SAME file
 binary ipl: inst.lod = row index in the COMPANION TEXT ipl - the stream
 file 'x_streamN.ipl' links into 'x.ipl' rows (SA quirk; a
 self-file read mislabels random details, caught on the pilot
 when the Grove bridge span got flagged is_lod and skipped).
 is_lod = the model name starts with 'lod' (every SA district proxy does)."""
    _, ipls = _gta_dat_files()
    all_rows = []            # (file_tag, is_text, per-file list of inst dicts)
    for path in ipls:
        if not os.path.exists(path):
            continue
        r = parse_ipl(open(path, "rb").read())
        all_rows.append((os.path.splitext(os.path.basename(path))[0].lower(),
                         True, r.get("inst", [])))
    for e in img.entries:
        if e.name.lower().endswith(".ipl"):
            try:
                r = parse_ipl(img.extract(e))
            except Exception:
                continue
            all_rows.append((os.path.splitext(e.name)[0].lower(),
                             False, r.get("inst", [])))

    text_rows = {tag: rows for tag, is_text, rows in all_rows if is_text}

    # pass 1: emit instances, remember (file,row) -> out index for link resolve
    out = []
    row_to_out = {}                    # (file_tag, row_idx) -> out index
    pending = []                       # (out_idx, target_tag, target_row)
    for tag, is_text, rows in all_rows:
        for k, inst in enumerate(rows):
            mid = inst["model_id"]
            d = defs.get(mid)
            name = (inst.get("name") or (d["dff"] if d else "")).lower()
            if d is None and not name:
                continue                          # unresolvable binary row
            x, y, z = inst["pos"]
            if x != x or y != y or z != z:        # NaN guard (whole-map lesson)
                continue
            interior = inst.get("interior", 0)
            if interior not in (0, 13):           # 13 = shared exterior world
                continue
            oi = len(out)
            row_to_out[(tag, k)] = oi
            out.append({
                "model_id": mid, "name": name,
                "pos": (float(x), float(y), float(z)),
                "quat": tuple(float(q) for q in inst["rot"]),
                "interior": interior,
                "is_lod": 1 if name.startswith("lod") else 0,
                "lod_ref": -1,
            })
            li = inst.get("lod", -1)
            if li is not None and li >= 0:
                if is_text:
                    pending.append((oi, tag, li))
                else:
                    base = tag.split("_stream")[0]
                    if base in text_rows and li < len(text_rows[base]):
                        pending.append((oi, base, li))

    # pass 2: resolve links to out indices (target may be filtered out -> -1)
    linked = 0
    for oi, ttag, trow in pending:
        ti = row_to_out.get((ttag, trow), -1)
        if ti >= 0:
            out[oi]["lod_ref"] = ti
            linked += 1
    out_linked = sum(1 for i in out if i["lod_ref"] >= 0)
    print("  lod links: %d resolved (%d live)" % (linked, out_linked))
    return out


def open_img():
    return ImgArchive.open(os.path.join(ROOT_PC, "MODELS/GTA3.IMG"))


def img_read(img, name):
    key = name.lower()
    for e in img.entries:
        if e.name.lower() == key:
            return img.extract(e)
    return None


if __name__ == "__main__":
    img = open_img()
    defs = load_defs()
    inst = load_instances(defs, img)
    lods = sum(i["is_lod"] for i in inst)
    print("defs=%d instances=%d (lod=%d)" % (len(defs), len(inst), lods))
    # the Grove bridge sanity anchor (spec): lae2_roads50/52 must resolve with dd
    for nm in ("lae2_roads50", "lae2_roads52", "lodlae2_roads50"):
        hit = [i for i in inst if i["name"] == nm]
        dd = next((defs[i["model_id"]]["dd"] for i in hit), None)
        print("  %-18s inst=%d dd=%s" % (nm, len(hit), dd))
