#!/usr/bin/env python3
"""interior_bake - bake ONE SA interior (by area code) into interior_<N>.pmap
+ interior_<N>.col for the PSP EntryExit runtime.

Interior geometry lives in gta_int.img (binary *stream*.ipl placements, DFF/TXD
in the same archive; defs come from the interior IDEs already listed in gta.dat
-- research/interior_enex_system.md). The map_export pipeline (geom/pack) is
reused as-is; the single tile is emitted via build_grid_pmaps with a huge tile
size and renamed. COL comes from the gta_int col libraries via col_bake's
model_geometry/build_blob.

 python interior_bake.py 3 # CARLS (CJ's house)
"""
import glob
import os
import shutil
import struct
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source
import geom
import pack as packm
from emit import emit_regions

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
# PS2-native codecs (the SAME swap the world chain validated, ps2world_pilot.py:
# 26,66): sa_txd decodes the disc's PS2 TXDs, ps2dff decodes the PS2-native DFF
# (day+night vertex colours). The PC sa_txd_d3d9 + formats.dff parse_dff readers
# choke on PS2 geometry (F_NATIVE), so they are gone from the interior path.
from gvcslib import psp_scene, sa_txd
from formats.ipl import parse_ipl
from core.imgarchive import ImgArchive

import ps2dff
import sa_col
import col_bake

# INPUT: SA_ROOT points at the user's extracted PS2 disc (Quarry sets it); the PC
# dev tree stays the fallback so the local loop is unchanged.
SA = os.environ.get("SA_ROOT", "")
GTA_INT = os.path.join(SA, "models", "gta_int.img")
# OUTPUT: bake into <data>/interiors. bake_all_interiors overrides OUT_DIR with the
# converter's own interiors dir (argv); no memstick/SA_PSP deploy from the baker.
OUT_DIR = os.environ.get("QUARRY_INT_OUT",
                         "")

# SA renders interior geometry as PRELIGHT + timecycle interior AMBIENT; our
# renderer only modulates the texel by the baked vertex colour, so shells with
# to material white (already 248+) and just clamp - their look is unchanged.
INT_AMBIENT_ADD = 90


def _add_interior_ambient(parts):
    out = []
    for sub in parts:
        tris = [tuple((p[0], p[1], (min(255, p[2][0] + INT_AMBIENT_ADD),
                                    min(255, p[2][1] + INT_AMBIENT_ADD),
                                    min(255, p[2][2] + INT_AMBIENT_ADD),
                                    p[2][3])) for p in t)
                for t in sub["tris"]]
        out.append({"mat": sub["mat"], "tris": tris})
    return out


class _MatShim:
    """Duck-typed map_export material (geom/pack read .color + .texture_name)."""
    __slots__ = ("color", "texture_name")

    def __init__(self, color, texture_name):
        self.color = color
        self.texture_name = texture_name


class _GeoShim:
    """Duck-typed SAW Geometry for map_export.geom.geometry_submeshes: exposes
 vertices / uvs / prelit_colors / splits / materials / num_vertices."""
    __slots__ = ("vertices", "uvs", "prelit_colors", "splits", "materials",
                 "num_vertices")


def _ps2_model_to_geo(model):
    """Adapt a ps2dff.decode_sa SaModel (already welded/triangulated .meshes) into
 the SAW-Geometry shape map_export.geom.process_geometry consumes - so the
 guard-band tessellation + striped-UV split + ambient/pack stay in the interior
 path UNCHANGED, only the DFF codec swapped PC->PS2. Colours are the PS2 DAY
 prelight (ps2dff packs day in the low byte); positions are frame-local, which
 for atomic IDE models (all interiors) is model-local (see ps2dff docstring)."""
    geo = _GeoShim()
    geo.vertices = []
    geo.uvs = [[]]
    geo.prelit_colors = []
    geo.splits = []
    geo.materials = []
    for mt in model.materials:
        c = mt["color"]                                  # RGBA8888: R<<24|G<<16|B<<8|A
        geo.materials.append(_MatShim(
            ((c >> 24) & 0xFF, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF),
            mt.get("texture_name", "")))
    uv0 = geo.uvs[0]
    for mesh in model.meshes:
        base = len(geo.vertices)
        geo.vertices.extend(mesh.positions)
        uv0.extend(mesh.uv)
        for col in mesh.colors:                          # per-vertex DAY RGBA8888
            geo.prelit_colors.append(
                ((col >> 24) & 0xFF, (col >> 16) & 0xFF, (col >> 8) & 0xFF, col & 0xFF))
        idx = []
        for (a, b, c) in mesh.triangles:                 # local -> global vertex idx
            idx.extend((base + a, base + b, base + c))
        geo.splits.append({"mat_index": mesh.material_index,
                           "indices": idx, "strip": False})
    geo.num_vertices = len(geo.vertices)
    return geo


def img_read(img, name):
    key = name.lower()
    for e in img.entries:
        if e.name.lower() == key:
            return img.extract(e)
    return None


def gather_instances(img_int, area, centre, radius):
    """Placements with interior == area INSIDE the pocket around `centre`
 (one area code is reused by many coordinate pockets across the map)."""
    cx, cy, cz = centre
    out = []
    for e in img_int.entries:
        if not e.name.lower().endswith(".ipl"):
            continue
        try:
            r = parse_ipl(img_int.extract(e))
        except Exception:
            continue
        for inst in r.get("inst", []):
            if inst.get("interior", 0) != area:
                continue
            x, y, z = inst["pos"]
            if x != x or y != y or z != z:
                continue
            if abs(x - cx) > radius or abs(y - cy) > radius or abs(z - cz) > 60.0:
                continue
            out.append(dict(model_id=inst["model_id"],
                            name=(inst.get("name") or "").lower(),
                            pos=(float(x), float(y), float(z)),
                            quat=tuple(float(q) for q in inst["rot"])))

    # TEXT interior IPLs (data/maps/interior/gen_int*.ipl etc.): the tattoo
    # parlours / sexshop / UFO bar / 24-7s are placed HERE, not in the
    # gta_int.img binary streams - they baked as "EMPTY pocket" before.
    # SA text inst row: id, model, interior, x,y,z, qx,qy,qz,qw, lod
    import glob as _glob
    for p in _glob.glob(os.path.join(SA, "data", "maps", "interior", "*.ipl")):
        ininst = False
        try:
            lines = open(p, encoding="latin-1").read().splitlines()
        except OSError:
            continue
        for ln in lines:
            s = ln.strip()
            if s == "inst": ininst = True; continue
            if s == "end":  ininst = False; continue
            if not ininst or not s or s.startswith("#"):
                continue
            parts = [t.strip() for t in s.split(",")]
            if len(parts) < 10:
                continue
            try:
                mid = int(parts[0]); nm = parts[1].lower()
                inter = int(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                qx, qy, qz, qw = (float(parts[6]), float(parts[7]),
                                  float(parts[8]), float(parts[9]))
            except ValueError:
                continue
            if inter != area:
                continue
            if x != x or y != y or z != z:
                continue
            if abs(x - cx) > radius or abs(y - cy) > radius or abs(z - cz) > 60.0:
                continue
            out.append(dict(model_id=mid, name=nm,
                            pos=(x, y, z), quat=(qx, qy, qz, qw)))
    return out


# module-level caches (reused across a batch of bakes)
_defs = None; _img_int = None; _img_main = None; _col_idx = None


def bake_interior(name, area, centre, radius=45.0):
    """Bake interior_<name>.pmap/.col for the pocket of `area` around `centre`.
 Returns (pmap_bytes, col_bytes) or None if the pocket is empty. Shared
 parsers are cached for batch calls."""
    global _defs, _img_int, _img_main, _col_idx
    if _defs is None:
        _defs = sa_source.load_defs()
        _img_int = ImgArchive.open(GTA_INT)
        _img_main = sa_source.open_img()
    defs, img_int, img_main = _defs, _img_int, _img_main

    inst = gather_instances(img_int, area, centre, radius)
    if not inst:
        print(f"  {name}: EMPTY pocket area {area} @ {centre[:2]}")
        return None
    xs = [i["pos"][0] for i in inst]; ys = [i["pos"][1] for i in inst]
    zs = [i["pos"][2] for i in inst]

    used = {}
    models = []
    texpool = packm.TexPool()
    txd_cache = {}
    scene_inst = []
    for i in inst:
        mid = i["model_id"]
        if mid not in used:
            used[mid] = -1
            d = defs.get(mid)
            if d:
                try:
                    blob = img_read(img_int, d["dff"] + ".dff") or \
                           img_read(img_main, d["dff"] + ".dff")
                    if blob:
                        dff = ps2dff.decode_sa(blob)         # PS2-native DFF
                        if d["txd"] not in txd_cache:
                            tblob = img_read(img_int, d["txd"] + ".txd") or \
                                    img_read(img_main, d["txd"] + ".txd")
                            txd_cache[d["txd"]] = (
                                {k.lower(): v for k, v in
                                 sa_txd.decode(tblob).items()}
                                if tblob else {})
                        parts = geom.process_geometry(_ps2_model_to_geo(dff))
                        parts = _add_interior_ambient(parts)
                        m = packm.pack_processed(parts, texpool,
                                                 txd_cache[d["txd"]], d["dd"],
                                                 txd_name=d["txd"])
                        if m is not None:
                            models.append(m)
                            used[mid] = len(models) - 1
                except Exception as ex:
                    print(f"  ! {d['dff']}: {ex}")
        gi = used[mid]
        if gi < 0:
            continue
        scene_inst.append(psp_scene.Instance(model=gi, pos=i["pos"],
                                             quat=i["quat"], interior=0))

    # single-tile emit: a huge tile guarantees one region file
    tmp = tempfile.mkdtemp(prefix="intbake_")
    emit_regions(models, texpool.list, scene_inst, tmp, tile=8192.0, cell=128.0)
    pmaps = glob.glob(os.path.join(tmp, "region_*.pmap"))
    if len(pmaps) != 1:
        print(f"  {name}: {len(pmaps)} tiles (pocket too spread) - skip")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    out_pmap = os.path.join(OUT_DIR, f"interior_{name}.pmap")
    shutil.copy(pmaps[0], out_pmap)
    shutil.rmtree(tmp, ignore_errors=True)

    # COL: the same instances against the gta_int col libraries
    if _col_idx is None:
        _col_idx, _ = sa_col.build_index(sa_col.ImgArchive(GTA_INT))
    int_idx = _col_idx
    cmodels = []; cindex = {}; cinsts = []
    id2name = {mid: defs[mid]["dff"] for mid in used if mid in defs}
    for i in inst:
        nm = id2name.get(i["model_id"], i["name"])
        cm = int_idx.get(nm)
        if cm is None or (not cm.faces and not cm.boxes):
            continue
        mi = cindex.get(nm)
        if mi is None:
            verts, faces = col_bake.model_geometry(cm)
            mi = len(cmodels); cindex[nm] = mi
            cmodels.append((verts, faces, cm.radius, cm.center, 0))  # b460: 5th = is_barrier
        (px, py, pz) = i["pos"]; (qx, qy, qz, qw) = i["quat"]
        cinsts.append((mi, col_bake.quat_to_matrix(qx, qy, qz, qw), (px, py, pz)))
    out_col = None
    if cinsts:
        blob, bi = col_bake.build_blob(cmodels, cinsts,
                                       min(xs) - 8.0, min(ys) - 8.0, 25.0, 8, 8)
        out_col = os.path.join(OUT_DIR, f"interior_{name}.col")
        open(out_col, "wb").write(blob)

    psz = os.path.getsize(out_pmap) // 1024
    csz = (os.path.getsize(out_col) // 1024) if out_col else 0
    print(f"  {name}: area {area} inst={len(scene_inst)} models={len(models)} "
          f"pmap={psz}KB col={csz}KB")
    return (psz, csz)


def main():
    # single-interior CLI: interior_bake.py NAME AREA CX CY CZ [R]
    if len(sys.argv) >= 6:
        name = sys.argv[1]; area = int(sys.argv[2])
        centre = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
        radius = float(sys.argv[6]) if len(sys.argv) > 6 else 45.0
        bake_interior(name, area, centre, radius)
        return
    # legacy default: CJ house
    bake_interior("CARLS", 3, (2496.0, -1692.0, 1014.0), 45.0)


if __name__ == "__main__":
    main()
