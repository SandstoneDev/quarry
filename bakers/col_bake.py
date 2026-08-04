#!/usr/bin/env python3
"""col_bake - bake the source game world collision for the PSP port's loaded patch.

The PSP world is a baked sub-region of Los Santos (see input.pmap grid). This
tool reproduces SA's static placement - read the text IPL `inst` sections (by
model name) AND the binary `bnry` IPL streams inside gta3.img (by model id),
resolve each ColModel (sa_col), apply the SA instance transform (position +
CONJUGATED IPL quaternion, exactly like CFileLoader::LoadObjectInstance), clip to
the pmap patch, and emit a compact per-model collision blob -> world_col.bin,
loaded by src/game_sa/Collision/Collision.c.

Layout is per-model (shared geometry) + an instance list, like SA keeps it - far
smaller than baking world-space triangles (2.8 MB vs 22 MB) and the runtime
transforms only the handful of instances near the ped. Boxes (lamp posts, walls
modelled as CBox) are triangulated to 12 faces so they collide too.

world_col.bin (little-endian) ===============================================
 (0x48 bytes):
 u32 magic 'WCOL' = 0x4C4F4357
 u32 version = 2
 u32 nModels, nInsts, nVerts, nFaces
 f32 gridMinX, gridMinY, gridCell
 u32 gridCX, gridCY
 u32 offModels, offVerts, offFaces, offInsts, offCellOff, offCellIdx, nCellIdx
Model (40B): u32 vFirst,vCount,fFirst,fCount; f32 vscale,radius,cx,cy,cz
Vert (6B): s16 x,y,z (local = s16 * model.vscale)
Face (8B): u16 a,b,c; u8 material; u8 flags
Inst (72B): u32 model; f32 m[9] (row-major, world = M*local + pos); f32 pos[3];
 f32 wc[3] (world bound centre); f32 wradius; u32 pad
Grid: cellOff[(CX*CY)+1] u32 prefix; cellIdx[nCellIdx] u32 instance indices.
 cell = gy*CX + gx, gx = (x-gridMinX)/gridCell
================================================================================

Modes:
 python col_bake.py measure # parse + filter, report stats / sizes
 python col_bake.py [out.bin] # full bake (default = deploy SA_PSP/world_col.bin)
"""
import os
import sys
import glob
import struct
import math

import sa_col

# SA_ROOT env override: Quarry points this at the user's extracted disc
# (sa_col honours SA_GTA3_IMG the same way). Defaults keep the dev loop.
SA = os.environ.get("SA_ROOT", "")
MAPS = SA + "/data/maps"
OUT_DEFAULT = ""

# pmap patch extent (from input.pmap grid header: min_x,min_y,cell,cx,cy)
PMIN_X, PMIN_Y = 400.09375, -2745.40625
PCELL, PCX, PCY = 400.0, 7, 5
PMAX_X = PMIN_X + PCELL * PCX
PMAX_Y = PMIN_Y + PCELL * PCY

# instance broadphase grid cell (world units)
GRID_CELL = 50.0
WCOL_MAGIC = 0x4C4F4357

# --- vegetation trunk cap ------------------------------------------------------
# SA models tall vegetation (palms, firs, pines) with a COL column that runs the
# FULL height of the tree - usually a single tall CBox, sometimes a tall mesh --
# so peds/cars can't walk through the trunk. But that column also blocks
# AIRCRAFT: a plane clips the invisible top of a 60u palm. We keep the trunk near
# the ground (cars/peds still stop) yet cap the COL height so aircraft clear the
# canopy. Classify by NAME only (material 43 is far broader) and err on the side
# of PRECISION - the SA 'vgs'/'vgn'/'vge' prefixes are Las Venturas map SECTIONS
# (vgn_corpbuild is a 154u skyscraper!), so we match vegetation tokens explicitly
# and never a bare 'vg*' prefix. Validated against every.col in gta3.img: 131
# models match, 0 buildings, and all 31 'fire*' props (fire_hydrant, firehouse,
# fire_esc, vgnfirestat...) are excluded by the 'fire' guard.
VEG_TRUNK_CAP = 7.0


def is_veg_name(name):
    """True for tall vegetation whose COL height should be capped. Name-only."""
    nm = name.lower()
    if nm.startswith("veg_"):            # canonical veg prefix (palms/trees/ferns)
        return True
    if "palm" in nm:                     # veg_palm*, vgs_palm*, sjmpalm*, veg_palmkb*
        return True
    if "pine" in nm:                     # pinetree*, Pinebg_*
        return True
    if ("_fir" in nm or nm.startswith("fir")) and "fire" not in nm:
        return True                      # sm_fir_*, vbg_fir_copse, firtree* (NOT fire_*)
    if nm.startswith("tree") or "_tree" in nm:
        return True                      # tree_hipoly*, veg_tree*, DEAD_TREE_*, des_treeline*
    return False


def quat_to_matrix(qx, qy, qz, qw):
    """SA conjugates the IPL quaternion (negate imag, keep real) then builds the
 rotation matrix. Returns row-major 3x3 R with v_world = R @ v_local."""
    x, y, z, w = -qx, -qy, -qz, qw
    n = x * x + y * y + z * z + w * w
    if n > 1e-12:
        s = 1.0 / math.sqrt(n)
        x *= s; y *= s; z *= s; w *= s
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    )


def parse_ipl_insts(path):
    """Yield (modelName, px,py,pz, qx,qy,qz,qw) from an IPL's `inst` section."""
    section = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if section is None:
                if low == "inst":
                    section = "inst"
                continue
            if low == "end":
                section = None
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 11:
                continue
            try:
                name = parts[1]
                px, py, pz = float(parts[3]), float(parts[4]), float(parts[5])
                qx, qy, qz, qw = (float(parts[6]), float(parts[7]),
                                  float(parts[8]), float(parts[9]))
            except ValueError:
                continue
            yield name, px, py, pz, qx, qy, qz, qw


def load_ide_id2name():
    """Map model id -> model name from every IDE (objs/tobj/anim sections).
 Binary IPL streams reference models by id; .col libraries key by name and
 often store modelId=0, so this mapping is required to resolve the bulk of the
 streamed map (roads / land / grass tiles especially)."""
    id2name = {}
    SECT = {"objs", "tobj", "anim"}
    ides = glob.glob(SA + "/data/**/*.ide", recursive=True) + \
           glob.glob(SA + "/data/**/*.IDE", recursive=True)
    for ide in sorted(set(os.path.normcase(p) for p in ides)):
        sec = None
        for line in open(ide, "r", errors="replace"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low in SECT:
                sec = low
                continue
            if low == "end":
                sec = None
                continue
            if sec:
                p = [x.strip() for x in s.split(",")]
                if len(p) >= 2:
                    try:
                        id2name[int(p[0])] = p[1]
                    except ValueError:
                        pass
    return id2name


def parse_bnry_insts(blob):
    """Yield (modelId, px,py,pz, qx,qy,qz,qw) from a binary `bnry` IPL stream.
 Header (after magic): numInst,_,_,_, numCars,_, offInst (7x u32). INST = 40B:
 pos(3f), rot(4f), modelId(i32), interior(i32), lod(i32)."""
    if blob[:4] != b"bnry":
        return
    numInst = struct.unpack_from("<I", blob, 4)[0]
    offInst = struct.unpack_from("<I", blob, 4 + 6 * 4)[0]
    for i in range(numInst):
        q = offInst + i * 40
        px, py, pz, rx, ry, rz, rw = struct.unpack_from("<7f", blob, q)
        mid = struct.unpack_from("<i", blob, q + 28)[0]
        yield mid, px, py, pz, rx, ry, rz, rw


def box_corners_faces(b, zcap=None):
    """8 corners + 12 (a,b,c,material) triangles of an axis-aligned CBox.
 zcap (veg trunk cap): lower the box top to zcap so aircraft clear tall
 vegetation, but never below the box floor - keeps the grounded trunk."""
    mnx, mny, mnz, mxx, mxy, mxz, mat = b
    if zcap is not None and mxz > zcap:
        mxz = max(mnz, zcap)
    v = [(mnx, mny, mnz), (mxx, mny, mnz), (mxx, mxy, mnz), (mnx, mxy, mnz),
         (mnx, mny, mxz), (mxx, mny, mxz), (mxx, mxy, mxz), (mnx, mxy, mxz)]
    idx = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 5, 1), (0, 4, 5),
           (1, 6, 2), (1, 5, 6), (2, 7, 3), (2, 6, 7), (3, 4, 0), (3, 7, 4)]
    return v, [(a, b_, c, mat) for (a, b_, c) in idx]


def model_geometry(cm):
    """Flatten a ColModel into (verts[(x,y,z)], faces[(a,b,c,mat)]) including its
 boxes-as-triangles. Returns model-local float coords. Tall vegetation is
 height-capped (VEG_TRUNK_CAP) so aircraft clear the canopy while the grounded
 trunk still collides - clamped on BOTH the mesh verts and the CBox tops."""
    zcap = VEG_TRUNK_CAP if is_veg_name(cm.name) else None
    if zcap is not None:
        verts = [(x, y, min(z, zcap)) for (x, y, z) in cm.verts]
    else:
        verts = list(cm.verts)
    faces = list(cm.faces)
    for box in cm.boxes:
        bv, bf = box_corners_faces(box, zcap)
        base = len(verts)
        verts.extend(bv)
        faces.extend((a + base, b + base, c + base, m) for (a, b, c, m) in bf)
    return verts, faces


def load_barrier_ids():
    """b464: model ids that are city-lock / bridge barriers, for the runtime "Barriers" toggle:
 every IDE object whose TXD is 'barrierblk' (the roadblock/bridge-barrier set 4510..4527:
 ce_fredbar, sfw/cn2/sfse_roadblock, ce_makospan - the LS<->SF<->countryside<->Vegas locks)
 PLUS the barriers.ide construction range 966..998. Both col_bake and barrier_sidecar_bake tag
 these the same way so the model render-skip and the collision-skip cover the same models."""
    ids = set(range(966, 999))
    SECT = {"objs", "tobj", "anim"}
    ides = glob.glob(SA + "/data/**/*.ide", recursive=True) + \
           glob.glob(SA + "/data/**/*.IDE", recursive=True)
    for ide in sorted(set(os.path.normcase(p) for p in ides)):
        sec = None
        for line in open(ide, "r", errors="replace"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low in SECT:
                sec = low; continue
            if low == "end":
                sec = None; continue
            if sec:
                p = [x.strip() for x in s.split(",")]
                if len(p) >= 3 and p[2].lower() == "barrierblk":
                    try:
                        ids.add(int(p[0]))
                    except ValueError:
                        pass
    return ids


def load_dyn_names():
    """Model names owned by the DYNAMIC object system (tools/dyn_names.txt from
 dynobj_bake.py). These are EXCLUDED from the static world collision: a felled
 lamp post must not leave an invisible wall (runtime capsules replace them)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dyn_names.txt")
    if not os.path.exists(path):
        return set()
    return {l.strip().lower() for l in open(path) if l.strip()}


def gather():
    """Collect unique models + in-patch instances.
 Returns (models, insts, stats):
 models[i] = (verts[(x,y,z)], faces[(a,b,c,mat)], radius, (cx,cy,cz))
 insts[j] = (model_idx, mat9, (px,py,pz))
 """
    img = sa_col.ImgArchive(sa_col.IMG)
    print("indexing col libraries...")
    idx, libs = sa_col.build_index(img)
    by_id = {}
    for cm in idx.values():
        by_id.setdefault(cm.model_id, cm)
    id2name = load_ide_id2name()
    dyn_names = load_dyn_names()
    barrier_ids = load_barrier_ids()          # b464: barrierblk (4510-4527) + 966-998
    print(f"  {len(idx)} named col models ({len(by_id)} by id) from {len(libs)} libs;"
          f" {len(id2name)} IDE id->name; {len(dyn_names)} dynamic (excluded)")

    def resolve_id(mid):
        nm = id2name.get(mid)
        cm = idx.get(nm.lower()) if nm else None
        return cm if cm is not None else by_id.get(mid)

    def candidates():
        ipls = glob.glob(MAPS + "/**/*.ipl", recursive=True) + \
               glob.glob(MAPS + "/**/*.IPL", recursive=True)
        for ipl in sorted(set(os.path.normcase(p) for p in ipls)):
            for (name, *rest) in parse_ipl_insts(ipl):
                yield idx.get(name.lower()), name, rest
        for nm in img.names(".ipl"):
            for (mid, *rest) in parse_bnry_insts(img.read(nm)):
                yield resolve_id(mid), mid, rest

    seen = set()
    model_index = {}          # model IDENTITY id(cm) -> index in `models`
                              # (NOT cm.model_id: 1009 country/terrain models all
                              # carry model_id==0, so keying by id collapses the
                              # whole back-country onto one model -> ground never
                              # emitted -> fall-through. id(cm) is stable here:
                              # every cm is pinned live by idx/by_id for the run.)
    models = []               # b460: each tuple is (verts, faces, radius, center, is_barrier)
    insts = []
    n_inst = n_in_patch = n_with_col = n_empty = n_missing = n_dup = 0
    missing = {}

    n_dyn_skip = 0
    for (cm, who, (px, py, pz, qx, qy, qz, qw)) in candidates():
        n_inst += 1
        if cm is None:
            n_missing += 1
            if isinstance(who, str):
                missing[who] = missing.get(who, 0) + 1
            continue
        nm = who.lower() if isinstance(who, str) else id2name.get(who, "").lower()
        if nm in dyn_names:
            n_dyn_skip += 1                    # runtime capsule owns this prop
            continue
        key = (round(px, 1), round(py, 1), round(pz, 1), id(cm))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        r = cm.radius
        if (px + r < PMIN_X or px - r > PMAX_X or
                py + r < PMIN_Y or py - r > PMAX_Y):
            continue
        n_in_patch += 1
        if not cm.faces and not cm.boxes:
            n_empty += 1
            continue
        n_with_col += 1

        mi = model_index.get(id(cm))
        if mi is None:
            verts, faces = model_geometry(cm)
            mi = len(models)
            model_index[id(cm)] = mi
            # b460/464: tag city-lock/bridge barrier models (barrierblk 4510-4527 + barriers.ide
            # 966..998) as the 5th tuple element so the runtime "Barriers" toggle skips their collision.
            models.append((verts, faces, cm.radius, cm.center,
                           1 if (cm.model_id in barrier_ids) else 0))
        insts.append((mi, quat_to_matrix(qx, qy, qz, qw), (px, py, pz)))

    used_v = sum(len(m[0]) for m in models)
    used_f = sum(len(m[1]) for m in models)
    mt = sorted(missing.items(), key=lambda kv: -kv[1])[:15]
    stats = dict(n_inst=n_inst, n_in_patch=n_in_patch, n_with_col=n_with_col,
                 n_empty=n_empty, n_missing=n_missing, n_dup=n_dup,
                 n_dyn_skip=n_dyn_skip,
                 n_models=len(models), n_insts=len(insts),
                 n_verts=used_v, n_faces=used_f, missing_top=mt)
    print(f"  dynamic props excluded from static COL: {n_dyn_skip}")
    return models, insts, stats


def build_blob(models, insts, gx0, gy0, gcell, gcx, gcy):
    # --- model table + shared vert/face pools (verts requantised to s16) ---
    model_rows = []
    vert_buf = bytearray()
    face_buf = bytearray()
    vfirst = ffirst = 0
    for (verts, faces, radius, center, _barr) in models:   # b460: 5th = is_barrier (read at emit)
        vmax = 0.0
        for (x, y, z) in verts:
            vmax = max(vmax, abs(x), abs(y), abs(z))
        vscale = (vmax / 32767.0) if vmax > 1e-6 else 1.0
        for (x, y, z) in verts:
            vert_buf += struct.pack("<3h",
                                    int(round(x / vscale)),
                                    int(round(y / vscale)),
                                    int(round(z / vscale)))
        for (a, b, c, mat) in faces:
            face_buf += struct.pack("<3HBB", a, b, c, mat & 0xFF, 0)
        model_rows.append((vfirst, len(verts), ffirst, len(faces),
                           vscale, radius, center[0], center[1], center[2]))
        vfirst += len(verts)
        ffirst += len(faces)

    model_buf = bytearray()
    for row in model_rows:
        model_buf += struct.pack("<4I5f", *row)

    # --- instances + world bound spheres ---
    ncells = gcx * gcy
    cell_lists = [[] for _ in range(ncells)]

    inst_buf = bytearray()
    for (j, (mi, m, (px, py, pz))) in enumerate(insts):
        cx, cy, cz = models[mi][3]
        radius = models[mi][2]
        wcx = m[0] * cx + m[1] * cy + m[2] * cz + px
        wcy = m[3] * cx + m[4] * cy + m[5] * cz + py
        wcz = m[6] * cx + m[7] * cy + m[8] * cz + pz
        inst_buf += struct.pack("<I9f3f3ffI", mi, *m, px, py, pz,
                                wcx, wcy, wcz, radius, models[mi][4])   # b460: offset-68 barrier flag
        # insert into all grid cells the world bound circle (xy) touches
        x0 = int((wcx - radius - gx0) / gcell)
        x1 = int((wcx + radius - gx0) / gcell)
        y0 = int((wcy - radius - gy0) / gcell)
        y1 = int((wcy + radius - gy0) / gcell)
        for yy in range(max(0, y0), min(gcy - 1, y1) + 1):
            for xx in range(max(0, x0), min(gcx - 1, x1) + 1):
                cell_lists[yy * gcx + xx].append(j)

    cell_off = bytearray()
    cell_idx = bytearray()
    acc = 0
    for c in cell_lists:
        cell_off += struct.pack("<I", acc)
        for j in c:
            cell_idx += struct.pack("<I", j)
        acc += len(c)
    cell_off += struct.pack("<I", acc)  # sentinel
    n_cellidx = acc

    # --- assemble (header 0x48, then sections, 16-byte aligned) ---
    HDR = 0x48

    def align(n):
        return (n + 15) & ~15

    off_models = align(HDR)
    off_verts = align(off_models + len(model_buf))
    off_faces = align(off_verts + len(vert_buf))
    off_insts = align(off_faces + len(face_buf))
    off_celloff = align(off_insts + len(inst_buf))
    off_cellidx = align(off_celloff + len(cell_off))
    total = align(off_cellidx + len(cell_idx))

    blob = bytearray(total)

    def put(off, data):
        blob[off:off + len(data)] = data

    struct.pack_into("<6I3f2I7I", blob, 0,
                     WCOL_MAGIC, 2, len(models), len(insts),
                     sum(len(m[0]) for m in models), sum(len(m[1]) for m in models),
                     gx0, gy0, gcell, gcx, gcy,
                     off_models, off_verts, off_faces, off_insts,
                     off_celloff, off_cellidx, n_cellidx)
    put(off_models, model_buf)
    put(off_verts, vert_buf)
    put(off_faces, face_buf)
    put(off_insts, inst_buf)
    put(off_celloff, cell_off)
    put(off_cellidx, cell_idx)
    return blob, dict(gcx=gcx, gcy=gcy, ncells=ncells, n_cellidx=n_cellidx,
                      sizes=dict(model=len(model_buf), vert=len(vert_buf),
                                 face=len(face_buf), inst=len(inst_buf),
                                 celloff=len(cell_off), cellidx=len(cell_idx),
                                 total=total))


def report(stats):
    for k in ("n_inst", "n_in_patch", "n_with_col", "n_empty", "n_missing",
              "n_dup", "n_models", "n_insts", "n_verts", "n_faces"):
        print(f"  {k:12} = {stats[k]}")
    if stats["missing_top"]:
        print("  top missing (no col, mostly LODs/interior props):")
        for nm, c in stats["missing_top"]:
            print(f"    {nm}  x{c}")


REGIONS_BIN = ""
REGION_OUT  = ""
GCELL_REGION = 50.0   # COL broadphase cell inside a region tile


def read_regions_bin(path):
    """Grid params from the .pmap region manifest (single source of truth -> the COL
 tiling matches Streaming.c's tile_of exactly)."""
    with open(path, "rb") as f:
        d = f.read(32)
    magic, ver, ox, oy, tile, nx, ny, cell = struct.unpack_from("<2I3f2If", d, 0)
    if magic != 0x4E475250:
        raise SystemExit("regions.bin: bad magic 0x%08X" % magic)
    return ox, oy, tile, nx, ny


def bake_regions(outdir):
    """Whole-map COL sliced into region_X_Y.col matching the .pmap region grid. An
 instance lands in EVERY tile its world bound circle intersects (not just its centre
 tile) so border overhangs are collidable from the neighbour -> no seams."""
    global PMIN_X, PMIN_Y, PMAX_X, PMAX_Y
    PMIN_X, PMIN_Y = -1e9, -1e9          # disable the patch clip -> gather the whole map
    PMAX_X, PMAX_Y =  1e9,  1e9
    models, insts, stats = gather()
    report(stats)
    ox, oy, tile, nx, ny = read_regions_bin(os.path.join(outdir, "regions.bin"))
    gc = int(math.ceil(tile / GCELL_REGION))
    print(f"\n  tiling {nx}x{ny} @ {tile}u (origin {ox:.1f},{oy:.1f}), region grid {gc}x{gc}@{GCELL_REGION}u")

    # precompute each instance's world bound centre once
    pre = []
    for (mi, m, (px, py, pz)) in insts:
        cx, cy, cz = models[mi][3]
        wcx = m[0]*cx + m[1]*cy + m[2]*cz + px
        wcy = m[3]*cx + m[4]*cy + m[5]*cz + py
        pre.append((mi, m, (px, py, pz), wcx, wcy, models[mi][2]))

    total = nfiles = peak = 0
    for ry in range(ny):
        for rx in range(nx):
            tminx, tmaxx = ox + rx*tile, ox + (rx+1)*tile
            tminy, tmaxy = oy + ry*tile, oy + (ry+1)*tile
            remap = {}; tmodels = []; tinsts = []
            for (mi, m, pos, wcx, wcy, r) in pre:
                if wcx + r < tminx or wcx - r > tmaxx or wcy + r < tminy or wcy - r > tmaxy:
                    continue
                nmi = remap.get(mi)
                if nmi is None:
                    nmi = len(tmodels); remap[mi] = nmi; tmodels.append(models[mi])
                tinsts.append((nmi, m, pos))
            if not tinsts:
                continue
            blob, bi = build_blob(tmodels, tinsts, tminx, tminy, GCELL_REGION, gc, gc)
            with open(os.path.join(outdir, f"region_{rx}_{ry}.col"), "wb") as f:
                f.write(blob)
            sz = bi["sizes"]["total"]; total += sz; nfiles += 1; peak = max(peak, sz)
            print(f"  region_{rx}_{ry}.col  {len(tinsts):5d} inst  {len(tmodels):4d} mdl  {sz/1024:7.0f}KB")
    print(f"\n  wrote {nfiles} region .col, total {total/1048576:.1f}MB, "
          f"largest {peak/1024:.0f}KB, ~3x3 resident <= {9*peak/1048576:.1f}MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "regions":
        outdir = sys.argv[2] if len(sys.argv) > 2 else REGION_OUT
        bake_regions(outdir)
        sys.exit(0)
    models, insts, stats = gather()
    report(stats)
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        sys.exit(0)
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT
    gcx = int(math.ceil((PMAX_X - PMIN_X) / GRID_CELL))
    gcy = int(math.ceil((PMAX_Y - PMIN_Y) / GRID_CELL))
    blob, bi = build_blob(models, insts, PMIN_X, PMIN_Y, GRID_CELL, gcx, gcy)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"\n  grid {bi['gcx']}x{bi['gcy']} ({bi['ncells']} cells), "
          f"cellidx={bi['n_cellidx']}")
    sz = bi["sizes"]
    print(f"  sizes(B): models={sz['model']} verts={sz['vert']} faces={sz['face']} "
          f"insts={sz['inst']} celloff={sz['celloff']} cellidx={sz['cellidx']}")
    print(f"  wrote {out}  ({sz['total']/1e6:.2f} MB)")
