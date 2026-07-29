#!/usr/bin/env python3
"""dynobj_bake - bake SA object.dat (dynamic/breakable props) for the PSP port.

Parses DATA/object.dat exactly like CObjectData::Initialise (static analysis the routine,
research/object_damage_system.md §1.2, BINARY behaviour incl. the default-row
slot mapping and the -500 FX sentinel), resolves model names to IDE ids, pulls
a collision capsule (r, h) per model from its ColModel bbox, and writes:

 data/dynobj.bin the runtime table (entries + modelId->entry map)
 tools/dyn_names.txt one model name per line - col_bake excludes these from
 the STATIC region collision (a felled lamp post must not
 leave an invisible wall behind).

dynobj.bin (little-endian):
 'DYN3' u16 nEntries u16 nMap
 entry (172B): f32 mass, turnMass, elasticity, uprootLimit, cdMult, smashMult,
 breakVel[3], breakVelRand, colR, colH;
 u8 nPrim, pad[3];
 prim[4] (28B each): f32 cx,cy,cz, hx,hy,hz, r
 r > 0: SPHERE centre (cx,cy,cz) radius r (hx/hy/hz = 0)
 r == 0: BOX centre + half-extents (model space)
 u8 cdEff, spCase, gunBreak, flags (flags: 1 sparks, 2 explodes)
 map (4B): u16 modelId, u16 entryIdx sorted by modelId (runtime bsearch)

The primitives are the REAL COL spheres+boxes (largest first, up to 4): a lamp
post is its narrow trunk box near the ground plus arm spheres 5u up - a 3D test
against these can't catch a car driving past under the arm (the b383 flat
XY-footprint did exactly that: lamppost2's arms made a 7.5u-wide ground box).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_col

SA = os.environ.get("SA_ROOT", "")   # extract root (data/object.dat); the chain sets SA_ROOT
# Output comes from argv so the converter can aim it at the user's data folder:
# dynobj_bake.py <out dynobj.bin> [out dyn_names.txt]
OUT_BIN = sys.argv[1] if len(sys.argv) > 1 else "dynobj.bin"
OUT_NAMES = (sys.argv[2] if len(sys.argv) > 2
             else os.path.join(os.path.dirname(os.path.abspath(__file__)), "dyn_names.txt"))


def parse_object_dat():
    """name -> 20-field tuple (mass..gunBreak), following the header column order.
 Returns (rows, defaults_used) with rows keyed by LOWER name."""
    rows = {}
    path = os.path.join(SA, "data", "object.dat")
    for line in open(path, "r", errors="replace"):
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("#"):
            continue
        if s.startswith("*"):
            break
        parts = s.replace(",", " ").split()   # commas+tabs mixed in the file
        if len(parts) < 13:
            continue
        name = parts[0].lower()
        try:
            mass = float(parts[1]); turn = float(parts[2]); air = float(parts[3])
            elast = float(parts[4]); sub = float(parts[5]); uproot = float(parts[6])
            cdmult = float(parts[7]); cdeff = int(parts[8]); spc = int(parts[9])
            camAvoid = int(parts[10]); expl = int(parts[11]); fxtype = int(parts[12])
        except ValueError:
            continue
        fxo = (0.0, 0.0, 0.0); fxname = "none"
        smash = 1.0; bvel = (0.0, 0.0, 0.0); bvr = 0.0; bgun = 0; bspk = 0
        if len(parts) >= 16:
            try: fxo = (float(parts[13]), float(parts[14]), float(parts[15]))
            except ValueError: pass
        if len(parts) >= 17: fxname = parts[16]
        if len(parts) >= 18:
            try: smash = float(parts[17])
            except ValueError: pass
        if len(parts) >= 21:
            try: bvel = (float(parts[18]), float(parts[19]), float(parts[20]))
            except ValueError: pass
        if len(parts) >= 22:
            try: bvr = float(parts[21])
            except ValueError: pass
        if len(parts) >= 23:
            try: bgun = int(parts[22])
            except ValueError: pass
        if len(parts) >= 24:
            try: bspk = int(parts[23])
            except ValueError: pass
        rows[name] = dict(mass=mass, turn=turn, air=air, elast=elast, sub=sub,
                          uproot=uproot, cdmult=cdmult, cdeff=cdeff, spc=spc,
                          camAvoid=camAvoid, expl=expl, fxtype=fxtype, fxo=fxo,
                          fxname=fxname, smash=smash, bvel=bvel, bvr=bvr,
                          bgun=bgun, bspk=bspk)
    return rows


def load_ide_id2name():
    import glob
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
                sec = low; continue
            if low == "end":
                sec = None; continue
            if sec:
                p = [x.strip() for x in s.split(",")]
                if len(p) >= 2:
                    try: id2name[int(p[0])] = p[1].lower()
                    except ValueError: pass
    return id2name


def gen_prims_from_mesh(cm):
    """No COL spheres/boxes -> derive up to 4 SPHERES from the collision MESH:
 slice the verts into z-layers, one sphere per layer at the layer's centroid
 (robust r = p90 of centroid distances, floored at half the layer height). A

 that ground traffic can't reach - the shape-following spheres asked
 for, placed where the object actually is."""
    verts = getattr(cm, "verts", None)
    if not verts or len(verts) < 3:
        return []
    z0 = min(v[2] for v in verts); z1 = max(v[2] for v in verts)
    h = max(0.05, z1 - z0)
    nlay = 4 if h > 3.0 else (3 if h > 1.6 else (2 if h > 0.7 else 1))
    step = h / nlay
    prims = []
    for li in range(nlay):
        lz0 = z0 + li * step; lz1 = lz0 + step
        lv = [v for v in verts if lz0 - 1e-4 <= v[2] <= lz1 + 1e-4]
        if len(lv) < 3:
            continue
        cx = sum(v[0] for v in lv) / len(lv)
        cy = sum(v[1] for v in lv) / len(lv)
        dists = sorted(((v[0]-cx)**2 + (v[1]-cy)**2) ** 0.5 for v in lv)
        r = dists[int(len(dists) * 0.9)] if len(dists) > 2 else dists[-1]
        r = max(0.12, min(1.6, max(r, step * 0.45)))
        prims.append((cx, cy, (lz0 + lz1) * 0.5, 0.0, 0.0, 0.0, r))
    return prims[:4]


def main():
    rows = parse_object_dat()
    print(f"object.dat rows: {len(rows)}")
    # default rows (mass 99999, cdmult 1, cdeff 0) are pure statics with special
    # col response - NOT interesting for phase 1 EXCEPT fenceparts; skip them so
    # they stay in the static world COL (they never move).
    dyn = {n: r for n, r in rows.items()
           if not (r["mass"] >= 99998.0 and r["cdeff"] == 0)}
    print(f"  dynamic-relevant (movable or breakable): {len(dyn)}")

    id2name = load_ide_id2name()
    name2ids = {}
    for mid, nm in id2name.items():
        name2ids.setdefault(nm, []).append(mid)

    # collision capsule from the ColModel bbox
    img = sa_col.ImgArchive(sa_col.IMG)
    idx, _libs = sa_col.build_index(img)
    # second lookup: COL headers carry the MODEL ID (v2/v3) - catches props whose
    # COL section name differs from the IDE model name (23 dyn props had no name
    # match -> they shipped a 0.3 stub sphere: "tiny collision floating over the
    # road" on the Grove St poles).
    by_id = {}
    for c in idx.values():
        if getattr(c, "model_id", 0):
            by_id.setdefault(c.model_id, c)

    entries = []          # deduped entry tuples
    entry_of = {}
    mapping = []          # (model_id, entry_idx)
    matched = 0
    stats = {}            # prim source census: col / mesh / bbox / stub
    for name, r in sorted(dyn.items()):
        ids = name2ids.get(name)
        if not ids:
            continue
        cm = idx.get(name)
        if cm is None:                # name miss -> try the COL-header model id
            for mid in ids:
                cm = by_id.get(mid)
                if cm is not None:
                    break
        col_r, col_h = 0.3, 2.0
        prims = []                    # (cx,cy,cz, hx,hy,hz, r)
        src = "stub"
        if cm is not None:
            try:
                (mnx, mny, mnz), (mxx, mxy, mxz) = cm.bmin, cm.bmax
                col_r = max(0.12, min(0.9, min(mxx - mnx, mxy - mny) * 0.5))
                col_h = max(0.3, min(12.0, mxz - mnz))
                # real COL primitives, biggest first: boxes (a pole's trunk is a
                # narrow tall box), then spheres (arm/lamp heads, usually high up).
                boxes = sorted(cm.boxes,
                               key=lambda b: -((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2])))
                for (x0, y0, z0, x1, y1, z1, _m) in boxes[:4]:
                    prims.append(((x0+x1)*0.5, (y0+y1)*0.5, (z0+z1)*0.5,
                                  max(0.05, (x1-x0)*0.5), max(0.05, (y1-y0)*0.5),
                                  max(0.05, (z1-z0)*0.5), 0.0))
                for (sx, sy, sz, sr, _m) in sorted(cm.spheres, key=lambda s: -s[3]):
                    if len(prims) >= 4:
                        break
                    prims.append((sx, sy, sz, 0.0, 0.0, 0.0, max(0.05, sr)))
                if prims:
                    src = "col"
                else:
                    # 59 dyn props ship a collision MESH only -> shape-following
                    # arm span a giant ground box: "huge collision / wrong place").
                    prims = gen_prims_from_mesh(cm)
                    if prims:
                        src = "mesh"
                if not prims:
                    # COL exists but has NO primitives and NO mesh: SA ships these
                    # as no-collision props (beach towels, banners, cigars). Bake
                    # ZERO prims - the b384 bbox/stub here is what produced the
                    # over the road" reports.
                    src = "empty"
            except (AttributeError, TypeError):
                pass
        # no COL at all (cj_ props, carcasses): also no car collision
        stats[src] = stats.get(src, 0) + 1
        prims = tuple(prims[:4])
        flags = (1 if r["bspk"] else 0) | (2 if r["expl"] else 0)
        key = (r["mass"], r["turn"], r["elast"], r["uproot"], r["cdmult"],
               r["smash"], r["bvel"], r["bvr"], col_r, col_h, prims,
               r["cdeff"], r["spc"], r["bgun"], flags)
        ei = entry_of.get(key)
        if ei is None:
            ei = len(entries)
            entry_of[key] = ei
            entries.append(key)
        for mid in ids:
            mapping.append((mid, ei))
            matched += 1

    mapping.sort()
    print(f"  entries: {len(entries)}, model mappings: {len(mapping)}")
    print(f"  prim sources: {stats}")

    buf = b"DYN3" + struct.pack("<HH", len(entries), len(mapping))
    for (mass, turn, elast, uproot, cdmult, smash, bvel, bvr,
         col_r, col_h, prims, cdeff, spc, bgun, flags) in entries:
        buf += struct.pack("<12f", mass, turn, elast, uproot, cdmult, smash,
                           bvel[0], bvel[1], bvel[2], bvr, col_r, col_h)
        buf += struct.pack("<4B", len(prims), 0, 0, 0)
        for pi in range(4):
            p = prims[pi] if pi < len(prims) else (0.0,) * 7
            buf += struct.pack("<7f", *p)
        buf += struct.pack("<4B", cdeff & 0xFF, spc & 0xFF, bgun & 0xFF, flags & 0xFF)
    for mid, ei in mapping:
        buf += struct.pack("<HH", mid, ei)

    os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
    open(OUT_BIN, "wb").write(buf)
    print(f"dynobj.bin: {len(buf)} bytes -> {OUT_BIN}")

    with open(OUT_NAMES, "w") as f:
        for name in sorted(dyn):
            if name in name2ids:
                f.write(name + "\n")
    print(f"dyn_names.txt: {sum(1 for n in dyn if n in name2ids)} names")


if __name__ == "__main__":
    main()
