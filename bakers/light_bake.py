#!/usr/bin/env python3
"""Bake ALL the source game map light sources (2dfx type-0) into lights.bin: for every Los
Santos model that carries 2dfx LIGHT effects (street lamps, traffic lights, neon...),
expand each IPL placement to world-space coronas + point lights with the model's real
attributes (colour, corona size, point-light range, show mode, day/night flags).

Pipeline (gvcslib READ-ONLY): IDE id->dff name; IPL placements (text + binary streams);
sa_2dfx parses each unique model's DFF once; the instance rotation places the light's
local position into the world (same heading/conjugate rule the .pmap baker used).

Out: assets_build/lights.bin = u32 count + per-light record (32B):
 f32 x,y,z | u8 r,g,b,a | f32 coronaSize | f32 farClip | f32 ptRange | u8 showMode,flagsLo,flagsHi,pad
 python tools/light_bake.py
"""
import os, struct, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_ide, sa_ipl, sa_img
import sa_2dfx, sa_col, ps2dff

GAME = os.environ.get("SA_ROOT", "")
DATA = os.path.join(GAME, "data")
GTA3 = os.environ.get("SA_GTA3_IMG") or os.path.join(GAME, "models", "gta3.img")
OUT  = sys.argv[1] if len(sys.argv) > 1 else "lights.bin"
# Whole map by default. The old bake clipped to a Los Santos box, from when the port
# streamed one city; the global map wants every district lit. Pass x0 x1 y0 y1 to clip.
BBOX = tuple(float(v) for v in sys.argv[2:6]) if len(sys.argv) >= 6 else None


def rot_apply(q, lp):
    """Rotate local pos lp by the IPL quaternion q=(x,y,z,w), the .pmap convention:
 heading-only for tiny x,y, else the conjugate full quaternion."""
    x, y, z, w = q
    if abs(x) > 0.05 or abs(y) > 0.05:
        x, y, z = -x, -y, -z                       # conjugate
        n = math.sqrt(x*x+y*y+z*z+w*w) or 1.0
        x/=n; y/=n; z/=n; w/=n
        r00=1-2*(y*y+z*z); r01=2*(x*y-w*z);  r02=2*(x*z+w*y)
        r10=2*(x*y+w*z);   r11=1-2*(x*x+z*z);r12=2*(y*z-w*x)
        r20=2*(x*z-w*y);   r21=2*(y*z+w*x);  r22=1-2*(x*x+y*y)
        return (r00*lp[0]+r01*lp[1]+r02*lp[2],
                r10*lp[0]+r11*lp[1]+r12*lp[2],
                r20*lp[0]+r21*lp[1]+r22*lp[2])
    a = math.acos(max(-1.0, min(1.0, w))) * (2.0 if z < 0 else -2.0)   # heading-only
    ca, sa = math.cos(a), math.sin(a)
    return (lp[0]*ca - lp[1]*sa, lp[0]*sa + lp[1]*ca, lp[2])


def main():
    defs = sa_ide.parse_maps(DATA)
    img = sa_img.SaImg(GTA3)

    # Ground under a static lamp = instance.z + the model's local base offset. SA places
    # an object at FindGroundZForCoord - boundBox.min.z (World.cpp 2652), so instance.z +
    # base = the ground-probe result by construction. The VISUAL base (lowest DFF vertex)
    # is what the rendered ground meets; the COL boundBox min often sits ~1-2m lower (a
    # buried collision skirt) which would push the pool under the road -> use DFF, COL only
    # as a fallback when the geometry won't decode.
    col_idx, _ = sa_col.build_index(sa_col.ImgArchive(GTA3))
    base_cache = {}
    def base_z(mid):
        if mid in base_cache:
            return base_cache[mid]
        d = defs.get(mid)
        v = 0.0
        if d:
            dff = d.dff if d.dff.lower().endswith(".dff") else d.dff + ".dff"
            try:
                mdl = ps2dff.decode_sa(img.extract(dff))
                v = min(p[2] for me in mdl.meshes for p in me.positions)
            except Exception:
                cm = col_idx.get(d.dff[:-4].lower() if d.dff.lower().endswith(".dff") else d.dff.lower())
                v = cm.bmin[2] if cm else 0.0
        base_cache[mid] = v
        return v

    insts = []
    maps = os.path.join(DATA, "maps")
    for root, _d, files in os.walk(maps):
        for fn in files:
            if fn.lower().endswith(".ipl"):
                try: insts += sa_ipl.parse_text_ipl(os.path.join(root, fn))
                except Exception: pass
    for nm in img.names():
        if nm.lower().endswith(".ipl"):
            try:
                blob = img.extract(nm)
                if blob[:4] == b"bnry": insts += sa_ipl.parse_binary_ipl(blob)
            except Exception: pass
    print("instances scanned:", len(insts))

    if BBOX:
        x0, x1, y0, y1 = BBOX
        insts = [it for it in insts if x0 <= it.pos[0] <= x1 and y0 <= it.pos[1] <= y1]
        print("instances inside the bbox:", len(insts))

    model_lights = {}   # model_id -> [light dicts] (cache; None = no lights)
    def lights_of(mid):
        if mid in model_lights:
            return model_lights[mid]
        d = defs.get(mid)
        L = []
        if d:
            try: L = sa_2dfx.parse_lights(img.extract(d.dff if d.dff.lower().endswith(".dff") else d.dff + ".dff"))
            except Exception: L = []
        model_lights[mid] = L
        return L

    out = []
    seen = set()
    for it in insts:
        L = lights_of(it.model_id)
        if not L:
            continue
        for lt in L:
            wl = rot_apply(it.rot, lt["pos"])
            wx, wy, wz = it.pos[0]+wl[0], it.pos[1]+wl[1], it.pos[2]+wl[2]
            key = (round(wx,1), round(wy,1), round(wz,1), lt["color"])
            if key in seen: continue
            seen.add(key)
            groundZ = it.pos[2] + base_z(it.model_id)   # precise ground under the lamp
            out.append((wx, wy, wz, groundZ, lt))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<I", len(out)))
        for (wx, wy, wz, bz, lt) in out:
            r, g, b, a = lt["color"]
            f.write(struct.pack("<3f", wx, wy, wz))
            f.write(struct.pack("<4B", r, g, b, a))
            f.write(struct.pack("<3f", lt["coronaSize"], lt["farClip"], lt["ptRange"]))
            fl = lt["flags"]
            f.write(struct.pack("<4B", lt["showMode"] & 0xFF, fl & 0xFF, (fl >> 8) & 0xFF, 0))
            f.write(struct.pack("<2f", bz, lt["shadowSize"]))   # ground z + 2dfx shadow (pool) size
    print("wrote %s  (%d world lights from %d lit models)" %
          (OUT, len(out), sum(1 for v in model_lights.values() if v)))
    g = sum(1 for (wx,wy,wz,bz,lt) in out if 2250<=wx<=2800 and -2000<=wy<=-1400)
    print("  in Ganton bbox:", g)


if __name__ == "__main__":
    main()
