#!/usr/bin/env python3
"""car_bake - bake a SA vehicle (DFF clump + TXD) into car.bin for the PSP port.

v3 (CAR3). Geometry codec = the in-repo PS2 decoder (tools/ps2dff.py): on a PS2 disc
the vehicle DFF is PS2-NATIVE VIF geometry (F_NATIVE) that the PC SAW formats.dff parser
chokes on, so ps2dff decodes the clump (frames + atomics + per-material triangles) and a
thin shim re-shapes it into the object the bake code already consumes. Paint + IMG stay on:
 - ps2dff.py : PS2-native clump parse (frames, _ok/_dam atomics, binmesh)
 - formats/carcols.py : the REAL paint pipeline - marker material colours
 (60,255,0)=primary (255,0,175)=secondary (0,255,255)=tertiary
 (255,0,255)=quaternary (confirmed from the source game
 ) -> carcols palette colour.
 - core/imgarchive.py : gta3.img

What CAR2 got wrong / lacked, fixed here:
 - paint was keyed on TEXTURE NAME (vehiclegeneric256/empty) -> chrome painted body
 colour, marker mats with other textures missed. Now: material colour marker match.
 - non-marker material colours were dropped (black underbody rendered white). Now:
 vertex colour = material colour always.
 - only the *_ok panels were baked; damage was a fake hinge tilt. Now: each panel
 bakes its *_ok AND *_dam mesh (real SA dents + vehiclescratch/shatter textures);
 the runtime swaps on damage status.
 - no LOD. Now: chassis_vlo baked as a separate group (far draw).
 - every prim embedded its own texture copy (vehiclegeneric256 duplicated ~20x in
 file AND RAM). Now: deduped texture table, prims reference by index.

car.bin layout (LE), all after a 'CAR3' magic:
 f32 handling[24] (raw handling.cfg numeric fields)
 u8 colPrim[3], colSec[3], pad[2] (resolved carcols combo 0)
 f32 seat[3] (ped_frontseat, car space)
 f32 wheelScale, wheelRadius
 f32 wheelMount[4][3] (lf, rf, lb, rb - car space)
 u32 nTex
 nTex texture blocks:
 u16 tw,th,numLevels(|alphaMode<<8),clutEntries; u32 texlen,clutlen
 <texels><clut>
 u32 nComp
 nComp component headers:
 char name[16]; u8 kind,axis,hasDam,pad; f32 pivot[3]
 f32 okScale, okCenter[3]; u32 okN (prim run, OK mesh)
 f32 dmScale, dmCenter[3]; u32 dmN (prim run, DAM mesh; 0 = none)
 vlo header: f32 scale, center[3]; u32 n
 wheel header: f32 scale, center[3]; u32 n
 prim blocks in order: comp0 ok run, comp0 dam run, comp1 ..., vlo run, wheel run
 prim block:
 i16 texIdx (-1 untextured); u8 amode (0 opaque / 1 alphatest / 2 blend); u8 pad
 u32 vbytes, ibytes; <verts><idx> (vertex = tex s16x2, color 5551, pos s16x3)

PLN1 (--plane, build 434): the CAR4 layout with two insertions (everything else
byte-identical, see the writer below + Vehicle.c load_carlike - they are the truth):
 'PLN2'
 f32 handling[24]
 f32 fly[21] (handling.cfg $-line, research plane_port.md par.4:
 Thrust FallOff Yaw YawStab SideSlip Roll RollStab
 Pitch PitchStab FormLift AttackLift GearUpR
 GearDownL WindMult MoveRes TurnRes[3] SpeedRes[3])
 u8 colPrim[3], colSec[3], pad[2]
 f32 seat[3]; f32 wheelScale, wheelRadius
 f32 wheelMount[4][3] (raw DFF dummies; coincident tandem pairs kept - the runtime spreads them like the bike +-0.3)
 u8 wheelParent[4] (b837: gear slot each wheel hangs from in the DFF - 0 gear_l, 1 gear_r, 2 misc_a, 3 misc_b, 0xFF none.
 The wheels are CHILDREN of the gear frames, so the
 gear rotation carries them; that is the whole
 visible retraction, and m_anWheelStatus MISSING is
 physics only - PreRender never reads it.)
 u8 nProp, pad[3]
 f32 propAnchor[2][4] (xyz = shaft pivot, w = spin axis code: 1.0 = Y)
 ... then COL box+spheres / tex table / comps / vlo / wheel / prim blocks as CAR4.
Plane component kinds (Vehicle.c render): 5 = prop (spins about Y through its pivot),
6 = control surface (rudder Z, elevator/aileron X; angle=0 until Block B),
7 = gear strut (X; angle=0 until Block B). static_prop* is NOT baked (v1: the prop
always spins, the blur-disc swap is deferred). COL spheres are SELECTED down to
<= 24 on bake (greedy farthest-point + forced X/Y extremes: wingtips/nose/tail)
because the runtime loader silently truncates at 32 and wq reduces to 8-12.
"""
import math
import os
import re
import struct
import sys
import deploy_util

# in-repo PS2 DFF codec (tools/ps2dff.py) - decodes the PS2-native VIF vehicle
# geometry the PC SAW formats.dff parser can't read (the world/interior swap).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps2dff
from wheel_pick import WHEEL_ALIASES, pick_wheel_node

SAW = os.environ.get("SAW_ROOT", "")
sys.path.insert(0, SAW)
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))

from core.imgarchive import ImgArchive                      # SAW (IMG reader - platform-neutral)
from formats.carcols import parse_carcols, resolve_colors, PAINT_MARKERS, LIGHT_MARKERS  # SAW (text carcols.dat)
from gvcslib import sa_txd, psp_mesh, psp_tex               # PS2 TXD decoder + proven PSP packers
from gvcslib.sa_dff import SaModel, SaMesh                  # pack-model containers

# INPUT: SA_ROOT points at the user's extracted PS2 disc (Quarry sets it; the world chain
# also sets SA_GTA3_IMG). The PC dev tree stays the fallback so the local loop is unchanged.
# gta3.img + DATA/ (carcols.dat, vehicles.ide, handling.cfg) + models/generic/vehicle.txd
# all resolve under SA_ROOT (Windows is case-insensitive, so the MODELS/ paths below match).
ROOT     = os.environ.get("SA_ROOT", "")
ROOT_PC  = ROOT                                             # legacy alias used throughout
GTA3_IMG = os.environ.get("SA_GTA3_IMG", ROOT + "/models/gta3.img")

# OUTPUT: default dev-loop targets; --out <dir> (parsed in __main__) redirects every writer
# into <dir> (Quarry passes <OutDir>/vehicles) and drops the memstick deploy mirror (None).
OUT     = ""
DEPLOY  = ""


# --------------------------------------------- PS2 DFF -> SAW-shape adapter ---
# The bake logic (frame_world / geometry_meshes / bake*) was written against the SAW
# formats.dff object: dff.frames, dff.atomics and dff.geometries[i] with.splits /
#.materials /.vertices /.uvs /.num_vertices. ps2dff.load_dff decodes the PS2-native
# clump (already welded + triangulated, frames named, _ok/_dam atomics intact); these
# thin shims re-shape it into that object so the bake code runs UNCHANGED. Vehicles are
# painted from MATERIAL colours (carcols markers), so the PS2 day/night vertex prelight
# ps2dff also carries is irrelevant here and dropped.
class _MatShim:
    __slots__ = ("color", "texture_name")
    def __init__(self, color, texture_name):
        self.color = color                 # (r,g,b,a); PAINT/LIGHT markers match on rgb
        self.texture_name = texture_name

class _GeoShim:
    __slots__ = ("vertices", "uvs", "materials", "splits", "num_vertices")

class _FrameShim:
    __slots__ = ("name", "rotation", "position", "parent")

class _AtomicShim:
    __slots__ = ("frame_index", "geometry_index")

class _DffShim:
    __slots__ = ("frames", "atomics", "geometries")


def load_ps2_dff(blob):
    """PS2-native vehicle DFF bytes -> a SAW-formats.dff-shaped object (frames / atomics /
 geometries) the bake code consumes. One atomic is synthesized per framed geometry."""
    # Vehicles are authored for a different VU1 pipeline than the world, and its
    # 16-bit positions carry 10 fractional bits rather than 7. Decoding them with
    # from handling, so the tyres looked correct on an oversized body.
    m = ps2dff.load_dff(bytes(blob), ps2dff.POS_SCALE_VEHICLE)
    d = _DffShim()
    d.frames = []
    for f in m.frames:
        fs = _FrameShim()
        fs.name = f.name
        r = f.rot                              # flat row-major 3x3 (librw right/up/at rows)
        fs.rotation = [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]
        fs.position = tuple(f.pos)
        fs.parent = f.parent
        d.frames.append(fs)
    d.geometries = []
    d.atomics = []
    for gi, geo in enumerate(m.geometries):
        gs = _GeoShim()
        gs.vertices = geo.verts
        gs.uvs = [geo.uvs]                      # channel-0 UV list (bake reads geo.uvs[0])
        gs.num_vertices = len(geo.verts)
        gs.materials = [_MatShim(mt.color, mt.texture) for mt in geo.materials]
        by_mat = {}
        for (a, b, c, mat) in geo.tris:
            by_mat.setdefault(mat, []).extend((a, b, c))
        gs.splits = [{"mat_index": mat, "indices": idx, "strip": False}
                     for mat, idx in sorted(by_mat.items())]
        d.geometries.append(gs)
        if geo.frame_index >= 0:               # frame_index<0 = orphan geometry (no atomic)
            at = _AtomicShim()
            at.frame_index = geo.frame_index
            at.geometry_index = gi
            d.atomics.append(at)
    return d


# ------------------------------------------------------------ frame helpers --
def frame_world(frames):
    """World 4x4 (row-major, row-vector p'=p*M) per frame from SAW Frame list."""
    W = [None] * len(frames)
    def local(f):
        r = f.rotation; p = f.position
        return [r[0][0], r[0][1], r[0][2], 0,
                r[1][0], r[1][1], r[1][2], 0,
                r[2][0], r[2][1], r[2][2], 0,
                p[0],    p[1],    p[2],    1]
    def mul(A, B):
        return [sum(A[r*4+k] * B[k*4+c] for k in range(4)) for r in range(4) for c in range(4)]
    def build(i):
        if W[i] is None:
            loc = local(frames[i])
            par = frames[i].parent
            W[i] = mul(loc, build(par)) if par >= 0 else loc
        return W[i]
    for i in range(len(frames)):
        build(i)
    return W


def xform(m, x, y, z):
    return (x*m[0] + y*m[4] + z*m[8]  + m[12],
            x*m[1] + y*m[5] + z*m[9]  + m[13],
            x*m[2] + y*m[6] + z*m[10] + m[14])


# --------------------------------------------------- geometry -> SaMeshes ----
def geometry_meshes(geo, world, paint_rgbs):
    """SAW Geometry -> list of (SaMesh, Material). Verts transformed by `world`
 (or left local if None). Vertex colour = the SA paint rule:
 marker colour -> carcols palette colour for that slot
 anything else -> the material colour itself (chrome white, underbody black...)
 """
    out = []
    uvs = geo.uvs[0] if geo.uvs else [(0.0, 0.0)] * geo.num_vertices
    for sp in geo.splits:
        mi = sp["mat_index"]
        mat = geo.materials[mi] if 0 <= mi < len(geo.materials) else None
        idx = list(sp["indices"])          # ps2dff pre-triangulates -> splits are strip=False
        tris = [(idx[i], idx[i+1], idx[i+2]) for i in range(0, len(idx) - 2, 3)]
        if not tris or mat is None:
            continue
        r, g, b, a = mat.color
        slot = PAINT_MARKERS.get((r, g, b))
        corner = None
        if slot is not None and slot < len(paint_rgbs):
            r, g, b = paint_rgbs[slot]
        elif (r, g, b) in LIGHT_MARKERS:
            # b739: neutralize the 4 corner light-lens markers (LF amber / RF turquoise / rears)
            # -> white. These are runtime light-state sentinels; a static bake must NOT tint by them,
            # SetEditableMaterialsCB forcing the lens material colour to white.
            # b740: ALSO capture WHICH corner (0=LF 1=RF 2=LR 3=RR) so the prim writer can tag
            # this lens prim (the free pad byte); the runtime redraws tagged lenses emissive at night.
            corner = LIGHT_MARKERS[(r, g, b)]
            r, g, b = 255, 255, 255
        col = (r << 24) | (g << 16) | (b << 8) | 0xFF
        remap = {}; lp = []; lu = []; lt = []
        for (ta, tb, tc) in tris:
            t = []
            ok = True
            for gi in (ta, tb, tc):
                if gi >= geo.num_vertices:
                    ok = False; break
                li = remap.get(gi)
                if li is None:
                    li = len(lp); remap[gi] = li
                    x, y, z = geo.vertices[gi]
                    lp.append(xform(world, x, y, z) if world else (x, y, z))
                    lu.append(uvs[gi] if gi < len(uvs) else (0.0, 0.0))
                t.append(li)
            if ok:
                lt.append(tuple(t))
        if not lt:
            continue
        me = SaMesh(material_index=0)
        me.positions = lp; me.uv = lu; me.triangles = lt
        me.colors = [col] * len(lp)
        me.lightCorner = corner          # b740: 0..3 head/tail lens corner, else None
        out.append((me, mat))
    return out


# ------------------------------------------------------------- textures ------
class TexTable:
    """Dedup textures by name; author once, hand out indices."""
    def __init__(self, txd):
        self.txd = {k.lower(): v for k, v in txd.items()}
        self.order = []          # authored dicts, file order
        self.index = {}          # name -> idx (or -1)

    def get(self, name):
        key = (name or "").strip().lower()
        if not key:
            return -1
        if key in self.index:
            return self.index[key]
        idx = -1
        entry = self.txd.get(key)
        if entry is not None:
            w, h, rgba = entry
            try:
                t = psp_tex.author_psp_texture(rgba, w, h, fmt="T8", mipmaps=True)
                idx = len(self.order)
                self.order.append((key, t))
            except Exception as e:
                print("  ! texture %s failed: %s" % (key, e))
        else:
            print("  ! texture %s MISSING from txd" % key)
        self.index[key] = idx
        return idx

    def blob(self):
        buf = bytearray()
        for _, t in self.order:
            nl = t["num_levels"] | (t.get("alpha_mode", 0) << 8)
            tx = t["texel_bytes"]; cl = t["clut_bytes"]
            buf += struct.pack("<HHHH", t["width"], t["height"], nl, t["clut_entries"])
            buf += struct.pack("<II", len(tx), len(cl))
            buf += tx + cl
        return buf


def pack_run(saw_meshes, textable):
    """[(SaMesh, Material)] -> (scale, center, [prim block bytes])."""
    if not saw_meshes:
        return 1.0, (0.0, 0.0, 0.0), []
    model = SaModel()
    for me, _ in saw_meshes:
        model.meshes.append(me)
    packed = psp_mesh.pack_model(model)
    prims = []
    for i, prim in enumerate(packed["prims"]):
        me  = saw_meshes[i][0]
        mat = saw_meshes[i][1]
        tidx = textable.get(mat.texture_name)
        alpha = mat.color[3]
        if alpha < 250:
            amode = 2                                       # glass / translucent lens
        elif tidx >= 0:
            amode = textable.order[tidx][1].get("alpha_mode", 0)
        else:
            amode = 0
        # b740: tag a head/tail lens prim by corner (0..3 -> pad 1..4); non-lens prims -> pad 0.
        # Zero format-size change (the pad byte already existed & was 0); Vehicle.c read_prim
        # reads it into VPrim.lightKind to drive the emissive lit-lens night pass.
        corner = getattr(me, "lightCorner", None)
        pad = (corner + 1) if corner is not None else 0
        blk = struct.pack("<hBB", tidx, amode, pad)
        blk += struct.pack("<II", len(prim["vertex_bytes"]), len(prim["index_bytes"]))
        blk += prim["vertex_bytes"] + prim["index_bytes"]
        prims.append(blk)
    return packed["scale"], packed["center"], prims


def _amode_of(mat, textable):
    tidx = textable.get(mat.texture_name)
    if mat.color[3] < 250:          return 2        # glass / translucent
    if tidx >= 0:                   return textable.order[tidx][1].get("alpha_mode", 0)
    return 0


def merge_meshes_by_texture(saw_meshes, textable, vcap=60000):
    """Concatenate meshes that share (texture, alpha-mode) into ONE SaMesh each, so a rigid
 body draws ~1 prim (draw call) per unique texture instead of 1 per source component.
 Per-vertex colours survive, so distinct paint slots are kept. A group splits once it
 would exceed vcap verts (u16 index safety). Opaque groups are ordered before glass so
 the translucent prims draw last."""
    groups = {}          # (texname, amode, corner) -> list of open (SaMesh, Material) buckets
    order = []
    for me, mat in saw_meshes:
        # b740: keep each head/tail-light lens corner (0..3) its OWN prim - the 4 corners share
        # the vehiclelights128 texture+amode and would otherwise merge into one, losing the
        # per-corner tag the runtime lit-lens pass needs. Non-lens prims: corner None (merge as before).
        lc = getattr(me, "lightCorner", None)
        key = ((mat.texture_name or "").strip().lower(), _amode_of(mat, textable), lc)
        bucket = groups.get(key)
        if bucket is None:
            bucket = []; groups[key] = bucket; order.append(key)
        if not bucket or len(bucket[-1][0].positions) + len(me.positions) > vcap:
            gm = SaMesh(material_index=0)
            gm.positions = []; gm.uv = []; gm.colors = []; gm.triangles = []
            gm.lightCorner = lc          # b740: propagate the corner tag to the merged mesh
            bucket.append((gm, mat))
        gm = bucket[-1][0]; base = len(gm.positions)
        gm.positions += me.positions
        gm.uv        += me.uv
        gm.colors    += me.colors
        gm.triangles += [(a + base, b + base, c + base) for (a, b, c) in me.triangles]
    order.sort(key=lambda k: 1 if k[1] == 2 else 0)   # glass last (stable)
    out = []
    for key in order:
        out += groups[key]
    return out


def pack_run_merged(meshes, textable):
    """pack_run after collapsing same-(texture,alpha) material-splits into one mesh -> ~1
 draw per unique texture in the component (a detailed door was ~8 material-split prims)."""
    return pack_run(merge_meshes_by_texture(meshes, textable), textable)


# ----------------------------------------------------- embedded vehicle COL --
# SA vehicle DFFs embed the collider in the clump's RW collision plugin (chunk
# 0x0253F2FA). On PC it is a standard COL3 library (FourCC "COL3"); on the PS2 disc it is
# a HEADERLESS COL1 body - the in-memory CColModel serialized straight from the TBounds,
# with NO "COL3"/name/modelId wrapper (verified byte-exact: the body consumes the chunk
# with zero slack across the roster). We only need the spheres + bound box:
# TBounds(40): f32 radius, center[3], min[3], max[3] (sphere-first, COL1 order)
# u32 nSph; nSph * { f32 radius, center[3]; u8 surface, PIECE, brightness, light } (0x14)
# u32 nLine / nBox / nVert / nFace... follow (unused here).
# Each sphere carries the eVehicleCollisionComponent PIECE id (1 bonnet 2 boot 3 bump_f
# 4 bump_r 5-8 doors 9-12 wings 17/19 windscreen; 0/255 = plain body) -> car.bin damage map.
_C_EXT, _C_CLUMP, _C_COLL = 0x03, 0x10, 0x0253F2FA


def _rw_find(b, off, end, cid):
    """First child chunk of id `cid` in [off, end) -> (bodyOff, bodySize) or (None, 0)."""
    while off + 12 <= end:
        c, size, _v = struct.unpack_from("<III", b, off)
        if size > end - off - 12:
            return None, 0
        if c == cid:
            return off + 12, size
        off += 12 + size
    return None, 0


class _VehCol:
    __slots__ = ("spheres", "bound_min", "bound_max")


def extract_vehicle_col(dff_bytes):
    """Decode the PS2 headerless-COL1 collision from the vehicle clump's extension.
 Returns a _VehCol (spheres=[{center,radius,piece}], bound_min, bound_max), or None
 when there is no collision chunk / the layout is not the expected one (caller then
 keeps its default bound box + empty sphere set)."""
    b = bytes(dff_bytes)
    cl_off, cl_size = _rw_find(b, 0, len(b), _C_CLUMP)
    if cl_off is None:
        return None
    e_off, e_size = _rw_find(b, cl_off, cl_off + cl_size, _C_EXT)     # clump-level extension list
    if e_off is None:
        return None
    co, cs = _rw_find(b, e_off, e_off + e_size, _C_COLL)
    if co is None:
        return None
    p = b[co:co + cs]
    try:
        bmin = struct.unpack_from("<3f", p, 16)
        bmax = struct.unpack_from("<3f", p, 28)
        nsph = struct.unpack_from("<I", p, 40)[0]
        o = 44
        if nsph > 4096 or o + nsph * 0x14 > len(p):      # sanity: reject a mis-identified chunk
            print("  ! embedded COL: unexpected layout (nsph=%d) - default box, no spheres" % nsph)
            return None
        col = _VehCol()
        col.bound_min = bmin
        col.bound_max = bmax
        col.spheres = []
        for _ in range(nsph):
            r = struct.unpack_from("<f", p, o)[0]
            c = struct.unpack_from("<3f", p, o + 4)
            col.spheres.append({"center": list(c), "radius": r, "piece": p[o + 17]})
            o += 0x14
        return col
    except struct.error as e:
        print("  ! embedded COL parse failed:", e)
        return None


# ------------------------------------------------------------- handling ------
# handling[24] baked layout (parsed by CFG column, letters -> codes) - the runtime
# rigid-body physics reads these by index (see Vehicle.c HND_*):
# 0 mass 1 turnMass 2 dragMult 3 cmX 4 cmY 5 cmZ 6 tractionMult
# 7 tractionLoss 8 tractionBias 9 numGears 10 maxVel(km/h) 11 engineAccel
# 12 engineInertia 13 driveType(0=F,1=R,2=4wd) 14 brakeDecel 15 brakeBias
# 16 steeringLock(deg) 17 suspForce 18 suspDamp 19 suspUpper 20 suspLower
# 21 suspBias 22 antiDive 23 handlingFlags(hex as float bits low 24)
def load_handling(name):
    for line in open(ROOT_PC + "/data/handling.cfg", "r", errors="replace"):
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("%"):
            continue
        t = s.split()
        if not t or t[0].upper() != name.upper():
            continue
        # CFG columns after the name (see research vehicle_physics.md §1/§8):
        # 1 mass 2 turnMass 3 drag 4 cmX 5 cmY 6 cmZ 7 subm 8 tracMul 9 tracLoss
        # 10 tracBias 11 nGears 12 maxV 13 engAcc 14 engInert 15 drive 16 engType
        # 17 brakeDec 18 brakeBias 19 ABS 20 steerLock 21 suspForce 22 suspDamp
        # 23 hiSpd 24 suspUp 25 suspLo 26 suspBias 27 antiDive 28 seatOff 29 dmgMul
        # 30 money 31 modelFlags 32 handlingFlags...
        def f(i, d=0.0):
            try: return float(t[i])
            except (IndexError, ValueError): return d
        drive = {"F": 0.0, "R": 1.0, "4": 2.0}.get(t[15].upper() if len(t) > 15 else "R", 1.0)
        try: hflags = float(int(t[32], 16) & 0xFFFFFF) if len(t) > 32 else 0.0
        except ValueError: hflags = 0.0
        return [
            f(1, 1400), f(2, 3000), f(3, 2.0), f(4), f(5), f(6),
            f(8, 0.7), f(9, 0.85), f(10, 0.5), f(11, 5), f(12, 160), f(13, 20),
            f(14, 10), drive, f(17, 8), f(18, 0.5), f(20, 35), f(21, 1.2),
            f(22, 0.1), f(24, 0.28), f(25, -0.15), f(26, 0.5), f(27, 0.0), hflags,
        ]
    return [1400, 3000, 2.0, 0, 0, 0, 0.7, 0.85, 0.5, 5, 160, 20,
            10, 1, 8, 0.5, 35, 1.2, 0.1, 0.28, -0.15, 0.5, 0.0, 0.0]


# fly[21] fallback = RUSTLER (handling.cfg:382) - a plane with no $-line still flies sanely.
FLY_RUSTLER = [0.5, 0.30, -0.0001, 0.004, 0.10, 0.002, -0.002, 0.0002, 0.0020,
               0.008, 0.10, 0.2, 1.2, 0.2, 1.0, 0.998, 0.998, 0.990, 10.0, 20.0, 0.0]

def load_flying_handling(name):
    """tFlyingHandlingData: the '$' lines of handling.cfg (research plane_port.md par.4).
 Token order after the name: fThrust fThrustFallOff fYaw fYawStab fSideSlip fRoll
 fRollStab fPitch fPitchStab fFormLift fAttackLift fGearUpR fGearDownL fWindMult
 fMoveRes vecTurnRes[3] vecSpeedRes[3] = 21 floats."""
    for line in open(ROOT_PC + "/data/handling.cfg", "r", errors="replace"):
        s = line.strip()
        if not s.startswith("$"):
            continue
        t = s.split()
        if len(t) < 2 or t[1].upper() != name.upper():
            continue
        vals = []
        for x in t[2:2 + 21]:
            try: vals.append(float(x))
            except ValueError: break
        if len(vals) == 21:
            return vals
        print("  ! $%s line has %d floats (want 21) - RUSTLER fallback" % (name, len(vals)))
        return list(FLY_RUSTLER)
    print("  ! no $-line for %s in handling.cfg - RUSTLER fallback" % name)
    return list(FLY_RUSTLER)


# --------------------------------------------------------------- bake ---------
# damageable panels: (base frame name, out name, kind, hingeAxis)
# kind: 0 static, 1 door, 2 bonnet, 3 boot, 4 bumper. hingeAxis: 0 X, 1 Y, 2 Z.
# each bakes <base>_ok as the OK run and <base>_dam (if present) as the DAM run.
PANELS = [
    ("door_lf",    "door_lf", 1, 2),
    ("door_rf",    "door_rf", 1, 2),
    ("door_lr",    "door_lr", 1, 2),
    ("door_rr",    "door_rr", 1, 2),
    ("bonnet",     "bonnet",  2, 0),
    ("boot",       "boot",    3, 0),
    ("bump_front", "bump_f",  4, 0),
    ("bump_rear",  "bump_r",  4, 0),
    ("windscreen", "ws",      0, 0),
    ("exhaust",    "exhaust", 0, 0),
    ("plate_front","plate_f", 0, 0),
]
# static, no _ok/_dam suffix pair:
STATICS = [("chassis", "chassis"), ("plate_rear", "plate_r")]


def bake(dff_name="bravura", txd_name="bravura", handling="BRAVURA",
         carcols_name="bravura", wheel_scale=0.74, out_paths=None):
    img = ImgArchive.open(GTA3_IMG)
    def img_read(nm):
        e = next((e for e in img.entries if e.name.lower() == nm.lower()), None)
        return img.extract(e) if e else None

    dff = load_ps2_dff(img_read(dff_name + ".dff"))
    txd = sa_txd.decode(img_read(txd_name + ".txd"))
    # shared generic vehicle textures (generic/grunge/lights/tyres/scratch/shatter/
    # plates) live in models/generic/vehicle.txd - merge, car txd wins.
    try:
        shared = sa_txd.decode(open(ROOT_PC + "/models/generic/vehicle.txd", "rb").read())
        for k, v in shared.items():
            txd.setdefault(k, v)
    except OSError:
        pass
    textable = TexTable(txd)

    # paint colours from carcols (combo 0)
    cc = parse_carcols(open(ROOT_PC + "/data/carcols.dat", "r", errors="replace").read())
    paint = resolve_colors(cc, carcols_name) or [(180, 180, 180), (180, 180, 180)]
    prim_rgb = paint[0]
    sec_rgb = paint[1] if len(paint) > 1 else paint[0]
    print("carcols %s: primary=%s secondary=%s (%d combos)"
          % (carcols_name, prim_rgb, sec_rgb, len(cc["cars"].get(carcols_name, []))))

    W = frame_world(dff.frames)
    frame_of = {}                      # frame name -> atomic (frame_idx, geo_idx)
    for a in dff.atomics:
        frame_of[dff.frames[a.frame_index].name.lower()] = a
    fidx = {f.name.lower(): i for i, f in enumerate(dff.frames)}

    def wpos(nm):
        i = fidx.get(nm.lower())
        return (W[i][12], W[i][13], W[i][14]) if i is not None else (0.0, 0.0, 0.0)

    def bake_atomic(frame_name):
        a = frame_of.get(frame_name.lower())
        if a is None:
            return []
        return geometry_meshes(dff.geometries[a.geometry_index],
                               W[a.frame_index], paint)

    mounts = [wpos("wheel_lf_dummy"), wpos("wheel_rf_dummy"),
              wpos("wheel_lb_dummy"), wpos("wheel_rb_dummy")]
    seat = wpos("ped_frontseat")

    # ---- components: OK + DAM runs ----
    comps = []   # (name, kind, axis, pivot, (okS,okC,okPrims), (dmS,dmC,dmPrims))
    for base, out_name, kind, axis in PANELS:
        ok = bake_atomic(base + "_ok")
        if not ok:
            continue
        dam = bake_atomic(base + "_dam")
        # hinge pivot = the panel's dummy frame origin (parent of _ok), falling back
        # to the _ok frame itself (exhaust_ok etc. hangs straight off chassis).
        pivot = wpos(base + "_dummy")
        if pivot == (0.0, 0.0, 0.0):
            pivot = wpos(base + "_ok")
        okR = pack_run_merged(ok, textable)
        dmR = pack_run_merged(dam, textable)
        comps.append((out_name, kind, axis, pivot, okR, dmR))
    # ---- STATIC BODY: chassis + plate + EVERY leftover atomic (turret/gun/rig parts the
    # whitelist doesn't know), MERGED by texture into ONE rigid component (kind 0, no hinge).
    # Was one draw PER component -> a detailed car was ~15-20 draws; merging collapses the
    # static part to ~1 prim (draw) per unique texture. The movable panels above
    # (doors/bonnet/boot/bumpers) + the wheels stay separate so they still hinge/spin. ----
    static_meshes = []
    for base, _out in STATICS:
        static_meshes += bake_atomic(base)
    used = set(s for s, _ in STATICS) | {"chassis_vlo", "wheel"}
    for base, _, _, _ in PANELS:
        used.add(base + "_ok"); used.add(base + "_dam")
    for fname in list(frame_of):
        if fname in used or fname.endswith("_dam") or fname.endswith("_vlo"):
            continue
        static_meshes += bake_atomic(fname)
    if static_meshes:
        comps.append(("body", 0, 0, (0.0, 0.0, 0.0),
                      pack_run_merged(static_meshes, textable), (1.0, (0, 0, 0), [])))
        print("  static body: %d source meshes -> %d merged prims (draws)"
              % (len(static_meshes), len(comps[-1][4][2])))

    # ---- chassis_vlo = the far LOD ----
    vloS, vloC, vloPrims = pack_run_merged(bake_atomic("chassis_vlo"), textable)

    # ---- wheel: single atomic, wheel-local (rotation kept, translation stripped) ----
    wa = pick_wheel_node(frame_of)
    wheel_prims = []; wS = 1.0; wC = (0, 0, 0); wrad = 0.0
    if wa is not None:
        wf = W[wa.frame_index]
        wlocal = [wf[0], wf[1], wf[2],  0,
                  wf[4], wf[5], wf[6],  0,
                  wf[8], wf[9], wf[10], 0,
                  0, 0, 0, 1]
        wmeshes = geometry_meshes(dff.geometries[wa.geometry_index], wlocal, paint)
        for me, _ in wmeshes:
            for (x, y, z) in me.positions:
                wrad = max(wrad, math.sqrt(x*x + y*y + z*z))
        wS, wC, wheel_prims = pack_run_merged(wmeshes, textable)

    hand = load_handling(handling)

    # ---- embedded vehicle COL (spheres with damage-piece ids) ----
    col = extract_vehicle_col(bytes(img_read(dff_name + ".dff")))
    spheres = []
    bbmin = (-1.1, -2.5, -0.5); bbmax = (1.1, 2.5, 0.8)
    if col is not None:
        for s in col.spheres:
            d = s if isinstance(s, dict) else s.__dict__
            c = d.get("center"); r = d.get("radius", 0.0)
            piece = d.get("piece", 0)
            if piece in (None, 255):
                piece = 0                       # plain body hit
            spheres.append((c[0], c[1], c[2], r, int(piece)))
        bbmin = tuple(col.bound_min); bbmax = tuple(col.bound_max)

    # ---- assemble CAR4 ----
    buf = bytearray()
    buf += b"CAR4"
    buf += struct.pack("<24f", *hand)
    buf += struct.pack("<3B3B2x", *prim_rgb, *sec_rgb)
    buf += struct.pack("<3f", *seat)
    buf += struct.pack("<2f", wheel_scale, wrad)
    for m in mounts:
        buf += struct.pack("<3f", *m)
    # COL section: bound box + spheres {center,radius,piece}
    buf += struct.pack("<3f3f", *bbmin, *bbmax)
    buf += struct.pack("<I", len(spheres))
    for (cx, cy, cz, r, piece) in spheres:
        buf += struct.pack("<4fBxxx", cx, cy, cz, r, piece)
    buf += struct.pack("<I", len(textable.order))
    buf += textable.blob()
    buf += struct.pack("<I", len(comps))
    for (name, kind, axis, pivot, (okS, okC, okP), (dmS, dmC, dmP)) in comps:
        nm = name.encode("latin1")[:15]; nm += b"\x00" * (16 - len(nm))
        buf += nm
        buf += struct.pack("<BBBB", kind, axis, 1 if dmP else 0, 0)
        buf += struct.pack("<3f", *pivot)
        buf += struct.pack("<f3fI", okS, *okC, len(okP))
        buf += struct.pack("<f3fI", dmS, *dmC, len(dmP))
    buf += struct.pack("<f3fI", vloS, *vloC, len(vloPrims))
    buf += struct.pack("<f3fI", wS, *wC, len(wheel_prims))
    for (_, _, _, _, (_, _, okP), (_, _, dmP)) in comps:
        for b in okP: buf += b
        for b in dmP: buf += b
    for b in vloPrims: buf += b
    for b in wheel_prims: buf += b

    # write targets: default = the single bravura car.bin (back-compat); else the
    # per-model veh_<name>.bin list passed by the roster driver.
    targets = out_paths if out_paths else ([OUT] + (deploy_util.targets(DEPLOY) if DEPLOY else []))
    dep = ""
    for p in targets:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(buf); dep += ("+" if dep else "") + os.path.basename(os.path.dirname(p))
        except OSError:
            pass
    dep = dep or "(no write)"

    nok = sum(len(c[4][2]) for c in comps)
    ndm = sum(len(c[5][2]) for c in comps)
    print("=== car.bin CAR4: %s  comps=%d (ok=%d dam=%d) vlo=%d wheel=%d tex=%d "
          "colSph=%d wheelR=%.2f  %d KB  %s ===" % (dff_name, len(comps), nok, ndm,
          len(vloPrims), len(wheel_prims), len(textable.order), len(spheres), wrad,
          len(buf) // 1024, dep))
    for (name, kind, axis, pivot, (_, _, okP), (_, _, dmP)) in comps:
        print("  comp %-9s kind=%d axis=%d pivot=(%+.2f,%+.2f,%+.2f) ok=%d dam=%d"
              % (name, kind, axis, pivot[0], pivot[1], pivot[2], len(okP), len(dmP)))
    for key, t in textable.order:
        print("  tex %-24s %dx%d nl=%d amode=%d" % (key, t["width"], t["height"],
              t["num_levels"], t.get("alpha_mode", 0)))


# ============================ BIKE / motorcycle bake =========================
# Bikes have a different DFF structure than cars: 2 separate wheel geometries
# (wheel_front / wheel_rear), a steering fork (forks_front + handlebars) and a
# rider sitting ON TOP (ped_frontseat, visible). v1 bakes the whole bike as ONE
# rigid body group + the chassis_vlo LOD; the runtime drives it on the dynamic-
# bicycle physics and LEANS it into turns. Steer/wheel articulation is a later pass.
#
# bike.bin layout (LE), after a 'BIKE' magic:
# f32 handling[24]
# u8 colPrim[3], colSec[3], pad[2]
# f32 riderSeat[3] (ped_frontseat, bike space)
# f32 steerPivot[3] (forks_front origin)
# f32 wheelFront[3], wheelRear[3] (wheel centres, bike space)
# f32 wheelRadius
# u32 nTex; nTex texture blocks (same block format as car.bin)
# body header: f32 scale, center[3]; u32 nPrims
# vlo header: f32 scale, center[3]; u32 nPrims
# COL: f32 boundMin[3], boundMax[3]; u32 nSph; nSph {f32 c[3],r; u8 piece,pad[3]}
# prim blocks: body run, then vlo run (same prim block format as car.bin)
def bake_bike(dff_name="pcj600", txd_name="pcj600", handling="BIKE",
              carcols_name="pcj600", wheel_scale=0.67, out_paths=None):
    img = ImgArchive.open(GTA3_IMG)
    def img_read(nm):
        e = next((e for e in img.entries if e.name.lower() == nm.lower()), None)
        return img.extract(e) if e else None

    dff = load_ps2_dff(img_read(dff_name + ".dff"))
    txd = sa_txd.decode(img_read(txd_name + ".txd"))
    try:
        shared = sa_txd.decode(open(ROOT_PC + "/models/generic/vehicle.txd", "rb").read())
        for k, v in shared.items():
            txd.setdefault(k, v)
    except OSError:
        pass
    textable = TexTable(txd)

    cc = parse_carcols(open(ROOT_PC + "/data/carcols.dat", "r", errors="replace").read())
    paint = resolve_colors(cc, carcols_name) or [(80, 80, 80), (80, 80, 80)]
    prim_rgb = paint[0]; sec_rgb = paint[1] if len(paint) > 1 else paint[0]

    W = frame_world(dff.frames)
    frame_of = {}
    for a in dff.atomics:
        frame_of[dff.frames[a.frame_index].name.lower()] = a
    fidx = {f.name.lower(): i for i, f in enumerate(dff.frames)}
    def wpos(nm):
        i = fidx.get(nm.lower())
        return (W[i][12], W[i][13], W[i][14]) if i is not None else (0.0, 0.0, 0.0)
    def bake_atomic(frame_name):
        a = frame_of.get(frame_name.lower())
        if a is None:
            return []
        return geometry_meshes(dff.geometries[a.geometry_index], W[a.frame_index], paint)

    # BODY = every atomic except the LOD shell (rigid v1: forks/wheels/handlebars all in).
    body_meshes = []
    for nm, a in frame_of.items():
        if nm == "chassis_vlo":
            continue
        body_meshes += geometry_meshes(dff.geometries[a.geometry_index], W[a.frame_index], paint)
    bodyS, bodyC, bodyPrims = pack_run(body_meshes, textable)
    vloS, vloC, vloPrims = pack_run_merged(bake_atomic("chassis_vlo"), textable)

    seat  = wpos("ped_frontseat")
    pivot = wpos("forks_front")
    wf = wpos("wheel_front"); wr = wpos("wheel_rear")
    # wheel radius from the front wheel geometry (verts relative to its frame origin).
    wrad = 0.3
    wa = frame_of.get("wheel_front")
    if wa is not None:
        wf_m = W[wa.frame_index]
        loc = [wf_m[0], wf_m[1], wf_m[2], 0, wf_m[4], wf_m[5], wf_m[6], 0,
               wf_m[8], wf_m[9], wf_m[10], 0, 0, 0, 0, 1]
        r = 0.0
        for me, _ in geometry_meshes(dff.geometries[wa.geometry_index], loc, paint):
            for (x, y, z) in me.positions:
                r = max(r, math.sqrt(x*x + y*y + z*z))
        if r > 0.05:
            wrad = r

    hand = load_handling(handling)
    col = extract_vehicle_col(bytes(img_read(dff_name + ".dff")))
    spheres = []
    bbmin = (-0.5, -1.1, -0.4); bbmax = (0.5, 1.1, 0.9)
    if col is not None:
        for s in col.spheres:
            d = s if isinstance(s, dict) else s.__dict__
            c = d.get("center"); rr = d.get("radius", 0.0); piece = d.get("piece", 0)
            if piece in (None, 255): piece = 0
            spheres.append((c[0], c[1], c[2], rr, int(piece)))
        bbmin = tuple(col.bound_min); bbmax = tuple(col.bound_max)

    buf = bytearray(b"BIKE")
    buf += struct.pack("<24f", *hand)
    buf += struct.pack("<3B3B2x", *prim_rgb, *sec_rgb)
    buf += struct.pack("<3f", *seat)
    buf += struct.pack("<3f", *pivot)
    buf += struct.pack("<3f", *wf)
    buf += struct.pack("<3f", *wr)
    buf += struct.pack("<f", wrad)
    buf += struct.pack("<I", len(textable.order))
    buf += textable.blob()
    buf += struct.pack("<f3fI", bodyS, *bodyC, len(bodyPrims))
    buf += struct.pack("<f3fI", vloS, *vloC, len(vloPrims))
    buf += struct.pack("<3f3f", *bbmin, *bbmax)
    buf += struct.pack("<I", len(spheres))
    for (cx, cy, cz, r, piece) in spheres:
        buf += struct.pack("<4fBxxx", cx, cy, cz, r, piece)
    for b in bodyPrims: buf += b
    for b in vloPrims: buf += b

    targets = out_paths if out_paths else ([OUT.replace("car.bin", "bike.bin")] +
              ([DEPLOY.replace("car.bin", "bike.bin")] if DEPLOY else []))
    dep = ""
    for p in targets:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(buf); dep += ("+" if dep else "") + os.path.basename(os.path.dirname(p))
        except OSError:
            pass
    print("=== bike.bin BIKE: %s  body=%d vlo=%d tex=%d colSph=%d wheelR=%.2f base=%.2f  %d KB  %s ==="
          % (dff_name, len(bodyPrims), len(vloPrims), len(textable.order), len(spheres),
             wrad, wf[1]-wr[1], len(buf)//1024, dep or "(no write)"))


# ============================ PLANE bake (PLN1, build 434) ===================
# Frame naming across the 11 baked planes (verified against the real DFFs):
# doors: door_lf_ok (+door_rf_ok on dodo/at400/androm/skimmer/beagle)
# surfaces: rudder; elevator_l+r (or elevator_r ONLY: rustler/cropdust/beagle/
# skimmer; or a single 'elevator': dodo); aileron_l+r
# gear doors: gear_l / gear_r (rustler/shamal/nevada/hydra/at400/androm)
# props: moving_prop (+moving_prop2: beagle/nevada; skimmer has ONLY
# moving_prop2); static_prop* = the sharp-blade twin -> NOT baked
# wheel mesh: wheel | wheel1 (beagle) | wheel_l/_r (stunt; _l baked, _r merged
# away would mirror - we just use _l for all mounts) | none (skimmer)
# LOD: chassis_vlo | chassis_vlo2 (stunt)
# leftovers: misc_a/b (Hydra nozzles etc.), wheel_lm/rm_dummy atomics (hydra
# mid gear) -> merged into the static body like the car bake.
PLANE_PANELS = [
    # (frame, out name, kind, axis) kind 5 prop / 6 surface / 7 gear / 8 nozzle;
    # axis 0 X, 1 Y, 2 Z
    ("rudder",       "rudder", 6, 2),
    ("elevator_l",   "elev_l", 6, 0),
    ("elevator_r",   "elev_r", 6, 0),
    ("elevator",     "elev",   6, 0),
    ("aileron_l",    "ail_l",  6, 0),
    ("aileron_r",    "ail_r",  6, 0),
    # b834: gear_l/gear_r are plane nodes 21/22. The rotation AXIS is per-MODEL in the
    # original (CPlane::PreRender 0x6FED50: one axis for nodes 21/22, another for 23/24),
    # not per-frame, so the axis written here is a placeholder - the engine overrides it
    # from its own PLANE_GEAR table. Angles: Rustler -85/+85 about Y; Shamal and AT-400
    # the same plus 130 about X on misc_a; Hydra -90/-90/-80/+130 about X; Nevada +75/+75
    # about X; Andromada +130/+130/-130 about X.
    ("gear_l",       "gear_l", 7, 0),
    ("gear_r",       "gear_r", 7, 0),
    ("moving_prop",  "prop",   5, 1),
    ("moving_prop2", "prop2",  5, 1),
    # misc_a / misc_b = plane nodes 23/24. b440 guessed these were the Hydra's VTOL
    # nozzles and named the outputs accordingly; b834 read PreRender (0x6FED50) and they
    # are GEAR parts on all four jets - Shamal/AT-400 rotate misc_a 130 deg about X with
    # the gear, and the Hydra rotates misc_a -80 and misc_b +130. The names are kept
    # because the engine dispatches on them and changing them would skew every baked
    # veh_*.bin already on a memstick.
    ("misc_a",       "nozzle_a", 8, 0),
    ("misc_b",       "nozzle_b", 8, 0),
    # b834: the REAL Hydra nozzle frames. PreRender rotates nodes 3 and 6 - wheel_rm_dummy
    # and wheel_lm_dummy, which the Hydra reuses as thrust-vector pivots - about X by
    # m_wMiscComponentAngle * (pi/2) / NOZZLE_ROTATE_LIMIT. b440 had the angle law right and
    # the frames wrong; until this bake runs, the Hydra's nozzles do not rotate. Other planes
    # have no mid-wheel atomics, so this is hydra-only in practice (the car bake merged them
    # into the static body before now). The pivots also anchor the jet-exhaust fx.
    ("wheel_rm_dummy", "nozzle_r", 8, 0),
    ("wheel_lm_dummy", "nozzle_l", 8, 0),
]
PLANE_DOORS = [("door_lf", "door_lf", 1, 2), ("door_rf", "door_rf", 1, 2)]
UV_LIMIT = 8.0            # |s16 UV * 4096| ceiling on the GE (psp-developer trap #1)


def _uv_extent(meshes):
    m = 0.0
    for me, _ in meshes:
        for (u, v) in me.uv:
            au = -u if u < 0 else u
            av = -v if v < 0 else v
            if au > m: m = au
            if av > m: m = av
    return m


def select_col_spheres(spheres, cap=24):
    """Greedy farthest-point SELECTION (build_wq_spheres style, Vehicle.c:899) down to
 `cap` REAL spheres + FORCED X/Y extremes (wingtips + nose/tail): a subset of the
 true hull never collides where the plane doesn't, and the extremes keep the wing
 span + the tail lever honest for Block B."""
    n = len(spheres)
    if n <= cap:
        return list(spheres)
    picked = []
    def add(i):
        if i not in picked:
            picked.append(i)
    add(max(range(n), key=lambda i: spheres[i][0]))          # +X wingtip
    add(min(range(n), key=lambda i: spheres[i][0]))          # -X wingtip
    add(max(range(n), key=lambda i: spheres[i][1]))          # nose
    add(min(range(n), key=lambda i: spheres[i][1]))          # tail
    add(max(range(n), key=lambda i: spheres[i][3]))          # seed: largest panel
    while len(picked) < cap:
        best, bs = -1, -1e30
        for i in range(n):
            if i in picked:
                continue
            mind = 1e30
            for k in picked:
                dx = spheres[i][0]-spheres[k][0]
                dy = spheres[i][1]-spheres[k][1]
                dz = spheres[i][2]-spheres[k][2]
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d < mind: mind = d
            score = mind + spheres[i][3]                     # radius bonus (big panels win)
            if score > bs: bs, best = score, i
        if best < 0:
            break
        picked.append(best)
    keep = set(picked)
    return [spheres[i] for i in range(n) if i in keep]       # stable source order


def bake_plane(dff_name, txd_name, handling, carcols_name, wheel_scale, out_paths):
    """Bake one plane into PLN1 (layout in the module docstring). Returns a report dict."""
    img = ImgArchive.open(GTA3_IMG)
    def img_read(nm):
        e = next((e for e in img.entries if e.name.lower() == nm.lower()), None)
        return img.extract(e) if e else None

    dff = load_ps2_dff(img_read(dff_name + ".dff"))
    txd = sa_txd.decode(img_read(txd_name + ".txd"))
    try:
        shared = sa_txd.decode(open(ROOT_PC + "/models/generic/vehicle.txd", "rb").read())
        for k, v in shared.items():
            txd.setdefault(k, v)
    except OSError:
        pass
    textable = TexTable(txd)

    cc = parse_carcols(open(ROOT_PC + "/data/carcols.dat", "r", errors="replace").read())
    paint = resolve_colors(cc, carcols_name) or [(200, 200, 200), (200, 200, 200)]
    prim_rgb = paint[0]; sec_rgb = paint[1] if len(paint) > 1 else paint[0]

    W = frame_world(dff.frames)
    frame_of = {}
    for a in dff.atomics:
        frame_of[dff.frames[a.frame_index].name.lower()] = a
    fidx = {f.name.lower(): i for i, f in enumerate(dff.frames)}
    def wpos(nm):
        i = fidx.get(nm.lower())
        return (W[i][12], W[i][13], W[i][14]) if i is not None else (0.0, 0.0, 0.0)
    def bake_atomic(frame_name):
        a = frame_of.get(frame_name.lower())
        if a is None:
            return []
        return geometry_meshes(dff.geometries[a.geometry_index], W[a.frame_index], paint)

    # wheel mounts: the RAW DFF dummies. Coincident tandem pairs (nose or tail strut:
    # lf==rf or lb==rb at x=0) are kept - the runtime spreads them +-0.3 like the bike.
    mounts = [wpos("wheel_lf_dummy"), wpos("wheel_rf_dummy"),
              wpos("wheel_lb_dummy"), wpos("wheel_rb_dummy")]
    seat = wpos("ped_frontseat")

    # b837: WHICH GEAR FRAME EACH WHEEL HANGS FROM.
    #
    # ★★ In the DFF the retracting wheels are CHILDREN of the gear frames, so when
    # CPlane::PreRender rotates gear_l/gear_r/misc_a/misc_b the wheels swing with them --
    # that is the whole visible gear animation. m_anWheelStatus going MISSING is physics
    # only; a scan of the retail binary finds ZERO reads of it anywhere in PreRender.
    # The port flattened the hierarchy at bake time and drew wheels at fixed mounts,
    #
    # The parenting read off the disc, and it corroborates the angle table exactly --
    # misc_a exists on precisely the models whose NOSE pair hangs from it:
    #
    # rustler lf,rf -> gear_l,gear_r lb,rb -> root (fixed tailwheel)
    # nevada lf,rf -> gear_l,gear_r lb,rb -> chassis_dummy (fixed)
    # shamal lf,rf -> misc_a (both) lb,rb -> gear_l,gear_r
    # at400 lf,rf -> misc_a (both) lb,rb -> gear_l,gear_r
    # androm lf,rf -> misc_a (both) lb,rb -> gear_l,gear_r
    # hydra lf -> misc_b, rf -> misc_a lb,rb -> gear_l,gear_r
    #
    # 0..3 index the runtime's gear slots (gear_l, gear_r, misc_a, misc_b); 0xFF = the
    # wheel does not move with the gear. Hydra's wheel_lm/rm_dummy sit at the ROOT, which
    # is the other half of the b834 finding that they are the VTOL nozzles, not gear.
    GEAR_SLOT = {"gear_l": 0, "gear_r": 1, "misc_a": 2, "misc_b": 3}
    def gear_parent(wheel_name):
        i = fidx.get(wheel_name)
        seen = 0
        while i is not None and i >= 0 and seen < 32:
            nm = dff.frames[i].name.lower()
            if nm in GEAR_SLOT:
                return GEAR_SLOT[nm]
            p = dff.frames[i].parent
            i = p if (p is not None and p >= 0 and p != i) else None
            seen += 1
        return 0xFF
    wheel_parent = [gear_parent(n) for n in
                    ("wheel_lf_dummy", "wheel_rf_dummy", "wheel_lb_dummy", "wheel_rb_dummy")]

    total_verts = 0
    def count(meshes):
        nonlocal total_verts
        total_verts += sum(len(me.positions) for me, _ in meshes)
        return meshes

    # ---- components ----
    comps = []      # (name, kind, axis, pivot, okRun, dmRun)
    uv_max = 0.0
    def track_uv(meshes):
        nonlocal uv_max
        e = _uv_extent(meshes)
        if e > uv_max: uv_max = e
        return meshes

    for base, out_name, kind, axis in PLANE_DOORS:
        ok = bake_atomic(base + "_ok")
        if not ok:
            continue
        dam = bake_atomic(base + "_dam")
        pivot = wpos(base + "_dummy")
        if pivot == (0.0, 0.0, 0.0):
            pivot = wpos(base + "_ok")
        comps.append((out_name, kind, axis, pivot,
                      pack_run_merged(track_uv(count(ok)), textable),
                      pack_run_merged(track_uv(count(dam)), textable)))
    prop_anchors = []                      # (x, y, z) in DFF order prop -> prop2
    for frame, out_name, kind, axis in PLANE_PANELS:
        meshes = bake_atomic(frame)
        if not meshes:
            continue
        pivot = wpos(frame)                # the part's own frame origin = the hinge/shaft
        if kind == 5:
            prop_anchors.append(pivot)
        comps.append((out_name, kind, axis, pivot,
                      pack_run_merged(track_uv(count(meshes)), textable),
                      (1.0, (0.0, 0.0, 0.0), [])))

    # ---- static body: chassis + EVERY leftover atomic (misc_*, hydra mid-gear...) ----
    used = {"chassis", "chassis_vlo", "chassis_vlo2",
            "static_prop", "static_prop2", "static_prop3"}
    used.update(WHEEL_ALIASES); used.add("wheel_r")
    for frame, _o, _k, _a in PLANE_PANELS:
        used.add(frame)
    for base, _o, _k, _a in PLANE_DOORS:
        used.add(base + "_ok"); used.add(base + "_dam")
    static_meshes = bake_atomic("chassis")
    for fname in list(frame_of):
        if fname in used or fname.endswith("_dam") or fname.endswith("_vlo"):
            continue
        static_meshes += bake_atomic(fname)
    if static_meshes:
        comps.append(("body", 0, 0, (0.0, 0.0, 0.0),
                      pack_run_merged(track_uv(count(static_meshes)), textable),
                      (1.0, (0.0, 0.0, 0.0), [])))

    # ---- LOD + wheel ----
    vlo_meshes = bake_atomic("chassis_vlo") or bake_atomic("chassis_vlo2")
    vloS, vloC, vloPrims = pack_run_merged(track_uv(count(vlo_meshes)), textable)
    wheel_prims = []; wS = 1.0; wC = (0.0, 0.0, 0.0); wrad = 0.0
    wa = pick_wheel_node(frame_of)
    if wa is not None:
        wf = W[wa.frame_index]
        wlocal = [wf[0], wf[1], wf[2],  0, wf[4], wf[5], wf[6],  0,
                  wf[8], wf[9], wf[10], 0, 0, 0, 0, 1]
        wmeshes = geometry_meshes(dff.geometries[wa.geometry_index], wlocal, paint)
        for me, _ in wmeshes:
            for (x, y, z) in me.positions:
                wrad = max(wrad, math.sqrt(x*x + y*y + z*z))
        wS, wC, wheel_prims = pack_run_merged(track_uv(count(wmeshes)), textable)

    if uv_max > UV_LIMIT:
        raise RuntimeError("%s: UV extent %.2f exceeds the s16 GE limit %.1f - retile the "
                           "source texture mapping" % (dff_name, uv_max, UV_LIMIT))

    hand = load_handling(handling)
    fly = load_flying_handling(handling)

    # ---- embedded COL: honest bbox (tail lever for Block B!) + sphere SELECTION ----
    col = extract_vehicle_col(bytes(img_read(dff_name + ".dff")))
    spheres = []
    bbmin = (-2.0, -4.0, -1.0); bbmax = (2.0, 4.0, 1.0)
    if col is not None:
        for s in col.spheres:
            d = s if isinstance(s, dict) else s.__dict__
            c = d.get("center"); r = d.get("radius", 0.0)
            piece = d.get("piece", 0)
            if piece in (None, 255):
                piece = 0
            spheres.append((c[0], c[1], c[2], r, int(piece)))
        bbmin = tuple(col.bound_min); bbmax = tuple(col.bound_max)
    sph_src = len(spheres)
    spheres = select_col_spheres(spheres, 24)

    # ---- assemble PLN1 ----
    # PLN2 (b837): PLN1 plus u8 wheelParent[4] after the wheel mounts. The magic moves so
    # an engine expecting one layout can never silently read the other.
    buf = bytearray(b"PLN2")
    buf += struct.pack("<24f", *hand)
    buf += struct.pack("<21f", *fly)
    buf += struct.pack("<3B3B2x", *prim_rgb, *sec_rgb)
    buf += struct.pack("<3f", *seat)
    buf += struct.pack("<2f", wheel_scale, wrad)
    for m in mounts:
        buf += struct.pack("<3f", *m)
    buf += struct.pack("<4B", *wheel_parent)      # b837 (PLN2): gear slot per wheel, 0xFF = none
    n_prop = min(len(prop_anchors), 2)
    buf += struct.pack("<B3x", n_prop)
    for k in range(2):
        a = prop_anchors[k] if k < n_prop else (0.0, 0.0, 0.0)
        buf += struct.pack("<4f", a[0], a[1], a[2], 1.0)     # w = spin axis code: 1.0 = Y
    buf += struct.pack("<3f3f", *bbmin, *bbmax)
    buf += struct.pack("<I", len(spheres))
    for (cx, cy, cz, r, piece) in spheres:
        buf += struct.pack("<4fBxxx", cx, cy, cz, r, piece)
    buf += struct.pack("<I", len(textable.order))
    buf += textable.blob()
    buf += struct.pack("<I", len(comps))
    for (name, kind, axis, pivot, (okS, okC, okP), (dmS, dmC, dmP)) in comps:
        nm = name.encode("latin1")[:15]; nm += b"\x00" * (16 - len(nm))
        buf += nm
        buf += struct.pack("<BBBB", kind, axis, 1 if dmP else 0, 0)
        buf += struct.pack("<3f", *pivot)
        buf += struct.pack("<f3fI", okS, *okC, len(okP))
        buf += struct.pack("<f3fI", dmS, *dmC, len(dmP))
    buf += struct.pack("<f3fI", vloS, *vloC, len(vloPrims))
    buf += struct.pack("<f3fI", wS, *wC, len(wheel_prims))
    for (_, _, _, _, (_, _, okP), (_, _, dmP)) in comps:
        for b in okP: buf += b
        for b in dmP: buf += b
    for b in vloPrims: buf += b
    for b in wheel_prims: buf += b

    dep = ""
    for p in out_paths:
        # b853: REFUSE a path with no directory. Release sanitisation left OUT/DEPLOY/
        # VEH_DIR_* as empty strings, so a run without --out built "veh/veh_hydra.bin" with
        # an empty root - which on Windows resolves against the CURRENT DRIVE and quietly
        # dropped eleven plane files into E:\. The bake still printed success, so nothing
        d = os.path.dirname(os.path.abspath(p))
        if not os.path.dirname(p) or os.path.splitdrive(d)[1] in ("\\", "/"):
            print("  !! refusing to write %r - no output directory (pass --out)" % p)
            continue
        try:
            os.makedirs(d, exist_ok=True)
            open(p, "wb").write(buf); dep += ("+" if dep else "") + os.path.basename(os.path.dirname(p))
        except OSError:
            pass

    print("=== PLN1: %-9s comps=%d prims(ok)=%d vlo=%d wheel=%d tex=%d "
          "colSph=%d->%d bb=(%.1f,%.1f,%.1f) verts=%d uvMax=%.2f nProp=%d  %d KB  %s ==="
          % (dff_name, len(comps), sum(len(c[4][2]) for c in comps), len(vloPrims),
             len(wheel_prims), len(textable.order), sph_src, len(spheres),
             bbmax[0], bbmax[1], bbmax[2], total_verts, uv_max, n_prop,
             len(buf)//1024, dep or "(no write)"))
    for (name, kind, axis, pivot, (_, _, okP), (_, _, dmP)) in comps:
        print("  comp %-9s kind=%d axis=%d pivot=(%+.2f,%+.2f,%+.2f) ok=%d dam=%d"
              % (name, kind, axis, pivot[0], pivot[1], pivot[2], len(okP), len(dmP)))
    return {"name": dff_name, "verts": total_verts, "comps": len(comps),
            "sph_src": sph_src, "sph": len(spheres), "kb": len(buf)//1024,
            "uv": uv_max, "nprop": n_prop, "wheel": len(wheel_prims)}


# ---- roster driver (Block B): bake many cars from vehicles.ide ----------------
VEH_DIR_LOCAL  = ""
VEH_DIR_DEPLOY = ""
IDX_LOCAL      = ""
IDX_DEPLOY     = ""

# phase-1 start set (research vehicle_roster.md): simplest civilian cars first.
SET12 = [401, 402, 411, 420, 429, 440, 445, 470, 492, 560, 596, 600]

# car-family types DRIVE on the car pipeline (4 wheels, standard handling columns):
# plain cars + monster trucks + quads. bike-family (2 wheels, forks, rider on top)
# use the bike bake/physics. planes bake to PLN1 (--plane); boats/helis later.
CAR_FAMILY   = ("car", "mtruck", "quad")
BIKE_FAMILY  = ("bike", "bmx")
PLANE_FAMILY = ("plane",)

def parse_vehicles_ide():
    """id -> {id,model,txd,handling,carcols,wheel,vtype} for drivable vehicles.
 Tokenised by commas AND whitespace ([,\\s]+): the stock dodo line is broken
 (TAB instead of the model,txd comma) and shifted every field one left under
 the plain comma split - dodo used to vanish into a phantom 'dodo' type.
 Planes: full 15-field rows, except skimmer = 11 fields (no wheel columns,
 no wheels on floats) -> wheelScale defaults 0.7."""
    path = ROOT_PC + "/data/vehicles.ide"
    out = {}
    for line in open(path, "r", errors="replace"):
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith(("end", "cars", "boats")):
            continue
        t = [x for x in re.split(r"[,\s]+", s) if x]
        if len(t) < 11:
            continue
        ty = t[3].lower()
        if ty in CAR_FAMILY:     vtype = "car"
        elif ty in BIKE_FAMILY:  vtype = "bike"
        elif ty in PLANE_FAMILY: vtype = "plane"
        else:                    continue
        if vtype != "plane" and len(t) < 13:
            continue
        try:
            mid = int(t[0])
        except ValueError:
            continue
        try:
            wheel = float(t[12])
        except (IndexError, ValueError):
            wheel = 0.7                       # skimmer: 11-field row, no wheel columns
        out[mid] = {"id": mid, "model": t[1], "txd": t[2], "handling": t[4],
                    "carcols": t[1], "wheel": wheel, "vtype": vtype}
    return out

def write_index(specs, deploy=True):
    """veh_index.bin: 'VIDX' u32 n, n x { u16 id, char name[16] }."""
    buf = bytearray(b"VIDX")
    buf += struct.pack("<I", len(specs))
    for sp in specs:
        nm = sp["model"].encode("latin1")[:15]; nm += b"\x00" * (16 - len(nm))
        buf += struct.pack("<H", sp["id"]) + nm
    paths = [IDX_LOCAL] + ([IDX_DEPLOY] if (deploy and IDX_DEPLOY) else [])
    for p in paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(buf)
        except OSError:
            pass
    print("=== veh_index.bin: %d models ===" % len(specs))

def _bake_fn(vtype):
    return bake_bike if vtype == "bike" else bake_plane if vtype == "plane" else bake

def bake_roster(ids):
    ide = parse_vehicles_ide()
    specs = []
    for mid in ids:
        sp = ide.get(mid)
        if not sp:
            print("  ! id %d not a car in vehicles.ide - skipped" % mid); continue
        name = sp["model"]
        outs = [VEH_DIR_LOCAL + "/veh_%s.bin" % name]
        if VEH_DIR_DEPLOY:
            outs.append(VEH_DIR_DEPLOY + "/veh_%s.bin" % name)
        print("--- bake %d %s [%s] (handling=%s wheel=%.3f) ---"
              % (sp["id"], name, sp["vtype"], sp["handling"], sp["wheel"]))
        try:
            _bake_fn(sp["vtype"])(name, sp["txd"], sp["handling"], sp["carcols"],
                                  sp["wheel"], out_paths=outs)
            specs.append(sp)                     # index only the ones that actually baked
        except Exception as e:
            print("  !! bake %s FAILED: %s" % (name, e))
    write_index(specs)

# --plane (Block A, build 434): the 11 flyable fixed-wings. vortex 539 (hovercraft)
# and rcbaron 464 (RC toy: no enter, mass 100) are deliberately OUT of v1.
PLANE_IDS = [476, 593, 513, 512, 511, 519, 553, 520, 577, 592, 460]

def bake_planes():
    """Bake all Block-A planes to assets_build ONLY (no memstick deploy - hand
 placement), then rebuild veh_index.bin (LOCAL only) from every veh_<name>.bin
 actually present on disk: existing car/bike roster + the fresh planes, and a
 failed bake can never shrink the index (the old single-entry-index accident)."""
    ide = parse_vehicles_ide()
    reports = []
    for mid in PLANE_IDS:
        sp = ide.get(mid)
        if not sp or sp["vtype"] != "plane":
            print("  ! id %d is not a plane in vehicles.ide - skipped" % mid); continue
        name = sp["model"]
        print("--- bake %d %s [plane] (handling=%s wheel=%.3f)%s ---"
              % (mid, name, sp["handling"], sp["wheel"],
                 "  (floats, no wheels)" if mid == 460 else ""))
        try:
            reports.append(bake_plane(name, sp["txd"], sp["handling"], sp["carcols"],
                                      sp["wheel"], out_paths=[VEH_DIR_LOCAL + "/veh_%s.bin" % name]))
        except Exception as e:
            print("  !! bake %s FAILED: %s" % (name, e))
    specs = [sp for mid, sp in sorted(ide.items())
             if os.path.exists(VEH_DIR_LOCAL + "/veh_%s.bin" % sp["model"])]
    write_index(specs, deploy=False)
    if reports:
        print("\n%-9s %6s %5s %9s %6s %5s %5s %5s" %
              ("model", "verts", "comps", "sph", "KB", "uvMax", "nProp", "whl"))
        for r in reports:
            print("%-9s %6d %5d %4d->%-4d %6d %5.2f %5d %5d" %
                  (r["name"], r["verts"], r["comps"], r["sph_src"], r["sph"],
                   r["kb"], r["uv"], r["nprop"], r["wheel"]))

if __name__ == "__main__":
    a = sys.argv[1:]
    # optional `--out <dir>` (Quarry passes <OutDir>/vehicles): redirect every writer into
    # <dir> - car.bin, veh/veh_*.bin, veh_index.bin - and drop the memstick deploy mirror.
    if "--out" in a:
        i = a.index("--out")
        outdir = a[i + 1]
        del a[i:i + 2]
        OUT            = os.path.join(outdir, "car.bin")
        DEPLOY         = None
        VEH_DIR_LOCAL  = os.path.join(outdir, "veh")
        VEH_DIR_DEPLOY = None
        IDX_LOCAL      = os.path.join(outdir, "veh_index.bin")
        IDX_DEPLOY     = None

    if not a:
        bake()                                   # default: bravura -> car.bin (back-compat)
    elif a[0] == "--set12":
        bake_roster(SET12)
    elif a[0] == "--all":
        bake_roster(sorted(parse_vehicles_ide().keys()))
    elif a[0] == "--plane":
        bake_planes()
    else:                                         # bake one model by name
        ide = {v["model"]: v for v in parse_vehicles_ide().values()}
        sp = ide.get(a[0])
        if not sp:
            print("unknown model %s" % a[0]); sys.exit(1)
        # SINGLE-model bake: write the veh_<name>.bin ONLY. bake_roster would
        # rewrite veh_index.bin with just this one entry and break the whole
        # roster (happened once: index shrank to 1 model, restored from F:).
        name = sp["model"]
        outs = [VEH_DIR_LOCAL + "/veh_%s.bin" % name]
        if VEH_DIR_DEPLOY:
            outs.append(VEH_DIR_DEPLOY + "/veh_%s.bin" % name)
        _bake_fn(sp["vtype"])(name, sp["txd"], sp["handling"], sp["carcols"],
                              sp["wheel"], out_paths=outs)
