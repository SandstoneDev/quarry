#!/usr/bin/env python3
"""hero_bake - bake a self-contained SKINNED ped (skeleton + mesh + skin weights +
textures + locomotion clips) into hero.bin for the PSP skeletal-animation runtime.

Self-contained so mesh vertices and skin weights share ONE order (the DFF geometry
vertex order, 0..nvert-1) by construction - the global vertex array IS the geometry
and submeshes reference it by global index, so no mesh-local remap can desync the
skin from the mesh (the roadmap's "highest-risk integration point").

Sources (gvcslib READ-ONLY): geometry parse mirrors gvcslib.sa_dff_pc but keeps
global indices; skin + skeleton from tools/sa_skin; clips from tools/sa_ifp;
textures via gvcslib.sa_txd_d3d9 + psp_tex (same T8 path as char_bake).

hero.bin layout (little-endian):
  'HRO1' | u16 numBones | u16 numClips | u16 numVerts | u16 numSub | u16 numTex | u16 pad
  bones[numBones]:  s16 parent | s16 nodeId | f32 bindQuat[4] | f32 bindPos[3] | f32 invBind[16]
  verts[numVerts]:  f32 pos[3] | f32 uv[2] | u32 rgba8888 | u8 boneIdx[4] | f32 boneW[4]
  submeshes[numSub]: s16 tex | u16 pad | u32 idxFirst | u32 idxCount   (indices are GLOBAL vert idx)
  indices[Σ idxCount]: u16
  textures[numTex]:  u16 tw,th | u16 numLevels|alpha<<8 | u16 clutEntries | u32 texelLen | u32 clutLen | <texel><clut>
  clips[numClips]:  char name[24] | f32 dur | u16 numTracks | u16 pad
     track:  s16 bone | u8 hasTrans | u8 pad | u16 numKeys
             key: s16 q[4] | s16 time | (s16 t[3] if hasTrans)
  (dequant: quat /4096, trans /1024, time /60s)
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))

import sa_skin
import ps2skin
import sa_ifp
import deploy_util
from gvcslib import sa_img, sa_txd, sa_txd_d3d9, psp_tex
from gvcslib.sa_dff import (parse_chunks, GEOMETRYLIST, GEOMETRY, STRUCT,
                            EXTENSION, _get_material_names)

# SA_ROOT env override: Quarry points this at the user's extracted PS2 disc; the
# PLAYER.IMG skinned models are stored PLATFORM-NEUTRAL on PS2 (standard RW skin +
# HAnim, byte-identical to PC - only the TXDs differ, PS2-native vs D3D9, handled by
# _decode_txd below). Defaults keep the PC dev loop (col_bake.py uses the same idiom).
SA_ROOT = os.environ.get("SA_ROOT", "")
GTA3 = os.environ.get("SA_GTA3_IMG", SA_ROOT + "/MODELS/GTA3.IMG")
PLAYER_IMG = SA_ROOT + "/MODELS/player.img"
OUT = ""
DEPLOY = ""  # game loads assetDir/ (data/) first


def _decode_txd(raw):
    """Decode a ped TXD to {name: (w,h,rgba)}. Picks the codec by the RW device id
    (TXD STRUCT: u16 numTex, u16 deviceId): 6 = PS2-native (sa_txd), else D3D8/9
    (sa_txd_d3d9). Lets one hero_bake serve both the PS2 disc and the PC dev loop."""
    raw = bytes(raw)
    devid = struct.unpack_from("<H", raw, 26)[0] if len(raw) >= 28 else 0
    prim, alt = (sa_txd, sa_txd_d3d9) if devid == 6 else (sa_txd_d3d9, sa_txd)
    try:
        return prim.decode(raw)
    except Exception:
        return alt.decode(raw)   # mixed/odd device id -> try the other decoder
# First 4 = locomotion (idle/walk/run/sprint), index-locked for the SkinAnim
# speed cross-fade. Appended >=4 are non-looping action clips (jump FSM), looked
# up by name in the runtime (CSkelAnim_FindClip).
CLIPS = ["IDLE_stance", "WALK_civi", "run_civi", "sprint_civi",
         "JUMP_launch", "JUMP_glide", "JUMP_land",
         "Turn_L", "Turn_R",
         "CAR_getin_LHS", "CAR_getout_LHS", "CAR_sit",   # driver enter/exit/seated
         "CAR_close_LHS",                                 # b309: shut the door from OUTSIDE (exit close gesture)
         "CAR_align_LHS", "CAR_open_LHS", "CAR_closedoor_LHS",  # real enter seq: align->open->getin->close
         "CAR_open_RHS", "CAR_getin_RHS", "CAR_closedoor_RHS", "CAR_shuffle_RHS",  # passenger-side entry + shuffle to driver
         "CAR_rollout_LHS", "getup",                      # bail-out tumble + stand up
         "BIKEs_Ride",                                    # motorcycle rider pose (fwd-lean, anim.img/bikes.ifp)
         "FightA_1", "FightA_2", "FightA_3",              # unarmed melee punch combo
         "FIGHTIDLE",                                     # boxing stance (target lock)
         "FightSh_FWD", "FightSh_BWD",                    # lock strafe shuffles
         "FightSh_Left", "FightSh_Right",
         # b438: SA plane boarding (anim.img rustler.ifp - the "rustler" anim group:
         # rustler/cropdust/stunt/hydra canopy climb; runtime MAX_CLIP raised to 40)
         "Plane_open", "Plane_getin", "Plane_close", "Plane_getout",
         # b443: the SA fall channel (ped.ifp). glide = held drop pose (CTaskSimpleInAir
         # fall-glide entry), fall = the flail LOOP once the drop is deep (vz < -0.1 SA
         # + ground > 4u below, 0x680600), land = on-feet touch-down, collapse = the
         # crumple landing after a flailed fall (CTaskComplexInAirAndLand 0x67CCB0).
         # The hard slam (minVz < -0.4 SA) re-uses CAR_rollout_LHS + getup, no new clip.
         # 39/40 of the runtime MAX_CLIP now used.
         "FALL_glide", "FALL_fall", "FALL_land", "FALL_collapse",
         # fat-gait locomotion (SA ANIM_GROUP_FAT): the runtime swaps these in for idle/walk/run
         # when the Fatness slider is high (waddle). Found by NAME at load. (ped.ifp)
         "Idlestance_fat", "WALK_fat", "run_fat",
         # b741: AMBIENT IDLE FIDGETS (SA IDLE anim group, ped.ifp) - periodic one-shot idle
         # gestures the player hero plays over the stance when standing still (look around /
         # scratch / check watch / stretch). The runtime (PlayerPed.c) resolves each by name via
         # CSkelAnim_FindClip and picks one at random after a few idle seconds; a name ped.ifp
         # lacks is simply skipped at bake (harmless). 42 -> 47 clips; runtime MAX_CLIP is 48.
         "IDLE_HBHB", "XPRESSscratch", "IDLE_chat", "IDLE_tired", "Idle_Gang1"]
BINMESH_PLG = 0x050E
RPGEOMETRY_PRELIT = 0x08
RPGEOMETRY_TEXTURED = 0x04
RPGEOMETRY_TEXTURED2 = 0x80
RPGEOMETRY_NATIVE = 0x01000000
STRIDE = {1: 20, 2: 32, 3: 10, 4: 16}


def mat3_to_quat(m):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25*s; x = (m21-m12)/s; y = (m02-m20)/s; z = (m10-m01)/s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0+m00-m11-m22)*2.0
        w = (m21-m12)/s; x = 0.25*s; y = (m01+m10)/s; z = (m02+m20)/s
    elif m11 > m22:
        s = math.sqrt(1.0+m11-m00-m22)*2.0
        w = (m02-m20)/s; x = (m01+m10)/s; y = 0.25*s; z = (m12+m21)/s
    else:
        s = math.sqrt(1.0+m22-m00-m11)*2.0
        w = (m10-m01)/s; x = (m02+m20)/s; y = (m12+m21)/s; z = 0.25*s
    n = math.sqrt(x*x+y*y+z*z+w*w) or 1.0
    # negate x,y,z so the runtime row-vector quat_pos_to_m4 reproduces the bind matrix
    # (round-trip 0; this is the convention that matches the raw IFP anim quats too).
    return (-x/n, -y/n, -z/n, w/n)


def std_quat_to_mat3(x, y, z, w):
    # standard quaternion -> row-major 3x3 (row-vector convention, matches sa_skin frame rot)
    xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
    return (1-2*(yy+zz), 2*(xy+wz),   2*(xz-wy),
            2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx),
            2*(xz+wy),   2*(yz-wx),   1-2*(xx+yy))

def mat3_mul(A, B):     # 3x3 row-major, C = A*B
    return tuple(sum(A[r*3+k]*B[k*3+c] for k in range(3)) for r in range(3) for c in range(3))

def mat3_T(A):          # transpose = inverse for a rotation matrix
    return (A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8])

def retarget_quat(qf, cs_bind, cj_bind):
    # qf = raw ANPK quat (x,y,z,w). The runtime uses the CONJUGATE (baked) quat, so the anim
    # local matrix cssmoke gets is A = std_quat_to_mat3(conj(qf))  [std_quat_to_mat3 is the exact
    # inverse of mat3_to_quat - same convention as the sa_skin frame rot]. Re-express the same
    # pose on CJ's bind: A' = A * inv(cs_bind) * cj_bind (verified 0-deg round-trip at rest),
    # then back to a baked quat via mat3_to_quat.
    A = std_quat_to_mat3(-qf[0], -qf[1], -qf[2], qf[3])
    Ap = mat3_mul(mat3_mul(A, mat3_T(cs_bind)), cj_bind)
    return mat3_to_quat(Ap)

def m4_mul(A, B):       # row-major 4x4, C = A*B (row-vector p*A*B)
    C = [0.0]*16
    for r in range(4):
        for c in range(4):
            C[r*4+c] = A[r*4+0]*B[0*4+c]+A[r*4+1]*B[1*4+c]+A[r*4+2]*B[2*4+c]+A[r*4+3]*B[3*4+c]
    return C

def frame_local16(rot, pos):   # row-vector: rows = basis, trans in row 3
    return [rot[0],rot[1],rot[2],0, rot[3],rot[4],rot[5],0,
            rot[6],rot[7],rot[8],0, pos[0],pos[1],pos[2],1]

def affine_inv(M):      # inverse of a row-vector affine 4x4 (R^-1=R^T, t'=-t*R^T)
    RT = [M[0],M[4],M[8], M[1],M[5],M[9], M[2],M[6],M[10]]   # transpose of the 3x3
    t = (M[12], M[13], M[14])
    tt = (-(t[0]*RT[0]+t[1]*RT[3]+t[2]*RT[6]),
          -(t[0]*RT[1]+t[1]*RT[4]+t[2]*RT[7]),
          -(t[0]*RT[2]+t[1]*RT[5]+t[2]*RT[8]))
    return [RT[0],RT[1],RT[2],0, RT[3],RT[4],RT[5],0,
            RT[6],RT[7],RT[8],0, tt[0],tt[1],tt[2],1]


def _num_tex_sets(flags):
    n = (flags >> 16) & 0xFF
    if n: return n
    if flags & RPGEOMETRY_TEXTURED2: return 2
    if flags & RPGEOMETRY_TEXTURED: return 1
    return 0


def _strip_to_tris(idx):
    tris = []
    for k in range(2, len(idx)):
        a, b, c = idx[k-2], idx[k-1], idx[k]
        if a == b or b == c or a == c: continue
        tris.append((b, a, c) if (k & 1) else (a, b, c))
    return tris


def _decode_skin(blob):
    """Skin + skeleton for one skinned DFF, routed by platform: PS2-native cutscene
    actors / ambient peds -> tools/ps2skin (native VIF skin); PC / platform-neutral
    skinned models (player.img: CJ, csplay, clothing) -> tools/sa_skin.  Both return
    {frames, nodes, geoms:[{nvert,numBones,numUsed,maxW,used,boneIdx,boneW,invBind}]},
    so bake_model consumes either unchanged."""
    if ps2skin.is_native_skinned(blob):
        return ps2skin.decode(blob)
    return sa_skin.decode(blob)


def parse_geometry(blob):
    """Return (positions, uvs, colors, submeshes[(matIndex,[tris...])], material_names).
    All vertex indices are GLOBAL (geometry order, == sa_skin order)."""
    if ps2skin.is_native_skinned(blob):
        # PS2-native skinned actor/ped: uninstance the VIF geometry + skin through
        # tools/ps2skin (shares ONE weld with _decode_skin so the vertex orders align).
        return ps2skin.geometry(blob)
    blob = bytes(blob)
    root = parse_chunks(blob)
    gl = root.find(GEOMETRYLIST)
    geo = next(iter(gl.find_all(GEOMETRY)))
    mat_names = _get_material_names(blob, geo)
    st = geo.find(STRUCT); o = st.data_off
    flags, ntri, nvert, nmorph = struct.unpack_from("<4I", blob, o); o += 16
    if flags & RPGEOMETRY_NATIVE:
        raise SystemExit("native geometry not handled")
    colors = [0xFFFFFFFF]*nvert
    if flags & RPGEOMETRY_PRELIT:
        for i in range(nvert):
            r, g, b, a = struct.unpack_from("<4B", blob, o+i*4)
            colors[i] = (r << 24)|(g << 16)|(b << 8)|a
        o += nvert*4
    nsets = _num_tex_sets(flags)
    uvs = [(0.0, 0.0)]*nvert
    for s in range(nsets):
        if s == 0:
            for i in range(nvert):
                uvs[i] = struct.unpack_from("<2f", blob, o+i*8)
        o += nvert*8
    tri_raw = []
    for i in range(ntri):
        v1, v0, matid, v2 = struct.unpack_from("<4H", blob, o+i*8)
        tri_raw.append((v0, v1, v2, matid))
    o += ntri*8
    positions = [(0.0, 0.0, 0.0)]*nvert
    normals = None
    for m in range(nmorph):
        o += 16
        has_v, has_n = struct.unpack_from("<2I", blob, o); o += 8
        if has_v:
            if m == 0:
                for i in range(nvert):
                    positions[i] = struct.unpack_from("<3f", blob, o+i*12)
            o += nvert*12
        if has_n:
            if m == 0:
                normals = [struct.unpack_from("<3f", blob, o+i*12) for i in range(nvert)]
            o += nvert*12
        if m == 0: break
    # bin-mesh splits
    ext = geo.find(EXTENSION)
    bm = None
    if ext:
        for c in ext.children:
            if c.type == BINMESH_PLG: bm = c; break
    submeshes = []
    if bm and bm.size >= 12:
        p = bm.data_off
        bmflags, num, total = struct.unpack_from("<3I", blob, p); p += 12
        tristrip = bool(bmflags & 1)
        for _ in range(num):
            numidx, mat = struct.unpack_from("<2i", blob, p); p += 8
            idx = [struct.unpack_from("<I", blob, p+4*i)[0] for i in range(numidx)]
            p += 4*numidx
            tris = _strip_to_tris(idx) if tristrip else \
                   [(idx[i], idx[i+1], idx[i+2]) for i in range(0, len(idx)-2, 3)]
            submeshes.append((mat, tris))
    else:
        by_mat = {}
        for (v0, v1, v2, matid) in tri_raw:
            by_mat.setdefault(matid, []).append((v0, v1, v2))
        submeshes = list(by_mat.items())
    if normals is None:
        normals = _compute_normals(positions, tri_raw, nvert)
    return positions, uvs, colors, submeshes, mat_names, nvert, normals


def _compute_normals(positions, tri_raw, nvert):
    """Per-vertex normals = area-weighted average of adjacent face normals (used
    only if the DFF morph target carries no normals; SA peds usually do)."""
    acc = [[0.0, 0.0, 0.0] for _ in range(nvert)]
    for (v0, v1, v2, _m) in tri_raw:
        if v0 >= nvert or v1 >= nvert or v2 >= nvert:
            continue
        a, b, c = positions[v0], positions[v1], positions[v2]
        ux, uy, uz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        vx, vy, vz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
        nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
        for vi in (v0, v1, v2):
            acc[vi][0] += nx; acc[vi][1] += ny; acc[vi][2] += nz
    import math
    out = []
    for n in acc:
        l = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) or 1.0
        out.append((n[0]/l, n[1]/l, n[2]/l))
    return out


# default-CJ body components (CClothes::RebuildPlayer). SA's NEW-GAME outfit --
# what CJ wears from the intro on: white vest, blue denim jeans, Binco sneakers
# (CPlayerClothes defaults: model+texture pairs vest/vest, jeans/jeansdenim,
# sneaker/sneakerbincblk). The clothed DFFs are skinned to the same 37-bone rig;
# each mesh takes one named texture from its own TXD in player.img.
# Base body + the SA new-game outfit. Skin FULLY covered by clothing is NOT
# baked (legs under the jeans, feet under the sneakers) and the torso is
# clipped against the vest's bind-pose bbox - near-coincident skin/cloth
# surfaces z-fight on the PSP's 16-bit depth buffer and knees/chest poked
# through as the skinning stretched the layers apart.
CJ_COMPONENTS = [
    ("torso.dff",   "player_torso.txd",   "torso"),   # arms/shoulders/neck (clipped by vest)
    ("head.dff",    "player_face.txd",    "face"),
    ("hands.dff",   "player_torso.txd",   "torso"),   # bare hands/fingers (skin tex)
    ("vest.dff",    "vest.txd",           "vest"),           # white tank top
    ("jeans.dff",   "jeansdenim.txd",     "legs"),           # blue jeans (TXD names its texture by the SLOT)
    ("sneaker.dff", "sneakerbincblk.txd", "sneakerbincblk"), # starting sneakers
]

# real CUTSCENE CJ (MODEL_CSPLAY): the 61-bone cutscene rig with the full face (nodeIds
# 5001-5026, same rig as Big Smoke). cs_head/cs_hands ARE 61-bone; the clothed body
# (torso/vest/jeans/sneaker, 37-bone) is re-indexed onto the 61-bone rig by nodeId at bake
# (remap_nodeid). The csplay ANPK then binds by INDEX like cssmoke -> correct arms + lip-sync.
CJ_CUT_COMPONENTS = [
    ("cs_head.dff",  "player_face.txd",    "face"),           # 61-bone head + full face rig
    ("cs_hands.dff", "player_torso.txd",   "torso"),          # 61-bone hands (skin tex)
    ("torso.dff",    "player_torso.txd",   "torso"),          # arms/shoulders/neck (37-bone -> remap)
    ("vest.dff",     "vest.txd",           "vest"),
    ("jeans.dff",    "jeansdenim.txd",     "legs"),
    ("sneaker.dff",  "sneakerbincblk.txd", "sneakerbincblk"),
]


# SA ped bone NAME -> HANIM nodeId (canonical). Used to bind a cutscene ANPK anim
# (bones named) onto the CJ hero rig (bones carry nodeIds, no names) for csplay.
CUT_NAME2ID = {
    "root":0, "normal":0, "pelvis":1, "spine 1":2, "spine 2":3, "spine":2,
    "neck":4, "head":5,
    "bip01 l clavicle":31, "l clavicle":31, "l upperarm":32, "l forearm":33, "l hand":34,
    "l finger":35, "l finger01":36,
    "bip01 r clavicle":21, "r clavicle":21, "r upperarm":22, "r forearm":23, "r hand":24,
    "r finger":25, "r finger01":26,
    "l thigh":41, "l calf":42, "l foot":43, "l toe0":44,
    "r thigh":51, "r calf":52, "r foot":53, "r toe0":54,
    "belly":201, "l breast":302, "r breast":301,   # (verified against cssmoke.dff nodeIds)
    "jaw1":8,                                       # BONE_JAW: CJ's mouth hinge (189 face verts).
    # ^ intro1a's csplay clip animates jaw1 ~14.5deg; the hero rig has only Jaw + 2 brows (no
    # separate lip bones), so this gives mouth open/close, not full lip-sync. Raw plain-conj map
    # (like the arms) may hinge on a skewed axis -> revert this one line if the jaw looks wrong.
    # Optional brows: "lbrow1":6, "rbrow1":7 (BONE_L_BROW / BONE_R_BROW).
}

def worldspace_retarget_tracks(a, id2bone, bones, numBones):
    """WORLD-SPACE self-referential retarget for csplay (CJ hero rig, no cutscene DFF).
    Per-bone local-delta retargeting broke the arms: the arm chain hangs off Spine2, whose
    bind differs ~178deg from the cutscene skeleton, and a per-bone delta doesn't preserve
    the COMPOSED world orientation. Instead: (1) build the anim's world-space deformation of
    each bone RELATIVE TO ITS OWN FRAME 0 (frame-independent, no external reference bind);
    (2) apply that world deformation to CJ's bind world; (3) convert back to a CJ-local rotation.
    Frame 0 -> exactly CJ's bind pose. Returns (tracks, dur)."""
    import struct as _st
    # gather each CJ bone's source anim (conjugated-quat matrices + times + translation)
    seq_by_bone = {}
    for s in a["seqs"]:
        nid = CUT_NAME2ID.get(s["bone"].strip().lower())
        if nid is None: continue
        b = id2bone.get(nid)
        if b is None: continue
        ht = 1 if s["keyType"] == "KRT0" else 0
        stride = 32 if ht else 20
        times = []; mats = []; trs = []
        for fi in range(s["numFrames"]):
            base = fi * stride
            qf = _st.unpack_from("<4f", s["kf"], base)
            mats.append(std_quat_to_mat3(-qf[0], -qf[1], -qf[2], qf[3]))   # runtime anim-local matrix
            if ht:
                tr = _st.unpack_from("<3f", s["kf"], base+16); t = _st.unpack_from("<f", s["kf"], base+28)[0]
            else:
                tr = (0.0, 0.0, 0.0);                          t = _st.unpack_from("<f", s["kf"], base+16)[0]
            trs.append(tr); times.append(t)
        seq_by_bone[b] = {"times": times, "mats": mats, "trs": trs, "ht": ht}
    cj_bl  = [std_quat_to_mat3(*bones[b]["q"]) for b in range(numBones)]   # CJ bind-local rot
    cj_par = [bones[b]["parent"] for b in range(numBones)]
    cj_bw  = [None]*numBones                                               # CJ bind WORLD (parent<b in skin order)
    for b in range(numBones):
        p = cj_par[b]; cj_bw[b] = cj_bl[b] if p < 0 else mat3_mul(cj_bl[b], cj_bw[p])
    root_b = next((b for b in range(numBones) if cj_par[b] < 0), 0)
    master = (seq_by_bone.get(root_b) or max(seq_by_bone.values(), key=lambda s: len(s["times"])))["times"]
    def samp_mat(b, t):
        s = seq_by_bone.get(b)
        if s is None: return cj_bl[b]                         # unmapped bone stays at bind
        ts = s["times"]; j = min(range(len(ts)), key=lambda k: abs(ts[k]-t)); return s["mats"][j]
    def samp_tr(b, t):
        s = seq_by_bone.get(b)
        if s is None or not s["ht"]: return (0.0,0.0,0.0)
        ts = s["times"]; j = min(range(len(ts)), key=lambda k: abs(ts[k]-t)); return s["trs"][j]
    def anim_world(t):
        aw = [None]*numBones
        for b in range(numBones):
            al = samp_mat(b, t); p = cj_par[b]; aw[b] = al if p < 0 else mat3_mul(al, aw[p])
        return aw
    aw0 = anim_world(master[0])
    out = {b: {"bone": b, "hasTrans": seq_by_bone[b]["ht"], "keys": []} for b in seq_by_bone}
    for t in master:
        awt = anim_world(t); tw = [None]*numBones
        for b in range(numBones):
            D  = mat3_mul(mat3_T(aw0[b]), awt[b])             # world deformation from frame 0 = inv(aw0)*awt
            tw[b] = mat3_mul(cj_bw[b], D)                      # target world = CJ bindWorld * D
        for b in out:
            p = cj_par[b]
            loc = tw[b] if p < 0 else mat3_mul(tw[b], mat3_T(tw[p]))   # local = TW * inv(TW_parent)
            q = mat3_to_quat(loc)
            tr = samp_tr(b, t)
            out[b]["keys"].append((tuple(int(round(c*4096.0)) for c in q),
                                   int(round(t*60.0)),
                                   tuple(int(round(c*1024.0)) for c in tr)))
    return list(out.values()), max(master)


def smooth_skin_weights(GV, radius=0.05, strength=0.7, lo=0):
    """Spatially smooth per-vertex bone weights: the gameplay CJ mesh is ~41% single-bone
    (rigid), which folds into blocky joints at extreme cutscene poses. Blend each vertex's
    weights with its neighbours within `radius` (linear falloff), mix `strength` toward the
    smoothed set, keep the top 4 bones, renormalise. Cutscene CJ only - cssmoke is already
    smooth and the gameplay/ambient peds must stay byte-identical.
    `lo`: only verts with index >= lo are smoothed (and only against each other) - used for the
    real cutscene CJ to soften the gameplay BODY (torso arms) without touching the already-smooth
    cs_head face verts (would blur the working lip-sync) or cs_hands."""
    n = len(GV); pos = [gv[0] for gv in GV]
    cell = radius; r2 = radius * radius
    grid = {}
    for i, p in enumerate(pos):
        if i < lo: continue                          # smooth verts >= lo only, and only vs each other
        grid.setdefault((int(p[0]//cell), int(p[1]//cell), int(p[2]//cell)), []).append(i)
    out = []
    for i, p in enumerate(pos):
        if i < lo:                                   # cs_head/cs_hands: keep as-is (don't blur the face)
            out.append(GV[i]); continue
        cx, cy, cz = int(p[0]//cell), int(p[1]//cell), int(p[2]//cell)
        acc = {}
        for dx in (-1,0,1):
          for dy in (-1,0,1):
            for dz in (-1,0,1):
              for j in grid.get((cx+dx, cy+dy, cz+dz), ()):
                q = pos[j]
                d2 = (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
                if d2 > r2: continue
                fo = 1.0 - d2/r2                      # linear distance falloff
                bidx, bw = GV[j][3], GV[j][4]
                for k in range(4):
                    if bw[k] > 0: acc[bidx[k]] = acc.get(bidx[k], 0.0) + bw[k]*fo
        s = sum(acc.values()) or 1.0
        for b in acc: acc[b] /= s
        orig = {}
        for k in range(4):
            if GV[i][4][k] > 0: orig[GV[i][3][k]] = orig.get(GV[i][3][k], 0.0) + GV[i][4][k]
        mixed = {}
        for b in set(list(acc) + list(orig)):
            mixed[b] = strength*acc.get(b, 0.0) + (1.0-strength)*orig.get(b, 0.0)
        top = sorted(mixed.items(), key=lambda x: -x[1])[:4]
        ts = sum(w for _, w in top) or 1.0
        bidx = [0,0,0,0]; bw = [0.0,0.0,0.0,0.0]
        for k, (b, w) in enumerate(top): bidx[k] = b; bw[k] = w/ts
        gv = GV[i]
        out.append((gv[0], gv[1], gv[2], tuple(bidx), tuple(bw), gv[5]))
    GV[:] = out


def bake_model(arg="fam1", clips=None, emit_clst=True, cut=None):
    """Bake one skinned ped into an HRO2 byte stream (multi-ped reuse: ped_bake.py
    concatenates several of these into peds.bin).  emit_clst=False drops the GE-skin
    cluster section (ambient peds stay on the CPU LBS path -> ~50KB/model less RAM).
    cut = {img, actor, anpk}: bake a CUTSCENE actor (DFF from cutscene.img, one ANPK
    clip from cuts.img, bones mapped by index, no stand-up) instead of a PED.IFP ped."""
    global CLIPS
    saved_clips = CLIPS
    if clips is not None:
        CLIPS = clips
    pkg = None if cut else sa_ifp.decode(open(sa_ifp._find_ped_ifp(), "rb").read())
    import dff_clumps
    remap_nodeid = False               # cjCut re-indexes 37-bone body comps onto the 61-bone rig

    if cut and cut.get("cjCut"):
        # real cutscene CJ: cs_head (61-bone + full face 5001-5026) + cs_hands + the clothed
        # body, all re-indexed onto the 61-bone cutscene rig by nodeId. The csplay ANPK binds by
        # INDEX (like cssmoke, NOT the useHero name-retarget) -> correct arms + full lip-sync.
        img = sa_img.SaImg(PLAYER_IMG)
        comps = CJ_CUT_COMPONENTS
        skel_blob = dff_clumps.split_clumps(img.extract("cs_head.dff")).get("normal")
        name = "cj"
        remap_nodeid = True
    elif cut and cut.get("useHero"):
        # csplay: the cutscene PLAYER actor reuses the game's CJ hero model (no csplay.dff
        # exists). Mesh + skeleton from player.img; the ANPK anim is bound by NAME->nodeId
        # onto CJ's 37-bone rig (the cutscene skeleton is 61-bone, so index mapping fails).
        img = sa_img.SaImg(PLAYER_IMG)
        comps = CJ_COMPONENTS
        skel_blob = img.extract("torso.dff")
        name = "cj"                       # -> up_mode 0 (Z-up), same as the hero
    elif cut:
        img = sa_img.SaImg(cut["img"])
        comps = [(cut["actor"] + ".dff", cut["actor"] + ".txd", None)]
        skel_blob = img.extract(cut["actor"] + ".dff")
        name = cut["actor"]
    elif arg.lower() == "cj":
        img = sa_img.SaImg(PLAYER_IMG)
        comps = CJ_COMPONENTS
        skel_blob = img.extract("torso.dff")
        name = "cj"
    else:
        img = sa_img.SaImg(GTA3)
        comps = [(arg + ".dff", arg + ".txd", None)]   # texname None -> by material name
        skel_blob = img.extract(arg + ".dff")
        name = arg

    # ---- skeleton (shared across components) ----
    sk = _decode_skin(skel_blob)
    nodes = sk["nodes"]; numBones = len(nodes)
    bone_nodeId = [nodes[b][0] for b in range(numBones)]
    nodeId_to_bone = {nid: b for b, nid in enumerate(bone_nodeId)}
    frames_all = sk["frames"]
    frame_by_node = {f["nodeId"]: f for f in frames_all if f["nodeId"] >= 0}
    frame_idx_by_node = {f["nodeId"]: i for i, f in enumerate(frames_all) if f["nodeId"] >= 0}
    fidx_node = {i: f["nodeId"] for i, f in enumerate(frames_all)}

    # cjCut: cs_head orders its 61 bones DIFFERENTLY from the csplay ANPK seq order (face bones
    # sit right after Head, not at the tail) -> a plain index-map animates the wrong bones (arms
    # = mush). Build a bone-NAME -> nodeId map from the skel's OWN frame node-names so the ANPK
    # binds by name -> nodeId -> bone instead (60/61 resolve; 'root' -> nodeId 0). Body + face.
    cjcut_name2nid = {}
    if remap_nodeid:
        _k = dff_clumps._children(skel_blob, 0, len(skel_blob))
        _cl = [x for x in _k if x[0] == 0x10]                        # CLUMP
        if _cl:
            _ck = dff_clumps._children(skel_blob, _cl[0][2], _cl[0][2] + _cl[0][1])
            _fl = [x for x in _ck if x[0] == 0x0E]                    # FRAMELIST
            if _fl:
                _nm = dff_clumps._frame_names(skel_blob, _fl[0][2], _fl[0][2] + _fl[0][1])
                for _i, _f in enumerate(frames_all):
                    _n = _nm.get(_i)
                    if _n and _f["nodeId"] >= 0:
                        cjcut_name2nid[_n.strip().lower()] = _f["nodeId"]
        cjcut_name2nid.setdefault("root", 0)

    fworld = [None]*len(frames_all)
    def build_fw(i):
        if fworld[i] is not None: return fworld[i]
        fl = frame_local16(frames_all[i]["rot"], frames_all[i]["pos"])
        p = frames_all[i]["parent"]
        fworld[i] = fl if (p is None or p < 0) else m4_mul(fl, build_fw(p))
        return fworld[i]
    for i in range(len(frames_all)): build_fw(i)

    # Use the DFF's ORIGINAL skin inverse-bind matrices (they are in MESH space and
    # bridge the FRAMELIST<->mesh convention gap; recomputing invBind from the frames
    # put the bones in a Y-up space that didn't coincide with the Z-up mesh verts ->
    # animation tore. See cj-skinning-rootcause). bindWorld = invBind^-1 (mesh space);
    # bind LOCAL = bindWorld[b] * bindWorld[parent]^-1 gives the mesh-space bone offset
    # + rest rotation; the keyframe rotation replaces the rest rotation at runtime.
    geo_inv = sk["geoms"][0]["invBind"]
    def fix_pad(m):
        m = list(m); m[3]=m[7]=m[11]=0.0; m[15]=1.0; return m
    bparent = []
    dffinv = []
    bindW = []
    bl_by_id = {}      # nodeId -> runtime bind-local rot (mesh-space); the cut retarget's CJ bind
    for b in range(numBones):
        nid = bone_nodeId[b]
        f = frame_by_node.get(nid)
        par = -1
        if f is not None:
            pf = f["parent"]
            if pf is not None and pf >= 0 and pf in fidx_node:
                par = nodeId_to_bone.get(fidx_node[pf], -1)
        bparent.append(par)
        iv = fix_pad(geo_inv[b]) if b < len(geo_inv) else [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
        dffinv.append(iv)
        bindW.append(affine_inv(iv))
    if os.environ.get("HERO_DBG"):
        # DIAG: framelist bind-world vs DFF-invBind bind-world per bone. T = framelist^-1 * mesh
        # is the framelist->mesh space transform; if it's a constant Y<->Z swap across bones, the
        # IFP anim (framelist space) must be rotated into mesh space to stop the walk/run tear.
        for b in [0, 1, 2, 3, 10, 20]:
            if b >= numBones: continue
            fi = frame_idx_by_node.get(bone_nodeId[b])
            if fi is None: continue
            fw = fworld[fi]; dw = bindW[b]
            T = m4_mul(affine_inv(fw), dw)
            print("BONE %2d nid %3d T(fl->mesh) rot[%.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f] t[%.2f %.2f %.2f]"
                  % (b, bone_nodeId[b], T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10], T[12],T[13],T[14]))
    bones = []
    for b in range(numBones):
        par = bparent[b]
        bl = m4_mul(bindW[b], dffinv[par]) if par >= 0 else bindW[b]   # mesh-space bind local
        blrot = [bl[0],bl[1],bl[2], bl[4],bl[5],bl[6], bl[8],bl[9],bl[10]]
        bl_by_id[bone_nodeId[b]] = tuple(blrot)   # runtime bind-local rot per nodeId (for cut retarget)
        bones.append({"parent": par, "nodeId": bone_nodeId[b],
                      "q": mat3_to_quat(blrot), "p": (bl[12], bl[13], bl[14]), "inv": dffinv[b]})

    # ---- geometry + skin across components into ONE global vertex array ----
    tex_list, tex_index = [], {}
    def author(txd, texname, txdname=""):
        # cache key includes the TXD: clothing TXDs reuse SLOT texture names
        # ("legs" lives in both player_legs.txd and jeansdenim.txd - a global
        # name key handed the jeans the underwear texture).
        nkey = (texname or "").strip().lower()
        key = (txdname.lower(), nkey)
        if key in tex_index: return tex_index[key]
        entry = txd.get(texname) or txd.get(nkey)
        if entry is None:
            for k, v in txd.items():
                if k.lower() == nkey: entry = v; break
        t = None
        if entry is not None:
            w, h, rgba = entry
            try: t = psp_tex.author_psp_texture(rgba, w, h, fmt="T8", mipmaps=True)
            except Exception: t = None
        ti = len(tex_list) if t is not None else -1
        if t is not None: tex_list.append(t)
        tex_index[key] = ti
        return ti

    # NO skin clipping. Clipping hidden skin kept either leaving z-fight or
    # carving holes (the shoulder gaps). Instead the clothing that only PARTIALLY
    # covers skin (the vest over the torso) is INFLATED along its vertex normals
    # so it always sits just outside the body -> the depth test hides the torso
    # skin under it, full body stays intact (no holes ever). Fully-covered parts
    # (legs under jeans, feet under sneakers) are dropped from CJ_COMPONENTS.
    INFLATE = {"vest.dff": 0.018}      # metres pushed out along the bind normal

    GV = []          # global verts: (pos3, uv2, color, bidx4, bw4)
    body_vstart = None   # cjCut: GV index where the gameplay BODY comps begin (after cs_head/cs_hands)
    # bake fat/muscle morph deltas for the gameplay CJ AND the real cutscene CJ (cjCut) so the
    # chosen body carries into cutscenes. cs_head/cs_hands + the body comps all have 3 clumps.
    want_morph = (arg.lower() == "cj" and not cut) or bool(cut and cut.get("cjCut"))
    GV_morph = []    # parallel to GV: (dPosFat[3], dPosRipped[3]) - the debug Player body sliders
    sub_out = []
    idx_pool = []
    for (dffname, txdname, texname) in comps:
        dff = img.extract(dffname)
        import dff_clumps                       # clothes DFFs = 3 CLUMPs (Normal/Fat/Ripped); parse_geometry
        _cl = dff_clumps.split_clumps(dff)      # would take the FIRST (="Ripped") -> use the true base "Normal".
        if "normal" in _cl:   dff = _cl["normal"]
        elif len(_cl) == 1:   dff = next(iter(_cl.values()))   # non-morph component: its only clump
        positions, uvs, colors, submeshes, mat_names, nvert, normals = parse_geometry(dff)
        mdelta = None
        if want_morph:                              # Fat/Ripped clump positions vs Normal (same topology)
            def _clpos(key):
                if key in _cl:
                    try:
                        pp = parse_geometry(_cl[key])[0]
                        return pp if len(pp) == nvert else None
                    except Exception:
                        return None
                return None
            fp = _clpos("fat"); rp = _clpos("ripped")
            mdelta = []
            for v in range(nvert):
                dfx = ((fp[v][0]-positions[v][0], fp[v][1]-positions[v][1], fp[v][2]-positions[v][2])
                       if fp else (0.0, 0.0, 0.0))
                drx = ((rp[v][0]-positions[v][0], rp[v][1]-positions[v][1], rp[v][2]-positions[v][2])
                       if rp else (0.0, 0.0, 0.0))
                mdelta.append((dfx, drx))
        infl = INFLATE.get(dffname, 0.0)
        if infl != 0.0:
            positions = [
                (positions[v][0] + normals[v][0] * infl,
                 positions[v][1] + normals[v][1] * infl,
                 positions[v][2] + normals[v][2] * infl)
                for v in range(nvert)
            ]
        _skc = _decode_skin(dff)
        cgeo = _skc["geoms"][0]
        if cgeo["nvert"] != nvert:
            raise SystemExit("%s skin nvert %d != geom %d" % (dffname, cgeo["nvert"], nvert))
        # cutscene CJ: this component's skin bone INDICES reference its OWN skeleton (the 37-bone
        # body rig, or cs_hands' own 61-bone order). Remap onto the main 61-bone rig by nodeId
        # (SA BuildBoneIndexConversionTable). Identity for a matching rig; real work for the body.
        comp_nid = None
        if remap_nodeid:
            _cn = _skc["nodes"]
            comp_nid = [_cn[b][0] for b in range(len(_cn))]
        txd = _decode_txd(img.extract(txdname))
        vbase = len(GV)
        if remap_nodeid and body_vstart is None and not dffname.startswith("cs_"):
            body_vstart = vbase              # first non-cutscene (body) component -> smooth from here

        for v in range(nvert):
            bidx = cgeo["boneIdx"][v]
            if comp_nid is not None:                       # remap component bone -> nodeId -> main-rig bone
                bidx = tuple(nodeId_to_bone.get(comp_nid[bi], 0) if bi < len(comp_nid) else 0
                             for bi in bidx)
            GV.append((positions[v], uvs[v], colors[v], bidx, cgeo["boneW"][v], normals[v]))
            if want_morph:
                GV_morph.append(mdelta[v] if mdelta else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        for (matidx, tris) in submeshes:
            if texname:
                ti = author(txd, texname, txdname)
            else:
                m = mat_names[matidx] if 0 <= matidx < len(mat_names) else ""
                nm = (m.get("texture_name") or "") if isinstance(m, dict) else (m or "")
                ti = author(txd, nm, txdname)
            first = len(idx_pool)
            for (a, b, c) in tris:
                if a < nvert and b < nvert and c < nvert:
                    idx_pool += [vbase + a, vbase + b, vbase + c]
            sub_out.append((ti, first, len(idx_pool) - first))
    nvert = len(GV)

    # cutscene CJ: soften the gameplay mesh's rigid weights so extreme cutscene poses don't
    # fold the shoulders/elbows into blocks (cssmoke is already smooth; leave it + peds alone).
    if cut and (cut.get("useHero") or cut.get("cjCut")):
        # cjCut: the bind MATCHES (arm bones share torso's bind) but torso's RIGID gameplay
        # weights fold the shoulder/upper-arm into blocks at extreme cutscene poses ("no
        # shoulders, stretched upper arm"). Smooth ONLY the body verts (lo=body_vstart) so the
        # working cs_head lip-sync + cs_hands stay untouched. (useHero smoothed everything.)
        _lo = (body_vstart or 0) if cut.get("cjCut") else 0
        _r0 = sum(1 for gv in GV[_lo:] if sum(1 for w in gv[4] if w > 0.01) == 1)
        smooth_skin_weights(GV, radius=0.065, strength=0.80, lo=_lo)
        _r1 = sum(1 for gv in GV[_lo:] if sum(1 for w in gv[4] if w > 0.01) == 1)
        print("  weight-smooth: single-bone verts %d -> %d (of %d, lo=%d)" % (_r0, _r1, nvert - _lo, _lo))

    # ==== GE hardware-skinning cluster partition (ge_hw_skinning_design.md) ====
    # The PSP GE skins in hardware but with <=8 bone matrices per draw; the SA rig is ~37
    # bones. Split each texture-submesh's triangles into clusters whose UNION of influencing
    # bones is <=8; per cluster remap the 4-bone weights POSITIONAL to the cluster's bone
    # list (u8 /128). Emitted as a separate 'CLST' section so the runtime can A/B vs the CPU
    # pre-skin path. Verified offline below (positional skin == direct LBS under random poses).
    def _tri_bones(tri):
        s = set()
        for v in tri:
            bi, bw = GV[v][3], GV[v][4]
            for k in range(4):
                if bw[k] > 1e-4: s.add(int(bi[k]))
        return s
    def _flush(clusters, ti, cbones, ctris):
        if not ctris: return
        bl = sorted(cbones); slot = {b: j for j, b in enumerate(bl)}
        loc = {}; cv = []; ci = []
        for tri in ctris:
            for v in tri:
                if v not in loc:
                    loc[v] = len(cv)
                    pos, uv, color, bi, bw, nrm = GV[v]
                    w8 = [0.0]*8
                    for k in range(4):
                        if bw[k] > 1e-4: w8[slot[int(bi[k])]] += bw[k]
                    tot = sum(w8) or 1.0
                    wq = [int(round(x/tot*128.0)) for x in w8]
                    wq[wq.index(max(wq))] += 128 - sum(wq)   # dump rounding on the largest
                    cv.append((pos, uv, color, wq, nrm, v))   # v = GV index (for the round-trip)
                ci.append(loc[v])
        clusters.append({"tex": ti, "bones": bl, "verts": cv, "idx": ci})
    def _tri_primary(tri):    # bone carrying the most weight across the triangle (locality key)
        acc = {}
        for v in tri:
            bi, bw = GV[v][3], GV[v][4]
            for k in range(4):
                if bw[k] > 1e-4: acc[int(bi[k])] = acc.get(int(bi[k]), 0.0) + bw[k]
        return max(acc, key=acc.get) if acc else 0
    clusters = []
    for (ti, first, cnt) in sub_out:
        tris = [(idx_pool[first+i], idx_pool[first+i+1], idx_pool[first+i+2])
                for i in range(0, cnt, 3)]
        # SORT by (primary bone, full bone set) so triangles of the same body part are
        # adjacent -> the greedy pass packs them into FAR fewer <=8-bone clusters (mesh order
        # mixed body parts -> a cluster filled 8 bones then the next tri needed others -> 63).
        tris.sort(key=lambda t: (_tri_primary(t), tuple(sorted(_tri_bones(t)))))
        cbones = set(); ctris = []
        for tri in tris:
            tb = _tri_bones(tri)
            if len(cbones | tb) > 8:
                _flush(clusters, ti, cbones, ctris); cbones = set(); ctris = []
            cbones |= tb; ctris.append(tri)
        _flush(clusters, ti, cbones, ctris)

    # OFFLINE ROUND-TRIP: positional-weight skinning (what the GE does) must equal the direct
    # 4-bone LBS for ANY bone pose. Test with random per-bone affines -> max error <= u8 quant.
    import random as _rnd
    _rnd.seed(1)
    def _rmat():
        import math as _m
        a, b, c = (_rnd.uniform(-1, 1) for _ in range(3))
        ca, sa = _m.cos(a), _m.sin(a); cb, sb = _m.cos(b), _m.sin(b)
        R = [[ca*cb, -sa, ca*sb], [sa*cb, ca, sa*sb], [-sb, 0, cb]]
        t = [_rnd.uniform(-2, 2) for _ in range(3)]
        return R, t
    _bm = [_rmat() for _ in range(numBones)]
    def _apply(m, p):
        R, t = m
        return [R[r][0]*p[0] + R[r][1]*p[1] + R[r][2]*p[2] + t[r] for r in range(3)]
    _maxerr = 0.0
    for cl in clusters:
        bl = cl["bones"]
        for (pos, uv, color, wq, nrm, gvi) in cl["verts"]:
            # GE path: sum over the cluster's slots of (wq[slot]/128) * bone[boneList[slot]] * pos
            ge = [0.0, 0.0, 0.0]
            for s in range(8):
                if wq[s] == 0: continue
                ap = _apply(_bm[bl[s]], pos); w = wq[s] / 128.0
                for r in range(3): ge[r] += w * ap[r]
            # ground truth: the ORIGINAL direct 4-bone float LBS for this exact vertex.
            bi0, bw0 = GV[gvi][3], GV[gvi][4]
            tw = sum(bw0) or 1.0
            lbs = [0.0, 0.0, 0.0]
            for k in range(4):
                if bw0[k] <= 1e-4: continue
                ap = _apply(_bm[int(bi0[k])], pos)
                for r in range(3): lbs[r] += (bw0[k] / tw) * ap[r]
            for r in range(3):
                e = abs(ge[r] - lbs[r]);  _maxerr = e if e > _maxerr else _maxerr
    ncv = sum(len(c["verts"]) for c in clusters)
    print("  CLST: %d clusters (from %d tex-subs), %d cluster-verts (%.1fx), max bones/cluster %d, round-trip err %.5f"
          % (len(clusters), len(sub_out), ncv, ncv/max(1, nvert),
             max(len(c["bones"]) for c in clusters), _maxerr))
    assert all(len(c["bones"]) <= 8 for c in clusters), "cluster exceeds 8 bones!"
    # tolerance = the u8 weight quantisation (1/128 per weight) x worst-case skin-matrix reach
    # (~2u ped) under RANDOM +-2 bone translations = ~2-3cm max vertex error, imperceptible on a
    # ped (PS2/PSP ship u8 skin weights). Real poses move bones far less -> smaller. Bump to
    # GU_WEIGHT_16BIT if a seam ever shows.
    # The guard only bites when the CLST section is actually SHIPPED (emit_clst, the hero):
    # cutscene actors + ambient peds (emit_clst=False) render via the CPU LBS path with the FULL
    # float weights, so the u8 clusters computed above are diagnostics only. PS2-native skin
    # weights carry ~13 mantissa bits (the low 10 are stolen for the bone index) so their
    # worst-case random-pose round-trip lands a hair over 3cm (cssmoke 0.0336) - fine for LBS,
    # and irrelevant unless those u8 clusters ride to the GE.
    assert _maxerr < 0.03 or not emit_clst, \
        "round-trip skinning mismatch %.4f (u8 weight quant too coarse?)" % _maxerr

    # CUTSCENE actor: one ANPK clip, bones bound by INDEX (ANPK bone i -> DFF bone i,
    # both are the same 61-bone skeleton in HAnim order), uncompressed float keyframes,
    # absolute-second times. No PED.IFP.
    if cut:
        a = next((x for x in cut["anpk"]["anims"] if x["name"].lower() == cut["actor"].lower()), None)
        out_clips = []
        # csplay (useHero): bind ANPK bones onto CJ's rig by NAME->nodeId->CJ-bone-index.
        # cssmoke: the DFF and ANPK are the SAME 61-bone skeleton -> bone i maps to i.
        id2bone = {bone_nodeId[b]: b for b in range(numBones)} if cut.get("useHero") else None
        use_selfref = bool(cut.get("useHero"))
        # csplay has no dedicated cutscene DFF, and each cutscene ped has its OWN custom bind
        # (measured: cssmoke vs csbjd/bear/bettina differ 54-72deg), so there is no external
        # reference bind. Instead retarget each NON-root bone RELATIVE TO THE ANIM'S OWN FRAME 0
        # onto CJ's bind: A' = A * inv(frame0) * cj_bind. CJ then starts in its own (natural) bind
        # at frame 0 and animates by the clip's delta-from-start - the same trick the runtime's
        # refRootQ does for the root, extended to every limb. No cssmoke reference needed.
        if a and cut.get("cjCut"):
            # real cutscene CJ: bind each ANPK bone by NAME -> nodeId (cjcut_name2nid, from
            # cs_head's own frame node-names) -> bone. Plain conjugation, absolute (isCut), like
            # cssmoke - but by NAME, because cs_head's bone ORDER != the csplay ANPK seq order
            # (face bones sit after Head, not at the tail), so an index-map drove the wrong bones
            # = arm mush. 60/61 seqs resolve; 'root' -> nodeId 0.
            tracks = []; maxtime = 0.0
            for s in a["seqs"]:
                nid = cjcut_name2nid.get(s["bone"].strip().lower())
                if nid is None: continue
                b = nodeId_to_bone.get(nid)
                if b is None: continue
                hasTrans = 1 if s["keyType"] == "KRT0" else 0
                stride = 32 if hasTrans else 20
                kf = s["kf"]; keys = []
                for fi in range(s["numFrames"]):
                    base = fi * stride
                    qf = struct.unpack_from("<4f", kf, base)
                    qb = (-qf[0], -qf[1], -qf[2], qf[3])              # ANPK conjugation
                    q = tuple(int(round(c * 4096.0)) for c in qb)
                    if hasTrans:
                        tr = tuple(int(round(c * 1024.0)) for c in struct.unpack_from("<3f", kf, base + 16))
                        tsec = struct.unpack_from("<f", kf, base + 28)[0]
                    else:
                        tr = (0, 0, 0); tsec = struct.unpack_from("<f", kf, base + 16)[0]
                    keys.append((q, int(round(tsec * 60.0)), tr))
                    maxtime = max(maxtime, tsec)
                # csplay's KRT0 tracks carry a bogus frame-0 TRANSLATION sentinel (~+10 on each axis)
                # that exploded the arms; the real per-bone offset is constant from frame 1 on (and the
                # runtime NEEDS it - dropping translation crumpled the arms). Copy frame 1's translation
                # onto frame 0 to kill the sentinel, keeping rotation + timing untouched.
                if hasTrans and len(keys) >= 2:
                    keys[0] = (keys[0][0], keys[0][1], keys[1][2])
                tracks.append({"bone": b, "hasTrans": hasTrans, "keys": keys})
            for tk in tracks:                                         # quat continuity (short-arc)
                ks = tk["keys"]
                for _k in range(1, len(ks)):
                    pq, cq = ks[_k-1][0], ks[_k][0]
                    if pq[0]*cq[0] + pq[1]*cq[1] + pq[2]*cq[2] + pq[3]*cq[3] < 0:
                        ks[_k] = ((-cq[0], -cq[1], -cq[2], -cq[3]), ks[_k][1], ks[_k][2])
            out_clips.append({"name": cut["actor"], "dur": maxtime, "tracks": tracks})
        elif a and cut.get("useHero"):
            # csplay PLAIN-CONJ via nodeId (path B). The world-space retarget referenced the
            # anim's FRAME 0 (D = inv(aw0)*awt), which ZEROED the authored world facing -> CJ
            # started at +X (screen-right) and spun (CUTDIAG b537: csplay yaw0=0.0 yaw1=-176.8
            # vs cssmoke yaw0=59.4). cssmoke works because it applies the raw ANPK track (plain
            # conj) on its own rig, keeping the ABSOLUTE orientation. Do the same for csplay but
            # map ANPK bone NAME -> nodeId -> CJ hero bone. Pelvis/spine binds match (~0deg) so
            # root/legs/facing come out right; arms hang off the ~178deg-different Spine2 and may
            # still need CJ's real cutscene bind (a dedicated csplay DFF) - but the facing, the
            # confirmed regression, is fixed here.
            tracks = []; maxtime = 0.0
            for s in a["seqs"]:
                nid = CUT_NAME2ID.get(s["bone"].strip().lower())
                if nid is None: continue
                b = id2bone.get(nid)
                if b is None: continue
                hasTrans = 1 if s["keyType"] == "KRT0" else 0
                stride = 32 if hasTrans else 20
                kf = s["kf"]; keys = []
                for fi in range(s["numFrames"]):
                    base = fi * stride
                    qf = struct.unpack_from("<4f", kf, base)
                    qb = (-qf[0], -qf[1], -qf[2], qf[3])          # ANPK conjugation (AnimManager.cpp:810)
                    q = tuple(int(round(c * 4096.0)) for c in qb)
                    if hasTrans:
                        tr = tuple(int(round(c * 1024.0)) for c in struct.unpack_from("<3f", kf, base + 16))
                        tsec = struct.unpack_from("<f", kf, base + 28)[0]
                    else:
                        tr = (0, 0, 0); tsec = struct.unpack_from("<f", kf, base + 16)[0]
                    keys.append((q, int(round(tsec * 60.0)), tr))
                    maxtime = max(maxtime, tsec)
                tracks.append({"bone": b, "hasTrans": hasTrans, "keys": keys})
            for tk in tracks:                                     # quat continuity (short-arc)
                ks = tk["keys"]
                for _k in range(1, len(ks)):
                    pq, cq = ks[_k-1][0], ks[_k][0]
                    if pq[0]*cq[0] + pq[1]*cq[1] + pq[2]*cq[2] + pq[3]*cq[3] < 0:
                        ks[_k] = ((-cq[0], -cq[1], -cq[2], -cq[3]), ks[_k][1], ks[_k][2])
            out_clips.append({"name": cut["actor"], "dur": maxtime, "tracks": tracks})
        elif a:
            # cssmoke: DFF + ANPK are the SAME 61-bone skeleton -> bone i maps to i, plain conjugation.
            tracks = []; maxtime = 0.0
            for bi, s in enumerate(a["seqs"]):
                if bi >= numBones: break
                bone_idx = bi
                hasTrans = 1 if s["keyType"] == "KRT0" else 0
                stride = 32 if hasTrans else 20
                kf = s["kf"]; keys = []
                for fi in range(s["numFrames"]):
                    base = fi * stride
                    qf = struct.unpack_from("<4f", kf, base)
                    qb = (-qf[0], -qf[1], -qf[2], qf[3])   # ANPK conjugation (AnimManager.cpp:810)
                    q = tuple(int(round(c * 4096.0)) for c in qb)
                    if hasTrans:
                        tr = tuple(int(round(c * 1024.0)) for c in struct.unpack_from("<3f", kf, base + 16))
                        tsec = struct.unpack_from("<f", kf, base + 28)[0]
                    else:
                        tr = (0, 0, 0); tsec = struct.unpack_from("<f", kf, base + 16)[0]
                    keys.append((q, int(round(tsec * 60.0)), tr))
                    maxtime = max(maxtime, tsec)
                tracks.append({"bone": bone_idx, "hasTrans": hasTrans, "keys": keys})
            out_clips.append({"name": cut["actor"], "dur": maxtime, "tracks": tracks})

    if not cut:
     # clips: PED.IFP anims + the bike-rider anims from anim.img/bikes.ifp (BIKEs_Ride
     # etc. - the forward-leaning motorcycle pose, a DIFFERENT skeleton posture than
     # CAR_sit). Merged so CLIPS can name either source.
     by_name = {a["name"].lower(): a for a in pkg["anims"]}
     # anim.img gotcha: bikes.ifp/rustler.ifp (motorcycle + plane rider clips) live in
     # the PC anim/anim.img, which DOES NOT EXIST on a PS2 disc (PS2 ships only ANIM/
     # PED.IFP + ANIM/CUTS.IMG). Source them from SA_ROOT/anim/anim.img when present
     # (PC dev loop); on a PS2 disc it is absent -> those clips are simply dropped (the
     # base CJ locomotion/idle set still comes from PED.IFP). Never crashes the bake.
     _anim_img = os.path.join(SA_ROOT, "anim", "anim.img")
     if os.path.exists(_anim_img):
         try:
             import sys as _sys
             _saw = os.environ.get("SAW_ROOT", "")
             if _saw not in _sys.path: _sys.path.insert(0, _saw)
             from core.imgarchive import ImgArchive
             aimg = ImgArchive.open(_anim_img)
             for blk in ("bikes.ifp", "rustler.ifp"):   # b438: + the Plane_* boarding set
                 be = next((e for e in aimg.entries if e.name.lower() == blk), None)
                 if be is None: continue
                 bpkg = sa_ifp.decode(aimg.extract(be))
                 for a in bpkg["anims"]:
                     by_name.setdefault(a["name"].lower(), a)
         except Exception as e:
             print("  ! bike anim load failed: %s" % e)
     else:
         print("  anim.img absent (PS2 disc) - bike/plane rider clips dropped; base anims from PED.IFP")
     out_clips = []
     for cname in CLIPS:
         a = by_name.get(cname.lower())
         if not a: continue
         tracks = []; maxtime = 0.0
         for s in a["seqs"]:
             bi = nodeId_to_bone.get(s["boneTag"], -1)
             if bi < 0: continue
             if os.environ.get("HERO_DBG") and not out_clips:
                 print("  seq bone=%2d nid=%3d name='%s'" % (bi, s["boneTag"], s["name"]))
             # SKIP bones whose component bind is IDENTITY but the IFP keyframe is a big
             # rotation (rig mismatch -> the verts spike): the "breast" jiggle bones and
             # the "finger" bones (their DFF bind is straight/identity, the anim assumes a
             # curled rest). Cosmetic for v1; keep them at bind (open hand, no jiggle).
             nm = s["name"].lower()
             # + face mimic bones (Jaw, L/R Brow): the locomotion IFP animates them, but our face
             # Keep them at bind = a neutral face; CJ doesn't need to emote during locomotion.
             if "breast" in nm or "finger" in nm or "jaw" in nm or "brow" in nm:
                 continue
             hasTrans = 1 if s["keyType"] in (2, 4) else 0
             stride = STRIDE[s["keyType"]]; comp = s["keyType"] in (3, 4)
             kf = s["kf"]; keys = []
             for fi in range(s["numFrames"]):
                 base = fi*stride
                 if comp:
                     q = struct.unpack_from("<4h", kf, base)
                     t16 = struct.unpack_from("<h", kf, base+8)[0]
                     tr = struct.unpack_from("<3h", kf, base+10) if hasTrans else (0, 0, 0)
                 else:
                     qf = struct.unpack_from("<4f", kf, base)
                     q = tuple(int(round(c*4096.0)) for c in qf)
                     t16 = int(round(struct.unpack_from("<f", kf, base+16)[0]*60.0))
                     tr = tuple(int(round(c*1024.0)) for c in struct.unpack_from("<3f", kf, base+20)) if hasTrans else (0, 0, 0)
                 keys.append((q, t16, tr))
                 maxtime = max(maxtime, t16/60.0)   # t16 is ABSOLUTE time; dur = last/max key
             tracks.append({"bone": bi, "hasTrans": hasTrans, "keys": keys})
         out_clips.append({"name": cname, "dur": maxtime, "tracks": tracks})

    # emit
    buf = bytearray()
    # upMode: 0 = already Z-up / no stand-up (CJ player.img components);
    #         1 = X-up authored -> runtime stands it via (x,y,z)->(-z,y,x) (gta3.img peds).
    # Cutscene actors (cutscene.img DFF) are authored Z-up like the player model, NOT X-up like
    # streamed gta3.img peds - measured from the baked invBind rest positions: the tallest bind
    # extent is along mesh +Z (~1.88m = human height), so upMode 0 (identity) stands them. upMode 1
    # mapped meshX->up and laid the actor flat on the ground. This is the "export rotation" bug.
    up_mode = 0 if (name == "cj" or cut) else 1
    buf += b"HRO2"   # HRO2 adds a per-vertex normal (f32[3]) after the bone weights
    buf += struct.pack("<HHHHHH", numBones, len(out_clips), nvert, len(sub_out), len(tex_list), up_mode)
    for b in bones:
        buf += struct.pack("<hh", b["parent"], b["nodeId"])
        buf += struct.pack("<4f", *b["q"]); buf += struct.pack("<3f", *b["p"]); buf += struct.pack("<16f", *b["inv"])
    for (pos, uv, color, bi, bw, nrm) in GV:
        buf += struct.pack("<5f", pos[0], pos[1], pos[2], uv[0], uv[1])
        buf += struct.pack("<I", color & 0xFFFFFFFF)
        buf += struct.pack("<4B", *(min(int(x2), numBones-1) for x2 in bi))
        buf += struct.pack("<4f", *bw)
        buf += struct.pack("<3f", nrm[0], nrm[1], nrm[2])
    for (ti, first, cnt) in sub_out:
        buf += struct.pack("<hHII", ti, 0, first, cnt)
    for gi in idx_pool:
        buf += struct.pack("<H", gi)
    for t in tex_list:
        nl = t["num_levels"] | (t.get("alpha_mode", 0) << 8)
        texel = t["texel_bytes"]; clut = t["clut_bytes"]
        buf += struct.pack("<HHHH", t["width"], t["height"], nl, t["clut_entries"])
        buf += struct.pack("<II", len(texel), len(clut))
        buf += texel + clut
    for c in out_clips:
        nm = c["name"].encode("ascii")[:23]; nm += b"\x00"*(24-len(nm))
        buf += nm
        buf += struct.pack("<fHH", c["dur"], len(c["tracks"]), 0)
        for t in c["tracks"]:
            buf += struct.pack("<hBBH", t["bone"], t["hasTrans"], 0, len(t["keys"]))
            for (q, tm, tr) in t["keys"]:
                buf += struct.pack("<4hh", q[0], q[1], q[2], q[3], tm)
                if t["hasTrans"]: buf += struct.pack("<3h", tr[0], tr[1], tr[2])

    # 'CLST' - GE-hardware-skinning clusters (<=8 bones each; u8 weights positional to
    # boneList /128). A SEPARATE section appended after the legacy pre-skin data so the
    # runtime can A/B (SKIN_HW flag). vert = f32 pos[3], f32 uv[2], u32 rgba, u8 w[8], f32 nrm[3].
    if emit_clst:
        buf += b"CLST" + struct.pack("<HH", len(clusters), 0)
        for cl in clusters:
            buf += struct.pack("<hHHH", cl["tex"], len(cl["bones"]), len(cl["verts"]), len(cl["idx"]))
            for b in cl["bones"]:
                buf += struct.pack("<H", b)
            for (pos, uv, color, wq, nrm, _gvi) in cl["verts"]:
                buf += struct.pack("<5f", pos[0], pos[1], pos[2], uv[0], uv[1])
                buf += struct.pack("<I", color & 0xFFFFFFFF)
                buf += struct.pack("<8B", *(max(0, min(255, w)) for w in wq))
                buf += struct.pack("<3f", nrm[0], nrm[1], nrm[2])
            for gi in cl["idx"]:
                buf += struct.pack("<H", gi)

    # 'MORF' - fat/muscle body-morph deltas (gameplay CJ only): per vertex, the position offset from
    # the Normal base to the Fat and Ripped clumps. The debug Player sliders blend these (Normal +
    # wFat*dFat + wMuscle*dRip) and re-skin on the CPU path. Absent -> no sliders. Appended last.
    if want_morph and GV_morph:
        buf += b"MORF" + struct.pack("<I", len(GV_morph))
        for (df, dr) in GV_morph:
            buf += struct.pack("<6f", df[0], df[1], df[2], dr[0], dr[1], dr[2])
        _mx = max((abs(c) for (df, dr) in GV_morph for t in (df, dr) for c in t), default=0.0)
        print("  MORF: %d verts, max|delta|=%.3f m" % (len(GV_morph), _mx))

    CLIPS = saved_clips
    print("=== baked: %s  bones=%d verts=%d sub=%d tex=%d clips=%d  %d bytes ==="
          % (name, numBones, nvert, len(sub_out), len(tex_list), len(out_clips), len(buf)))
    print("  indices=%d  submeshes(tex,first,count):" % len(idx_pool))
    for (ti, fst, cnt) in sub_out:
        print("    tex=%d first=%d count=%d" % (ti, fst, cnt))
    for c in out_clips:
        print("  clip %-14s dur=%.2fs tracks=%d" % (c["name"], c["dur"], len(c["tracks"])))
    return buf


def main():
    # DEFAULT = cj: this tool bakes the PLAYER hero. A bare `hero_bake.py` used to
    # default to "fam1" (a Families gang ped, green hoodie) -> re-running it without the
    # "cj" arg silently replaced CJ with fam1 (b309 regression). Default to cj so that
    # can't happen; pass another id explicitly to bake a different ped.
    arg = sys.argv[1] if len(sys.argv) > 1 else "cj"
    # argv[2] = explicit output path (Quarry passes <OutDir>/peds/hero.bin). When
    # given we write ONLY there and skip the dev-loop memstick mirror - the converter
    # must not touch a live install.
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    buf = bake_model(arg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(buf)
    if len(sys.argv) > 2:
        print("hero.bin written: %d bytes -> %s" % (len(buf), out))
    else:
        try: dep = "deployed x%d" % deploy_util.mirror(DEPLOY, buf)
        except OSError: dep = "(no deploy)"
        print("hero.bin written: %d bytes %s" % (len(buf), dep))


if __name__ == "__main__":
    main()
