"""the source game DFF model (RenderWare RpClump) decoder.

Parses the PC platform-independent variant (RW 3.6.0.3, libid 0x1803FFFF) that the
the source game readers consume: the clump tree (frame/bone hierarchy + geometry list +
atomics + materials + texture-name refs) plus the binMesh render index buffers.

What it decodes for v1:
 * FRAMELIST -> frames (name, parent, 3x3 rotation, position, HAnim node id).
 * GEOMETRYLIST -> geometries: format, vertices, per-set UVs, prelit RGBA, normals,
 triangles (with material id), materials (modulate color + texture/mask names),
 binMesh splits (render-ready index batches, trilist or tristrip).
 * SKIN 0x116 -> per-vertex bone indices/weights + inverse bind (skin-to-bone)
 matrices on the geometry (geo.skin), for CPU bind-pose skinning of peds.
 * HANIM 0x11E -> frame.hanim_id per bone frame plus the hierarchy-root node
 array (dff.hanim_nodes) whose order IS the skin bone-index order.
 * ATOMIC -> (frameIndex, geometryIndex) bindings.
 * Export -> to_mesh_json() / to_gltf() (positions + UVs + de-stripped indices)
 for a Three.js viewer.

Deferred for v1 (noted, parsed-around, do not break the clump): MATFX 0x120,
2dfx 0x253F2F8, night-vertex-colors 0x253F2F9, embedded COLLISION, morph targets
beyond the base (morph 0). The PS2-native geometry path (format bit24 NATIVE) is
a different decode and is rejected here.

 (confirmed, byte-validated vs gta3.img
helipad/barrel1). Cross-checked against librw-master/src/clump.cpp, geometry.cpp,
geoplg.cpp (mirrored parse logic).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core import rwstream as rw
from formats import dff_ps2

# --- chunk ids in the SA developer range (not in core.rwstream registry) ---
_FRAME_NAME = 0x0253F2FE
_NIGHT_COLORS = 0x0253F2F9

# --- geometry format flag bits ---
F_TRISTRIP = 0x00000001
F_POSITIONS = 0x00000002
F_TEXTURED = 0x00000004
F_PRELIT = 0x00000008
F_NORMALS = 0x00000010
F_LIGHT = 0x00000020
F_MODULATE = 0x00000040
F_TEXTURED2 = 0x00000080
F_NATIVE = 0x01000000

_U32 = struct.Struct("<I")
_4U32 = struct.Struct("<4I")
_2U32 = struct.Struct("<2I")
_FRAME = struct.Struct("<9f3f i I")  # right, up, at, pos, parent, flags
_3F = struct.Struct("<3f")
_2F = struct.Struct("<2f")
_4F = struct.Struct("<4f")


# =========================================================================
# data model
# =========================================================================

@dataclass
class Frame:
    name: str
    parent: int
    rotation: List[List[float]]   # 3x3: rows = right, up, at axis vectors
    position: List[float]         # x, y, z
    hanim_id: Optional[int] = None  # HAnim (0x11E) node id, None if not a bone


@dataclass
class Material:
    color: tuple              # (r, g, b, a) bytes
    textured: int
    texture_name: str = ""
    mask_name: str = ""
    ambient: float = 1.0
    specular: float = 1.0
    diffuse: float = 1.0
    addr_u: int = 1           # RW texture addressing: 1 wrap, 2 mirror,
    addr_v: int = 1           # 3 clamp, 4 border (from filterAddressing)
    matfx: Optional[dict] = None  # MatFX 0x0120: {"effect": int, "bump_coef"/"env_coef"/... }


@dataclass
class Geometry:
    format: int
    num_triangles: int
    num_vertices: int
    num_morph_targets: int
    vertices: List[tuple] = field(default_factory=list)          # [(x,y,z)]
    uvs: List[List[tuple]] = field(default_factory=list)         # [set][ (u,v) ]
    prelit_colors: Optional[List[tuple]] = None                  # [(r,g,b,a)] or None
    normals: Optional[List[tuple]] = None                        # [(x,y,z)] or None
    triangles: List[tuple] = field(default_factory=list)         # [(a,b,c,matId)]
    materials: List[Material] = field(default_factory=list)
    splits: List[dict] = field(default_factory=list)             # binMesh batches
    bounding_sphere: Optional[tuple] = None                      # (cx,cy,cz,r)
    skin: Optional[dict] = None                                  # SKIN 0x116 (see _parse_skin)
    effects_2d: List[dict] = field(default_factory=list)         # 2dfx 0x0253F2F8 entries
    night_colors: Optional[List[tuple]] = None                   # 0x0253F2F9 [(r,g,b,a)] or None


@dataclass
class Atomic:
    frame_index: int
    geometry_index: int
    flags: int = 0


@dataclass
class Dff:
    frames: List[Frame] = field(default_factory=list)
    geometries: List[Geometry] = field(default_factory=list)
    atomics: List[Atomic] = field(default_factory=list)
    version: int = 0
    # HAnim hierarchy-root node ids, in node order: skin bone index i refers to
    # the frame whose hanim_id == hanim_nodes[i]. Empty for non-skinned clumps.
    hanim_nodes: List[int] = field(default_factory=list)


# =========================================================================
# helpers
# =========================================================================

def _ver(lib_id: int) -> int:
    return rw.unpack_version(lib_id)[0]


def _read_string(buf, ch: rw.ChunkHeader) -> str:
    """Decode a STRING (0x02) or UNICODESTRING (0x13) chunk body. SA uses 0x02."""
    raw = buf[ch.body_offset:ch.end]
    if ch.type == rw.UNICODE_STRING:
        raw = raw[0::2]  # keep low byte of each u16
    return raw.split(b"\x00", 1)[0].decode("latin-1")


def _num_tex_sets(fmt: int) -> int:
    n = (fmt >> 16) & 0xFF
    if n:
        return n
    if fmt & F_TEXTURED2:
        return 2
    if fmt & F_TEXTURED:
        return 1
    return 0


# =========================================================================
# top level
# =========================================================================

def parse_dff(data: bytes) -> Dff:
    root = rw.read_header(data, 0)
    if root.type != rw.CLUMP:
        found = rw.find_chunk(data, rw.CLUMP, 0, len(data))
        if not found:
            raise ValueError("no CLUMP chunk")
        root = found

    dff = Dff(version=_ver(root.lib_id))
    body, end = root.body_offset, root.end

    # --- clump STRUCT: {numAtomics, numLights, numCameras} (12B if ver>0x33000) ---
    st = rw.find_chunk(data, rw.STRUCT, body, end)
    if st is None:
        raise ValueError("clump has no STRUCT")
    num_atomics = _U32.unpack_from(data, st.body_offset)[0]
    clump_ver = _ver(st.lib_id)
    num_lights = _U32.unpack_from(data, st.body_offset + 4)[0] if (clump_ver > 0x33000 and st.size >= 8) else 0
    num_cameras = _U32.unpack_from(data, st.body_offset + 8)[0] if (clump_ver > 0x33000 and st.size >= 12) else 0

    # --- FRAMELIST ---
    fl = rw.find_chunk(data, rw.FRAME_LIST, st.end, end)
    if fl is not None:
        dff.frames, dff.hanim_nodes = _parse_framelist(data, fl)

    # --- GEOMETRYLIST (ver >= 0x30400) ---
    gl = rw.find_chunk(data, rw.GEOMETRY_LIST, st.end, end)
    geom_end = st.end
    if gl is not None:
        gst = rw.find_chunk(data, rw.STRUCT, gl.body_offset, gl.end)
        num_geoms = _U32.unpack_from(data, gst.body_offset)[0]
        cursor = gst.end
        for _ in range(num_geoms):
            gc = rw.find_chunk(data, rw.GEOMETRY, cursor, gl.end)
            if gc is None:
                break
            try:
                dff.geometries.append(_parse_geometry(data, gc))
            except Exception as e:  # one bad geometry must not kill the clump
                dff.geometries.append(Geometry(0, 0, 0, 0))
                dff.geometries[-1].error = f"{e}"  # type: ignore[attr-defined]
            cursor = gc.end
        geom_end = gl.end

    # --- ATOMICS (after the geometry list) ---
    cursor = geom_end
    for _ in range(num_atomics):
        ac = rw.find_chunk(data, rw.ATOMIC, cursor, end)
        if ac is None:
            break
        try:
            dff.atomics.append(_parse_atomic(data, ac))
        except Exception:
            pass
        cursor = ac.end

    return dff


# =========================================================================
# framelist
# =========================================================================

def _parse_framelist(data: bytes, fl: rw.ChunkHeader) -> tuple:
    """Returns (frames, hanim_nodes) - the node-id array from the hierarchy root."""
    fst = rw.find_chunk(data, rw.STRUCT, fl.body_offset, fl.end)
    if fst is None:
        return [], []
    num_frames = _U32.unpack_from(data, fst.body_offset)[0]

    frames: List[Frame] = []
    off = fst.body_offset + 4
    for _ in range(num_frames):
        rx, ry, rz, ux, uy, uz, ax, ay, az, px, py, pz, parent, _flags = _FRAME.unpack_from(data, off)
        frames.append(Frame(
            name="",
            parent=parent,
            rotation=[[rx, ry, rz], [ux, uy, uz], [ax, ay, az]],
            position=[px, py, pz],
        ))
        off += _FRAME.size

    # per-frame EXTENSION blocks (in order) carry FRAME_NAME / HANIM
    hanim_nodes: List[int] = []
    cursor = off
    for i in range(num_frames):
        ext = rw.find_chunk(data, rw.EXTENSION, cursor, fl.end)
        if ext is None:
            break
        nm = rw.find_chunk(data, _FRAME_NAME, ext.body_offset, ext.end)
        if nm is not None:
            frames[i].name = data[nm.body_offset:nm.end].split(b"\x00", 1)[0].decode("latin-1")
        ha = rw.find_chunk(data, rw.HANIM_PLG, ext.body_offset, ext.end)
        if ha is not None:
            try:
                node_id, node_ids = _parse_hanim(data, ha)
                frames[i].hanim_id = node_id
                if node_ids and not hanim_nodes:   # the hierarchy root carries the array
                    hanim_nodes = node_ids
            except Exception:
                pass  # a bad HAnim block must not kill the framelist
        cursor = ext.end
    return frames, hanim_nodes


def _parse_hanim(data: bytes, ha: rw.ChunkHeader) -> tuple:
    """Decode a HANIM_PLG (0x011E) frame-extension body. Mirrors librw readHAnim:

 i32 hAnimVersion (0x100), i32 nodeId, i32 numNodes
 if numNodes: (only on the hierarchy ROOT bone frame)
 i32 flags, i32 maxInterpKeyFrameSize
 numNodes x { i32 nodeId, i32 nodeIndex, i32 nodeFlags }

 Returns (nodeId, [nodeId,...]) - the array position IS the skin bone index
 (nodeIndex is written sequentially and unused by librw on read).
 """
    if ha.size < 12:
        raise ValueError("hanim chunk too small")
    ver, node_id, num_nodes = struct.unpack_from("<3i", data, ha.body_offset)
    if ver != 0x100:
        raise ValueError(f"unexpected hAnim version 0x{ver:X}")
    node_ids: List[int] = []
    if num_nodes:
        if ha.size < 12 + 8 + num_nodes * 12:
            raise ValueError("hanim node array truncated")
        p = ha.body_offset + 20            # skip i32 flags + i32 maxKeyFrameSize
        for _ in range(num_nodes):
            node_ids.append(struct.unpack_from("<i", data, p)[0])
            p += 12
    return node_id, node_ids


# =========================================================================
# geometry
# =========================================================================

def _parse_geometry(data: bytes, gc: rw.ChunkHeader) -> Geometry:
    geom_ver = _ver(gc.lib_id)
    gst = rw.find_chunk(data, rw.STRUCT, gc.body_offset, gc.end)
    if gst is None:
        raise ValueError("geometry has no STRUCT")

    fmt, n_tri, n_vert, n_morph = _4U32.unpack_from(data, gst.body_offset)
    geo = Geometry(fmt, n_tri, n_vert, n_morph)

    if fmt & F_NATIVE:
        # PS2/console-native geometry (rpGEOMETRYNATIVE 0x01000000): vertex/index data lives in a
        # nativeData plugin chunk (0x0510) as VU-ready DMA/VIF clusters, not the generic arrays below.
        # Layout:; decoder: formats/dff_ps2.py.
        gst = rw.find_chunk(data, rw.STRUCT, gc.body_offset, gc.end)  # (re-find for clarity)
        ml = rw.find_chunk(data, rw.MATERIAL_LIST, gst.end, gc.end)
        if ml is not None:
            geo.materials = _parse_materiallist(data, ml)
        ext_start = ml.end if ml is not None else gst.end
        ext = rw.find_chunk(data, rw.EXTENSION, ext_start, gc.end)
        if ext is not None:
            native = dff_ps2.parse_native_geometry(data, ext)
            if native is not None:
                verts, uv0, cols, tris, splits = native
                geo.vertices = verts
                geo.uvs = [uv0] if uv0 else []
                geo.prelit_colors = cols if cols else None
                geo.triangles = tris
                geo.splits = splits
                # native counts come from the decoded strips, not the (zeroed) native header
                geo.num_vertices = len(verts)
                geo.num_triangles = len(tris)
            fx = rw.find_chunk(data, rw.TWO_D_EFFECT, ext.body_offset, ext.end)
            if fx is not None:
                try:
                    geo.effects_2d = _parse_2dfx(data, fx)
                except Exception:
                    geo.effects_2d = []
        return geo

    p = gst.body_offset + 16
    # legacy surface props (ver < 0x34000): 3 f32 right after the 16B header
    if geom_ver < 0x34000:
        p += 12

    # 1. prelit RGBA
    if fmt & F_PRELIT:
        cols = []
        for _ in range(n_vert):
            cols.append((data[p], data[p + 1], data[p + 2], data[p + 3]))
            p += 4
        geo.prelit_colors = cols

    # 2. texcoords: numTexSets x numVertices x (u,v)
    n_sets = _num_tex_sets(fmt)
    for _s in range(n_sets):
        uv_set = []
        for _ in range(n_vert):
            u, v = _2F.unpack_from(data, p)
            uv_set.append((u, v))
            p += 8
        geo.uvs.append(uv_set)

    # 3. triangles: word-pair byteswap decode
    tris = []
    for _ in range(n_tri):
        t0, t1 = _2U32.unpack_from(data, p)
        p += 8
        v0 = t0 >> 16
        v1 = t0 & 0xFFFF
        v2 = t1 >> 16
        mat = t1 & 0xFFFF
        tris.append((v0, v1, v2, mat))
    geo.triangles = tris

    # 4. morph targets (decode base morph 0 positions/normals; skip extras' payloads)
    for m in range(n_morph):
        cx, cy, cz, radius = _4F.unpack_from(data, p)
        p += 16
        has_v, has_n = _2U32.unpack_from(data, p)
        p += 8
        verts = []
        if has_v:
            for _ in range(n_vert):
                x, y, z = _3F.unpack_from(data, p)
                verts.append((x, y, z))
                p += 12
        norms = []
        if has_n:
            for _ in range(n_vert):
                x, y, z = _3F.unpack_from(data, p)
                norms.append((x, y, z))
                p += 12
        if m == 0:
            geo.bounding_sphere = (cx, cy, cz, radius)
            if has_v:
                geo.vertices = verts
            if has_n:
                geo.normals = norms

    # --- MATERIALLIST ---
    ml = rw.find_chunk(data, rw.MATERIAL_LIST, gst.end, gc.end)
    if ml is not None:
        geo.materials = _parse_materiallist(data, ml)

    # --- geometry EXTENSION: binMesh + skin (2dfx/night deferred) ---
    ext_start = ml.end if ml is not None else gst.end
    ext = rw.find_chunk(data, rw.EXTENSION, ext_start, gc.end)
    if ext is not None:
        bm = rw.find_chunk(data, rw.BIN_MESH_PLG, ext.body_offset, ext.end)
        if bm is not None:
            try:
                geo.splits = _parse_binmesh(data, bm)
            except Exception:
                geo.splits = []
        sk = rw.find_chunk(data, rw.SKIN_PLG, ext.body_offset, ext.end)
        if sk is not None:
            try:
                geo.skin = _parse_skin(data, sk, n_vert)
            except Exception:
                geo.skin = None  # corrupt skin must not kill the geometry
        fx = rw.find_chunk(data, rw.TWO_D_EFFECT, ext.body_offset, ext.end)
        if fx is not None:
            try:
                geo.effects_2d = _parse_2dfx(data, fx)
            except Exception:
                geo.effects_2d = []
        nc = rw.find_chunk(data, _NIGHT_COLORS, ext.body_offset, ext.end)
        if nc is not None:
            try:
                # u32 magic (0 = no night set) + RwRGBA[n_vert]
                magic = struct.unpack_from("<I", data, nc.body_offset)[0]
                if magic != 0 and nc.size >= 4 + n_vert * 4:
                    o = nc.body_offset + 4
                    geo.night_colors = [tuple(data[o + i*4: o + i*4 + 4])
                                        for i in range(n_vert)]
            except Exception:
                geo.night_colors = None

    return geo


def _parse_skin(data: bytes, sk: rw.ChunkHeader, n_vert: int) -> Optional[dict]:
    """Decode a SKIN_PLG (0x0116) geometry-extension body (PC, non-native).

 Mirrors librw readSkin (skin.cpp):
 u8 numBones, u8 numUsedBones, u8 maxWeightsPerVertex, u8 pad
 numUsedBones x u8 usedBoneIds (new format only)
 numVertices x 4 u8 vertex bone indices (into the HAnim node array)
 numVertices x 4 f32 vertex weights
 numBones x 16 f32 inverse bind (skin-to-bone) matrices, RW Matrix
 order: right(3)+flags, up(3)+pad, at(3)+pad,
 pos(3)+pad - words 3/7/11/15 are garbage, ignore.
 i32 boneLimit, i32 numMeshes, i32 rleSize [+ split payload] (new format)

 numUsedBones == 0 marks the pre-3.4 legacy layout: no usedBones table, no
 split data, and each matrix is prefixed by a u32 0xdeaddead marker.
 """
    if n_vert <= 0:
        return None
    num_bones, num_used, max_weights, _pad = struct.unpack_from("<4B", data, sk.body_offset)
    if num_bones == 0:
        return None
    old_format = num_used == 0
    need = 4 + (0 if old_format else num_used) + n_vert * (4 + 16) \
        + num_bones * (68 if old_format else 64)
    if sk.size < need:
        raise ValueError(f"skin plugin truncated: size {sk.size} < {need}")

    p = sk.body_offset + 4
    used_bones: List[int] = []
    if not old_format:
        used_bones = list(data[p:p + num_used])
        p += num_used

    bone_indices = [[data[q], data[q + 1], data[q + 2], data[q + 3]]
                    for q in range(p, p + n_vert * 4, 4)]
    p += n_vert * 4

    weights = [list(_4F.unpack_from(data, q)) for q in range(p, p + n_vert * 16, 16)]
    p += n_vert * 16

    inverse_bind: List[List[float]] = []
    for _ in range(num_bones):
        if old_format:
            p += 4  # 0xdeaddead marker word
        inverse_bind.append(list(struct.unpack_from("<16f", data, p)))
        p += 64

    return {"num_bones": num_bones, "num_used_bones": num_used,
            "max_weights": max_weights, "used_bones": used_bones,
            "bone_indices": bone_indices, "weights": weights,
            "inverse_bind": inverse_bind}


def _parse_materiallist(data: bytes, ml: rw.ChunkHeader) -> List[Material]:
    mst = rw.find_chunk(data, rw.STRUCT, ml.body_offset, ml.end)
    if mst is None:
        return []
    num_mat = _U32.unpack_from(data, mst.body_offset)[0]
    index_table = [
        struct.unpack_from("<i", data, mst.body_offset + 4 + i * 4)[0]
        for i in range(num_mat)
    ]

    parsed: List[Material] = []   # the inline-read materials, in stream order
    out: List[Material] = []      # one entry per index-table slot (resolves refs)
    cursor = mst.end
    for idx in index_table:
        if idx >= 0:
            # reuse an already-read material
            out.append(parsed[idx] if idx < len(parsed) else Material((255, 255, 255, 255), 0))
            continue
        mc = rw.find_chunk(data, rw.MATERIAL, cursor, ml.end)
        if mc is None:
            out.append(Material((255, 255, 255, 255), 0))
            continue
        mat = _parse_material(data, mc)
        parsed.append(mat)
        out.append(mat)
        cursor = mc.end
    return out


def _parse_material(data: bytes, mc: rw.ChunkHeader) -> Material:
    mst = rw.find_chunk(data, rw.STRUCT, mc.body_offset, mc.end)
    flags, r, g, b, a, _unused, textured = struct.unpack_from("<I4BII", data, mst.body_offset)
    amb = spec = diff = 1.0
    if mst.size >= 28:
        amb, spec, diff = _3F.unpack_from(data, mst.body_offset + 16)

    mat = Material((r, g, b, a), textured, ambient=amb, specular=spec, diffuse=diff)

    if textured:
        tc = rw.find_chunk(data, rw.TEXTURE, mst.end, mc.end)
        if tc is not None:
            tst = rw.find_chunk(data, rw.STRUCT, tc.body_offset, tc.end)
            if tst is not None and tst.size >= 4:
                # RW rwTEXTURE struct word: bits 0-7 filter, 8-11 addressU,
                # 12-15 addressV (1 wrap / 2 mirror / 3 clamp / 4 border).
                fa = _U32.unpack_from(data, tst.body_offset)[0]
                au = (fa >> 8) & 0xF
                av = (fa >> 12) & 0xF
                mat.addr_u = au if au else 1
                mat.addr_v = av if av else 1
            name_start = tst.end if tst is not None else tc.body_offset
            names = []
            for s in rw.iter_chunks(data, name_start, tc.end):
                if s.type in (rw.STRING, rw.UNICODE_STRING):
                    names.append(_read_string(data, s))
                if len(names) == 2:
                    break
            if names:
                mat.texture_name = names[0]
            if len(names) > 1:
                mat.mask_name = names[1]

    # material EXTENSION: MatFX 0x0120 (env/bump/dual)
    ext = rw.find_chunk(data, rw.EXTENSION, mst.end, mc.end)
    if ext is not None:
        mfx = rw.find_chunk(data, rw.MATERIAL_EFFECTS_PLG, ext.body_offset, ext.end)
        if mfx is not None and mfx.size >= 4:
            try:
                mat.matfx = _parse_matfx(data, mfx)
            except Exception:
                mat.matfx = None  # a bad matfx block must not kill the material
    return mat


# MatFX effect ids as stored in the material extension chunk.
_MATFX_NAMES = {0: "none", 1: "bump", 2: "env", 3: "bumpenv", 4: "dual",
                5: "uvtransform", 6: "dualuvtransform"}


def _parse_matfx(data: bytes, mfx: rw.ChunkHeader) -> dict:
    """MatFX 0x0120 material extension. Body starts with u32 effectType; the effect
 coefficient(s) follow per RpMatFXMaterialFlags. We record the effect id/name and
 the leading coefficient (best-effort; full env/bump/dual slot decode is optional)."""
    effect = _U32.unpack_from(data, mfx.body_offset)[0]
    out = {"effect": effect, "effect_name": _MATFX_NAMES.get(effect, f"0x{effect:X}")}
    # For env/bump the next dword is typically a second type tag then an f32 coefficient;
    # expose the raw tail so downstream can refine without re-reading the stream.
    out["raw"] = data[mfx.body_offset:mfx.end]
    return out


def _parse_2dfx(data: bytes, fx: rw.ChunkHeader) -> List[dict]:
    """2dfx 0x0253F2F8 geometry extension: light/particle/effect placement.

 Body: u32 count, then per entry {f32 x,y,z; u32 entry_type; u32 data_size; data[]}.
 entry_type: 0=light/corona, 1=particle, 3=ped-attractor, 6=enter/exit, 7=sun-glare,
 8=escalator (SA set). We keep position + type + raw data (order-independent of verts)."""
    count = _U32.unpack_from(data, fx.body_offset)[0]
    entries: List[dict] = []
    p = fx.body_offset + 4
    for _ in range(count):
        if p + 20 > fx.end:
            break
        x, y, z = _3F.unpack_from(data, p)
        entry_type, data_size = struct.unpack_from("<II", data, p + 12)
        p += 20
        blob = data[p:p + data_size] if p + data_size <= fx.end else b""
        entries.append({"pos": (x, y, z), "type": entry_type, "data": blob})
        p += data_size
    return entries


def _parse_binmesh(data: bytes, bm: rw.ChunkHeader) -> List[dict]:
    flags, num_meshes, _total = struct.unpack_from("<3I", data, bm.body_offset)
    strip = bool(flags & 1)
    splits: List[dict] = []
    p = bm.body_offset + 12
    for _ in range(num_meshes):
        num_idx, mat_index = struct.unpack_from("<Ii", data, p)
        p += 8
        idx = list(struct.unpack_from("<%dI" % num_idx, data, p))
        p += num_idx * 4
        splits.append({"mat_index": mat_index, "indices": idx, "strip": strip})
    return splits


def _parse_atomic(data: bytes, ac: rw.ChunkHeader) -> Atomic:
    ast = rw.find_chunk(data, rw.STRUCT, ac.body_offset, ac.end)
    frame_idx, geom_idx = _2U32.unpack_from(data, ast.body_offset)
    flags = _U32.unpack_from(data, ast.body_offset + 8)[0] if ast.size >= 12 else 0
    return Atomic(frame_idx, geom_idx, flags)


# =========================================================================
# export
# =========================================================================

def _destrip(indices: List[int]) -> List[int]:
    """Convert a triangle strip to a flat triangle list (skip degenerates)."""
    out: List[int] = []
    for i in range(len(indices) - 2):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        if a == b or b == c or a == c:
            continue
        if i & 1:
            out.extend((a, c, b))
        else:
            out.extend((a, b, c))
    return out


def _flat_indices(geo: Geometry) -> List[int]:
    """Render-ready flat triangle indices: from binMesh splits if present, else triangles."""
    if geo.splits:
        out: List[int] = []
        for s in geo.splits:
            out.extend(_destrip(s["indices"]) if s["strip"] else list(s["indices"]))
        return out
    flat: List[int] = []
    for a, b, c, _mat in geo.triangles:
        flat.extend((a, b, c))
    return flat


def to_mesh_json(dff: Dff, geom_index: int = 0) -> dict:
    """Flat buffers a Three.js BufferGeometry can consume (first geometry by default)."""
    if not dff.geometries:
        return {"positions": [], "uvs": [], "normals": [], "indices": [], "texture_names": []}
    geo = dff.geometries[geom_index]

    positions: List[float] = []
    for x, y, z in geo.vertices:
        positions.extend((x, y, z))

    uvs: List[float] = []
    if geo.uvs:
        for u, v in geo.uvs[0]:
            uvs.extend((u, v))

    normals: List[float] = []
    if geo.normals:
        for x, y, z in geo.normals:
            normals.extend((x, y, z))

    return {
        "positions": positions,
        "uvs": uvs,
        "normals": normals,
        "indices": _flat_indices(geo),
        "texture_names": [m.texture_name for m in geo.materials],
    }


def to_gltf(dff: Dff, geom_index: int = 0) -> dict:
    """Minimal glTF 2.0 (first geometry): POSITION (+ TEXCOORD_0) + indices, one mesh.

 Single embedded binary buffer (data URI), interleaved-free: positions, then uvs,
 then indices, each in its own bufferView.
 """
    mj = to_mesh_json(dff, geom_index)
    pos = mj["positions"]
    uvs = mj["uvs"]
    idx = mj["indices"]
    n_vert = len(pos) // 3

    import base64

    blob = bytearray()
    buffer_views = []
    accessors = []

    # POSITION
    pos_off = len(blob)
    blob += struct.pack("<%df" % len(pos), *pos) if pos else b""
    # min/max for POSITION (required by glTF)
    if n_vert:
        xs = pos[0::3]; ys = pos[1::3]; zs = pos[2::3]
        pmin = [min(xs), min(ys), min(zs)]
        pmax = [max(xs), max(ys), max(zs)]
    else:
        pmin = pmax = [0.0, 0.0, 0.0]
    buffer_views.append({"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos) * 4, "target": 34962})
    accessors.append({
        "bufferView": 0, "componentType": 5126, "count": n_vert,
        "type": "VEC3", "min": pmin, "max": pmax,
    })
    pos_accessor = 0

    attributes = {"POSITION": pos_accessor}

    # TEXCOORD_0 (optional)
    if uvs:
        uv_off = len(blob)
        blob += struct.pack("<%df" % len(uvs), *uvs)
        bv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": uv_off, "byteLength": len(uvs) * 4, "target": 34962})
        accessors.append({"bufferView": bv, "componentType": 5126, "count": len(uvs) // 2, "type": "VEC2"})
        attributes["TEXCOORD_0"] = len(accessors) - 1

    # indices (u32)
    idx_off = len(blob)
    # pad position/uv region to 4 bytes (already f32-aligned); indices are u32-aligned
    blob += struct.pack("<%dI" % len(idx), *idx) if idx else b""
    bv = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx) * 4, "target": 34963})
    accessors.append({"bufferView": bv, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
    idx_accessor = len(accessors) - 1

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")

    return {
        "asset": {"version": "2.0", "generator": "SAW dff.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "indices": idx_accessor,
                "mode": 4,  # TRIANGLES
            }]
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob), "uri": uri}],
    }
