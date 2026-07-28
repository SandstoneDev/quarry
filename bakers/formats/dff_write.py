"""the source game DFF model (RenderWare RpClump) *writer* + OBJ->DFF importer.

The inverse of formats/dff.py: serialize a `DFF.Dff` dataclass back into a valid
RenderWare clump byte stream that `DFF.parse_dff` reads back equivalently, and turn
a Wavefront OBJ into one such clump.

Target format is the SA PC platform-independent variant the readers consume:
RW 3.6.0.3, libraryID 0x1803FFFF (`rwstream.pack_version(0x36003)`), the same id
seen on every model in gta3.img. Every chunk is written with a correct 12-byte
header `{u32 type, u32 size, u32 libraryID}` whose size is measured from the
children buffer, mirroring librw `writeChunkHeader` / `clump.cpp` / `geometry.cpp`.

LOSSY - this is a *geometry-only* writer. It reproduces the clump skeleton
(FrameList + GeometryList + Atomics), per-geometry positions / UVs / prelit RGBA /
normals / triangles / bounding sphere, the MaterialList (colour + texture-name +
ambient/specular/diffuse), the binMesh render-split plugin, and the SKIN plugin
(bone weights/indices/inverse-bind, when the geometry is skinned). It deliberately
DROPS the remaining geometry / atomic extensions the reader parses around:

 * HANIM_PLG 0x011E (animation node table) -> only frame names kept
 (reader is lossy on per-node flags/root)
 * MATERIAL_EFFECTS 0x0120(env / bump map fx) -> dropped
 * 2dfx 0x0253F2F8, night-colours 0x0253F2F9 -> dropped
 * RIGHT_TO_RENDER 0x1F / 0x0253F2FD, embedded COL -> dropped
 * morph targets beyond base (morph 0) -> single morph written

so a rebuilt SA prop/weapon comes out as a static, un-rigged, single-morph mesh.
Round-trips are therefore SEMANTIC (re-parse equivalent), not byte-identical.

Reference: librw-master/src/clump.cpp, geometry.cpp, plg.cpp; formats/dff.py.
"""
from __future__ import annotations

import struct
from typing import List, Optional

from core import rwstream as rw
from formats import dff as DFF
from formats.dff import (
    F_TRISTRIP, F_POSITIONS, F_TEXTURED, F_PRELIT, F_NORMALS, F_LIGHT,
    F_MODULATE, F_TEXTURED2, F_NATIVE, _FRAME_NAME,
)

# canonical SA retail D3D9 library id (RW 3.6.0.3) used for every chunk we emit
SA_VERSION = 0x36003
SA_LIB_ID = rw.pack_version(SA_VERSION)  # == 0x1803FFFF

_U32 = struct.Struct("<I")


# =========================================================================
# low-level chunk helpers
# =========================================================================

def _chunk(type_: int, body: bytes, lib_id: int = SA_LIB_ID) -> bytes:
    """A complete RW chunk: 12-byte header {type, size, lib_id} + body."""
    return struct.pack("<III", type_, len(body), lib_id) + body


def _string_chunk(s: str) -> bytes:
    """A STRING (0x02) chunk: latin-1 bytes, NUL-terminated, padded to 4 bytes.

 librw rounds the stored length up to a multiple of 4 (with at least one NUL).
 """
    raw = s.encode("latin-1", "replace") + b"\x00"
    pad = (-len(raw)) % 4
    raw += b"\x00" * pad
    return _chunk(rw.STRING, raw)


def _empty_extension() -> bytes:
    return _chunk(rw.EXTENSION, b"")


# =========================================================================
# build_dff (Dff dataclass -> clump bytes)
# =========================================================================

def build_dff(dff: "DFF.Dff") -> bytes:
    """Serialize a `DFF.Dff` into a RenderWare clump byte stream.

 Structure (all chunks tagged 0x1803FFFF):
 CLUMP
 STRUCT { numAtomics, numLights=0, numCameras=0 }
 FRAMELIST { STRUCT(count + per-frame matrix/pos/parent/flags) + Ext each }
 GEOMETRYLIST { STRUCT(count) + GEOMETRY... }
 ATOMIC... { STRUCT(frameIdx, geomIdx, flags, 0) + Ext }
 """
    n_atomics = len(dff.atomics)

    # --- CLUMP STRUCT (12 bytes for the 0x36003 layout) ---
    clump_struct = _chunk(rw.STRUCT, struct.pack("<III", n_atomics, 0, 0))

    parts: List[bytes] = [clump_struct]
    parts.append(_build_framelist(dff.frames))
    parts.append(_build_geometrylist(dff.geometries))
    for atom in dff.atomics:
        parts.append(_build_atomic(atom))

    return _chunk(rw.CLUMP, b"".join(parts))


def _build_framelist(frames: List["DFF.Frame"]) -> bytes:
    """FRAMELIST: STRUCT(count + frame records) then one EXTENSION per frame.

 Frame record = 3x3 rotation (right/up/at rows) + position + parentIndex + flags,
 matching the reader's `<9f3f i I`. A frame name (when present) goes in its
 EXTENSION as a 0x0253F2FE chunk.
 """
    n = len(frames)
    body = bytearray(_U32.pack(n))
    for f in frames:
        rot = f.rotation
        body += struct.pack(
            "<9f3f i I",
            rot[0][0], rot[0][1], rot[0][2],
            rot[1][0], rot[1][1], rot[1][2],
            rot[2][0], rot[2][1], rot[2][2],
            f.position[0], f.position[1], f.position[2],
            f.parent, 0,
        )
    struct_chunk = _chunk(rw.STRUCT, bytes(body))

    ext_chunks = bytearray()
    for f in frames:
        inner = b""
        if f.name:
            raw = f.name.encode("latin-1", "replace")
            inner = _chunk(_FRAME_NAME, raw)
        ext_chunks += _chunk(rw.EXTENSION, inner)

    return _chunk(rw.FRAME_LIST, struct_chunk + bytes(ext_chunks))


def _build_atomic(atom: "DFF.Atomic") -> bytes:
    """ATOMIC: STRUCT { frameIndex, geometryIndex, flags, 0 } + empty EXTENSION."""
    st = _chunk(rw.STRUCT, struct.pack("<IIII",
                                       atom.frame_index, atom.geometry_index,
                                       atom.flags, 0))
    return _chunk(rw.ATOMIC, st + _empty_extension())


def _build_geometrylist(geometries: List["DFF.Geometry"]) -> bytes:
    st = _chunk(rw.STRUCT, _U32.pack(len(geometries)))
    body = bytearray(st)
    for g in geometries:
        body += _build_geometry(g)
    return _chunk(rw.GEOMETRY_LIST, bytes(body))


# =========================================================================
# geometry serialization
# =========================================================================

def _normalize_format(g: "DFF.Geometry") -> int:
    """Recompute the format flags from what the geometry actually carries.

 Keeps the low render flags (tristrip/light/modulate) the source had, but makes
 the data-presence bits (positions/textured/prelit/normals) and the tex-set
 count honest about what we are about to write, so the reader walks our buffer
 correctly. NATIVE is always cleared (we only emit the generic path).
 """
    fmt = g.format & (F_TRISTRIP | F_LIGHT | F_MODULATE)
    fmt &= ~F_NATIVE

    if g.vertices:
        fmt |= F_POSITIONS
    if g.prelit_colors is not None:
        fmt |= F_PRELIT
    if g.normals is not None:
        fmt |= F_NORMALS

    # tex-set count is exactly the UV sets we are going to write
    n_sets = len(g.uvs)
    if n_sets >= 1:
        fmt |= F_TEXTURED
    if n_sets >= 2:
        fmt |= F_TEXTURED2
    # high byte = explicit tex-set count (matches reader's `(fmt>>16)&0xFF`)
    fmt = (fmt & 0x00FFFFFF) | ((n_sets & 0xFF) << 16)
    return fmt & 0xFFFFFFFF


def _build_geometry(g: "DFF.Geometry") -> bytes:
    fmt = _normalize_format(g)
    n_vert = len(g.vertices) if g.vertices else g.num_vertices
    n_tri = len(g.triangles) if g.triangles else g.num_triangles
    n_morph = 1  # we only ever emit the base morph target

    payload = bytearray()
    payload += struct.pack("<IIII", fmt, n_tri, n_vert, n_morph)
    # ver >= 0x34000 -> NO legacy surface props block here

    # 1. prelit RGBA
    if fmt & F_PRELIT:
        cols = g.prelit_colors or [(255, 255, 255, 255)] * n_vert
        for i in range(n_vert):
            c = cols[i] if i < len(cols) else (255, 255, 255, 255)
            payload += struct.pack("<4B", c[0] & 0xFF, c[1] & 0xFF, c[2] & 0xFF, c[3] & 0xFF)

    # 2. texcoords: numTexSets x numVertices x (u, v)
    n_sets = (fmt >> 16) & 0xFF
    for s in range(n_sets):
        uv_set = g.uvs[s] if s < len(g.uvs) else []
        for i in range(n_vert):
            u, v = uv_set[i] if i < len(uv_set) else (0.0, 0.0)
            payload += struct.pack("<2f", u, v)

    # 3. triangles (the reader's word-pair swap: t0=(v0<<16)|v1, t1=(v2<<16)|mat)
    for tri in g.triangles:
        a, b, c, mat = tri
        t0 = ((a & 0xFFFF) << 16) | (b & 0xFFFF)
        t1 = ((c & 0xFFFF) << 16) | (mat & 0xFFFF)
        payload += struct.pack("<II", t0, t1)

    # 4. morph target 0: bounding sphere + hasVertices/hasNormals + arrays
    sphere = g.bounding_sphere or _bounding_sphere(g.vertices)
    payload += struct.pack("<4f", *sphere)
    has_v = 1 if g.vertices else 0
    has_n = 1 if g.normals else 0
    payload += struct.pack("<II", has_v, has_n)
    if has_v:
        for x, y, z in g.vertices:
            payload += struct.pack("<3f", x, y, z)
    if has_n:
        for x, y, z in g.normals:
            payload += struct.pack("<3f", x, y, z)

    geo_struct = _chunk(rw.STRUCT, bytes(payload))
    mat_list = _build_materiallist(g.materials)
    extension = _build_geometry_extension(g)

    return _chunk(rw.GEOMETRY, geo_struct + mat_list + extension)


def _bounding_sphere(verts) -> tuple:
    """A correct (centre, radius) bounding sphere over the vertices (AABB centre)."""
    if not verts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    r = 0.0
    for x, y, z in verts:
        d = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5
        if d > r:
            r = d
    return (cx, cy, cz, r)


def _build_materiallist(materials: List["DFF.Material"]) -> bytes:
    """MATERIALLIST: STRUCT(count + index table, all -1) + each MATERIAL inline.

 We never share materials, so every index-table slot is -1 (a fresh material
 follows), matching the simplest stream the reader handles.
 """
    n = len(materials)
    st_body = bytearray(_U32.pack(n))
    for _ in range(n):
        st_body += struct.pack("<i", -1)
    struct_chunk = _chunk(rw.STRUCT, bytes(st_body))

    mats = bytearray()
    for m in materials:
        mats += _build_material(m)

    return _chunk(rw.MATERIAL_LIST, struct_chunk + bytes(mats))


def _build_material(m: "DFF.Material") -> bytes:
    """MATERIAL: STRUCT(flags, rgba, unused, textured, ambient/specular/diffuse)
 + optional TEXTURE chunk + empty EXTENSION.

 STRUCT is the 28-byte ver>=0x34000 form: 16-byte head + 3 surface floats.
 """
    color = tuple(m.color) if m.color else (255, 255, 255, 255)
    r, g, b, a = (color + (255, 255, 255, 255))[:4]
    textured = 1 if m.textured else 0
    st_body = struct.pack("<I4BII", 0, r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF, 0, textured)
    st_body += struct.pack("<3f", m.ambient, m.specular, m.diffuse)
    struct_chunk = _chunk(rw.STRUCT, st_body)

    tex_chunk = b""
    if textured:
        tex_chunk = _build_texture(m.texture_name, m.mask_name)

    return _chunk(rw.MATERIAL, struct_chunk + tex_chunk + _empty_extension())


def _build_texture(name: str, mask: str = "") -> bytes:
    """TEXTURE: STRUCT(filter|addressing dword) + name STRING + mask STRING + Ext.

 Filter/addressing packed dword: filter=linear(2), addrU=addrV=wrap(1) ->
 librw layout `filter | (addrU<<8) | (addrV<<12)` = 0x1102. (The reader only
 reads names, so the exact value is cosmetic but kept game-valid.)
 """
    filter_addr = 0x00001102
    st = _chunk(rw.STRUCT, _U32.pack(filter_addr))
    body = st + _string_chunk(name or "") + _string_chunk(mask or "") + _empty_extension()
    return _chunk(rw.TEXTURE, body)


def _build_geometry_extension(g: "DFF.Geometry") -> bytes:
    """Geometry EXTENSION carrying binMesh (0x050E) + skin (0x0116) when present."""
    body = _build_binmesh(g)
    if getattr(g, "skin", None):
        try:
            body += _build_skin(g)
        except Exception:
            pass  # a bad skin must not break the export
    return _chunk(rw.EXTENSION, body)


def _build_skin(g: "DFF.Geometry") -> bytes:
    """SKIN_PLG 0x0116 (new/SA format), inverse of dff._parse_skin.

 u8 numBones, numUsedBones, maxWeightsPerVertex, pad;
 numUsedBones x u8 usedBoneIds; nVert x 4 u8 boneIndices; nVert x 4 f32 weights;
 numBones x 16 f32 inverse-bind matrices; then i32 boneLimit, numMeshes, rleSize.
 We emit an UNSPLIT skin (boneLimit=numMeshes=rleSize=0) - the reader ignores the
 trailer, and matrices are written back verbatim so weights/indices round-trip exactly.
 """
    s = g.skin
    n_vert = len(g.vertices) if g.vertices else g.num_vertices
    num_bones = s["num_bones"]
    used = s.get("used_bones") or []
    body = bytearray(struct.pack("<4B", num_bones, len(used), s.get("max_weights", 4), 0))
    body += bytes(used)

    idx = s["bone_indices"]
    for i in range(n_vert):
        bi = idx[i] if i < len(idx) else (0, 0, 0, 0)
        body += struct.pack("<4B", bi[0] & 0xFF, bi[1] & 0xFF, bi[2] & 0xFF, bi[3] & 0xFF)

    wts = s["weights"]
    for i in range(n_vert):
        w = wts[i] if i < len(wts) else (0.0, 0.0, 0.0, 0.0)
        body += struct.pack("<4f", w[0], w[1], w[2], w[3])

    for m in s["inverse_bind"]:
        vals = list(m) + [0.0] * (16 - len(m))
        body += struct.pack("<16f", *vals[:16])

    body += struct.pack("<iii", 0, 0, 0)   # boneLimit, numMeshes, rleSize (unsplit)
    return _chunk(rw.SKIN_PLG, bytes(body))


def _build_binmesh(g: "DFF.Geometry") -> bytes:
    """BIN_MESH_PLG 0x050E: {faceType, numSplits, totalIndices} + per-split records.

 Uses the geometry's existing splits when present; otherwise synthesizes one
 trilist split per material from the triangle list. Synthesized splits are always
 flat trilists (faceType 0) so they need no de-strip on read-back.
 """
    if g.splits:
        flags = 1 if any(s.get("strip") for s in g.splits) else 0
        splits = [(s["mat_index"], list(s["indices"])) for s in g.splits]
    else:
        flags = 0  # trilist
        splits = _synth_splits(g)

    total = sum(len(idx) for _mat, idx in splits)
    body = bytearray(struct.pack("<III", flags, len(splits), total))
    for mat_index, idx in splits:
        body += struct.pack("<Ii", len(idx), mat_index)
        if idx:
            body += struct.pack("<%dI" % len(idx), *idx)
    return _chunk(rw.BIN_MESH_PLG, bytes(body))


def _synth_splits(g: "DFF.Geometry") -> List[tuple]:
    """One flat-trilist split per material, gathering that material's triangles."""
    n_mat = max(1, len(g.materials))
    buckets: List[List[int]] = [[] for _ in range(n_mat)]
    for a, b, c, mat in g.triangles:
        mi = mat if 0 <= mat < n_mat else 0
        buckets[mi].extend((a, b, c))
    # keep only non-empty material splits, in material order; ensure at least one
    out = [(mi, idx) for mi, idx in enumerate(buckets) if idx]
    if not out:
        out = [(0, [])]
    return out


# =========================================================================
# obj_to_dff (Wavefront OBJ -> clump bytes)
# =========================================================================

def obj_to_dff(obj_text: str, default_texture: str = "") -> bytes:
    """Parse a Wavefront OBJ and serialize it as a geometry-only SA DFF.

 Handles `v`, `vt`, `vn`, `usemtl`, and `f` faces with `a`, `a/vt`, `a/vt/vn`,
 or `a//vn` vertex refs (1-based, negatives relative-to-end). De-indexes into a
 single geometry: one Material per distinct `usemtl` (texture_name = the label,
 textured when the mesh has UVs/a name), triangles tagged with their material
 index, one binMesh split per material, one identity Frame, one Atomic.

 The exporter stored UV V as `1 - v`; we re-flip it back to `v` on import so the
 stored texcoord matches the original asset. n-gon faces are fan-triangulated.
 """
    positions: List[tuple] = []
    tex_coords: List[tuple] = []
    normals: List[tuple] = []

    # de-indexed geometry: a unique vertex per distinct (v, vt, vn) corner ref
    out_pos: List[tuple] = []
    out_uv: List[tuple] = []
    out_norm: List[tuple] = []
    corner_map: dict = {}

    has_uv = False
    has_norm = False

    # materials, in first-seen order
    mat_names: List[str] = []
    mat_index: dict = {}
    triangles: List[tuple] = []  # (a, b, c, matIdx)

    cur_mat = -1  # index into mat_names; -1 = none declared yet

    def _ensure_default_mat() -> int:
        nonlocal cur_mat
        if cur_mat < 0:
            label = default_texture or ""
            cur_mat = _get_mat(label)
        return cur_mat

    def _get_mat(label: str) -> int:
        if label not in mat_index:
            mat_index[label] = len(mat_names)
            mat_names.append(label)
        return mat_index[label]

    def _corner(ref: str) -> int:
        """Resolve one `v/vt/vn` face corner to a de-indexed output vertex index."""
        nonlocal has_uv, has_norm
        parts = ref.split("/")
        vi = int(parts[0])
        ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        ni = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        # OBJ indices are 1-based; negatives count from the end
        vi = vi - 1 if vi > 0 else len(positions) + vi
        if ti:
            ti = ti - 1 if ti > 0 else len(tex_coords) + ti
        if ni:
            ni = ni - 1 if ni > 0 else len(normals) + ni

        key = (vi, ti if (len(parts) > 1 and parts[1]) else -1,
               ni if (len(parts) > 2 and parts[2]) else -1)
        idx = corner_map.get(key)
        if idx is not None:
            return idx
        idx = len(out_pos)
        corner_map[key] = idx
        out_pos.append(positions[vi] if 0 <= vi < len(positions) else (0.0, 0.0, 0.0))
        if len(parts) > 1 and parts[1] and 0 <= ti < len(tex_coords):
            u, v = tex_coords[ti]
            out_uv.append((u, v))
            has_uv = True
        else:
            out_uv.append((0.0, 0.0))
        if len(parts) > 2 and parts[2] and 0 <= ni < len(normals):
            out_norm.append(normals[ni])
            has_norm = True
        else:
            out_norm.append((0.0, 0.0, 0.0))
        return idx

    for raw_line in obj_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        tag = tok[0]
        if tag == "v" and len(tok) >= 4:
            positions.append((float(tok[1]), float(tok[2]), float(tok[3])))
        elif tag == "vt" and len(tok) >= 3:
            # exporter wrote (u, 1 - v); invert to recover v
            u = float(tok[1])
            v = 1.0 - float(tok[2])
            tex_coords.append((u, v))
        elif tag == "vn" and len(tok) >= 4:
            normals.append((float(tok[1]), float(tok[2]), float(tok[3])))
        elif tag == "usemtl":
            label = line[len("usemtl"):].strip()
            cur_mat = _get_mat(label)
        elif tag == "f" and len(tok) >= 4:
            mi = _ensure_default_mat()
            corners = [_corner(r) for r in tok[1:]]
            # fan-triangulate any polygon
            for k in range(1, len(corners) - 1):
                triangles.append((corners[0], corners[k], corners[k + 1], mi))

    if not mat_names:  # a face-less or material-less OBJ still needs one material
        _get_mat(default_texture or "")

    geo = _assemble_geometry(out_pos, out_uv if has_uv else None,
                             out_norm if has_norm else None,
                             triangles, mat_names, has_uv)

    dff = DFF.Dff(
        frames=[DFF.Frame(name="", parent=-1,
                          rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                          position=[0.0, 0.0, 0.0])],
        geometries=[geo],
        atomics=[DFF.Atomic(frame_index=0, geometry_index=0, flags=0)],
        version=SA_VERSION,
    )
    return build_dff(dff)


def _assemble_geometry(positions, uvs, normals, triangles, mat_names, has_uv) -> "DFF.Geometry":
    """Build a Geometry dataclass from de-indexed OBJ arrays + material labels."""
    materials: List[DFF.Material] = []
    for label in mat_names:
        textured = 1 if (has_uv or label) else 0
        materials.append(DFF.Material(
            color=(255, 255, 255, 255),
            textured=textured,
            texture_name=label,
            mask_name="",
            ambient=1.0, specular=1.0, diffuse=1.0,
        ))
    if not materials:
        materials.append(DFF.Material((255, 255, 255, 255), 0))

    fmt = F_POSITIONS | F_LIGHT | F_MODULATE
    if has_uv:
        fmt |= F_TEXTURED | (1 << 16)
    if normals is not None:
        fmt |= F_NORMALS

    geo = DFF.Geometry(
        format=fmt,
        num_triangles=len(triangles),
        num_vertices=len(positions),
        num_morph_targets=1,
        vertices=list(positions),
        uvs=[list(uvs)] if uvs is not None else [],
        prelit_colors=None,
        normals=list(normals) if normals is not None else None,
        triangles=list(triangles),
        materials=materials,
        splits=[],  # synthesized per-material in _build_binmesh
        bounding_sphere=None,
    )
    return geo
