"""the source game COL collision decoder (COLL / COL2 / COL3 / COL4).

A .col library is a bare concatenation of self-describing FourCC chunks - NO global
header or count. One IMG .col entry (or a loose .col file) holds many per-model chunks
back-to-back; `parse_col_library` walks them all.

Each chunk: FourCC(4) + u32 size + name[22] + u16 modelId + bound block (40 B), then a
version-specific body. The whole sub-file occupies 8 + size bytes; `size` counts everything
after the size field.

CRITICAL OFFSET RULE (COL2/3/4): every section offset stored in the body header is measured
from the SIZE FIELD (chunk byte +4), NOT from the body start. In-file index = chunk+4+off.
An offset of 0 means the section is NULL/absent. `num_vertices` is NOT stored - derive it as
(off_faces - off_vertices) / 6 (minus the u32-prefixed face-group blob when flag 0x08 is set).

Body layouts are byte-confirmed against the the reference sources readers
(ColHelpers.h: V1/V2/V3/V4 Header VALIDATE_SIZE 0x20/0x4C/0x58/0x5C; FileLoader.cpp
, LoadCollisionModelVer2/3/4) and cross-checked on real bytes:
 * COL1/COLL: sphere-first bound; float32 verts; body order is
 spheres -> LINES -> boxes -> verts -> tris (lines are read then discarded by SA).
 * COL2/3/4: box-first bound; CompressedVector verts (s16, dequant = s16 / 128.0); body order
 spheres -> boxes -> lines -> verts -> tris, with face groups (flag 0x08) between verts and
 tris, then triangle planes, then (COL3/4) the shadow mesh.

 
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_TAGS = {b"COLL": 1, b"COL2": 2, b"COL3": 3, b"COL4": 4}

VERTEX_SCALE = 1.0 / 128.0       # CompressedVector dequant (s16 -> metres)
NORMAL_SCALE = 1.0 / 4096.0      # CColTrianglePlane normal dequant (COL3/4)
DIST_SCALE = 1.0 / 128.0         # CColTrianglePlane dist dequant

# flag bits (low byte of the flags dword)
FLAG_DISKS = 0x01
FLAG_NOT_EMPTY = 0x02
FLAG_FACE_GROUPS = 0x08
FLAG_SHADOW = 0x10

Vec3 = Tuple[float, float, float]


@dataclass
class ColModel:
    """One decoded collision model. Vertices are always dequantized floats (metres)."""
    name: str
    model_id: int
    version: int                                   # 1 / 2 / 3 / 4
    bound_radius: float
    bound_center: Vec3
    bound_min: Vec3
    bound_max: Vec3
    flags: int = 0
    spheres: List[Dict] = field(default_factory=list)   # {center, radius, surface, piece, brightness, light}
    boxes: List[Dict] = field(default_factory=list)     # {min, max, surface, piece, brightness, light}
    lines: List[Dict] = field(default_factory=list)     # {p0, p1}
    vertices: List[Vec3] = field(default_factory=list)
    faces: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (a, b, c, surface)
    face_light: List[int] = field(default_factory=list)                   # parallel to faces
    planes: List[Dict] = field(default_factory=list)        # COL3/4 {normal, dist}
    face_groups: List[Dict] = field(default_factory=list)   # COL3/4 {min, max, start, end}
    shadow_vertices: List[Vec3] = field(default_factory=list)
    shadow_faces: List[Tuple[int, int, int, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # genuine anomalies (corrupt / out-of-range)
    info: List[str] = field(default_factory=list)      # benign notes (expected on-disk slack)


# ----------------------------- helpers -----------------------------

def _cstr(buf: bytes, off: int, n: int) -> str:
    return buf[off:off + n].split(b"\x00", 1)[0].decode("latin-1")


def _vec3f(buf: bytes, off: int) -> Vec3:
    return struct.unpack_from("<3f", buf, off)


# ----------------------------- top-level walk -----------------------------

def parse_col_library(data: bytes) -> List[ColModel]:
    """Parse a whole .col library (one IMG entry's bytes or a raw .col file).

 Walks chunk-by-chunk until the next FourCC is not a COL tag or < 8 bytes remain
 (sector / zero padding). Each chunk occupies 8 + size bytes.
 """
    models: List[ColModel] = []
    pos = 0
    n = len(data)
    while n - pos >= 8:
        fourcc = bytes(data[pos:pos + 4])
        version = _TAGS.get(fourcc)
        if version is None:
            break  # padding or end of meaningful data
        size = struct.unpack_from("<I", data, pos + 4)[0]
        total = 8 + size
        if total < 8 or pos + total > n:
            break  # truncated final chunk; stop rather than read past the buffer
        chunk = data[pos:pos + total]
        try:
            models.append(_parse_chunk(chunk, version))
        except Exception as e:  # one bad chunk must not kill the library
            models.append(_error_model(chunk, version, str(e)))
        pos += total
    return models


def _error_model(chunk: bytes, version: int, msg: str) -> ColModel:
    try:
        name = _cstr(chunk, 8, 22)
        model_id = struct.unpack_from("<H", chunk, 0x1E)[0]
    except Exception:
        name, model_id = "<error>", 0
    m = ColModel(name, model_id, version, 0.0, (0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    m.warnings.append(f"parse error: {msg}")
    return m


def _parse_chunk(chunk: bytes, version: int) -> ColModel:
    name = _cstr(chunk, 8, 22)
    model_id = struct.unpack_from("<H", chunk, 0x1E)[0]

    if version == 1:
        # COL1/COLL bound is SPHERE-FIRST:,, box
        bound_radius = struct.unpack_from("<f", chunk, 0x20)[0]
        bound_center = _vec3f(chunk, 0x24)
        bound_min = _vec3f(chunk, 0x30)
        bound_max = _vec3f(chunk, 0x3C)
        m = ColModel(name, model_id, version, bound_radius, bound_center, bound_min, bound_max)
        _parse_col1_body(chunk, m)
        return m

    # COL2/3/4 bound is BOX-FIRST:,,, 
    bound_min = _vec3f(chunk, 0x20)
    bound_max = _vec3f(chunk, 0x2C)
    bound_center = _vec3f(chunk, 0x38)
    bound_radius = struct.unpack_from("<f", chunk, 0x44)[0]
    m = ColModel(name, model_id, version, bound_radius, bound_center, bound_min, bound_max)
    _parse_col23_body(chunk, m, version)
    return m


# ----------------------------- COL1 (float path) -----------------------------

def _parse_col1_body(chunk: bytes, m: ColModel) -> None:
    """Legacy float32 body, built element-by-element (the reference sources LoadCollisionModel).

 Reader order: u32 nSpheres; TSphere[0x14] (radius-first); u32 nLines; TLine[24] (SKIPPED
 by SA - kept here for completeness); u32 nBoxes; TBox[0x1C]; u32 nVerts; f32[3][nVerts];
 u32 nFaces; TFace[16] (u32 a,b,c; u8 surface; u8 light; 2 pad).

 Each count is sanity-checked against the remaining chunk bytes; an implausible count
 (overflowing the chunk - seen in some third-party-extracted .col files) is treated as 0.
 """
    pos = 0x48  # body starts right after the 0x48-byte chunk header
    end = len(chunk)

    def count(stride: int) -> int:
        nonlocal pos
        if pos + 4 > end:
            return 0
        n = struct.unpack_from("<I", chunk, pos)[0]
        pos += 4
        if n < 0 or pos + n * stride > end:  # junk in an unused field -> skip the section
            m.warnings.append(f"COL1 count {n} (stride {stride}) overflows chunk; treated as 0")
            return 0
        return n

    n_sph = count(0x14)
    for _ in range(n_sph):
        radius = struct.unpack_from("<f", chunk, pos)[0]
        center = _vec3f(chunk, pos + 4)
        m.spheres.append({"center": list(center), "radius": radius,
                          "surface": chunk[pos + 0x10], "piece": chunk[pos + 0x11],
                          "brightness": chunk[pos + 0x12], "light": chunk[pos + 0x13]})
        pos += 0x14

    n_lines = count(24)
    for _ in range(n_lines):
        m.lines.append({"p0": list(_vec3f(chunk, pos)), "p1": list(_vec3f(chunk, pos + 0x0C))})
        pos += 24

    n_box = count(0x1C)
    for _ in range(n_box):
        m.boxes.append({"min": list(_vec3f(chunk, pos)), "max": list(_vec3f(chunk, pos + 0x0C)),
                        "surface": chunk[pos + 0x18], "piece": chunk[pos + 0x19],
                        "brightness": chunk[pos + 0x1A], "light": chunk[pos + 0x1B]})
        pos += 0x1C

    n_verts = count(12)
    for _ in range(n_verts):
        m.vertices.append(_vec3f(chunk, pos))  # already float metres
        pos += 12

    n_faces = count(16)
    for _ in range(n_faces):
        a, b, c = struct.unpack_from("<3i", chunk, pos)
        m.faces.append((a, b, c, chunk[pos + 12]))
        m.face_light.append(chunk[pos + 13])
        pos += 16


# ----------------------------- COL2 / COL3 / COL4 -----------------------------

def _parse_col23_body(chunk: bytes, m: ColModel, version: int) -> None:
    """Body for COL2/3/4 (the reference sources LoadCollisionModelVer2/3/4).

 Header (V2::Header sizeof 0x4C, chunk-absolute offsets):
 u16 , u16 , u16 , u8 , u8 ,
 u32 , then six u32 .. : offSpheres, offBoxes, offLines,
 offVerts, offFaces, offPlanes. (COL3 adds u32 , ,
 ; COL4 adds one unknown .) ALL offsets are relative to the
 size field at chunk+4 -> in-chunk index = 4 + stored.
 """
    end = len(chunk)
    base = 4  # the size field; section offsets are relative to here

    num_spheres = struct.unpack_from("<H", chunk, 0x48)[0]
    num_boxes = struct.unpack_from("<H", chunk, 0x4A)[0]
    num_faces = struct.unpack_from("<H", chunk, 0x4C)[0]
    num_lines = chunk[0x4E]
    flags = struct.unpack_from("<I", chunk, 0x50)[0]
    off_spheres, off_boxes, off_lines, off_vertices, off_faces, off_planes = \
        struct.unpack_from("<6I", chunk, 0x54)

    off_shadow_verts = off_shadow_faces = 0
    num_shadow_faces = 0
    if version >= 3:
        num_shadow_faces, off_shadow_verts, off_shadow_faces = \
            struct.unpack_from("<3I", chunk, 0x6C)
        # COL4 has one extra (unknown/unused) -> ignored.

    m.flags = flags

    def at(off: int) -> int:
        return base + off

    def fits(off: int, n: int, stride: int) -> bool:
        return bool(off) and n > 0 and at(off) + n * stride <= end

    # --- derive num_vertices (NOT stored) ---
    num_vertices = 0
    n_groups = 0
    vert_end = off_faces
    if off_vertices and off_faces and off_faces >= off_vertices:
        if (flags & FLAG_FACE_GROUPS):
            # Face groups are a V2::Header feature (flag bit 0x08) - present in COL2 too,
            # not just COL3/4. A u32 group count sits immediately before off_faces and the
            # CColFaceGroup[] block grows downward from there, BETWEEN vertices and faces.
            n_groups = struct.unpack_from("<I", chunk, at(off_faces - 4))[0]
            cand_end = (off_faces - 4) - n_groups * 28
            if cand_end >= off_vertices:
                vert_end = cand_end
            else:
                m.warnings.append(f"face-group count {n_groups} inconsistent; ignored for vert derive")
                n_groups = 0
        span = vert_end - off_vertices
        if span >= 0:
            # Floor division is exact for the real vertex count: off_faces is consistently a
            # couple of bytes past the true end of the vertex array (on-disk alignment slack),
            # so a non-6-multiple span is expected and benign (verified: derived count always
            # equals max-face-index + 1, on plain and face-grouped COL2/3 chunks alike).
            num_vertices = span // 6
            if span % 6:
                m.info.append(f"vertex span {span} has {span % 6} byte(s) of trailing slack")

    # --- spheres (stride 0x14: center, radius, surf/piece/brightness/light) ---
    if fits(off_spheres, num_spheres, 0x14):
        o = at(off_spheres)
        for _ in range(num_spheres):
            m.spheres.append({"center": list(_vec3f(chunk, o)),
                              "radius": struct.unpack_from("<f", chunk, o + 0x0C)[0],
                              "surface": chunk[o + 0x10], "piece": chunk[o + 0x11],
                              "brightness": chunk[o + 0x12], "light": chunk[o + 0x13]})
            o += 0x14

    # --- boxes (stride 0x1C: min, max, surf/piece/brightness/light) ---
    if fits(off_boxes, num_boxes, 0x1C):
        o = at(off_boxes)
        for _ in range(num_boxes):
            m.boxes.append({"min": list(_vec3f(chunk, o)), "max": list(_vec3f(chunk, o + 0x0C)),
                            "surface": chunk[o + 0x18], "piece": chunk[o + 0x19],
                            "brightness": chunk[o + 0x1A], "light": chunk[o + 0x1B]})
            o += 0x1C

    # --- lines (stride 24: p0, p1; no w on disk) ---
    if fits(off_lines, num_lines, 24):
        o = at(off_lines)
        for _ in range(num_lines):
            m.lines.append({"p0": list(_vec3f(chunk, o)), "p1": list(_vec3f(chunk, o + 0x0C))})
            o += 24

    # --- vertices (6B CompressedVector, dequant /128) ---
    if fits(off_vertices, num_vertices, 6):
        o = at(off_vertices)
        for _ in range(num_vertices):
            x, y, z = struct.unpack_from("<3h", chunk, o)
            m.vertices.append((x * VERTEX_SCALE, y * VERTEX_SCALE, z * VERTEX_SCALE))
            o += 6

    # --- faces (stride 8: u16 a,b,c; u8 surface; u8 light) ---
    if fits(off_faces, num_faces, 8):
        o = at(off_faces)
        for _ in range(num_faces):
            a, b, c = struct.unpack_from("<3H", chunk, o)
            m.faces.append((a, b, c, chunk[o + 6]))
            m.face_light.append(chunk[o + 7])
            o += 8

    # --- triangle planes (8B disk: s16 nx,ny,nz,dist) - off_planes is a V2::Header field but
    # ships only on COL3/4 in practice; guarded by off_planes != 0 regardless of version ---
    if fits(off_planes, num_faces, 8):
        o = at(off_planes)
        for _ in range(num_faces):
            nx, ny, nz, dist = struct.unpack_from("<4h", chunk, o)
            m.planes.append({"normal": [nx * NORMAL_SCALE, ny * NORMAL_SCALE, nz * NORMAL_SCALE],
                             "dist": dist * DIST_SCALE})
            o += 8

    # --- face groups (28B: s16[3] min, s16[3] max, u16 start, u16 end); COL2+ when flag 0x08 ---
    # Optional broad-phase accel, NOT needed for a wireframe viewer. The block carries a couple
    # of alignment bytes whose exact placement isn't pinned, so the per-record start/end face
    # indices are best-effort; the count and bound corners are reliable. The mesh (verts/faces)
    # is unaffected - its vertex count is already corrected for this block above.
    if (flags & FLAG_FACE_GROUPS) and n_groups and fits(vert_end, n_groups, 28):
        o = at(vert_end)  # groups grow downward from off_faces-8; lowest record sits at vert_end
        for _ in range(n_groups):
            gmin = struct.unpack_from("<3h", chunk, o)
            gmax = struct.unpack_from("<3h", chunk, o + 6)
            start, fin = struct.unpack_from("<2H", chunk, o + 0x12)
            m.face_groups.append({"min": [v * VERTEX_SCALE for v in gmin],
                                  "max": [v * VERTEX_SCALE for v in gmax],
                                  "start": start, "end": fin})
            o += 28

    # --- COL3/4 shadow mesh (verts 6B, faces 8B; shadow vert count derived) ---
    if version >= 3 and off_shadow_verts and off_shadow_faces and off_shadow_faces >= off_shadow_verts:
        num_shadow_verts = (off_shadow_faces - off_shadow_verts) // 6
        if fits(off_shadow_verts, num_shadow_verts, 6):
            o = at(off_shadow_verts)
            for _ in range(num_shadow_verts):
                x, y, z = struct.unpack_from("<3h", chunk, o)
                m.shadow_vertices.append((x * VERTEX_SCALE, y * VERTEX_SCALE, z * VERTEX_SCALE))
                o += 6
        if fits(off_shadow_faces, num_shadow_faces, 8):
            o = at(off_shadow_faces)
            for _ in range(num_shadow_faces):
                a, b, c = struct.unpack_from("<3H", chunk, o)
                m.shadow_faces.append((a, b, c, chunk[o + 6]))
                o += 8


# ----------------------------- export -----------------------------

def to_json(m: ColModel) -> Dict:
    """Plain-dict view for the web wireframe viewer (all floats, JSON-serializable).

 mesh.vertices is a flat [[x,y,z], ...] list; mesh.faces is [{v:[a,b,c], surface, light}, ...].
 """
    faces = [{"v": [a, b, c], "surface": s,
              "light": m.face_light[i] if i < len(m.face_light) else 0}
             for i, (a, b, c, s) in enumerate(m.faces)]
    out: Dict = {
        "version": m.version,
        "name": m.name,
        "model_id": m.model_id,
        "flags": m.flags,
        "bound": {
            "min": list(m.bound_min),
            "max": list(m.bound_max),
            "center": list(m.bound_center),
            "radius": m.bound_radius,
        },
        "spheres": m.spheres,
        "boxes": m.boxes,
        "lines": m.lines,
        "mesh": {
            "vertices": [list(v) for v in m.vertices],
            "faces": faces,
        },
    }
    if m.planes:
        out["planes"] = m.planes
    if m.face_groups:
        out["face_groups"] = m.face_groups
    if m.shadow_vertices or m.shadow_faces:
        out["shadow"] = {
            "vertices": [list(v) for v in m.shadow_vertices],
            "faces": [{"v": [a, b, c], "surface": s} for (a, b, c, s) in m.shadow_faces],
        }
    if m.warnings:
        out["warnings"] = m.warnings
    if m.info:
        out["info"] = m.info
    return out


def wireframe(m: ColModel) -> Dict:
    """Minimal wireframe payload: {vertices:[[x,y,z]...], faces:[[a,b,c]...]}."""
    return {
        "vertices": [list(v) for v in m.vertices],
        "faces": [[a, b, c] for (a, b, c, _s) in m.faces],
    }
