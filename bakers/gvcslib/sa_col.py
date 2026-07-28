"""Read-only decoder for the source game collision files (COL2 / COL3 / COLL).

the source game PS2 ships collision as COL2 (and a few COL3) FourCC TCollFile
records. One IMG ".col" entry concatenates MANY models back-to-back, each its
own FourCC'd sub-file, so parse_col_file() loops until the bytes run out.

Byte layout (verified against real GTA3.IMG bytes, see tests/test_sa_col.py):

  Common file header (all versions)
    +0   char[4]   FourCC      'COLL' | 'COL2' | 'COL3' | 'COL4'
    +4   u32       size        bytes of THIS sub-file AFTER the +8 header
                               (i.e. sub-file total length == 8 + size)
  --- everything below is at sub[8:]; ALL section offsets in the COL2/3 header
      are measured from sub[4] (the size field), so an in-file byte index is
      (4 + stored_offset). ---
    +8   char[22]  name        nul-padded model name
    +30  u16       model_id    (TObjectID; zone-local in SA)
    +32  f32[3]    bbox_min
    +44  f32[3]    bbox_max
    +56  f32[3]    bound_center
    +68  f32       bound_radius

  COLL (version 1) header continues differently (bbox first, then sphere) and
  uses UNCOMPRESSED float vertices. It is rare in this corpus; supported.

  COL2 / COL3 header continues:
    +72  u16       num_spheres
    +74  u16       num_boxes
    +76  u32       num_faces
    +80  u32       flags        (bit1 0x02 = not-empty, bit3 0x08 = face groups)
    +84  u32       off_spheres
    +88  u32       off_boxes
    +92  u32       off_cones    (COL3 cones / COL2 unused)
    +96  u32       off_vertices
    +100 u32       off_faces
    (COL3 only) +104 u32 off_triangle_planes  - not needed for geometry
    num_vertices is NOT stored: it is (off_faces - off_vertices) // 6.

  Sections (offset = 4 + stored value, within the sub-file blob):
    Sphere   20 bytes: f32 cx,cy,cz, f32 radius, u8 mat, u8 flag, u8 brightness, u8 light
    Box      28 bytes: f32 min[3], f32 max[3], u8 mat, u8 flag, u8 brightness, u8 light
    Vertex    6 bytes: i16 x,y,z  -> divide by 128.0 to get float metres (COL2/3)
    Face      8 bytes: u16 a,b,c, u8 material, u8 light
    FaceGroup 28 bytes: i16 min[3], i16 max[3], u16 start, u16 end  (when flag 0x08)
              preceded by a u32 count stored immediately before off_faces.

This module is read-only. It never mutates the input bytes.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_FOURCCS = (b"COLL", b"COL2", b"COL3", b"COL4")
_VERSION = {b"COLL": 1, b"COL2": 2, b"COL3": 3, b"COL4": 4}


@dataclass
class ColSphere:
    center: Tuple[float, float, float]
    radius: float
    material: int = 0
    flag: int = 0
    brightness: int = 0
    light: int = 0


@dataclass
class ColBox:
    bmin: Tuple[float, float, float]
    bmax: Tuple[float, float, float]
    material: int = 0
    flag: int = 0
    brightness: int = 0
    light: int = 0


@dataclass
class ColFace:
    a: int
    b: int
    c: int
    material: int = 0
    light: int = 0


@dataclass
class ColFaceGroup:
    bmin: Tuple[int, int, int]
    bmax: Tuple[int, int, int]
    start: int
    end: int


@dataclass
class ColModel:
    name: str
    model_id: int
    version: int
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    bound_center: Tuple[float, float, float]
    bound_radius: float
    spheres: List[ColSphere] = field(default_factory=list)
    boxes: List[ColBox] = field(default_factory=list)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[ColFace] = field(default_factory=list)
    facegroups: List[ColFaceGroup] = field(default_factory=list)

    @property
    def bounds(self):
        """(sphere(center,radius), box(min,max))."""
        return (
            (self.bound_center, self.bound_radius),
            (self.bbox_min, self.bbox_max),
        )


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _vec3f(b, o):
    return struct.unpack_from("<3f", b, o)


def _parse_surface(b, o):
    # 4 bytes: material, flag, brightness, light
    return b[o], b[o + 1], b[o + 2], b[o + 3]


def _parse_v1(sub: bytes) -> ColModel:
    """Parse a COLL (version 1) sub-file. Uncompressed float vertices."""
    name = sub[8:30].split(b"\x00")[0].decode("latin1")
    model_id = _u16(sub, 30)
    # version 1 header: bbox(min,max) then sphere(center,radius) order varies;
    # SA uses: bbox_min, bbox_max, sphere_center, sphere_radius (same as COL2 prefix)
    bbox_min = _vec3f(sub, 32)
    bbox_max = _vec3f(sub, 44)
    center = _vec3f(sub, 56)
    radius = struct.unpack_from("<f", sub, 68)[0]
    p = 72
    m = ColModel("", model_id, 1, bbox_min, bbox_max, center, radius)
    m.name = name
    num_spheres = _u32(sub, p); p += 4
    p += 4  # unused
    for _ in range(num_spheres):
        cx, cy, cz, r = struct.unpack_from("<4f", sub, p)
        mat, flag, bright, light = _parse_surface(sub, p + 16)
        m.spheres.append(ColSphere((cx, cy, cz), r, mat, flag, bright, light))
        p += 20
    num_boxes = _u32(sub, p); p += 4
    for _ in range(num_boxes):
        bmin = _vec3f(sub, p); bmax = _vec3f(sub, p + 12)
        mat, flag, bright, light = _parse_surface(sub, p + 24)
        m.boxes.append(ColBox(bmin, bmax, mat, flag, bright, light))
        p += 28
    num_verts = _u32(sub, p); p += 4
    verts = []
    for _ in range(num_verts):
        verts.append(struct.unpack_from("<3f", sub, p)); p += 12
    m.vertices = verts
    num_faces = _u32(sub, p); p += 4
    for _ in range(num_faces):
        a, b2, c = struct.unpack_from("<3I", sub, p)
        mat, light = sub[p + 12], sub[p + 13]
        m.faces.append(ColFace(a, b2, c, mat, light))
        p += 16
    return m


def _parse_v23(sub: bytes, version: int) -> ColModel:
    name = sub[8:30].split(b"\x00")[0].decode("latin1")
    model_id = _u16(sub, 30)
    bbox_min = _vec3f(sub, 32)
    bbox_max = _vec3f(sub, 44)
    center = _vec3f(sub, 56)
    radius = struct.unpack_from("<f", sub, 68)[0]

    num_spheres = _u16(sub, 72)
    num_boxes = _u16(sub, 74)
    num_faces = _u32(sub, 76)
    flags = _u32(sub, 80)
    off_spheres = _u32(sub, 84)
    off_boxes = _u32(sub, 88)
    # off_cones = _u32(sub, 92)   # COL3 cones / COL2 unused
    off_verts = _u32(sub, 96)
    off_faces = _u32(sub, 100)

    # All section offsets are measured from sub[4]; in-blob index = 4 + stored.
    base = 4
    m = ColModel(name, model_id, version, bbox_min, bbox_max, center, radius)

    o = base + off_spheres
    for _ in range(num_spheres):
        cx, cy, cz, r = struct.unpack_from("<4f", sub, o)
        mat, flag, bright, light = _parse_surface(sub, o + 16)
        m.spheres.append(ColSphere((cx, cy, cz), r, mat, flag, bright, light))
        o += 20

    o = base + off_boxes
    for _ in range(num_boxes):
        bmin = _vec3f(sub, o); bmax = _vec3f(sub, o + 12)
        mat, flag, bright, light = _parse_surface(sub, o + 24)
        m.boxes.append(ColBox(bmin, bmax, mat, flag, bright, light))
        o += 28

    # When face groups are present (flag bit 0x08) they are stored BETWEEN the
    # vertex section and the face section, prefixed by a u32 group count that
    # sits immediately before off_faces. The vertex section therefore ends at
    # (off_faces - 4 - ngroups*28), NOT at off_faces. Getting this right is what
    # makes the derived vertex count exact and keeps verts inside the bbox.
    vert_end = base + off_faces
    ngroups = 0
    fg_start = None
    if (flags & 0x08) and num_faces:
        cnt_pos = base + off_faces - 4
        if cnt_pos >= base + off_verts:
            cand = _u32(sub, cnt_pos)
            gp = cnt_pos - cand * 28
            if cand and base + off_verts <= gp <= cnt_pos and cand < 100000:
                ngroups = cand
                fg_start = gp
                vert_end = gp

    # num_vertices is NOT stored: derive it from the (face-group-corrected) gap
    # between the vertex section and whatever follows it.
    num_verts = 0
    if num_faces and vert_end > base + off_verts:
        num_verts = (vert_end - (base + off_verts)) // 6
    o = base + off_verts
    verts = []
    for _ in range(num_verts):
        x, y, z = struct.unpack_from("<3h", sub, o)
        verts.append((x / 128.0, y / 128.0, z / 128.0))
        o += 6
    m.vertices = verts

    # Face groups (raw int16 AABB + start/end face index range).
    if fg_start is not None:
        gp = fg_start
        for _ in range(ngroups):
            gmin = struct.unpack_from("<3h", sub, gp)
            gmax = struct.unpack_from("<3h", sub, gp + 6)
            start = _u16(sub, gp + 12)
            end = _u16(sub, gp + 14)
            m.facegroups.append(ColFaceGroup(gmin, gmax, start, end))
            gp += 28

    # COL2/3 face record is 8 bytes: a,b,c (u16) + material (u8) + light (u8).
    o = base + off_faces
    for _ in range(num_faces):
        a, b2, c = struct.unpack_from("<3H", sub, o)
        mat = sub[o + 6]
        light = sub[o + 7]
        m.faces.append(ColFace(a, b2, c, mat, light))
        o += 8

    return m


def parse_col_file(blob: bytes) -> List[ColModel]:
    """Parse a (possibly concatenated) collision blob into a list of ColModel.

    Stops cleanly at the first non-COL FourCC or when fewer than 8 header
    bytes remain (trailing IMG sector padding is common).
    """
    out: List[ColModel] = []
    off = 0
    n = len(blob)
    while off + 8 <= n:
        fcc = blob[off:off + 4]
        if fcc not in _FOURCCS:
            break
        size = _u32(blob, off + 4)
        end = off + 8 + size
        if size <= 0 or end > n:
            break
        sub = blob[off:end]
        version = _VERSION[fcc]
        try:
            if version == 1:
                out.append(_parse_v1(sub))
            else:
                out.append(_parse_v23(sub, version))
        except (struct.error, IndexError):
            # Tolerate a malformed tail sub-file; keep what parsed.
            break
        off = end
    return out


def is_finite_model(m: ColModel) -> bool:
    vals = list(m.bbox_min) + list(m.bbox_max) + list(m.bound_center) + [m.bound_radius]
    return all(math.isfinite(v) for v in vals)
