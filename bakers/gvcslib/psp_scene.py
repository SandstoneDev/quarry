"""'.pmap' v2 - a self-contained little-endian world container the PSP homebrew
engine (``work/psp_engine``) memory-maps and feeds DIRECTLY to ``sceGu`` with
zero conversion.

This module is the **writer/reader half of the engine-side contract** defined in
``work/psp_engine/pmap.h``.  Unlike the v1 format (which stored opaque per-model
blobs the engine could not consume), v2 explodes geometry into the exact shared
GE pools ``main.c`` draws from:

  * one **vertex pool**  - ``PmapVertex`` (12 B: s16 u,v · u16 color5551 · s16 x,y,z)
  * one **index pool**   - u16, GU_INDEX_16BIT, indices are submesh-LOCAL
  * one **texel pool**   - PSP-swizzled texel planes (from ``psp_tex``)
  * one **clut pool**    - RGBA8888 CLUT entries (alpha last) for T4/T8

and tables that slice those pools: models → a run of submeshes; each submesh →
one texture + a vertex/index slice; instances place a model with pos/quat/scale;
a zone grid buckets instances into cells for streaming/culling.

Coordinate system: **the source game native** (X east, Y north, **Z up**).  The viewer is
Z-up, the grid tiles the horizontal **XY** plane.  Geometry, instance positions
and quaternions pass through verbatim - no axis conversion.

Two correctness fixes over the original ``pmap.h`` draft (both carried here and
in the C structs):
  * per-model **center** (f32×3) is stored - ``psp_mesh`` quantises positions
    about the AABB centre, so the engine must translate by it.  world_local =
    center + pos_i16 * scale.
  * per-model **scale** is an f32 (the original i32 12-frac fixed underflowed to
    0 for sub-8-metre models).

================================================================================
BYTE LAYOUT (all little-endian; every table/pool 16-byte aligned in-file)
================================================================================

HEADER (0x50 = 80 bytes) @ 0
  +0x00 char[4] magic 'PMAP'
  +0x04 u32 version            = 2
  +0x08 u32 file_size
  +0x0c u32 model_count
  +0x10 u32 model_off
  +0x14 u32 submesh_count
  +0x18 u32 submesh_off
  +0x1c u32 texture_count
  +0x20 u32 texture_off
  +0x24 u32 instance_count
  +0x28 u32 instance_off
  +0x2c u32 grid_off
  +0x30 u32 vertex_off  +0x34 u32 vertex_bytes
  +0x38 u32 index_off   +0x3c u32 index_bytes
  +0x40 u32 texel_off   +0x44 u32 texel_bytes
  +0x48 u32 clut_off    +0x4c u32 clut_bytes

MODEL rec (0x1c = 28)  array[model_count] @ model_off
  +0x00 u32 first_submesh  +0x04 u32 submesh_count
  +0x08 f32 scale          +0x0c f32 center_x +0x10 center_y +0x14 center_z
  +0x18 f32 bound_radius

SUBMESH rec (0x14 = 20)  array[submesh_count] @ submesh_off
  +0x00 i32 texture (-1 = untextured)
  +0x04 u32 vertex_first  +0x08 u32 vertex_count   (vertex units, 12 B each)
  +0x0c u32 index_first   +0x10 u32 index_count    (u16 units, mult of 3)

TEXTURE rec (0x1c = 28)  array[texture_count] @ texture_off
  +0x00 u16 width +0x02 u16 height
  +0x04 u32 format (PMAP_FMT_* == GU_PSM_*)
  +0x08 u32 texel_first +0x0c u32 texel_bytes
  +0x10 u32 buffer_width (TEXELS)
  +0x14 u32 clut_first   +0x18 u32 clut_entries (0/16/256)

INSTANCE rec (0x24 = 36)  array[instance_count] @ instance_off
  +0x00 u32 model (index into MODEL table)
  +0x04 f32 pos_x +0x08 pos_y +0x0c pos_z
  +0x10 s16 qx +0x12 qy +0x14 qz +0x16 qw   (unit quat, fixed 1.15, 1.0==32767)
  +0x18 f32 scale
  +0x1c i32 interior (0 = world, drawn; else culled)
  +0x20 i32 cell

GRID (0x1c = 28) @ grid_off
  +0x00 f32 min_x +0x04 f32 min_y +0x08 f32 cell_size
  +0x0c u32 cells_x +0x10 u32 cells_y +0x14 u32 inst_index_count +0x18 u32 pad
  then  i32 cell_off[cells_x*cells_y + 1]
        u16 inst_index[inst_index_count]
  cell c owns inst_index[cell_off[c] .. cell_off[c+1]); the +1 sentinel == count.

then the four pools, each 16-byte aligned: vertex, index, texel, clut.
================================================================================
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MAGIC = b"PMAP"
PMAP_VERSION = 2

# PmapTexture.format / GU_PSM_* (identical numbering on the PSP).
PMAP_FMT_5650 = 0
PMAP_FMT_5551 = 1
PMAP_FMT_4444 = 2
PMAP_FMT_8888 = 3
PMAP_FMT_T4 = 4
PMAP_FMT_T8 = 5

PMAP_INTERIOR_WORLD = 0
VERTEX_SIZE = 12          # PmapVertex
INDEX_SIZE = 2            # u16

_ALIGN = 16


def _align(n: int) -> int:
    return (n + _ALIGN - 1) & ~(_ALIGN - 1)


# struct layouts (little-endian)
_HEADER = struct.Struct("<4s19I")            # 0x50
_MODEL = struct.Struct("<IIffffff")          # 0x20 (adds draw_dist)
_SUBMESH = struct.Struct("<iIIII")           # 0x14
_TEXTURE = struct.Struct("<HHIIIIIII")       # 0x20 (adds num_levels)
_INSTANCE = struct.Struct("<Ifffhhhhfii")    # 0x24
_GRID = struct.Struct("<fffIIII")            # 0x1c

HEADER_SIZE = _HEADER.size
assert HEADER_SIZE == 0x50, HEADER_SIZE
assert _MODEL.size == 0x20, _MODEL.size
assert _SUBMESH.size == 0x14, _SUBMESH.size
assert _TEXTURE.size == 0x20, _TEXTURE.size
assert _INSTANCE.size == 0x24, _INSTANCE.size
assert _GRID.size == 0x1c, _GRID.size


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Submesh:
    """One material/texture slice: a vertex block + a u16 (mesh-local) index
    block.  ``texture`` indexes the TEXTURE table, or -1 for untextured."""
    texture: int
    vertex_bytes: bytes        # PmapVertex[] (12 B each), from psp_mesh prim
    index_bytes: bytes         # u16[] local indices, from psp_mesh prim
    uvscroll: tuple = None      # (du_dt, dv_dt) UV/sec animated-texture scroll, or None (build-time only, -> .anim sidecar)
    # filled by write_scene:
    vertex_first: int = 0
    vertex_count: int = 0
    index_first: int = 0
    index_count: int = 0


@dataclass
class Model:
    """A run of submeshes + the uniform int16-position dequant (scale, center)
    from ``psp_mesh.pack_model``.  world_local = center + pos_i16 * scale."""
    submeshes: List[Submesh]
    scale: float = 1.0
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bound_radius: float = 0.0
    draw_dist: float = 300.0   # SA IDE per-model draw distance (LOD like original)
    sway_class: int = 0        # wind-sway: 0 none / 1 tree / 2 palm (IDE bIsTree/bIsPalm + name)
    sway_min_z: float = 0.0    # model-local MIN world-Z (base pivot for the matrix-shear)
    spin: tuple = None         # CAnimatedBuilding rotator (axis, mode, rate_deg_s, amplitude_deg)
                               # or None (build-time only, -> .spin sidecar)
    tobj: tuple = None         # SA IDE `tobj` hour window (on, off); bit 7 of `on` = IDE
                               # ADDITIVE, or None (build-time only, -> .tobj sidecar)
    # filled by write_scene:
    first_submesh: int = 0


@dataclass
class Texture:
    """A swizzled texel plane + (T4/T8) a CLUT.  ``format`` is a GU_PSM_* id."""
    width: int
    height: int
    format: int
    texel_bytes: bytes         # ALL mip levels concatenated (level 0 first)
    buffer_width: int          # GE texture buffer width in TEXELS (level 0)
    clut_bytes: bytes = b""    # RGBA8888, alpha last (empty if not paletted)
    clut_entries: int = 0      # 0 / 16 / 256
    num_levels: int = 1        # mip levels in texel_bytes (1 = no mips)
    # filled by write_scene:
    texel_first: int = 0
    clut_first: int = 0


@dataclass
class Instance:
    """One world placement of a model (index into the MODEL table)."""
    model: int
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # xyzw
    scale: float = 1.0
    interior: int = PMAP_INTERIOR_WORLD
    # filled by write_scene:
    cell: int = 0


@dataclass
class Grid:
    """Uniform XY cell grid (the source game ground plane) for streaming / culling."""
    cell_size: float
    min_x: float = 0.0
    min_y: float = 0.0
    cells_x: int = 1
    cells_y: int = 1

    @property
    def cell_count(self) -> int:
        return self.cells_x * self.cells_y

    def cell_of(self, x: float, y: float) -> int:
        cx = int((x - self.min_x) // self.cell_size)
        cy = int((y - self.min_y) // self.cell_size)
        cx = 0 if cx < 0 else (self.cells_x - 1 if cx >= self.cells_x else cx)
        cy = 0 if cy < 0 else (self.cells_y - 1 if cy >= self.cells_y else cy)
        return cy * self.cells_x + cx


@dataclass
class Scene:
    models: List[Model]
    textures: List[Texture]
    instances: List[Instance]
    grid: Grid


_S16 = 32767


def _q15(v: float) -> int:
    q = int(round(v * _S16))
    return -32768 if q < -32768 else (_S16 if q > _S16 else q)


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def write_scene(
    models: List[Model],
    textures: List[Texture],
    instances: List[Instance],
    grid: Optional[Grid] = None,
) -> bytes:
    """Serialise into a v2 '.pmap' the engine consumes verbatim.

    Instances are sorted by their grid cell so each cell owns a contiguous run
    of the instance array (inst_index is then the identity permutation).
    """
    if grid is None:
        grid = Grid(cell_size=1.0, cells_x=1, cells_y=1)
    cells = grid.cell_count

    # ---- pools ----
    vertex_pool = bytearray()
    index_pool = bytearray()
    for m in models:
        m.first_submesh = -1  # set below in submesh enumeration order
    # submeshes are laid out model-major so a model owns a contiguous run.
    submeshes: List[Submesh] = []
    for m in models:
        m.first_submesh = len(submeshes)
        for sm in m.submeshes:
            vcount = len(sm.vertex_bytes) // VERTEX_SIZE
            icount = len(sm.index_bytes) // INDEX_SIZE
            sm.vertex_first = len(vertex_pool) // VERTEX_SIZE
            sm.vertex_count = vcount
            sm.index_first = len(index_pool) // INDEX_SIZE
            sm.index_count = icount
            vertex_pool += sm.vertex_bytes
            index_pool += sm.index_bytes
            submeshes.append(sm)

    texel_pool = bytearray()
    clut_pool = bytearray()
    for t in textures:
        # align each swizzled plane / clut to 16 within its pool so the absolute
        # address (pool base is 16-aligned) is GE-friendly.
        if len(texel_pool) % _ALIGN:
            texel_pool += b"\x00" * (_align(len(texel_pool)) - len(texel_pool))
        t.texel_first = len(texel_pool)
        texel_pool += t.texel_bytes
        if t.clut_bytes:
            if len(clut_pool) % _ALIGN:
                clut_pool += b"\x00" * (_align(len(clut_pool)) - len(clut_pool))
            t.clut_first = len(clut_pool)
            clut_pool += t.clut_bytes
        else:
            t.clut_first = 0

    # ---- instances bucketed by cell ----
    for inst in instances:
        inst.cell = grid.cell_of(inst.pos[0], inst.pos[1])
    order = sorted(range(len(instances)), key=lambda i: instances[i].cell)
    sorted_inst = [instances[i] for i in order]

    cell_off = [0] * (cells + 1)
    for inst in sorted_inst:
        cell_off[inst.cell + 1] += 1
    for c in range(cells):
        cell_off[c + 1] += cell_off[c]
    inst_index = list(range(len(sorted_inst)))  # identity (instances pre-sorted)

    # ---- offsets ----
    off = HEADER_SIZE
    model_off = off;     off = _align(off + _MODEL.size * len(models))
    submesh_off = off;   off = _align(off + _SUBMESH.size * len(submeshes))
    texture_off = off;   off = _align(off + _TEXTURE.size * len(textures))
    instance_off = off;  off = _align(off + _INSTANCE.size * len(sorted_inst))
    grid_off = off
    grid_blk = _GRID.size + 4 * (cells + 1) + 2 * len(inst_index)
    off = _align(off + grid_blk)
    vertex_off = off;    off = _align(off + len(vertex_pool))
    index_off = off;     off = _align(off + len(index_pool))
    texel_off = off;     off = _align(off + len(texel_pool))
    clut_off = off;      off = _align(off + len(clut_pool))
    file_size = off

    out = bytearray(file_size)
    _HEADER.pack_into(
        out, 0, MAGIC, PMAP_VERSION, file_size,
        len(models), model_off,
        len(submeshes), submesh_off,
        len(textures), texture_off,
        len(sorted_inst), instance_off,
        grid_off,
        vertex_off, len(vertex_pool),
        index_off, len(index_pool),
        texel_off, len(texel_pool),
        clut_off, len(clut_pool),
    )

    p = model_off
    for m in models:
        _MODEL.pack_into(out, p, m.first_submesh, len(m.submeshes),
                         m.scale, m.center[0], m.center[1], m.center[2],
                         m.bound_radius, m.draw_dist)
        p += _MODEL.size

    p = submesh_off
    for sm in submeshes:
        _SUBMESH.pack_into(out, p, sm.texture, sm.vertex_first, sm.vertex_count,
                           sm.index_first, sm.index_count)
        p += _SUBMESH.size

    p = texture_off
    for t in textures:
        _TEXTURE.pack_into(out, p, t.width, t.height, t.format,
                           t.texel_first, len(t.texel_bytes),
                           t.buffer_width, t.clut_first, t.clut_entries,
                           t.num_levels)
        p += _TEXTURE.size

    p = instance_off
    for inst in sorted_inst:
        q = inst.quat
        _INSTANCE.pack_into(out, p, inst.model,
                            inst.pos[0], inst.pos[1], inst.pos[2],
                            _q15(q[0]), _q15(q[1]), _q15(q[2]), _q15(q[3]),
                            inst.scale, inst.interior, inst.cell)
        p += _INSTANCE.size

    _GRID.pack_into(out, grid_off, grid.min_x, grid.min_y, grid.cell_size,
                    grid.cells_x, grid.cells_y, len(inst_index), 0)
    p = grid_off + _GRID.size
    for v in cell_off:
        struct.pack_into("<i", out, p, v); p += 4
    for v in inst_index:
        struct.pack_into("<H", out, p, v); p += 2

    out[vertex_off:vertex_off + len(vertex_pool)] = vertex_pool
    out[index_off:index_off + len(index_pool)] = index_pool
    out[texel_off:texel_off + len(texel_pool)] = texel_pool
    out[clut_off:clut_off + len(clut_pool)] = clut_pool
    return bytes(out)


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def read_scene(data: bytes) -> Scene:
    """Parse a v2 '.pmap'.  ``read_scene(write_scene(...))`` re-serialises
    byte-exact."""
    data = bytes(data)
    if len(data) < HEADER_SIZE:
        raise ValueError("truncated PMAP header")
    (magic, version, file_size,
     model_count, model_off,
     submesh_count, submesh_off,
     texture_count, texture_off,
     instance_count, instance_off,
     grid_off,
     vertex_off, vertex_bytes,
     index_off, index_bytes,
     texel_off, texel_bytes,
     clut_off, clut_bytes) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != PMAP_VERSION:
        raise ValueError(f"unsupported PMAP version {version}")

    vpool = data[vertex_off:vertex_off + vertex_bytes]
    ipool = data[index_off:index_off + index_bytes]
    tpool = data[texel_off:texel_off + texel_bytes]
    cpool = data[clut_off:clut_off + clut_bytes]

    # submeshes
    raw_sm = []
    p = submesh_off
    for _ in range(submesh_count):
        tex, vf, vc, if_, ic = _SUBMESH.unpack_from(data, p)
        raw_sm.append(Submesh(
            texture=tex,
            vertex_bytes=vpool[vf * VERTEX_SIZE:(vf + vc) * VERTEX_SIZE],
            index_bytes=ipool[if_ * INDEX_SIZE:(if_ + ic) * INDEX_SIZE],
            vertex_first=vf, vertex_count=vc, index_first=if_, index_count=ic,
        ))
        p += _SUBMESH.size

    models = []
    p = model_off
    for _ in range(model_count):
        fs, sc, scale, cx, cy, cz, br, dd = _MODEL.unpack_from(data, p)
        models.append(Model(
            submeshes=raw_sm[fs:fs + sc], scale=scale,
            center=(cx, cy, cz), bound_radius=br, draw_dist=dd, first_submesh=fs,
        ))
        p += _MODEL.size

    textures = []
    p = texture_off
    for _ in range(texture_count):
        w, h, fmt, tf, tb, bw, cf, ce, nl = _TEXTURE.unpack_from(data, p)
        textures.append(Texture(
            width=w, height=h, format=fmt,
            texel_bytes=tpool[tf:tf + tb], buffer_width=bw,
            clut_bytes=(cpool[cf:cf + ce * 4] if ce else b""),
            clut_entries=ce, num_levels=nl, texel_first=tf, clut_first=cf,
        ))
        p += _TEXTURE.size

    instances = []
    p = instance_off
    for _ in range(instance_count):
        (model, px, py, pz, qx, qy, qz, qw,
         scale, interior, cell) = _INSTANCE.unpack_from(data, p)
        instances.append(Instance(
            model=model, pos=(px, py, pz),
            quat=(qx / _S16, qy / _S16, qz / _S16, qw / _S16),
            scale=scale, interior=interior, cell=cell,
        ))
        p += _INSTANCE.size

    mnx, mny, csz, cax, cay, iic, _pad = _GRID.unpack_from(data, grid_off)
    grid = Grid(cell_size=csz, min_x=mnx, min_y=mny, cells_x=cax, cells_y=cay)

    return Scene(models=models, textures=textures,
                 instances=instances, grid=grid)


def read_cell_ranges(data: bytes) -> List[Tuple[int, int]]:
    """Return [(first, count), ...] per cell from the grid cell_off table."""
    data = bytes(data)
    fields = _HEADER.unpack_from(data, 0)
    grid_off = fields[11]
    _mnx, _mny, _csz, cax, cay, _iic, _pad = _GRID.unpack_from(data, grid_off)
    cells = cax * cay
    base = grid_off + _GRID.size
    offs = [struct.unpack_from("<i", data, base + 4 * i)[0]
            for i in range(cells + 1)]
    return [(offs[c], offs[c + 1] - offs[c]) for c in range(cells)]
