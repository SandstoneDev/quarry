"""PSP GE model geometry - VTYPE 0x000115 static-world meshes (the console title).

Format notes, derived from the retail files:
geometry lives inside a zone's `.IMG` as **quantized LOCAL/model-space** model blobs,
addressed by 0x20-byte 'DLRW' streaming descriptors embedded in the relocated `.LVZ`.

Pipeline
--------
* The `.LVZ` is the shared relocatable container (see :mod:`gvcslib.container`); its
  payload root is ``payload+0x20``.  ``root+0x04`` is the first category pair whose
  pointer (a payload-relative offset on disk, e.g. BEACH 0x364d4c) starts a contiguous
  table of 0x20-byte streaming descriptors.
* Each descriptor (engine ``FUN_00152498``, dispatch type 1 = MODEL):
    ``desc[+0x00]='DLRW'``, ``desc[+0x08]=read_size``, ``desc[+0x0c]=mem_size``,
    ``desc[+0x14]=section/sub-object count``, ``desc[+0x18]=IMG byte offset`` (2 KiB-aligned).
* The IMG is a headerless 2 KiB-aligned concatenation of raw payloads - read
  ``read_size`` bytes from ``IMG[img_off]`` to get one model blob.

Model blob layout (verified BEACH.IMG)
----------------------------------------------
::

    +0x00 u32   mesh-table offset   (e.g. 0x4f6c)  -> end of the vertex stream
    +0x04 u32   section count       (e.g. 0x0c)
    +0x08 u32 * ascending RW-style region offsets (start==end pairs)
    +0x2c u32   type/version marker  0x00950002 / 0x00950003
    +0x30..     per-mesh block header (16 bytes), then
    +0x40       3x fp16 SCALE triple  (local bounding half-extents)
    +0x44       packed VTYPE-0x115 vertex stream, runs up to the mesh-table offset

Vertex - VTYPE 0x000115 (GE cmd 0x12000115), **stride 10 bytes**, non-indexed tri-strip:
    +0x00 u8   U        (texcoord, /255 .. scaled at draw by GE cmd 0x48/0x49)
    +0x01 u8   V
    +0x02 u16  COLOR    RGBA5551
    +0x04 s16  X
    +0x06 s16  Y
    +0x08 s16  Z

Decoded LOCAL position: ``pos = (s16 / 32768) * per_mesh_scale``.  Coordinates clamp to
``+-scale`` per axis and are origin-centred (VC world space is thousands of units; these
are single-digit), proving the world is *instanced*, not baked.  The per-instance world
transform lives in GAME.DTZ (see :mod:`gvcslib.dtz_instances`), not here.

decode(blob) -> Model.  encode(model) -> bytes re-emits an EXISTING decoded blob
BYTE-EXACT (``encode(decode(blob)) == blob``): the structured float decode is lossy
(quantisation + a sub-stride vertex remainder + a 77 KiB post-vertex draw/material
table that is not yet reversed), so the Model retains the raw byte segments it cannot
losslessly reconstruct and ``encode`` re-serialises them grouped as header / per-mesh
header / vertex blocks / draw table.  This is the re-emit path for blobs that already
exist in the IMG - NOT a triangle-soup -> tri-strip generator (that is the converter,
built later).  ``to_obj`` exports the decoded mesh.
"""
import struct

try:  # numpy makes the bulk s16 decode fast; fall back to pure-python if absent.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from .container import Container

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MAGIC_DLRW = 0x57524C44               # 'DLRW' streaming-descriptor magic
# blob +0x2c type/version marker = <family> | <low-byte sub-type>. The two families below
# are seen on many VTYPE-0x115 model-blobs (12-u32 header, mesh_table_off=header[0],
# section_count=header[1], fp16 scale @+0x40, vertex stream @+0x44):
#   * 0x009500xx - BEACH 0x02/0x03, MAINLA 0x05-0x08, MALL 0x00
#   * 0x001200xx - much of BEACH (low bytes 0x02-0x09); e.g. BEACH.IMG
#                  is 0x00120005 and decodes to a 43692-vertex stream.
# WARNING (workflow w31opaze8): the +0x2c marker is NOT a reliable geometry/version gate.
# Many valid geometry bundles carry a marker OUTSIDE these families, so gating on it
# wrongly rejects them (it dropped placed-model coverage to ~53%). Detection is therefore
# done STRUCTURALLY (a plausible {model_id, in_blob_off} directory at mesh_table_off; see
# is_geometry_blob / model_map.parse_bundle_directory). The constants and _is_geom_marker
# below are RETAINED for reference / as a soft hint only, and for back-compat
# (model_map.is_geom_marker) - they must NOT be used as a hard accept/reject gate.
GEOM_MARKER_MASK = 0xFFFFFF00
GEOM_MARKER_FAMILIES = (0x00950000, 0x00120000)
GEOM_MARKER_MAX_SUBTYPE = 0x20
GEOM_MARKERS = (0x00950002, 0x00950003)  # retained for reference (original BEACH sub-types)


def _is_geom_marker(marker):
    """Soft hint: does +0x2c look like a known geometry marker family?

    Reference / hint ONLY - NOT a reliable version or geometry gate (workflow
    w31opaze8): valid bundles exist with markers outside these families. Real
    detection is structural (see :func:`is_geometry_blob`). Kept for back-compat
    (``model_map.is_geom_marker``) and diagnostics.
    """
    return ((marker & GEOM_MARKER_MASK) in GEOM_MARKER_FAMILIES
            and (marker & 0xFF) <= GEOM_MARKER_MAX_SUBTYPE)
VTYPE = 0x000115                       # PSP GE vertex type word (TEX u8x2, COLOR 5551, POS s16x3)
VERTEX_STRIDE = 10                     # bytes per VTYPE-0x115 vertex
DESC_SIZE = 0x20                       # streaming descriptor record size

HDR_FIELDS = 12                        # u32 entries in the blob offset-table header
SCALE_OFF = 0x40                       # per-mesh fp16 scale triple
VERT_START = 0x44                      # first VTYPE-0x115 vertex (UV@+0, color@+2, pos@+4)

# LVZ payload root (== Container.root) layout
ROOT = 0x20
CAT0_PTR_OFF = ROOT + 0x04             # first category pair -> streaming descriptor table


# ---------------------------------------------------------------------------
# fp16 helper (PSP half-floats); avoid hard numpy dependency for a 6-byte read
# ---------------------------------------------------------------------------
def _half_to_float(h):
    """IEEE-754 binary16 -> python float."""
    sign = (h >> 15) & 0x1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        val = frac / 1024.0 * (2.0 ** -14)
    elif exp == 0x1F:
        val = float('inf') if frac == 0 else float('nan')
    else:
        val = (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def _read_scale(blob, off=SCALE_OFF):
    h = struct.unpack_from('<3H', blob, off)
    return tuple(_half_to_float(x) for x in h)


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------
class Mesh:
    """One model-blob's local-space vertex stream (VTYPE 0x000115)."""
    __slots__ = ('scale', 'positions', 'uv', 'colors', 'marker', 'mesh_table_off',
                 'section_count', 'region_offsets', 'vert_start',
                 'raw_scale', 'raw_vertices')

    def __init__(self, scale, positions, uv, colors, marker, mesh_table_off,
                 section_count, region_offsets, vert_start=VERT_START,
                 raw_scale=b'', raw_vertices=b''):
        self.scale = scale                  # (sx, sy, sz) local half-extents
        self.positions = positions          # list[(x,y,z)] LOCAL-space floats
        self.uv = uv                        # list[(u,v)] floats in 0..1 (pre tex-scale)
        self.colors = colors                # list[int] RGBA5551
        self.marker = marker
        self.mesh_table_off = mesh_table_off
        self.section_count = section_count
        self.region_offsets = region_offsets
        self.vert_start = vert_start
        # Raw byte segments kept for BYTE-EXACT re-encode (see geometry.encode).
        # The float decode (positions/uv/colors) is lossy; raw_vertices holds the
        # exact ``n * VERTEX_STRIDE`` packed stream and raw_scale the exact 3 fp16.
        self.raw_scale = bytes(raw_scale)
        self.raw_vertices = bytes(raw_vertices)

    @property
    def vertex_count(self):
        return len(self.positions)

    def bounds(self):
        """((minx,miny,minz),(maxx,maxy,maxz)) over local positions."""
        if not self.positions:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


class Model:
    """A decoded model blob (one or more meshes; currently one vertex section).

    For BYTE-EXACT re-encode the model retains the *raw* byte segments that the
    structured float decode does not capture losslessly (the 0x30-byte offset/
    per-mesh header prefix, and the post-vertex draw/material table tail).  See
    :func:`encode` - the blob is rebuilt segment-by-segment from the decoded
    structure, not echoed as one opaque buffer.
    """
    __slots__ = ('meshes', 'marker', 'header', 'raw_prefix', 'raw_tail',
                 'blob_size')

    def __init__(self, meshes, marker, header, raw_prefix=b'', raw_tail=b'',
                 blob_size=0):
        self.meshes = meshes
        self.marker = marker
        self.header = header            # raw 12-u32 offset-table header
        # raw_prefix : bytes [0x00 .. scale_off)  (12-u32 header + per-mesh hdr)
        # raw_tail   : bytes [vert_start + n*stride .. end)  (leftover + draw table)
        self.raw_prefix = bytes(raw_prefix)
        self.raw_tail = bytes(raw_tail)
        self.blob_size = blob_size

    @property
    def vertex_count(self):
        return sum(m.vertex_count for m in self.meshes)

    def all_positions(self):
        out = []
        for m in self.meshes:
            out.extend(m.positions)
        return out


# ---------------------------------------------------------------------------
# core decode
# ---------------------------------------------------------------------------
def is_geometry_blob(blob):
    """Cheap structural test: does this IMG payload look like a model-geometry bundle?

    Judged by a plausible {model_id, in_blob_off} directory at mesh_table_off, NOT
    by the +0x2c marker (which is not a reliable version constant; workflow w31opaze8).
    """
    if len(blob) < VERT_START + VERTEX_STRIDE:
        return False
    try:
        mesh_table_off, _section_count = struct.unpack_from('<2I', blob, 0)
    except struct.error:
        return False
    if not (VERT_START + VERTEX_STRIDE <= mesh_table_off <= len(blob) - 8):
        return False
    try:
        model_id, in_off = struct.unpack_from('<2I', blob, mesh_table_off)
    except struct.error:
        return False
    return 1 <= model_id < 8192 and in_off < len(blob)


def _decode_positions_np(verts, n, scale):
    a = _np.frombuffer(verts[:n * VERTEX_STRIDE], dtype=_np.uint8).reshape(n, VERTEX_STRIDE)
    pos = _np.empty((n, 3), dtype=_np.int16)
    for i in range(3):
        lo = a[:, 4 + 2 * i].astype(_np.uint16)
        hi = a[:, 5 + 2 * i].astype(_np.uint16)
        pos[:, i] = (lo | (hi << 8)).astype(_np.int16)
    f = pos.astype(_np.float32) / 32768.0 * _np.asarray(scale, dtype=_np.float32)
    uv = a[:, 0:2].astype(_np.float32) / 255.0
    colors = (a[:, 2].astype(_np.uint16) | (a[:, 3].astype(_np.uint16) << 8))
    return (
        [tuple(map(float, r)) for r in f],
        [tuple(map(float, r)) for r in uv],
        [int(c) for c in colors],
    )


def _decode_positions_py(verts, n, scale):
    positions, uv, colors = [], [], []
    sx, sy, sz = scale
    for k in range(n):
        o = k * VERTEX_STRIDE
        u = verts[o]
        v = verts[o + 1]
        color = verts[o + 2] | (verts[o + 3] << 8)
        x, y, z = struct.unpack_from('<3h', verts, o + 4)
        positions.append((x / 32768.0 * sx, y / 32768.0 * sy, z / 32768.0 * sz))
        uv.append((u / 255.0, v / 255.0))
        colors.append(color)
    return positions, uv, colors


def decode(blob):
    """Decode a model blob (bytes) into a :class:`Model`.

    The vertex stream is decoded as VTYPE-0x115 (10-byte interleaved) from
    ``+0x44`` up to the per-mesh material/draw-table offset (header[0]).  Positions
    are dequantised to LOCAL/model space via the per-mesh fp16 scale triple.
    """
    blob = bytes(blob)
    if len(blob) < 0x30:
        raise ValueError("blob too small to be a model (%d bytes)" % len(blob))
    header = list(struct.unpack_from('<%dI' % HDR_FIELDS, blob, 0))
    mesh_table_off = header[0]
    section_count = header[1]
    region_offsets = header[2:11]
    marker = header[11]
    if not is_geometry_blob(blob):
        raise ValueError("not a geometry blob (no plausible directory at mesh_table_off)")
    if not (VERT_START + VERTEX_STRIDE <= mesh_table_off <= len(blob)):
        raise ValueError("implausible mesh-table offset 0x%x (blob %d bytes)"
                         % (mesh_table_off, len(blob)))

    scale = _read_scale(blob, SCALE_OFF)
    vstart = VERT_START
    vbytes = mesh_table_off - vstart
    n = vbytes // VERTEX_STRIDE
    vend = vstart + n * VERTEX_STRIDE
    verts = blob[vstart:vend]

    if _np is not None:
        positions, uv, colors = _decode_positions_np(verts, n, scale)
    else:  # pragma: no cover
        positions, uv, colors = _decode_positions_py(verts, n, scale)

    # Byte-exact partition of the blob into three NON-overlapping segments:
    #   raw_prefix  = [0x00 .. vstart)   12-u32 header + per-mesh hdr + scale lead
    #   raw_vertices= [vstart .. vend)   exact n * VERTEX_STRIDE packed VTYPE-0x115
    #   raw_tail    = [vend .. end)      stride remainder + the draw/material table
    # ``encode`` concatenates these to reproduce the source blob byte-for-byte.
    # (SCALE_OFF=0x40 straddles vstart=0x44, so the scale's leading bytes live in
    #  raw_prefix; we never re-pack scale for output, the prefix already holds it.)
    raw_prefix = blob[:vstart]
    raw_tail = blob[vend:]
    raw_scale = blob[SCALE_OFF:SCALE_OFF + 6]

    mesh = Mesh(scale, positions, uv, colors, marker, mesh_table_off,
                section_count, region_offsets, vstart,
                raw_scale=raw_scale, raw_vertices=verts)
    return Model([mesh], marker, header,
                 raw_prefix=raw_prefix, raw_tail=raw_tail, blob_size=len(blob))


# ---------------------------------------------------------------------------
# LVZ streaming-descriptor extraction (self-contained; uses gvcslib.lvz if present)
# ---------------------------------------------------------------------------
class ModelDescriptor:
    """A 0x20-byte 'DLRW' streaming descriptor (MODEL, dispatch type 1)."""
    __slots__ = ('img_off', 'read_size', 'mem_size', 'count', 'desc_off')

    def __init__(self, img_off, read_size, mem_size, count, desc_off):
        self.img_off = img_off          # desc[+0x18]  IMG byte offset (2 KiB-aligned)
        self.read_size = read_size      # desc[+0x08]  bytes to read from IMG
        self.mem_size = mem_size        # desc[+0x0c]  in-RAM size
        self.count = count              # desc[+0x14]  section/sub-object count
        self.desc_off = desc_off        # payload offset of this descriptor

    def __repr__(self):
        return ("ModelDescriptor(img_off=0x%x, read_size=0x%x, count=%d)"
                % (self.img_off, self.read_size, self.count))


def iter_streaming_descriptors(lvz_data):
    """Yield every contiguous 0x20-byte 'DLRW' streaming descriptor from a `.LVZ`.

    Accepts raw `.LVZ` bytes (zlib-wrapped) or an already-loaded
    :class:`gvcslib.container.Container`.  The descriptor table starts at the first
    category-pair pointer (``root+0x04``), which is a payload-relative offset.
    """
    if isinstance(lvz_data, Container):
        c = lvz_data
    else:
        c = Container.load(lvz_data)
    p = c.payload
    table_off = struct.unpack_from('<I', p, CAT0_PTR_OFF)[0]
    if not (0 < table_off < len(p)):
        return
    o = table_off
    while o + DESC_SIZE <= len(p):
        magic = struct.unpack_from('<I', p, o)[0]
        if magic != MAGIC_DLRW:
            break
        read_size = struct.unpack_from('<I', p, o + 0x08)[0]
        mem_size = struct.unpack_from('<I', p, o + 0x0c)[0]
        count = struct.unpack_from('<I', p, o + 0x14)[0]
        img_off = struct.unpack_from('<I', p, o + 0x18)[0]
        yield ModelDescriptor(img_off, read_size, mem_size, count, o)
        o += DESC_SIZE


def extract_blob(img_data, desc):
    """Slice one model blob out of an IMG given a :class:`ModelDescriptor`.

    ``img_data`` may be bytes or an open binary file object (recommended for the
    200 MB BEACH.IMG - avoids loading the whole archive into RAM).
    """
    off = desc.img_off
    size = desc.read_size
    if hasattr(img_data, 'seek') and hasattr(img_data, 'read'):
        img_data.seek(off)
        return img_data.read(size)
    return bytes(img_data[off:off + size])


def find_model_descriptors(lvz_data, img_data, limit=None, scan=None,
                           min_vertices=16, max_scale=10000.0):
    """Return ``[(ModelDescriptor, Model), ...]`` for blobs that decode as real meshes.

    Scans the LVZ streaming-descriptor table, reads each blob from the IMG, keeps only
    those that pass :func:`is_geometry_blob`, decode to a non-degenerate mesh
    (``>= min_vertices`` vertices, at least one non-zero scale axis, and no absurd
    scale ``> max_scale`` that would indicate a billboard/LOD/placeholder rather than
    a LOCAL-space model), then decodes them.  ``limit`` caps the number returned;
    ``scan`` caps how many descriptors are inspected.

    The road-strip case (one legitimately long axis, e.g. ~1135 units) is preserved:
    ``max_scale`` defaults well above it but rejects the multi-thousand placeholder
    scales seen on degenerate blobs.
    """
    out = []
    inspected = 0
    for desc in iter_streaming_descriptors(lvz_data):
        if scan is not None and inspected >= scan:
            break
        inspected += 1
        if desc.img_off == 0:
            continue
        try:
            blob = extract_blob(img_data, desc)
        except Exception:
            continue
        if not is_geometry_blob(blob):
            continue
        try:
            model = decode(blob)
        except Exception:
            continue
        mesh = model.meshes[0]
        sc = mesh.scale
        if mesh.vertex_count < min_vertices:
            continue
        if not any(abs(s) > 1e-6 for s in sc):
            continue
        if any(abs(s) > max_scale for s in sc):
            continue
        out.append((desc, model))
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------
def to_obj(model_or_mesh, path, name="vcs_model"):
    """Write decoded LOCAL-space vertices to a Wavefront .OBJ file.

    Emits ``v``/``vt`` records.  Triangle-strip topology (PRIM type 4) is not yet fully
    reversed, so faces are emitted as a best-effort strip per mesh (degenerate tris that
    collapse to a line/point are skipped); the vertex cloud + UVs are exact.
    """
    if isinstance(model_or_mesh, Model):
        meshes = model_or_mesh.meshes
    elif isinstance(model_or_mesh, Mesh):
        meshes = [model_or_mesh]
    else:
        meshes = list(model_or_mesh)

    lines = ["# console PSP GE geometry (VTYPE 0x000115) - gvcslib.geometry",
             "o %s" % name]
    vbase = 1
    for mi, m in enumerate(meshes):
        lines.append("g mesh_%d" % mi)
        for (x, y, z) in m.positions:
            lines.append("v %.6f %.6f %.6f" % (x, y, z))
        for (u, v) in m.uv:
            lines.append("vt %.6f %.6f" % (u, v))
        # tri-strip faces (degenerate-safe). topology is approximate, see docstring.
        pos = m.positions
        for k in range(len(pos) - 2):
            a, b, c = vbase + k, vbase + k + 1, vbase + k + 2
            ia, ib, ic = k, k + 1, k + 2
            if pos[ia] == pos[ib] or pos[ib] == pos[ic] or pos[ia] == pos[ic]:
                continue  # degenerate strip joint
            if k & 1:
                lines.append("f %d/%d %d/%d %d/%d" % (b, b, a, a, c, c))
            else:
                lines.append("f %d/%d %d/%d %d/%d" % (a, a, b, b, c, c))
        vbase += len(pos)

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# encode - byte-exact re-emission of an EXISTING decoded blob
# ---------------------------------------------------------------------------
def encode(model):
    """Re-emit a decoded model blob (bytes), BYTE-EXACT with the source.

    Scope (per the codec contract): re-serialise the structure that :func:`decode`
    produced - ``encode(decode(blob)) == blob`` byte-for-byte.  This is the
    re-emit path for blobs that *already exist* in the IMG; it is **not** a
    triangle-soup -> tri-strip generator (that is the converter, built later).

    Grouped emission, matching the on-disk model layout:

        1. header        : 12 u32 re-packed from ``model.header``
        2. per-mesh hdr  : the bytes between the header and the vertex stream
                           (carried verbatim in ``model.raw_prefix``; includes the
                           leading fp16 scale, which overlaps the vertex start)
        3. vertex blocks : every mesh's exact ``raw_vertices`` (VTYPE-0x115, stride
                           10), concatenated in submesh order
        4. draw table    : the post-vertex material/draw-table tail
                           (``model.raw_tail`` - stride remainder + table)

    The 12-u32 header is genuinely re-built from the decoded ``header`` field and
    must match the prefix's leading bytes (validated below); the remaining segments
    are the lossless raw bytes the float decode cannot reconstruct (quantisation),
    so they are re-emitted from the structure verbatim rather than faked.
    """
    if not isinstance(model, Model):
        raise TypeError("geometry.encode expects a Model, got %r" % type(model))
    if not model.raw_prefix or not model.meshes:
        raise ValueError(
            "Model was not produced by geometry.decode (no raw segments to re-emit); "
            "synthesising a blob from arbitrary geometry is out of scope - that is "
            "the converter, not this re-emit codec.")

    # 1. header: re-pack the 12 u32 from the decoded structure and confirm it
    #    reproduces the leading 0x30 bytes of the captured prefix exactly.
    header_bytes = struct.pack('<%dI' % HDR_FIELDS, *model.header)
    hlen = HDR_FIELDS * 4  # 0x30
    if model.raw_prefix[:hlen] != header_bytes:
        raise ValueError("header re-pack mismatch: decoded header does not match blob")

    out = bytearray()
    # 1 + 2: re-packed header followed by the per-mesh-header/scale-lead prefix tail.
    out += header_bytes
    out += model.raw_prefix[hlen:]

    # 3: all vertex blocks, contiguous in submesh order.
    for m in model.meshes:
        out += m.raw_vertices

    # 4: the draw/material table tail (and any sub-stride vertex remainder).
    out += model.raw_tail

    blob = bytes(out)
    if model.blob_size and len(blob) != model.blob_size:
        raise ValueError("re-emitted size 0x%x != source 0x%x"
                         % (len(blob), model.blob_size))
    return blob
