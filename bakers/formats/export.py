"""Asset EXPORT - decoded the source game assets -> standard interchange formats for download.

Thin conversion layer over the tested decoders (txd / dff / col). Every function
takes the RAW asset bytes and returns a ready-to-serve artifact:

 * txd_to_pngs(data) -> [(name+'.png', png_bytes)] one PNG per TextureNative.
 * txd_to_zip(data) -> bytes all PNGs in a single .zip.
 * dff_to_gltf(data) -> bytes glTF 2.0 JSON, full model,
 embedded base64 buffer.
 * dff_to_obj(data) -> str Wavefront OBJ, full model,
 grouped per part with usemtl.
 * col_to_obj(data) -> str OBJ of all collision meshes.

The DFF exporters assemble EVERY atomic into one mesh, applying each atomic's frame
world-transform (same math as server.app.assemble_model) so a multi-part model
(vehicles, weapons) comes out whole and correctly posed.

House style matches formats/txd.py: dataclasses where useful, defensive (one bad
element never kills the whole export), module docstring per public function.
"""
from __future__ import annotations

import base64
import io
import json
import struct
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

from formats import col as COL
from formats import dff as DFF
from formats import txd as TXD

_IDENTITY = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])


# =========================================================================
# TXD -> PNG
# =========================================================================

def _safe_name(name: str, index: int) -> str:
    """A filesystem-safe base name for a texture (no separators / blanks)."""
    nm = (name or "").strip()
    nm = nm.replace("/", "_").replace("\\", "_").replace(":", "_")
    if not nm:
        nm = f"texture_{index}"
    return nm


def txd_to_pngs(data: bytes) -> List[Tuple[str, bytes]]:
    """Decode every texture in a TXD to a PNG.

 Returns a list of (filename, png_bytes); filename is the texture name with a
 '.png' suffix (deduplicated, separators stripped). A texture that fails to decode
 is skipped rather than aborting the whole dictionary.
 """
    txd = TXD.parse_txd(data)
    out: List[Tuple[str, bytes]] = []
    used: dict = {}
    for i, tex in enumerate(txd.textures):
        base = _safe_name(tex.name, i)
        # de-duplicate collisions (TXDs can repeat names / carry <error> placeholders)
        n = used.get(base.lower(), 0)
        used[base.lower()] = n + 1
        fname = f"{base}.png" if n == 0 else f"{base}_{n}.png"
        try:
            png = _texture_to_png(tex)
        except Exception:
            continue
        out.append((fname, png))
    return out


def _texture_to_png(tex: "TXD.Texture") -> bytes:
    """One Texture's level-0 raster -> PNG bytes (RGBA)."""
    w = max(int(tex.width), 1)
    h = max(int(tex.height), 1)
    rgba = tex.rgba(0)
    expected = w * h * 4
    if len(rgba) < expected:
        rgba = bytes(rgba) + b"\x00" * (expected - len(rgba))
    elif len(rgba) > expected:
        rgba = rgba[:expected]
    img = Image.frombytes("RGBA", (w, h), bytes(rgba))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def txd_to_zip(data: bytes) -> bytes:
    """All textures of a TXD as PNGs inside a single .zip (stdlib zipfile)."""
    pngs = txd_to_pngs(data)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, png in pngs:
            zf.writestr(name, png)
    return buf.getvalue()


# =========================================================================
# DFF assembly (shared by OBJ + glTF)
# =========================================================================

@dataclass
class _Part:
    """One assembled atomic's contribution to the merged model."""
    start: int                      # first flat index into the shared index array
    count: int                      # number of flat indices
    texture: str = ""               # texture name (group / usemtl label), may be 


@dataclass
class _Assembled:
    positions: List[float] = field(default_factory=list)   # x,y,z flat
    uvs: List[float] = field(default_factory=list)         # u,v flat (parallel to positions)
    indices: List[int] = field(default_factory=list)       # flat triangle indices (0-based)
    parts: List[_Part] = field(default_factory=list)
    has_any_uv: bool = False

    @property
    def num_vertices(self) -> int:
        return len(self.positions) // 3


def _frame_worlds(frames):
    """Resolve each frame's world transform (R, T) by walking parents.

 Mirrors server.app._frame_worlds: cached, identity fallback for an out-of-range
 parent or a model with no frames.
 """
    n = len(frames)
    cache: List[Optional[tuple]] = [None] * n

    def world(i: int):
        if i < 0 or i >= n:
            return _IDENTITY
        if cache[i] is not None:
            return cache[i]
        f = frames[i]
        R = [list(f.rotation[0]), list(f.rotation[1]), list(f.rotation[2])]
        T = list(f.position)
        parent = f.parent
        if 0 <= parent < n and parent != i:
            PR, PT = world(parent)
            # world = parent * local
            Rw = [[sum(PR[r][k] * R[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
            Tw = [PR[r][0] * T[0] + PR[r][1] * T[1] + PR[r][2] * T[2] + PT[r] for r in range(3)]
            cache[i] = (Rw, Tw)
        else:
            cache[i] = (R, T)
        return cache[i]

    return [world(i) for i in range(n)]


def _assemble(d: "DFF.Dff") -> _Assembled:
    """Assemble all atomics into one mesh with frame world-transforms applied.

 Same algorithm as server.app.assemble_model: positions + parallel uvs +
 flat (de-stripped) indices, plus a part record per atomic for OBJ grouping.
 """
    worlds = _frame_worlds(d.frames)
    asm = _Assembled()
    for atom in d.atomics:
        gi = atom.geometry_index
        if gi < 0 or gi >= len(d.geometries):
            continue
        g = d.geometries[gi]
        if not g.vertices:
            continue
        Rw, Tw = worlds[atom.frame_index] if 0 <= atom.frame_index < len(worlds) else _IDENTITY
        base = asm.num_vertices
        has_uv = bool(g.uvs and g.uvs[0])
        if has_uv:
            asm.has_any_uv = True
        for vi, v in enumerate(g.vertices):
            x = Rw[0][0] * v[0] + Rw[0][1] * v[1] + Rw[0][2] * v[2] + Tw[0]
            y = Rw[1][0] * v[0] + Rw[1][1] * v[1] + Rw[1][2] * v[2] + Tw[1]
            z = Rw[2][0] * v[0] + Rw[2][1] * v[1] + Rw[2][2] * v[2] + Tw[2]
            asm.positions += [x, y, z]
            if has_uv:
                u, w = g.uvs[0][vi]
                asm.uvs += [u, w]
            else:
                asm.uvs += [0.0, 0.0]
        flat = DFF._flat_indices(g)
        start = len(asm.indices)
        for ix in flat:
            asm.indices.append(ix + base)
        tex = ""
        if g.materials and g.materials[0].texture_name:
            tex = g.materials[0].texture_name
        asm.parts.append(_Part(start=start, count=len(flat), texture=tex))
    return asm


# =========================================================================
# DFF -> OBJ
# =========================================================================

def dff_to_obj(data: bytes) -> str:
    """Wavefront OBJ of the full assembled model.

 Emits all v / vt lines first (1-based indexing), then one face group per atomic
 (`g part_NN` + `usemtl <texture>`), with `f a/a b/b c/c` triangles. UVs are written
 only when at least one geometry is textured (else plain `f a b c`).
 """
    d = DFF.parse_dff(data)
    asm = _assemble(d)
    write_uv = asm.has_any_uv

    lines: List[str] = ["# SAW DFF export (assembled model)",
                        f"# vertices {asm.num_vertices} triangles {len(asm.indices) // 3}"]

    for i in range(0, len(asm.positions), 3):
        x, y, z = asm.positions[i], asm.positions[i + 1], asm.positions[i + 2]
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    if write_uv:
        for i in range(0, len(asm.uvs), 2):
            u, v = asm.uvs[i], asm.uvs[i + 1]
            # OBJ V axis is bottom-up; RW UVs are top-down -> flip for correct texturing
            lines.append(f"vt {u:.6f} {1.0 - v:.6f}")

    idx = asm.indices
    for pi, part in enumerate(asm.parts):
        if part.count <= 0:
            continue
        lines.append(f"g part_{pi}")
        lines.append(f"usemtl {part.texture or f'material_{pi}'}")
        end = part.start + part.count
        j = part.start
        while j + 2 < end + 1 and j + 2 < len(idx):
            a = idx[j] + 1
            b = idx[j + 1] + 1
            c = idx[j + 2] + 1
            if write_uv:
                lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            else:
                lines.append(f"f {a} {b} {c}")
            j += 3
    return "\n".join(lines) + "\n"


# =========================================================================
# DFF -> glTF 2.0
# =========================================================================

def dff_to_gltf(data: bytes) -> bytes:
    """Full assembled model as glTF 2.0 JSON bytes with an embedded base64 buffer.

 One mesh / one primitive: POSITION (+ TEXCOORD_0 when textured) + u32 indices,
 each attribute in its own bufferView of a single binary buffer (data URI).
 """
    d = DFF.parse_dff(data)
    asm = _assemble(d)
    pos = asm.positions
    uvs = asm.uvs if asm.has_any_uv else []
    idx = asm.indices
    n_vert = len(pos) // 3

    blob = bytearray()
    buffer_views: List[dict] = []
    accessors: List[dict] = []

    # POSITION
    pos_off = len(blob)
    blob += struct.pack("<%df" % len(pos), *pos) if pos else b""
    if n_vert:
        xs = pos[0::3]
        ys = pos[1::3]
        zs = pos[2::3]
        pmin = [min(xs), min(ys), min(zs)]
        pmax = [max(xs), max(ys), max(zs)]
    else:
        pmin = pmax = [0.0, 0.0, 0.0]
    buffer_views.append({"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos) * 4, "target": 34962})
    accessors.append({
        "bufferView": 0, "componentType": 5126, "count": n_vert,
        "type": "VEC3", "min": pmin, "max": pmax,
    })
    attributes = {"POSITION": 0}

    # TEXCOORD_0 (optional)
    if uvs:
        uv_off = len(blob)
        blob += struct.pack("<%df" % len(uvs), *uvs)
        bv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": uv_off, "byteLength": len(uvs) * 4, "target": 34962})
        accessors.append({"bufferView": bv, "componentType": 5126, "count": len(uvs) // 2, "type": "VEC2"})
        attributes["TEXCOORD_0"] = len(accessors) - 1

    # indices (u32) - byte offset is already 4-aligned (f32 region precedes it)
    idx_off = len(blob)
    blob += struct.pack("<%dI" % len(idx), *idx) if idx else b""
    bv = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx) * 4, "target": 34963})
    accessors.append({"bufferView": bv, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
    idx_accessor = len(accessors) - 1

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")

    gltf = {
        "asset": {"version": "2.0", "generator": "SAW export.py"},
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
    return json.dumps(gltf).encode("utf-8")


# =========================================================================
# COL -> OBJ
# =========================================================================

def col_to_obj(data: bytes) -> str:
    """OBJ of every collision mesh in a .col library, merged into one object.

 Vertices of all models are concatenated (already dequantized metres); face indices
 are rebased per model and written 1-based. Models with no mesh (sphere/box-only
 collision) contribute nothing.
 """
    models = COL.parse_col_library(data)
    lines: List[str] = ["# SAW COL export (merged collision meshes)",
                        f"# models {len(models)}"]
    vert_lines: List[str] = []
    face_lines: List[str] = []
    base = 0
    for m in models:
        for (x, y, z) in m.vertices:
            vert_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        nv = len(m.vertices)
        if m.faces:
            face_lines.append(f"g {_safe_name(m.name, m.model_id)}")
        for (a, b, c, _s) in m.faces:
            # guard against any stray out-of-range index from a corrupt chunk
            if 0 <= a < nv and 0 <= b < nv and 0 <= c < nv:
                face_lines.append(f"f {a + base + 1} {b + base + 1} {c + base + 1}")
        base += nv
    lines.extend(vert_lines)
    lines.extend(face_lines)
    return "\n".join(lines) + "\n"
