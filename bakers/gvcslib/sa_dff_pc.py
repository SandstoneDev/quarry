"""the source game PC (Direct3D, generic RW) DFF geometry decoder -> SaModel.

PC SA DFFs store *platform-independent* RW geometry (the native bit
rpGEOMETRYNATIVE 0x01000000 is CLEAR): plain f32 positions/UVs, RwRGBA prelit,
optional f32 normals, a triangle list in the GEOMETRY STRUCT, and a Bin-Mesh PLG
(0x050E) that splits the triangles per material as index lists. This is unlike
the PS2-native DMA/VIF path in sa_dff; it is the format the PC decomp renders.

Output reuses gvcslib.sa_dff.SaModel / SaMesh so psp_mesh.pack_model is unchanged:
one SaMesh per Bin-Mesh split (one material), vertices remapped to mesh-local.
"""
from __future__ import annotations

import struct
from typing import List

from . import sa_dff
from .sa_dff import (SaModel, SaMesh, parse_chunks, GEOMETRYLIST, GEOMETRY,
                     STRUCT, EXTENSION, _get_material_names)

BINMESH_PLG = 0x050E

# rpGeometry format flags
RPGEOMETRY_TRISTRIP = 0x00000001
RPGEOMETRY_PRELIT   = 0x00000008
RPGEOMETRY_NORMALS  = 0x00000010
RPGEOMETRY_TEXTURED = 0x00000004
RPGEOMETRY_TEXTURED2 = 0x00000080
RPGEOMETRY_NATIVE   = 0x01000000


def _num_tex_sets(flags: int) -> int:
    n = (flags >> 16) & 0xFF
    if n:
        return n
    if flags & RPGEOMETRY_TEXTURED2:
        return 2
    if flags & RPGEOMETRY_TEXTURED:
        return 1
    return 0


def _strip_to_tris(idx: List[int]) -> List[tuple]:
    """Triangle-strip index list -> triangles, alternating winding, skip degenerate."""
    tris = []
    for k in range(2, len(idx)):
        a, b, c = idx[k-2], idx[k-1], idx[k]
        if a == b or b == c or a == c:
            continue
        if k & 1:
            tris.append((b, a, c))
        else:
            tris.append((a, b, c))
    return tris


def _parse_binmesh(blob: bytes, ext) -> list:
    """Return [(matIndex, [indices...]), ...] from Bin-Mesh PLG, or []."""
    if not ext:
        return []
    bm = None
    for c in ext.children:
        if c.type == BINMESH_PLG:
            bm = c
            break
    if not bm or bm.size < 12:
        return []
    o = bm.data_off
    flags, num, total = struct.unpack_from("<3I", blob, o)
    o += 12
    out = []
    end = bm.data_off + bm.size
    for _ in range(num):
        if o + 8 > end:
            break
        numidx, mat = struct.unpack_from("<2i", blob, o)
        o += 8
        idx = []
        for _i in range(numidx):
            if o + 4 > end:
                break
            idx.append(struct.unpack_from("<I", blob, o)[0])
            o += 4
        out.append((mat, idx))
    return out, flags


def decode(blob) -> SaModel:
    """Decode a PC generic-geometry DFF blob into a SaModel."""
    blob = bytes(blob)
    root = parse_chunks(blob)
    model = SaModel()

    gl = root.find(GEOMETRYLIST)
    if not gl:
        raise ValueError("No GeometryList in DFF")

    for geo in gl.find_all(GEOMETRY):
        mat_base = len(model.materials)
        model.materials.extend(_get_material_names(blob, geo))

        st = geo.find(STRUCT)
        if not st:
            continue
        o = st.data_off
        flags, ntri, nvert, nmorph = struct.unpack_from("<4I", blob, o)
        o += 16
        if flags & RPGEOMETRY_NATIVE:
            # native (PS2/Xbox) geometry -> not handled here; use sa_dff.
            continue

        # --- prelit colours (RwRGBA, R,G,B,A bytes) ---
        colors = [0xFFFFFFFF] * nvert
        if flags & RPGEOMETRY_PRELIT:
            for i in range(nvert):
                r, g, b, a = struct.unpack_from("<4B", blob, o + i*4)
                colors[i] = (r << 24) | (g << 16) | (b << 8) | a
            o += nvert * 4

        # --- texcoords (f32 u,v per set); keep set 0 ---
        nsets = _num_tex_sets(flags)
        uvs = [(0.0, 0.0)] * nvert
        for s in range(nsets):
            if s == 0:
                for i in range(nvert):
                    u, v = struct.unpack_from("<2f", blob, o + i*8)
                    uvs[i] = (u, v)
            o += nvert * 8

        # --- triangles: on disk [v1, v0, matId, v2] (u16) ---
        tri_raw = []
        for i in range(ntri):
            v1, v0, matid, v2 = struct.unpack_from("<4H", blob, o + i*8)
            tri_raw.append((v0, v1, v2, matid))
        o += ntri * 8

        # --- morph target 0: bounding sphere(4 f32) + hasVerts + hasNormals ---
        positions = [(0.0, 0.0, 0.0)] * nvert
        for m in range(nmorph):
            o += 16  # bounding sphere x,y,z,radius
            has_v, has_n = struct.unpack_from("<2I", blob, o)
            o += 8
            if has_v:
                if m == 0:
                    for i in range(nvert):
                        x, y, z = struct.unpack_from("<3f", blob, o + i*12)
                        positions[i] = (x, y, z)
                o += nvert * 12
            if has_n:
                o += nvert * 12   # normals (unused for now)
            if m == 0:
                break  # only first morph target

        # --- Bin-Mesh splits (per material index lists) ---
        bm = _parse_binmesh(blob, geo.find(EXTENSION))
        if bm:
            meshes, bm_flags = bm
            tristrip = bool(bm_flags & 1)
        else:
            # fall back to the geometry triangle list grouped by matId
            meshes = None
            tristrip = bool(flags & RPGEOMETRY_TRISTRIP)

        def emit(mat_index, tri_list):
            """tri_list: list of (a,b,c) global vert indices -> mesh-local SaMesh."""
            if not tri_list:
                return
            remap = {}
            loc_pos = []; loc_uv = []; loc_col = []
            loc_tris = []
            for (a, b, c) in tri_list:
                tri = []
                for gi in (a, b, c):
                    if gi >= nvert:
                        tri = None
                        break
                    li = remap.get(gi)
                    if li is None:
                        li = len(loc_pos)
                        remap[gi] = li
                        loc_pos.append(positions[gi])
                        loc_uv.append(uvs[gi])
                        loc_col.append(colors[gi])
                    tri.append(li)
                if tri:
                    loc_tris.append(tuple(tri))
            if not loc_tris:
                return
            mesh = SaMesh(material_index=mat_base + mat_index)
            mesh.positions = loc_pos
            mesh.uv = loc_uv
            mesh.colors = loc_col
            mesh.triangles = loc_tris
            model.meshes.append(mesh)

        if meshes is not None:
            for matidx, idx in meshes:
                tris = _strip_to_tris(idx) if tristrip else [
                    (idx[i], idx[i+1], idx[i+2]) for i in range(0, len(idx) - 2, 3)]
                emit(matidx, tris)
        else:
            # no binMesh: use raw triangle list, grouped by matId
            by_mat = {}
            for (v0, v1, v2, matid) in tri_raw:
                by_mat.setdefault(matid, []).append((v0, v1, v2))
            for matidx, tris in by_mat.items():
                emit(matidx, tris)

    return model
