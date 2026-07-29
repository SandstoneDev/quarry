"""Step 1 of the region day/night prelight pipeline (see docs/.../dn_prelight_plan.md).

Produce a NIGHT vertex-colour stream aligned 1:1 to the PRE-tessellation whole-map
`sa_full.pmap` vertex pool. We replicate `sa_export_pmap.build_pmap`'s enumeration EXACTLY
(same IMG/IDE/IPL, `want = sorted(model_ids)`, same per-model skip filters, same
`sa_dff_pc.decode` + `psp_mesh.pack_model`) so the night stream lands in the same
model-major / submesh / vertex order the baker wrote. `pack_model` is 1:1 (packed vertex
i == mesh vertex i), so per model we match each mesh vertex back to its geometry vertex by
(position, day-colour) and read the night RGBA from the GEOMETRY->EXTENSION chunk 0x0253F2F9
(magic u32 + RwRGBA[nVert]; magic 0 -> night = day).

Output: `night_pre.bin` = u16 GU_COLOR_5551 per sa_full.pmap vertex.
Validation: vertex count must equal sa_full.pmap's.

Run: cd <gvcslib root> && PYTHONPATH=. python <this>/night_bake.py
"""
import os
import struct
import sys

# gvcslib must be importable (PYTHONPATH = the dir that contains the `gvcslib` package).
GVCSLIB = os.environ.get("GVCS_ROOT", "")
if GVCSLIB not in sys.path:
    sys.path.insert(0, GVCSLIB)

from gvcslib import sa_ipl, sa_ide, psp_mesh, sa_dff_pc          # noqa: E402
from gvcslib.sa_img import SaImg                                  # noqa: E402
from gvcslib.sa_dff import (parse_chunks, GEOMETRYLIST, GEOMETRY,  # noqa: E402
                            STRUCT, EXTENSION)
from gvcslib.sa_dff_pc import (_num_tex_sets, RPGEOMETRY_NATIVE,   # noqa: E402
                               RPGEOMETRY_PRELIT)

ROOT_PC   = ""
SA_FULL   = ""
OUT       = ""
NIGHT_CHUNK = 0x0253F2F9


def _img_name(stem, ext):
    s = stem.lower()
    return s if s.endswith(ext) else s + ext


def rgba8888_to_5551(c):
    """RwRGBA (r<<24|g<<16|b<<8|a) -> GU_COLOR_5551 (matches psp_mesh's day packing)."""
    r = (c >> 24) & 0xFF; g = (c >> 16) & 0xFF; b = (c >> 8) & 0xFF; a = c & 0xFF
    return (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10) | ((1 if a >= 128 else 0) << 15)


def _parse_night(blob, geo, nvert):
    """night RGBA[nVert] from GEOMETRY->EXTENSION 0x0253F2F9, or None (= night is day)."""
    ext = geo.find(EXTENSION)
    if not ext:
        return None
    nc = ext.find(NIGHT_CHUNK)
    if not nc:
        return None
    o = nc.data_off
    if struct.unpack_from("<I", blob, o)[0] == 0:   # magic 0 -> no night set
        return None
    o += 4
    night = [0xFFFFFFFF] * nvert
    for i in range(nvert):
        r, g, b, a = struct.unpack_from("<4B", blob, o + i * 4)
        night[i] = (r << 24) | (g << 16) | (b << 8) | a
    return night


def geom_pos_day_night(blob, geo):
    """Walk one GEOMETRY STRUCT to get per-vertex (position, day-colour) + the night array.
 Mirrors sa_dff_pc.decode's STRUCT walk exactly so offsets line up."""
    st = geo.find(STRUCT)
    if not st:
        return None
    o = st.data_off
    flags, ntri, nvert, nmorph = struct.unpack_from("<4I", blob, o); o += 16
    if flags & RPGEOMETRY_NATIVE:
        return None
    colors = [0xFFFFFFFF] * nvert
    if flags & RPGEOMETRY_PRELIT:
        for i in range(nvert):
            r, g, b, a = struct.unpack_from("<4B", blob, o + i * 4)
            colors[i] = (r << 24) | (g << 16) | (b << 8) | a
        o += nvert * 4
    o += nvert * 8 * _num_tex_sets(flags)            # skip texcoord sets
    o += ntri * 8                                    # skip triangle list
    positions = [(0.0, 0.0, 0.0)] * nvert
    for m in range(nmorph):
        o += 16                                      # bounding sphere
        has_v, has_n = struct.unpack_from("<2I", blob, o); o += 8
        if has_v:
            if m == 0:
                for i in range(nvert):
                    positions[i] = struct.unpack_from("<3f", blob, o + i * 12)
            o += nvert * 12
        if has_n:
            o += nvert * 12
        if m == 0:
            break
    night = _parse_night(blob, geo, nvert)
    return positions, colors, night


def night_lookup_for_model(blob):
    """{(position, day_colour): night_colour} over all geoms of a model (night=day if absent)."""
    root = parse_chunks(blob)
    gl = root.find(GEOMETRYLIST)
    lut = {}
    if not gl:
        return lut
    for geo in gl.find_all(GEOMETRY):
        pdn = geom_pos_day_night(blob, geo)
        if not pdn:
            continue
        positions, colors, night = pdn
        for gi in range(len(positions)):
            nc = (night[gi] if night else colors[gi])
            lut[(positions[gi], colors[gi])] = nc
    return lut


def main():
    im   = SaImg(ROOT_PC + "/MODELS/GTA3.IMG")
    ide  = sa_ide.parse_maps(ROOT_PC + "/DATA")
    insts = sa_ipl.load_all(ROOT_PC + "/DATA", im)
    have = set(n.lower() for n in im.names())
    sel  = [i for i in insts if i.interior in (0, -1)]   # no_interior=True, --all (no bbox)
    want = sorted({i.model_id for i in sel})
    print("instances:", len(insts), "selected:", len(sel), "unique models:", len(want))

    out = bytearray()
    ok = fail = nverts = with_night = 0
    for n, mid in enumerate(want):
        d = ide.get(mid)
        if not d:
            fail += 1; continue
        dn = _img_name(d.dff, ".dff")
        if dn not in have:
            fail += 1; continue
        try:
            blob = im.extract(dn)
            model = sa_dff_pc.decode(blob)
        except Exception:
            fail += 1; continue
        if not any(me.triangles for me in model.meshes):
            fail += 1; continue
        try:
            packed = psp_mesh.pack_model(model)
        except Exception:
            fail += 1; continue
        if not packed["prims"]:
            fail += 1; continue

        lut = night_lookup_for_model(blob)
        had_night = any(v != k[1] for k, v in lut.items())
        if had_night:
            with_night += 1
        # pack_model is 1:1 per mesh vertex, prims in mesh order -> emit night the same way.
        for mesh in model.meshes:
            for i in range(len(mesh.positions)):
                day = mesh.colors[i] if i < len(mesh.colors) else 0xFFFFFFFF
                nc = lut.get((mesh.positions[i], day), day)
                out += struct.pack("<H", rgba8888_to_5551(nc))
                nverts += 1
        ok += 1
        if (n + 1) % 500 == 0:
            print("  %d/%d ok=%d fail=%d verts=%d night-models=%d"
                  % (n + 1, len(want), ok, fail, nverts, with_night))

    with open(OUT, "wb") as f:
        f.write(out)
    print("wrote %s: %d verts (%d bytes), models ok=%d fail=%d, with night-set=%d"
          % (OUT, nverts, len(out), ok, fail, with_night))

    # validation: vertex count must equal sa_full.pmap's
    if os.path.exists(SA_FULL):
        vc = pmap_vertex_count(SA_FULL)
        print("sa_full.pmap vertex count =", vc,
              "->", "MATCH" if vc == nverts else "MISMATCH (alignment broken!)")


def pmap_vertex_count(path):
    """vertex pool bytes / 12 (PmapVertex = 12B). vertex_bytes is the 14th u32 of the
 PmapHeader (offset 52) - see src/platform_psp/pmap.h."""
    with open(path, "rb") as f:
        h = f.read(56)
    return struct.unpack_from("<I", h, 52)[0] // 12


if __name__ == "__main__":
    main()
