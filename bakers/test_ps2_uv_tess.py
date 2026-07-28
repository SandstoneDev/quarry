# Standalone test for ps2_uv_tess (run: python tools/test_ps2_uv_tess.py).
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ps2_uv_tess import cap_uv_span, UV_ONE, UV_EDGE_MAX, VFMT   # noqa: E402


def pack_vert(u, v, colour, x, y, z):
    """Pack one vertex. UVs go in as RAW 16 bits, so an authored (signed,
    centred) coordinate is stored as its two's-complement pattern - exactly
    what the packer upstream of us writes and what the GE reads back unsigned."""
    return struct.pack(VFMT, u & 0xFFFF, v & 0xFFFF, colour, x, y, z)


def uv_signed(w):
    return w - 65536 if w > 32767 else w


class Submesh(object):
    """Stands in for the scene submesh cap_uv_span rebuilds: it only needs the
    four attributes the real one is constructed with."""

    def __init__(self, texture, vertex_bytes, index_bytes, uvscroll=None):
        self.texture = texture
        self.vertex_bytes = vertex_bytes
        self.index_bytes = index_bytes
        self.uvscroll = uvscroll


class Model(object):
    def __init__(self, submeshes):
        self.submeshes = submeshes


def quad(u_tiles, v_tiles=1.0, colour=0x7FFF, centred=True):
    """One texture-space quad (2 triangles, 4 shared vertices) tiling u_tiles x
    v_tiles. The two triangles share the diagonal, so a crack-free subdivision
    must give that edge a single midpoint.

    Centred by default because that is how the source authors these: the UVs sit
    in a symmetric band around zero, which is exactly what makes them straddle
    the GE's unsigned window seam. `centred=False` gives a quad already inside
    tile 0, i.e. one that needs no work at all."""
    du = int(u_tiles * UV_ONE)
    dv = int(v_tiles * UV_ONE)
    u0, v0 = (-du // 2, -dv // 2) if centred else (0, 0)
    verts = [(u0, v0, colour, 0, 0, 0),
             (u0 + du, v0, colour, 1000, 0, 0),
             (u0 + du, v0 + dv, colour, 1000, 1000, 0),
             (u0, v0 + dv, colour, 0, 1000, 0)]
    vb = b"".join(pack_vert(*v) for v in verts)
    ib = struct.pack("<6H", 0, 1, 2, 0, 2, 3)
    return Submesh("road", vb, ib)


def read_back(sm):
    verts = list(struct.iter_unpack(VFMT, sm.vertex_bytes))
    n = len(sm.index_bytes) // 2
    idx = struct.unpack("<%dH" % n, sm.index_bytes)
    tris = [(idx[i * 3], idx[i * 3 + 1], idx[i * 3 + 2]) for i in range(n // 3)]
    return verts, tris


def worst_span(model):
    """Largest per-triangle UV extent across a model, in tiles."""
    worst = 0
    for sm in model.submeshes:
        verts, tris = read_back(sm)
        for a, b, c in tris:
            us = (verts[a][0], verts[b][0], verts[c][0])
            vs = (verts[a][1], verts[b][1], verts[c][1])
            worst = max(worst, max(us) - min(us), max(vs) - min(vs))
    return worst / float(UV_ONE)


def test_wide_span_is_capped():
    m = Model([quad(14.0)])
    st = cap_uv_span([m], verbose=False)
    assert worst_span(m) <= UV_EDGE_MAX + 1e-6, worst_span(m)
    assert st["tris_after"] > st["tris_before"], st
    assert st["wrapped"] == 0, st
    # every piece must land inside the GE's [0,16) tile window
    for sm in m.submeshes:
        verts, _ = read_back(sm)
        assert min(v[0] for v in verts) >= 0 and min(v[1] for v in verts) >= 0
        assert max(v[0] for v in verts) < 16 * UV_ONE
        assert max(v[1] for v in verts) < 16 * UV_ONE


def test_shared_edge_gets_one_midpoint():
    """Both triangles share the diagonal, so a crack-free split must reuse its
    midpoint. If every triangle got private vertices the count would be 3x the
    triangle count - that is the T-junction bug this guards against."""
    # 7 tiles inside cell 0: wide enough to subdivide, narrow enough to stay one
    # bucket, so the vertex count reflects sharing alone.
    m = Model([quad(7.0, centred=False)])
    cap_uv_span([m], verbose=False)
    assert len(m.submeshes) == 1, len(m.submeshes)
    verts, tris = read_back(m.submeshes[0])
    assert len(verts) < 3 * len(tris), (len(verts), len(tris))


def test_widest_representable_span_is_handled():
    """The worst case the packed s16 domain can express is the full range,
    65535 raw = exactly 16 tiles. Two passes must bring it under the cap without
    the geometry blowing up."""
    verts = [(0, 0, 0x7FFF, 0, 0, 0),
             (32767, 0, 0x7FFF, 1000, 0, 0),
             (-32768, 32767, 0x7FFF, 1000, 1000, 0)]     # span 65535 raw = 16 tiles
    vb = b"".join(pack_vert(*v) for v in verts)
    m = Model([Submesh("junk", vb, struct.pack("<3H", 0, 1, 2))])
    st = cap_uv_span([m], verbose=False)
    assert worst_span(m) <= UV_EDGE_MAX + 1e-6, worst_span(m)
    assert st["wrapped"] == 0, st              # the work bound must not fire
    assert st["tris_after"] <= 64, st          # 16 -> 8 -> 4 tiles = 2 passes


def test_compliant_mesh_is_left_alone():
    """A quad already inside tile 0 needs neither subdivision nor a shift."""
    m = Model([quad(1.0, centred=False)])
    before = m.submeshes[0].vertex_bytes
    st = cap_uv_span([m], verbose=False)
    assert len(m.submeshes) == 1, len(m.submeshes)
    assert st["tris_after"] == st["tris_before"] == 2, st
    assert st["wrapped"] == 0, st
    assert m.submeshes[0].vertex_bytes == before, "compliant mesh was rewritten"


def test_centred_mesh_is_moved_off_the_window_seam():
    """A 1-tile quad centred on zero is small enough to need no subdivision, but
    half of it reads as ~15.5 tiles in the GE's unsigned view - it straddles the
    seam, which is the striped/stretched artefact. It must come out non-negative."""
    m = Model([quad(1.0)])
    st = cap_uv_span([m], verbose=False)
    assert st["tris_after"] == 2, st                  # no subdivision needed
    verts, _ = read_back(m.submeshes[0])
    # the half that was negative read as ~15.5 tiles before; after the shift the
    # whole quad must sit in the first couple of tiles, not up against the seam
    assert max(max(v[0], v[1]) for v in verts) <= 2 * UV_ONE, verts


def test_idempotent():
    m = Model([quad(11.0)])
    cap_uv_span([m], verbose=False)
    first = [(sm.vertex_bytes, sm.index_bytes) for sm in m.submeshes]
    cap_uv_span([m], verbose=False)
    second = [(sm.vertex_bytes, sm.index_bytes) for sm in m.submeshes]
    assert first == second, "second pass changed the data"


def test_per_submesh_attributes_survive():
    sm = quad(12.0)
    sm.uvscroll = (0.5, 0.25)
    m = Model([sm])
    cap_uv_span([m], verbose=False)
    assert len(m.submeshes) >= 1
    for piece in m.submeshes:
        assert piece.texture == "road", piece.texture
        assert piece.uvscroll == (0.5, 0.25), piece.uvscroll


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("  ok  %s" % name)
    print("ps2_uv_tess: all tests passed")


if __name__ == "__main__":
    main()
