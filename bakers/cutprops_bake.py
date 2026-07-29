#!/usr/bin/env python3
"""cutprops_bake - bake the RIGID cutscene props (csbat/csframe/csmomchair) for intro1a.

Each prop is a rigid DFF (cutscene.img) whose Root bone is animated by intro1a.ifp (the .cut
`motion` block maps csbat:Root(tag0) etc.). So a prop = [mesh] + [Root track of KRT0 keyframes
(quat + trans + time)]. The runtime samples the Root at the cutscene phase and draws the mesh at
offset + trans rotated by the quat - same world frame as the skinned actors (cssmoke/csplay).

Output data/cutscene/cutprops.bin:
 'CPRP' u32 nprops
 per prop:
 char name[16]
 u16 nvert, nidx, texW, texH, amode, clutEntries ; u32 texelLen, clutLen
 vert[nvert]: f32 u,v ; u32 colorABGR ; f32 x,y,z
 idx[nidx]: u16
 texels[texelLen] (swizzled T8), clut[clutLen] RGBA8888
 u16 nframes ; frame[nframes]: f32 qx,qy,qz,qw, tx,ty,tz, time
"""
import os, sys, struct
TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS); sys.path.insert(0, os.path.join(TOOLS, "map_export"))
GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path: sys.path.insert(0, GVCS)
sys.path.insert(0, GVCS + "/gvcslib")
import ps2dff, sa_ifp_anpk
# geom belongs to the PC dev loop only (it pulls the external 'formats' package for
# strip conversion). The disc path reads PS2-native VIF through ps2dff and never calls
# it, so a tree without that package - the shipped converter bundle, for one - must
# still be able to bake the props.
try:
    import geom
except Exception:
    geom = None
from gvcslib import sa_txd, sa_txd_d3d9, psp_tex
from gvcslib.sa_dff import parse_chunks, GEOMETRYLIST, GEOMETRY, STRUCT
from sa_img import SaImg
# PC RpGeometry parser (external 'formats' package). Absent from the current tree AND the
# PS2 cutscene props are PS2-native VIF geometry anyway -> guard the import so a missing
# parser doesn't crash the process; bake_mesh defers (skips) each prop it can't read.
try:
    from formats.dff import parse_dff
except Exception:
    parse_dff = None

# SA_ROOT env override: Quarry points this at the extracted PS2 disc. On the disc the
# cutscene props (csbat/csframe/csmomchair) are PS2-NATIVE VIF geometry (flags 0x01010037,
# native bit 0x01000000); tools/ps2dff reads them, so all three bake. Defaults keep the
# PC dev loop, which still uses the neutral parser when one is present.
SA_ROOT      = os.environ.get("SA_ROOT", "")
CUTSCENE_IMG = SA_ROOT + "/models/cutscene.img"
CUTS_IMG     = SA_ROOT + "/anim/cuts.img"
OUT = ""
DEPLOY = ["",
          "",
          ""]
PROPS = ["csbat", "csframe", "csmomchair"]


def _decode_txd(raw):
    """Decode a prop TXD, picking the codec by RW device id (TXD STRUCT @26: 6 = PS2-native
 sa_txd, else D3D8/9 sa_txd_d3d9). Mirrors hero_bake._decode_txd so one baker serves both
 the PS2 disc and the PC dev loop."""
    raw = bytes(raw)
    devid = struct.unpack_from("<H", raw, 26)[0] if len(raw) >= 28 else 0
    prim, alt = (sa_txd, sa_txd_d3d9) if devid == 6 else (sa_txd_d3d9, sa_txd)
    try:
        return prim.decode(raw)
    except Exception:
        return alt.decode(raw)


def _is_native_dff(raw):
    """True if the DFF's first RpGeometry carries the PS2-native flag (0x01000000) - VIF
 geometry the PC parse_dff path can't read (needs the ambient-ped codec, task #36)."""
    try:
        raw = bytes(raw)
        root = parse_chunks(raw)
        gl = root.find(GEOMETRYLIST)
        geo = next(iter(gl.find_all(GEOMETRY)))
        flags = struct.unpack_from("<I", raw, geo.find(STRUCT).data_off)[0]
        return bool(flags & 0x01000000)
    except Exception:
        return False


def _bake_mesh_ps2(raw, img, txdname):
    """PS2-native prop -> the same (verts, idx, texture) the PC path returns.

 The props are rigid VIF geometry, which tools/ps2dff already uninstances for the
 world and for vehicles; nothing about a prop needs the skin pipeline. Their vertex
 positions come out in metres directly (a baseball bat measures 0.887 m, mom's chair
 0.95 x 0.56 x 0.58, the picture frame 0.28 x 0.29), so no fixed-point divisor
 applies here the way it does for world and character geometry.

 CPRP carries ONE texture per prop, while csframe and csmomchair each reference two.
 The dominant material - the one covering the most triangles - is used for the whole
 mesh; the alternative is dropping half the geometry, which is worse for a prop that
 is on screen for a few seconds.
 """
    model = ps2dff.load_dff(bytes(raw))
    if not model.geometries:
        raise SystemExit("no geometry")
    geo = model.geometries[0]
    counts = {}
    for (_a, _b, _c, mat) in geo.tris:
        counts[mat] = counts.get(mat, 0) + 1
    if not counts:
        raise SystemExit("no triangles")
    dom = max(counts, key=counts.get)
    texname = ""
    if dom < len(geo.materials) and geo.materials[dom].texture:
        texname = geo.materials[dom].texture.lower()

    verts, vmap, idx = [], {}, []
    for (a, b, c, _mat) in geo.tris:
        for vi_src in (a, b, c):
            pos = geo.verts[vi_src]
            uv = geo.uvs[vi_src] if vi_src < len(geo.uvs) else (0.0, 0.0)
            key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                   round(uv[0], 5), round(uv[1], 5))
            vi = vmap.get(key)
            if vi is None:
                vi = len(verts); vmap[key] = vi
                verts.append((uv[0], uv[1], 0xFFFFFFFF, pos[0], pos[1], pos[2]))
            idx.append(vi)

    txd = {k.lower(): v for k, v in _decode_txd(img.extract(txdname + ".txd")).items()}
    entry = txd.get(texname) or next(iter(txd.values()))
    tw, th, rgba = entry
    t = psp_tex.author_psp_texture(rgba, tw, th, fmt="T8", mipmaps=False)
    return verts, idx, t


def bake_mesh(img, dffname, txdname):
    raw = img.extract(dffname + ".dff")
    if _is_native_dff(raw):
        return _bake_mesh_ps2(raw, img, txdname)
    if parse_dff is None or geom is None:
        raise SystemExit("neutral DFF parser unavailable in this tree")
    dff = parse_dff(raw)
    txd = {k.lower(): v for k, v in _decode_txd(img.extract(txdname + ".txd")).items()}
    verts, vmap, idx, texname = [], {}, [], None
    for a in dff.atomics:
        for part in geom.process_geometry(dff.geometries[a.geometry_index]):
            m = part.get("mat")
            if texname is None and m is not None and getattr(m, "texture_name", ""):
                texname = m.texture_name.lower()
            for tri in part["tris"]:
                for (pos, uv, col) in tri:
                    key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                           round(uv[0], 5), round(uv[1], 5))
                    vi = vmap.get(key)
                    if vi is None:
                        vi = len(verts); vmap[key] = vi
                        verts.append((uv[0], uv[1], 0xFFFFFFFF, pos[0], pos[1], pos[2]))
                    idx.append(vi)
    entry = txd.get(texname or "") or next(iter(txd.values()))
    tw, th, rgba = entry
    t = psp_tex.author_psp_texture(rgba, tw, th, fmt="T8", mipmaps=False)
    return verts, idx, t


def root_track(anpk, name):
    """Root KRT0 keyframes (quat conjugated like the actors, trans) for prop `name`."""
    a = next((x for x in anpk["anims"] if x["name"].lower() == name.lower()), None)
    if a is None: return []
    s = next((q for q in a["seqs"] if "root" in q["bone"].lower() or "tag0" in q["bone"].lower()),
             a["seqs"][0] if a["seqs"] else None)
    if s is None: return []
    ht = 1 if s["keyType"] == "KRT0" else 0
    stride = 32 if ht else 20
    frames = []
    for fi in range(s["numFrames"]):
        base = fi * stride
        qf = struct.unpack_from("<4f", s["kf"], base)
        qb = (-qf[0], -qf[1], -qf[2], qf[3])                 # ANPK conjugation (AnimManager.cpp:810)
        if ht:
            tr = struct.unpack_from("<3f", s["kf"], base + 16); tm = struct.unpack_from("<f", s["kf"], base + 28)[0]
        else:
            tr = (0.0, 0.0, 0.0); tm = struct.unpack_from("<f", s["kf"], base + 16)[0]
        frames.append((qb, tr, tm))
    return frames


def main():
    # argv[1] = explicit output path (Quarry passes <OutDir>/cutscene/cutprops.bin). When
    # given we write ONLY there and skip the dev-loop memstick mirror (ped_bake idiom).
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    quarry = len(sys.argv) > 1

    img = SaImg(CUTSCENE_IMG)
    anpk = sa_ifp_anpk.decode(SaImg(CUTS_IMG).extract("intro1a.ifp").rstrip(b"\x00"))
    # bake per-prop bodies first, skipping any that can't be read (PS2-native VIF on the disc,
    # or an absent PC parser) -> the CPRP count reflects what actually baked.
    bodies = []
    for name in PROPS:
        try:
            verts, idx, t = bake_mesh(img, name, name)
        except (SystemExit, Exception) as e:
            print("  ! %-12s skipped: %s" % (name, e)); continue
        trk = root_track(anpk, name)
        body = bytearray()
        nm = name.encode("ascii")[:15]; nm += b"\x00" * (16 - len(nm))
        body += nm
        body += struct.pack("<6H2I", len(verts), len(idx), t["width"], t["height"],
                            (t.get("alpha_mode", 0) & 3), t["clut_entries"],
                            len(t["texel_bytes"]), len(t["clut_bytes"]))
        for (u, v, c, x, y, z) in verts: body += struct.pack("<2fI3f", u, v, c, x, y, z)
        for i in idx: body += struct.pack("<H", i)
        body += t["texel_bytes"] + t["clut_bytes"]
        body += struct.pack("<H", len(trk))
        for (q, tr, tm) in trk: body += struct.pack("<8f", q[0], q[1], q[2], q[3], tr[0], tr[1], tr[2], tm)
        bodies.append(bytes(body))
        print("  %-12s verts=%4d idx=%5d tex=%dx%d rootframes=%d" %
              (name, len(verts), len(idx), t["width"], t["height"], len(trk)))
    buf = bytearray(b"CPRP"); buf += struct.pack("<I", len(bodies))
    for b in bodies: buf += b
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(buf)
    if quarry:
        print("cutprops.bin: %d/%d props, %d bytes -> %s" % (len(bodies), len(PROPS), len(buf), out))
        if len(bodies) < len(PROPS):
            print("    (cutscene.img props are PS2-native VIF - codec pending #36; empty CPRP is valid)")
        return
    n = 0
    for d in DEPLOY:
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True); open(d, "wb").write(buf); n += 1
        except OSError: pass
    print("cutprops.bin: %d props, %d bytes, deployed to %d dir(s)" % (len(bodies), len(buf), n))


if __name__ == "__main__":
    main()
