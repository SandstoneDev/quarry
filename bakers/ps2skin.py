#!/usr/bin/env python3
"""ps2skin - PS2-native SKINNED-DFF codec for the Quarry converter.

the source game ships skinned CHARACTER models (CJ / csplay in player.img) platform-NEUTRAL,
so those read fine with the PC codecs (tools/sa_skin + hero_bake.parse_geometry).
But the cutscene actors (cssmoke = Big Smoke, csbat, ... in cutscene.img) and the
ambient civilians (fam1, ... in gta3.img) are stored PS2-NATIVE: VIF-instanced
geometry (RpGeometry NATIVE bit, NativeDataPLG 0x510) + a NATIVE skin plugin
(RpSkin 0x116, ps2::readNativeSkin).  tools/ps2dff already uninstances PS2-native
STATIC world geometry; this module drives it through the SKIN pipeline and reshapes
the result into exactly what the skinned bakers already consume, so cutscene_bake /
ped_bake feed a native actor through hero_bake unchanged.

Codec (cross-checked against librw src/ps2/ps2skin.cpp + src/skin.cpp and byte-for-
byte against the PC twin of cssmoke - see the module self-test):

  * NATIVE SKIN plugin (0x116) -> ps2dff._read_native_skin: numBones, numUsedBones,
    numWeights, usedBones[], inverseMatrices[numBones][16].  The inverse-bind
    matrices are byte-identical to the PC twin; usedBones matches the PC twin's
    used-bone set exactly.
  * PER-VERTEX weights/indices are NOT in the plugin - they ride the geometry VIF
    stream as a 5th attribute (AT_NORMAL+1, V4_32), captured by ps2dff._parse_chain
    and welded parallel to pos/uv/colour (verts differing only in bone binding do
    NOT weld).  weight = float(w & ~0x3FF); boneIndex = ((w & 0x3FF) >> 2) - 1.
  * FRAMELIST + HANIM are platform-neutral -> reuse sa_skin.decode_skeleton (the
    same frame/node hierarchy the hero uses).  Verified: node table + the
    nodeId->parent map are identical on the PC and PS2 twins.

API (mirrors the neutral path so the bakers are agnostic):
  is_native_skinned(blob) -> bool
  geometry(blob)          -> (positions, uvs, colors, submeshes, mat_names, nvert, normals)
                             == hero_bake.parse_geometry's return
  decode(blob)            -> {frames, nodes, geoms:[{nvert,numBones,numUsed,maxW,
                             used,boneIdx,boneW,invBind}]} == sa_skin.decode's return
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps2dff
import sa_skin


def is_native_skinned(blob):
    """True iff this DFF's first geometry is PS2-native AND carries a skin plugin
    (cutscene actors / ambient peds).  PC/neutral skinned DFFs -> False (the caller
    keeps the sa_skin path); world/vehicle native DFFs -> False (no skin plugin)."""
    try:
        b = bytes(blob)
        cl_off, cl_size = ps2dff._find(b, 0, len(b), ps2dff.C_CLUMP)
        if cl_off is None:
            return False
        gl_off, gl_size = ps2dff._find(b, cl_off, cl_off + cl_size, ps2dff.C_GEOMLIST)
        if gl_off is None:
            return False
        gg = ps2dff._find_all(b, gl_off, gl_off + gl_size, ps2dff.C_GEOMETRY)
        if not gg:
            return False
        g_off, g_size = gg[0]
        s_off, _ = ps2dff._find(b, g_off, g_off + g_size, ps2dff.C_STRUCT)
        if s_off is None:
            return False
        fmt = struct.unpack_from("<I", b, s_off)[0]
        if not (fmt & ps2dff.FMT_NATIVE):
            return False
        e_off, e_size = ps2dff._find(b, g_off, g_off + g_size, ps2dff.C_EXT)
        if e_off is None:
            return False
        sk_off, _ = ps2dff._find(b, e_off, e_off + e_size, ps2dff.C_SKIN)
        return sk_off is not None
    except Exception:
        return False


def _compute_normals(positions, tris, nvert):
    """Per-vertex normals = area-weighted average of adjacent face normals.  The PS2
    skinned stream carries NO normal attribute (cutscene actors drop it, so the weld
    can't split on it either), so we synthesise smooth normals from the triangles --
    exactly hero_bake._compute_normals, kept here to avoid a circular import."""
    acc = [[0.0, 0.0, 0.0] for _ in range(nvert)]
    for tri in tris:
        v0, v1, v2 = tri[0], tri[1], tri[2]
        if v0 >= nvert or v1 >= nvert or v2 >= nvert:
            continue
        a, b, c = positions[v0], positions[v1], positions[v2]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        for vi in (v0, v1, v2):
            acc[vi][0] += nx; acc[vi][1] += ny; acc[vi][2] += nz
    out = []
    for n in acc:
        ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        out.append((n[0] / ln, n[1] / ln, n[2] / ln))
    return out


# ps2dff.load_dff + sa_skin.decode_skeleton are both non-trivial, and hero_bake calls
# geometry() and decode() back-to-back on the SAME component bytes.  Memoise the
# combined decode so both halves come from ONE weld (guaranteeing identical vertex
# order/count) and the work isn't repeated.  Keyed by content hash; a few entries.
_CACHE = {}
_CACHE_ORDER = []


def _full(blob):
    b = bytes(blob)
    key = (len(b), hash(b))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == b:
        return hit[1]
    res = _decode_full(b)
    _CACHE[key] = (b, res)
    _CACHE_ORDER.append(key)
    if len(_CACHE_ORDER) > 6:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)
    return res


def _decode_full(b):
    model = ps2dff.load_dff(b)
    if not model.geometries:
        raise ValueError("ps2skin: DFF has no geometry")
    # peds/actors are single-geometry; mirror hero_bake.parse_geometry (first geom).
    geo = model.geometries[0]
    if geo.skin is None:
        raise ValueError("ps2skin: geometry carries no native skin plugin")

    positions = list(geo.verts)
    uvs = list(geo.uvs)
    colors = [(d[0] << 24) | (d[1] << 16) | (d[2] << 8) | d[3] for d in geo.day]
    nvert = len(positions)

    by_mat = {}
    for (a, b3, c, mat) in geo.tris:
        by_mat.setdefault(mat, []).append((a, b3, c))
    submeshes = sorted(by_mat.items())                       # [(matIndex, [(a,b,c)...])]
    mat_names = [{"texture_name": m.texture} for m in geo.materials]
    normals = _compute_normals(positions, geo.tris, nvert)
    geometry = (positions, uvs, colors, submeshes, mat_names, nvert, normals)

    sk = geo.skin
    skgeo = {
        "nvert": nvert,
        "numBones": sk["numBones"],
        "numUsed": sk["numUsed"],
        "maxW": sk["numWeights"],
        "used": sk["used"],
        "boneIdx": list(geo.boneIdx),
        "boneW": list(geo.boneW),
        "invBind": sk["invBind"],
    }
    skel = sa_skin.decode_skeleton(b)                        # neutral FRAMELIST + HANIM
    return {
        "geometry": geometry,
        "frames": skel["frames"],
        "nodes": skel["nodes"],
        "geoms": [skgeo],
    }


def geometry(blob):
    """hero_bake.parse_geometry drop-in for a PS2-native skinned DFF:
    (positions, uvs, colors, submeshes, mat_names, nvert, normals)."""
    return _full(blob)["geometry"]


def decode(blob):
    """sa_skin.decode drop-in for a PS2-native skinned DFF:
    {frames, nodes, geoms:[{nvert,numBones,numUsed,maxW,used,boneIdx,boneW,invBind}]}."""
    f = _full(blob)
    return {"frames": f["frames"], "nodes": f["nodes"], "geoms": f["geoms"]}


if __name__ == "__main__":
    # Self-test: decode a PS2-native actor and cross-check against the PC twin.
    sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
    from gvcslib.sa_img import SaImg
    PS2 = os.environ.get("SA_ROOT",
        "")
    PC = ""
    actor = sys.argv[1] if len(sys.argv) > 1 else "cssmoke"
    img = "cutscene.img" if actor.startswith("cs") else "gta3.img"
    blob = SaImg(PS2 + "/models/" + img).extract(actor + ".dff")
    print("native-skinned:", is_native_skinned(blob))
    d = decode(blob)
    g = d["geoms"][0]
    positions, uvs, colors, submeshes, mat_names, nvert, normals = geometry(blob)
    wsum = [sum(w) for w in g["boneW"]]
    used = set()
    for bi, bw in zip(g["boneIdx"], g["boneW"]):
        for k in range(4):
            if bw[k] > 0:
                used.add(bi[k])
    print("=== PS2 %s ===" % actor)
    print(" frames=%d nodes=%d  numBones=%d numUsed=%d maxW=%d"
          % (len(d["frames"]), len(d["nodes"]), g["numBones"], g["numUsed"], g["maxW"]))
    print(" verts=%d tris=%d submeshes=%d textures=%s"
          % (nvert, sum(len(t) for _, t in submeshes), len(submeshes),
             [m["texture_name"] for m in mat_names]))
    print(" weight-sum min=%.4f max=%.4f  idx-out-of-range=%d  used==plugin.used:%s"
          % (min(wsum), max(wsum),
             sum(1 for bi in g["boneIdx"] for x in bi if x < 0 or x >= g["numBones"]),
             sorted(used) == sorted(g["used"])))
    try:
        pc = sa_skin.decode(SaImg(PC + "/models/" + img).extract(actor + ".dff"))
        pg = pc["geoms"][0]
        print("=== PC twin cross-check ===")
        print(" numBones %d==%d  numUsed %d==%d  maxW %d==%d  usedBones-set-equal:%s"
              % (g["numBones"], pg["numBones"], g["numUsed"], pg["numUsed"],
                 g["maxW"], pg["maxW"], sorted(g["used"]) == sorted(pg["used"])))
    except Exception as e:
        print(" (no PC twin: %s)" % e)
