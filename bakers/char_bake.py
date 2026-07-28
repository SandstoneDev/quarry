"""Bake a SA ped DFF+TXD into char.bin - a resident character model for the PSP
port's playable hero (bind-pose geometry; skeletal anim via IFP is a later stage).

Port-side COPY of gvcslib/work/char_bake.py (gvcslib is READ-ONLY; we never run
from / write into it). Reads the gvcslib codecs + the PC SA install, writes
char.bin into this project. Usage:
    python tools/char_bake.py [ped] [out.bin]
    (ped default 'cj' -> the player model; falls back is the caller's job)

char.bin layout (little-endian):
  'CHAR' | scale f32 | center 3*f32 | min_z f32 | radius f32 | nprims u32
  per prim: tw,th u16 | num_levels u16 (lo=mips, hi=alpha_mode) | clut_entries u16
            vbytes,ibytes u32 | texel_len,clut_len u32 | <verts><idx><texels><clut>
"""
import math
import os
import struct
import sys

# gvcslib package lives here (READ-ONLY): its parent dir must be importable.
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # dff_clumps (local)

from gvcslib import sa_img, sa_dff_pc, sa_txd, sa_txd_d3d9, psp_mesh, psp_tex
import dff_clumps

# SA_ROOT env override: Quarry points this at the extracted PS2 disc (col_bake.py idiom).
SA_ROOT = os.environ.get("SA_ROOT", "")
ROOT_PC = SA_ROOT
OUT_DEFAULT = ""


def _decode_txd(raw):
    """Pick the TXD codec by RW device id (26: u16 deviceId): 6 = PS2-native (sa_txd),
    else D3D8/9 (sa_txd_d3d9). One char_bake serves the PS2 disc and the PC dev loop."""
    raw = bytes(raw)
    devid = struct.unpack_from("<H", raw, 26)[0] if len(raw) >= 28 else 0
    prim, alt = (sa_txd, sa_txd_d3d9) if devid == 6 else (sa_txd_d3d9, sa_txd)
    try:
        return prim.decode(raw)
    except Exception:
        return alt.decode(raw)


def bake(ped="cj", root=ROOT_PC, out=OUT_DEFAULT):
    # 'cj' has no gta3.img model (there is no cj.dff on either platform) and the PS2
    # gta3.img ambient peds are PS2-NATIVE (unreadable by the static sa_dff_pc), so the
    # resident player char is baked from PLAYER.IMG's torso - a PLATFORM-NEUTRAL skinned
    # DFF that sa_dff_pc reads as a bind-pose mesh on both platforms. Other named peds
    # still come from gta3.img (works on a PC disc; PS2-native ones fail -> caller's call).
    if ped.lower() == "cj":
        im = sa_img.SaImg(root + "/MODELS/player.img")
        dff_name, txd_name = "torso.dff", "player_torso.txd"
    else:
        im = sa_img.SaImg(root + "/MODELS/GTA3.IMG")
        dff_name, txd_name = ped + ".dff", ped + ".txd"
    raw = im.extract(dff_name)
    _cl = dff_clumps.split_clumps(raw)     # clothes DFFs are 3-clump (Normal/Fat/Ripped)
    dff = _cl.get("normal", raw if len(_cl) <= 1 else next(iter(_cl.values())))
    model = sa_dff_pc.decode(dff)
    txd = _decode_txd(im.extract(txd_name))

    # SA skinned peds are authored LYING ALONG X; with no skeleton yet, stand the
    # bind pose up: rotate -90 about Y so model-X -> world-Z (up): (x,y,z)->(-z,y,x).
    for me in model.meshes:
        me.positions = [(-z, y, x) for (x, y, z) in me.positions]

    packed = psp_mesh.pack_model(model)
    scale = packed["scale"]
    cx, cy, cz = packed["center"]

    zs = [p[2] for me in model.meshes for p in me.positions]
    min_z = min(zs)
    allp = [p for me in model.meshes for p in me.positions]
    radius = max(math.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2) for x, y, z in allp)

    tex_cache = {}

    def author(name):
        key = (name or "").strip().lower()
        if key in tex_cache:
            return tex_cache[key]
        entry = txd.get(name) or txd.get(key)
        if entry is None:
            for k, v in txd.items():
                if k.lower() == key:
                    entry = v
                    break
        if entry is None:
            tex_cache[key] = None
            return None
        w, h, rgba = entry
        try:
            t = psp_tex.author_psp_texture(rgba, w, h, fmt="T8", mipmaps=True)
        except Exception:
            t = None
        tex_cache[key] = t
        return t

    prims_out = []
    for prim in packed["prims"]:
        mi = prim["material_index"]
        name = ""
        if 0 <= mi < len(model.materials):
            name = model.materials[mi].get("texture_name") or ""
        t = author(name)
        prims_out.append((t, prim["vertex_bytes"], prim["index_bytes"]))

    buf = bytearray()
    buf += b"CHAR"
    buf += struct.pack("<f", scale)
    buf += struct.pack("<3f", cx, cy, cz)
    buf += struct.pack("<f", min_z)
    buf += struct.pack("<f", radius)
    buf += struct.pack("<I", len(prims_out))
    for t, vb, ib in prims_out:
        if t is None:
            buf += struct.pack("<HHHH", 0, 0, 0, 0)
            buf += struct.pack("<II", len(vb), len(ib))
            buf += struct.pack("<II", 0, 0)
            buf += vb + ib
        else:
            nl = t["num_levels"] | (t.get("alpha_mode", 0) << 8)
            texel = t["texel_bytes"]; clut = t["clut_bytes"]
            buf += struct.pack("<HHHH", t["width"], t["height"], nl, t["clut_entries"])
            buf += struct.pack("<II", len(vb), len(ib))
            buf += struct.pack("<II", len(texel), len(clut))
            buf += vb + ib + texel + clut

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(buf)
    print("wrote %s : %d bytes  ped=%s prims=%d scale=%.5f center=(%.3f,%.3f,%.3f) "
          "min_z=%.3f radius=%.3f"
          % (out, len(buf), ped, len(prims_out), scale, cx, cy, cz, min_z, radius))


if __name__ == "__main__":
    ped = sys.argv[1] if len(sys.argv) > 1 else "cj"
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT
    bake(ped, out=out)
