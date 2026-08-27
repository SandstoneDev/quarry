#!/usr/bin/env python3
"""weapon_model_bake - bake the weapon meshes AND the HUD icons, per weapon type.

Why one file each rather than one container: SA streams weapon models, and it is right
to. Forty-odd weapons at ~20 KB baked is most of a megabyte, which a PSP cannot hold
resident for something only one of which is in your hand at a time. `data/weapons/w22.bin`
is loaded when the pistol is equipped and freed when it is not.

Two assets per weapon, and the second one is easy to look for in the wrong place:

 * the MESH serves the weapon in the ped's hand and the pickup lying in its corona;
 * the HUD ICON is a 2D SPRITE whose texture lives in the WEAPON'S OWN TXD under the
 name "<modelName>ICON" - `colt45.txd` holds `colt45icon`, `ak47.txd` holds
 `ak47icon`, and so on. CHud::DrawWeaponIcon (0x58D7D0) looks it up with
 RwTexDictionaryFindHashNamedTexture(txd, AppendStringToKey(mi->m_nKey, "ICON"))
 and draws one XLU sprite. models/hud.txd carries only `fist`, which is why it looks
 at first as though SA renders the model into the corner - it does not.

Model and TXD names come from DATA/DEFAULT.IDE's weapon rows, which are
 id, modelName, txdName, animGroup, drawDist, flags
so the TXD is column 2 and column 3 is the ANIM group (that is why `ak47` reads
`ak47, ak47, rifle` and not a `rifle.txd` that does not exist).

 SA_ROOT=... GVCS_ROOT=... python tools/weapon_model_bake.py [--out data]

Output:
 data/weapons/w<type>.bin 'PRP1' - identical to prop_bake.py's, read by CProp_Load:
 u16 nvert, nidx, texW, texH, nlevels|amode<<8, clutEntries ; u32 texelLen, clutLen
 vert[nvert]: f32 u,v ; u32 colorABGR ; f32 x,y,z
 idx[nidx] u16 ; texels ; clut
 data/weapons/i<type>.bin 'WICO' u32 w, h ; w*h RGBA8888 - the same shape hud.bin
 uses for the fist, so the HUD draws it through the sprite path that already works.
"""
import argparse
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS and GVCS not in sys.path:
    sys.path.insert(0, GVCS)
    sys.path.insert(0, GVCS + "/gvcslib")

import cutprops_bake                                    # bake_mesh, PS2-native + PC paths
from sa_img import SaImg

SA_ROOT = os.environ.get("SA_ROOT", "")

# eWeaponType -> the model id in weapon.dat's modelId1 column. Kept here rather than
# re-parsing weapon.dat: the two must agree, and weapon_bake.py already asserts the.dat
# side, so a mismatch shows up as a missing file rather than a silently wrong mesh.
# The muzzle flash lives in every gun's own TXD under this name; the `gunflash` atomic
# inside the weapon DFF is the quad set it is mapped to.
MUZZLE_TEX = "muzzle_texture4"

WEAPON_MODEL = {
    1: 331, 2: 333, 3: 334, 4: 335, 5: 336, 6: 337, 7: 338, 8: 339, 9: 341,
    10: 321, 11: 322, 12: 323, 13: 324, 14: 325, 15: 326,
    16: 342, 17: 343, 18: 344, 19: 345, 20: 345, 21: 345,
    22: 346, 23: 347, 24: 348, 25: 349, 26: 350, 27: 351, 28: 352, 29: 353,
    30: 355, 31: 356, 32: 372, 33: 357, 34: 358, 35: 359, 36: 360, 37: 361,
    38: 362, 39: 363, 40: 364, 41: 365, 42: 366, 43: 367, 44: 368, 45: 369,
    46: 371,
}


def ide_models(root):
    """{model id: (dffName, txdName)} from DATA/DEFAULT.IDE."""
    out = {}
    path = os.path.join(root, "data", "default.ide")
    if not os.path.isfile(path):
        sys.exit("weapon_model_bake: %s not found" % path)
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "," not in line:
                continue
            tok = [t.strip() for t in line.split(",")]
            if len(tok) < 3 or not tok[0].isdigit():
                continue
            out[int(tok[0])] = (tok[1], tok[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--root", default=SA_ROOT)
    args = ap.parse_args()
    if not args.root:
        sys.exit("weapon_model_bake: set SA_ROOT (or pass --root)")

    ide = ide_models(args.root)
    img = SaImg(os.path.join(args.root, "models", "gta3.img"))

    outdir = os.path.join(args.out, "weapons")
    os.makedirs(outdir, exist_ok=True)

    made, skipped, total = 0, [], 0
    icons, noicon, flashes = 0, [], 0
    for wtype in sorted(WEAPON_MODEL):
        mid = WEAPON_MODEL[wtype]
        names = ide.get(mid)
        if not names:
            skipped.append("%d(model %d not in ide)" % (wtype, mid))
            continue
        dffname, txdname = names
        try:
            # merge=True: an SA weapon DFF is several atomics and only the first was
            # being taken. The minigun's barrel cluster (`minigun2`, 208 tris), the
            # sawn-off's barrels (`sawbarl`, 122) and the flower's petals (100) share
            # the weapon's own sheet and belong in the mesh; `gunflash` does not share
            # it and is baked separately below.
            verts, idx, tex = cutprops_bake.bake_mesh(img, dffname, txdname, merge=True)
        except Exception as e:                          # a weapon we cannot read is not fatal
            skipped.append("%d/%s(%s)" % (wtype, dffname, type(e).__name__))
            continue
        if not verts or not idx:
            skipped.append("%d/%s(empty)" % (wtype, dffname))
            continue

        buf = bytearray()
        buf += b"PRP1"
        buf += struct.pack("<HHHHHH", len(verts), len(idx), tex["width"], tex["height"],
                           tex["num_levels"] | (tex.get("alpha_mode", 0) << 8),
                           tex["clut_entries"])
        buf += struct.pack("<II", len(tex["texel_bytes"]), len(tex["clut_bytes"]))
        for (u, v, c, x, y, z) in verts:
            buf += struct.pack("<ffIfff", u, v, c, x, y, z)
        for i in idx:
            buf += struct.pack("<H", i)
        buf += tex["texel_bytes"] + tex["clut_bytes"]

        dst = os.path.join(outdir, "w%d.bin" % wtype)
        with open(dst, "wb") as f:
            f.write(buf)
        made += 1
        total += len(buf)

        # the HUD icon out of the same TXD: "<modelName>ICON", 64x64 in every weapon
        # that has one. A weapon without one keeps the fist, which is what SA does when
        # the lookup fails.
        try:
            txd = {k.lower(): v for k, v in
                   cutprops_bake._decode_txd(img.extract(txdname + ".txd")).items()}
        except Exception:
            txd = {}
        ent = txd.get(dffname.lower() + "icon")
        if ent is None:
            ent = next((v for k, v in txd.items() if k.endswith("icon")), None)
        if ent is None:
            noicon.append("%d/%s" % (wtype, dffname))
        else:
            iw, ih, irgba = ent
            with open(os.path.join(outdir, "i%d.bin" % wtype), "wb") as f:
                f.write(b"WICO")
                f.write(struct.pack("<II", iw, ih))
                f.write(bytes(irgba))
            icons += 1
            total += 8 + len(irgba)
        # the MUZZLE FLASH mesh: SA keeps it inside the weapon DFF as a separate
        # `gunflash` atomic textured with muzzle_texture4, hidden except on the frame a
        # shot goes off. Same PRP1 container, its own file, loaded with the weapon.
        try:
            fv, fi, ft = cutprops_bake.bake_mesh(img, dffname, txdname,
                                                 texture=MUZZLE_TEX)
        except (Exception, SystemExit):   # SystemExit is not an Exception; melee
            fv = None                     # weapons simply have no gunflash atomic
        if fv:
            fb = bytearray(b"PRP1")
            fb += struct.pack("<HHHHHH", len(fv), len(fi), ft["width"], ft["height"],
                              ft["num_levels"] | (ft.get("alpha_mode", 0) << 8),
                              ft["clut_entries"])
            fb += struct.pack("<II", len(ft["texel_bytes"]), len(ft["clut_bytes"]))
            for (u, v, c, x, y, z) in fv:
                fb += struct.pack("<ffIfff", u, v, c, x, y, z)
            for i in fi:
                fb += struct.pack("<H", i)
            fb += ft["texel_bytes"] + ft["clut_bytes"]
            with open(os.path.join(outdir, "f%d.bin" % wtype), "wb") as f:
                f.write(fb)
            flashes += 1
            total += len(fb)

        print("  w%-3d %-14s %5d vert %5d idx  %3dx%-3d  %6d B%s"
              % (wtype, dffname, len(verts), len(idx), tex["width"], tex["height"],
                 len(buf), "  +flash" if fv else ""))

    print("weapon_model_bake: %d models + %d icons + %d muzzle flashes -> %s (%d B total, "
          "one weapon resident at a time)" % (made, icons, flashes, outdir, total))
    if skipped:
        print("  skipped: %s" % ", ".join(skipped))
    if noicon:
        print("  no ICON texture (keeps the fist): %s" % ", ".join(noicon))
    # The pistol is the one the port is built against first; a tree without it is broken
    # in a way that would only show up as "no gun in hand" three layers later.
    if not os.path.isfile(os.path.join(outdir, "w22.bin")):
        sys.exit("weapon_model_bake: PISTOL (w22) did not bake - refusing to report success")


if __name__ == "__main__":
    main()
