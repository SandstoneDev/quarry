#!/usr/bin/env python3
"""pickup_bake - the pickup ICON models (dollar, bribe, info, health, armour, ...).

Weapon pickups reuse the weapon meshes weapon_model_bake.py already writes; what is
missing is the other half of the system - the icon band. Those models are declared in
`data/maps/generic/dynamic.ide` on the disc, not in default.ide, which is why they are
easy to miss when you go looking for them by name.

The list below is the icon set recovered in 
section 5, model id -> model name -> TXD, verified against the disc.

Output: data/pickups/p<modelId>.bin, the same 'PRP1' container CProp_Load already reads
(so the pickup renderer is the prop renderer, exactly as SA's is CObject).

 SA_ROOT=... GVCS_ROOT=... python tools/pickup_bake.py [--out data]
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

import cutprops_bake
from sa_img import SaImg

SA_ROOT = os.environ.get("SA_ROOT", "")

# model id -> what it is. Names/TXDs come from the IDE at bake time so a typo here shows
# up as "not in ide" rather than as a wrong mesh.
ICONS = {
    1239: "info",
    1240: "health",
    1241: "adrenaline",
    1242: "bodyarmour",
    1247: "bribe (star removal)",
    1248: "bonus",
    1252: "explosive barrel",
    1253: "camera",
    1254: "killfrenzy",
    1272: "property locked",
    1273: "property for sale",
    1274: "bigdollar",
    1275: "clothes",
    1276: "package1 (collectable)",
    1277: "save",
    1310: "parachute",
    1313: "killfrenzy 2p",
    1314: "two player",
    1212: "money",
    1210: "briefcase",
    # ★ b984: the jetpack. MODEL_JETPACK = 370 (eModelID.h). It is not an "icon" like the
    # rest of this table - it is the real wearable prop - but it reaches the world the
    # same way (CPickups::GenerateNewOne with PICKUP_ONCE_TIMEOUT_SLOW when the player
    # drops it, TaskSimpleJetPack.cpp:447), and the runtime loads it through the same
    # pickups/p<id>.bin path, so it belongs here rather than in a baker of its own.
    370:  "jetpack",
}

# Where the icon band is declared. default.ide holds the weapons; the icons live here.
IDE_FILES = (
    os.path.join("data", "maps", "generic", "dynamic.ide"),
    os.path.join("data", "maps", "generic", "propext.ide"),
    os.path.join("data", "default.ide"),
)


def ide_models(root):
    """{model id: (dffName, txdName)} across every IDE that declares pickups."""
    out = {}
    for rel in IDE_FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or "," not in line:
                    continue
                tok = [t.strip() for t in line.split(",")]
                if len(tok) < 3 or not tok[0].isdigit():
                    continue
                out.setdefault(int(tok[0]), (tok[1], tok[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--root", default=SA_ROOT)
    a = ap.parse_args()
    if not a.root:
        sys.exit("pickup_bake: set SA_ROOT (or pass --root)")

    ide = ide_models(a.root)
    img = SaImg(os.path.join(a.root, "models", "gta3.img"))
    outdir = os.path.join(a.out, "pickups")
    os.makedirs(outdir, exist_ok=True)

    made, total, skipped = 0, 0, []
    for mid in sorted(ICONS):
        names = ide.get(mid)
        if not names:
            skipped.append("%d(%s: not in ide)" % (mid, ICONS[mid]))
            continue
        dff, txd = names
        try:
            # merge=True for the same reason the weapons need it: an icon can be several
            # atomics, and each sits at its own frame matrix (b880/b882).
            verts, idx, tex = cutprops_bake.bake_mesh(img, dff, txd, merge=True)
        except (Exception, SystemExit) as e:
            skipped.append("%d/%s(%s)" % (mid, dff, type(e).__name__))
            continue
        if not verts or not idx:
            skipped.append("%d/%s(empty)" % (mid, dff))
            continue

        buf = bytearray(b"PRP1")
        buf += struct.pack("<HHHHHH", len(verts), len(idx), tex["width"], tex["height"],
                           tex["num_levels"] | (tex.get("alpha_mode", 0) << 8),
                           tex["clut_entries"])
        buf += struct.pack("<II", len(tex["texel_bytes"]), len(tex["clut_bytes"]))
        for (u, v, c, x, y, z) in verts:
            buf += struct.pack("<ffIfff", u, v, c, x, y, z)
        for i in idx:
            buf += struct.pack("<H", i)
        buf += tex["texel_bytes"] + tex["clut_bytes"]

        with open(os.path.join(outdir, "p%d.bin" % mid), "wb") as f:
            f.write(buf)
        made += 1
        total += len(buf)
        print("  p%-5d %-16s %-14s %4d vert %5d idx %3dx%-3d %6d B"
              % (mid, dff, ICONS[mid], len(verts), len(idx),
                 tex["width"], tex["height"], len(buf)))

    print("pickup_bake: %d icons -> %s (%d B)" % (made, outdir, total))
    if skipped:
        print("  skipped: %s" % ", ".join(skipped))
    # The dollar is the one the user asked for by name; a tree without it is broken in a
    # way that would only show as "no money pickup" much later.
    if not os.path.isfile(os.path.join(outdir, "p1274.bin")):
        sys.exit("pickup_bake: bigdollar (1274) did not bake - refusing to report success")


if __name__ == "__main__":
    main()
