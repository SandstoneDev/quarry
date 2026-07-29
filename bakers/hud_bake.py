#!/usr/bin/env python3
"""hud_bake.py - extract HUD sprite assets for the GTASA_PSP port.

The unarmed weapon-icon slot uses the real SA "fist" texture from models/hud.txd
(64x64 RGBA). Packed into hud.bin for CHud to draw as the weapon icon.

hud.bin:
 'HUD1'
 u32 fistW, fistH (64, 64)
 fistW*fistH RGBA8888 (fist)
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (Quarry: user's PS2 disc)

import numpy as np

# INPUT: SA_ROOT env (Quarry points it at the user's extracted PS2 disc); the PC
# install stays as the dev-loop fallback.
SA_ROOT = os.environ.get("SA_ROOT", "")
# OUTPUT: argv[1] dir (Quarry passes <data>/hud), else the dev assets_build tree.
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else ""


def main():
    d = sa_txd.decode(open(SA_ROOT + "/models/hud.txd", "rb").read())
    w, h, rgba = d["fist"]
    fist = np.frombuffer(rgba, np.uint8).reshape(h, w, 4)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "hud.bin")
    with open(out, "wb") as f:
        f.write(b"HUD1")
        f.write(struct.pack("<2I", w, h))
        f.write(fist.tobytes())
    print("hud.bin: fist %dx%d (%.1f KB) -> %s" % (w, h, os.path.getsize(out)/1e3, out))


if __name__ == "__main__":
    main()
