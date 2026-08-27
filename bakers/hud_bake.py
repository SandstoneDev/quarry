#!/usr/bin/env python3
"""hud_bake.py - extract HUD sprite assets for the GTASA_PSP port.

The unarmed weapon-icon slot uses the real SA "fist" texture from models/hud.txd
(64x64 RGBA), and the free-aim CROSSHAIR is `sitem16` out of the same TXD - SA draws
that ONE 64x64 sprite four times, once per corner (CHud::DrawCrossHairs 0x58E020).
`siterocket` (32x32) is the rocket-launcher reticle, baked for the launcher aim mode.

hud.bin:
 'HUD2' ('HUD1' = fist only, still read)
 u32 fistW, fistH ; fistW*fistH RGBA8888
 u32 siteW, siteH ; siteW*siteH RGBA8888 (sitem16)
 u32 rockW, rockH ; rockW*rockH RGBA8888 (siterocket)
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
    lower = {k.lower(): k for k in d}

    def grab(name):
        k = lower.get(name)
        if k is None:
            return None
        w, h, rgba = d[k]
        return w, h, np.frombuffer(rgba, np.uint8).reshape(h, w, 4)

    fist = grab("fist")
    if fist is None:
        sys.exit("hud_bake: hud.txd has no `fist` - refusing to write a HUD without it")
    site = grab("sitem16")
    rock = grab("siterocket")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "hud.bin")
    with open(out, "wb") as f:
        f.write(b"HUD2")
        for name, tex in (("fist", fist), ("sitem16", site), ("siterocket", rock)):
            if tex is None:                       # absent: a 0x0 entry, the HUD keeps its fallback
                f.write(struct.pack("<2I", 0, 0))
                print("  ! hud.txd has no %s - written as empty" % name)
                continue
            w, h, px = tex
            f.write(struct.pack("<2I", w, h))
            f.write(px.tobytes())
    print("hud.bin: fist %dx%d, sitem16 %s, siterocket %s (%.1f KB) -> %s"
          % (fist[0], fist[1],
             "%dx%d" % site[:2] if site else "-",
             "%dx%d" % rock[:2] if rock else "-",
             os.path.getsize(out) / 1e3, out))


if __name__ == "__main__":
    main()
