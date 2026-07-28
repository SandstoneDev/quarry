#!/usr/bin/env python3
"""fxtex_bake - bake the SA particle textures (effectsPC.txd) the CarFx/fx
renderers need into data/fxtex.bin.

Order is FIXED (the runtime indexes by position):
  0 smokeii_3   engine/explosion smoke puff (the SA prt_smokeII prototype tex)
  1 fireball6   fire lick / explosion fireball

fxtex.bin: 'FXT1' u16 count u16 pad, then per texture:
  u16 w,h, nlevels|amode<<8, clutEntries; u32 texelLen, clutLen; texels; clut
"""
import os
import struct
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
if GVCS not in sys.path:
    sys.path.insert(0, GVCS)
from gvcslib import sa_txd, psp_tex   # PS2-native TXD codec (device id 6)

# The disc carries these in models/effects.txd. The PC build's effectsPC.txd, which
# this used to open, does not exist on a PS2 disc.
SA  = os.environ.get("SA_ROOT", "")
TXD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SA, "models", "effects.txd")
OUT = sys.argv[2] if len(sys.argv) > 2 else "fxtex.bin"
NAMES = ["smokeii_3", "fireball6"]


def main():
    texs = sa_txd.decode(open(TXD, "rb").read())
    texs = {k.lower(): v for k, v in texs.items()}
    buf = b"FXT1" + struct.pack("<HH", len(NAMES), 0)
    for nm in NAMES:
        w, h, rgba = texs[nm]
        t = psp_tex.author_psp_texture(rgba, w, h, fmt="T8", mipmaps=False)
        buf += struct.pack("<4H2I", t["width"], t["height"],
                           (t["num_levels"] & 0xFF) | ((t.get("alpha_mode", 0) & 3) << 8),
                           t["clut_entries"], len(t["texel_bytes"]), len(t["clut_bytes"]))
        buf += t["texel_bytes"] + t["clut_bytes"]
        print(f"  {nm}: {t['width']}x{t['height']}")
    open(OUT, "wb").write(buf)
    print(f"fxtex.bin: {len(buf)} bytes")


if __name__ == "__main__":
    main()
