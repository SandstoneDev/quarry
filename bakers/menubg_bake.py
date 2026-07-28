#!/usr/bin/env python3
"""menubg_bake.py - per-city pause-menu backgrounds -> data/menubg.bin (PSP port feature).

Vanilla SA shows ONE fixed menu art (Vinewood) behind the pause menu. Our port swaps it by the
player's city. The arts are the real SA menu backgrounds from fronten2.txd (the "back" photos):

  city 0  LOS SANTOS   -> back8  ("VINEWOOD" sign - the vanilla SA menu art)
  city 1  SAN FIERRO   -> back5  (blue city skyline silhouette)
  city 2  LAS VENTURAS -> back3  (desert sunset / billboard)
  city 3  COUNTRYSIDE  -> back2  (green hills)            <- also the default if city unknown

Each art is centre-cropped to 2:1, box-downscaled to 256x128 RGBA8888 (one PSP texture, 128 KB),
so the runtime loads ONLY the active city's bg (lazy) and draws it full-screen under the UI.

menubg.bin (little-endian) ====================================================
  u32 magic 'MBG1'
  u32 count, u32 w, u32 h            # 4, 256, 128
  count * (w*h*4) RGBA8888           # GU_PSM_8888, city order above
================================================================================
Usage: python menubg_bake.py
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (Quarry: user's PS2 disc)

# INPUT: SA_ROOT env (Quarry -> user's extracted PS2 disc); PC install = dev fallback.
SA = os.environ.get("SA_ROOT", "") + "/models"
W, H = 256, 128                     # 2:1, close to the 480x272 (1.76:1) screen
CITIES = ["back8", "back5", "back3", "back2"]   # LS, SF, LV, COUNTRYSIDE
# OUTPUT: argv[1] dir (Quarry passes <data>/hud), else the dev assets_build tree.
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else ""


def crop_2to1(rgba, w, h):
    """centre-crop to a 2:1 (w' = 2*h') region, keeping the most pixels."""
    if w >= 2 * h:                  # too wide -> crop width
        nw = 2 * h; x0 = (w - nw) // 2
        out = bytearray(nw * h * 4)
        for y in range(h):
            s = (y * w + x0) * 4
            out[y*nw*4:(y+1)*nw*4] = rgba[s:s + nw*4]
        return bytes(out), nw, h
    nh = w // 2; y0 = (h - nh) // 2  # too tall -> crop height (centre band)
    return bytes(rgba[y0*w*4:(y0+nh)*w*4]), w, nh


def resize(rgba, w, h, nw, nh):
    """nearest-area downscale to nw x nh RGBA8888."""
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        sy = y * h // nh
        for x in range(nw):
            sx = x * w // nw
            s = (sy * w + sx) * 4; d = (y * nw + x) * 4
            out[d:d+4] = rgba[s:s+4]
    return bytes(out)


def main():
    t2 = sa_txd.decode(open(SA + "/fronten2.txd", "rb").read())
    out = bytearray()
    out += b"MBG1" + struct.pack("<III", len(CITIES), W, H)
    for name in CITIES:
        w, h, rgba = t2[name]
        rgba, w, h = crop_2to1(bytes(rgba), w, h)
        rgba = resize(rgba, w, h, W, H)
        assert len(rgba) == W * H * 4
        out += rgba
        print("  %-6s %dx%d -> %dx%d" % (name, w, h, W, H))

    os.makedirs(OUT_DIR, exist_ok=True)
    outp = os.path.join(OUT_DIR, "menubg.bin")
    open(outp, "wb").write(out)
    print("menubg.bin %d bytes (%d cities x %dx%d) -> %s" % (len(out), len(CITIES), W, H, outp))


if __name__ == "__main__":
    main()
