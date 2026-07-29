#!/usr/bin/env python3
"""frontend_bake.py - the source game pause-menu sprites -> data/frontend.bin for the PSP port.

Pulls the real front-end sprites out of the SA TXDs (the same ones CMenuManager loads
from FrontEndFilenames[] in the original):
 - "arrow" (fronten1.txd, 64x128) - the blue cross/4-arrow SELECTION CURSOR drawn
 next to the highlighted menu item.
 - one menu BACKGROUND photo (fronten2.txd "back2"..., 512x512) - box-downscaled to
 128x128, shown dimmed in the top-right corner like the SA pause screen.

frontend.bin (little-endian) ==================================================
 u32 magic 'FEND'
 u32 arrowW, arrowH # 64,128
 u32 bgW, bgH # 128,128
 arrowW*arrowH*4 RGBA8888 # GU_PSM_8888 (alpha last)
 bgW*bgH*4 RGBA8888
================================================================================
Usage: python frontend_bake.py
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (Quarry: user's PS2 disc)

# INPUT: SA_ROOT env (Quarry -> user's extracted PS2 disc); PC install = dev fallback.
SA = os.environ.get("SA_ROOT", "") + "/models"
BG_NAME = "back2"          # menu background photo to show in the corner
BG_SIZE = 128
# OUTPUT: argv[1] dir (Quarry passes <data>/hud), else the dev assets_build tree.
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else ""


def box_half(rgba, w, h):
    nw, nh = w // 2, h // 2
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        s0 = (2 * y) * w * 4
        s1 = (2 * y + 1) * w * 4
        for x in range(nw):
            i0 = s0 + (2 * x) * 4
            i1 = s1 + (2 * x) * 4
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = (rgba[i0 + c] + rgba[i0 + 4 + c] +
                              rgba[i1 + c] + rgba[i1 + 4 + c]) >> 2
    return bytes(out), nw, nh


def fit(rgba, w, h, target):
    while w > target:
        rgba, w, h = box_half(rgba, w, h)
    return rgba, w, h


def main():
    t1 = sa_txd.decode(open(SA + "/fronten1.txd", "rb").read())
    t2 = sa_txd.decode(open(SA + "/fronten2.txd", "rb").read())

    aw, ah, arrow = t1["arrow"]
    bw, bh, bg = t2[BG_NAME]
    bg, bw, bh = fit(bytes(bg), bw, bh, BG_SIZE)
    print("arrow %dx%d  bg %s %dx%d" % (aw, ah, BG_NAME, bw, bh))

    out = bytearray()
    out += b"FEND" + struct.pack("<IIII", aw, ah, bw, bh)
    out += bytes(arrow)
    out += bytes(bg)

    os.makedirs(OUT_DIR, exist_ok=True)
    outp = os.path.join(OUT_DIR, "frontend.bin")
    open(outp, "wb").write(out)
    print("frontend.bin %d bytes -> %s" % (len(out), outp))


if __name__ == "__main__":
    main()
