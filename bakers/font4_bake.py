#!/usr/bin/env python3
"""Bake font4 = the game's thick DISPLAY face (the condensed heavy font it uses for the
money counter and the clock) into a PSP atlas in plain ASCII order, so CFont can use it
exactly like font1/font2/font3 (ASCII 0x20..0x7F at tile c-0x20).

This is the disc-native "Pricedown-style" HUD face - extracted straight from the game,
no downloaded TTF. Source = fonts.txd "font1" texture (16x13 grid). The display glyphs
live in the LOWER block of that atlas (same layout as font2's Bank Gothic block):
 row 9 : '0'..'9' at cols 0..9, ':' at col 10, 'A'..'E' at cols 11..15
 row 10: 'F'..'U' at cols 0..15
 row 11: 'V'..'Z' at cols 0..4
It is digits + ':' + CAPS only (no other punctuation), so lowercase maps to uppercase.
Two HUD glyphs the block lacks are synthesized IN THE SAME WEIGHT so money/clock read as
one font (the alternative - borrowing them from the thin clean block - is exactly the
mismatch flagged):
 '$' = the display 'S' (row10 col13) with a centered vertical bar of the digit stroke.
 ':' = the display colon at row9 col10 (chunky, matches the digits).
Any remaining punctuation/space falls back to font1's clean upper block (rare in the HUD).

Out: <outdir>/font4.bin (512x512 RGBA8888, white RGB + glyph alpha) +
 src/game_sa/font4_widths.h (FONT4_W[96] proportional advances, dev repo only). Run:
 python tools/font4_bake.py [outdir]
"""
import os, sys
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (Quarry: user's PS2 disc)
from PIL import Image

# INPUT: SA_ROOT env (Quarry -> user's extracted PS2 disc); PC install = dev fallback.
SA_ROOT = os.environ.get("SA_ROOT", "")
TXD   = SA_ROOT + "/models/fonts.txd"
# OUTPUT: argv[1] dir (Quarry passes <data>/hud), else the dev assets_build tree.
OUTD  = sys.argv[1] if len(sys.argv) > 1 else ""
HDR   = ""
A, COLS, ROWS = 512, 16, 13
CW, CHH = A / COLS, A / ROWS
# Side bearing added to the ink width to get the advance. The display face is
# CONDENSED: a generous bearing makes the clock wide enough to run under the
# weapon icon. Ink is ~18 px in a 32 px cell, so 3 px is a real gap already.
BEARING = 3
GUTTER_X, GUTTER_Y = 1, 2      # texels of empty space kept on each cell edge


def src_tile(c):
    """(row,col) in font1's display block for ASCII code c, or None for blank.
 '$' (0x24) is handled by synth_dollar(), not here."""
    ch = chr(c)
    if ch == ' ':
        return None
    if '0' <= ch <= '9':
        return (9, ord(ch) - ord('0'))
    if ch == ':':
        return (9, 10)                            # chunky display colon (clock)
    L = ch.upper()
    if 'A' <= L <= 'E':
        return (9, 11 + (ord(L) - ord('A')))
    if 'F' <= L <= 'U':
        return (10, ord(L) - ord('F'))
    if 'V' <= L <= 'Z':
        return (11, ord(L) - ord('V'))
    # other punctuation/symbols: fall back to the clean upper block at the ASCII position
    if 0x20 <= c <= 0x3F:
        t = c - 0x20
        return (t // COLS, t % COLS)
    return None


def _crop(sa, row, col):
    return sa.crop((int(col * CW), int(row * CHH), int(col * CW + CW), int(row * CHH + CHH)))


def synth_dollar(sa):
    """Chunky '$' = the display 'S' (row10 col13) with a centred vertical bar.

 The bar must stay THIN: at HUD size the glyph is only ~18 px of ink, so a bar
 sized off the digit stroke swallows the S and the sign reads as a blob. A
 fifth of the glyph width matches how the display face draws its own stems."""
    S = _crop(sa, 10, 13).copy()
    px = S.load()
    bb = S.getbbox()
    if bb:
        cx = (bb[0] + bb[2]) // 2
        stroke = max(3, (bb[2] - bb[0]) // 5)
        top = max(0, bb[1] - 4)
        bot = min(int(CHH), bb[3] + 4)
        for y in range(top, bot):
            for x in range(cx - stroke // 2, cx - stroke // 2 + stroke):
                if 0 <= x < int(CW):
                    px[x, y] = 255
    return S


def main():
    texs = sa_txd.decode(open(TXD, "rb").read())
    w, h, rgba = texs["font1"]
    src = Image.frombytes("RGBA", (w, h), rgba).resize((A, A), Image.BILINEAR)
    sa = src.split()[3]                          # alpha = glyph coverage
    dst = Image.new("L", (A, A), 0)

    widths = [0] * 96
    for c in range(0x20, 0x80):
        di = c - 0x20
        dr, dc = di // COLS, di % COLS
        cell = synth_dollar(sa) if c == 0x24 else None   # '$' has no display glyph
        if cell is None:
            st = src_tile(c)
            if st is None:
                widths[di] = int(CW * 0.42) if c == 0x20 else int(CW * 0.30)
                continue
            cell = _crop(sa, *st)
        # GUTTER: most display glyphs sit flush against the left and bottom of
        # their source cell. Pasted flush, GU_LINEAR sampling at a cell edge
        # reaches into the neighbouring glyph and draws it as a faint stripe
        # beside the text. Shifting the cell content by a couple of texels puts
        # empty space on every edge. The shift is the SAME for every glyph, so
        # relative baselines are untouched.
        dst.paste(cell, (int(dc * CW) + GUTTER_X, int(dr * CHH) - GUTTER_Y))
        # Advance is measured from the CELL origin, not from the ink: CFont draws the
        # whole 32px cell starting at the pen, so a glyph with a left bearing appears
        # that many texels in. Using the ink WIDTH ignored that bearing and pulled the
        # next glyph over the tail of this one - visible on ':' (bearing 5, ink 8),
        # whose advance of 11 overlapped its own right edge at 13 and welded the colon
        # to the minutes. Right ink edge + bearing is the correct advance.
        bb = cell.getbbox()
        widths[di] = (bb[2] + GUTTER_X + BEARING) if bb else int(CW * 0.5)

    # white RGB + alpha, emit raw RGBA8888
    out = bytearray(A * A * 4)
    ap = dst.load()
    for y in range(A):
        for x in range(A):
            o = (y * A + x) * 4
            out[o] = out[o + 1] = out[o + 2] = 255
            out[o + 3] = ap[x, y]
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, "font4.bin"), "wb").write(out)

    # FONT4_W is engine source (already compiled in): regenerate it only in the dev
    # repo. Under the Quarry converter src/ is absent -> skip (data/ needs only the.bin).
    if HDR and os.path.isdir(os.path.dirname(HDR)):
        with open(HDR, "w") as f:
            f.write("/* generated by tools/font4_bake.py - display face advances. */\n")
            f.write("#ifndef FONT4_W_H\n#define FONT4_W_H\n")
            f.write("static const unsigned char FONT4_W[96] = {\n")
            for i in range(0, 96, 16):
                f.write("    " + ",".join("%2d" % widths[j] for j in range(i, i + 16)) + ",\n")
            f.write("};\n#endif\n")

    print("wrote %s" % os.path.join(OUTD, "font4.bin"))


if __name__ == "__main__":
    main()
