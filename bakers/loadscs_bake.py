#!/usr/bin/env python3
"""loadscs_bake.py - the disc's painted loading arts -> loadscs.bin for the engine.

The arts live in ONE texture dictionary, models/txd/LOADS<region>.txd (LOADSUK.txd on
a PAL disc), holding fifteen 512x512 textures: `loadsc1`..`loadsc14` plus a title card.
Earlier revisions of this baker looked first for a PC-style LOADSCS.txd and then for
MODELS/TXD/INTRO*.TXD; neither is right. The INTRO files are the opening cutscene
backdrops, which is why a converted build showed a caption card where a painting
belonged.

The title card is packed FIRST. The engine treats index 0 as the title: it shows it
once at boot with the progress bar hidden, then cycles the remaining arts at random
and never returns to it.

 header : u32 magic 'LDSC', u32 count, u32 width(512), u32 height(512)
 texels : count * (512*512*4) RGBA8888 (alpha last == PSP GU_PSM_8888)

1 MB resident per art; only one is held at a time, before the streaming cache is live.

Usage: python loadscs_bake.py <LOADS*.txd, or the txd dir> <out loadscs.bin>
"""
import glob
import os
import re
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd            # PS2-native TXD codec (Quarry: user's PS2 disc)
from PIL import Image                 # resize any art to the 512x512 the engine draws

SIZE = 512


def find_txd(src):
    """The LOADS<region>.txd next to src (src may be that file or the txd dir).
 Also probes SA_ROOT/models/txd so an env-only call resolves."""
    dirs = []
    if src:
        dirs.append(src if os.path.isdir(src) else os.path.dirname(src))
    sa = os.environ.get("SA_ROOT", "")
    if sa:
        dirs += [os.path.join(sa, "models", "txd"), os.path.join(sa, "MODELS", "TXD")]
    found = []
    for d in dirs:
        if d and os.path.isdir(d):
            found += glob.glob(os.path.join(d, "[Ll][Oo][Aa][Dd][Ss]*.[Tt][Xx][Dd]"))
    return sorted(set(os.path.normcase(p) for p in found))


def art_order(names):
    """Title card first, then loadscN by number. The title is the entry carrying no
 digits (loadscuk on a PAL disc); a disc without one simply has no title card and
 every art joins the rotation."""
    numbered = sorted((int(re.search(r"(\d+)", n).group(1)), n)
                      for n in names if re.search(r"(\d+)", n))
    title = [n for n in names if not re.search(r"(\d+)", n)]
    return title + [n for _, n in numbered]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SA_ROOT", "")
    out = sys.argv[2] if len(sys.argv) > 2 else "loadscs.bin"

    txds = find_txd(src)
    if not txds:
        raise SystemExit("no LOADS*.txd (the loading arts) found near: " + str(src))

    arts = []
    for path in txds:
        txd = sa_txd.decode(open(path, "rb").read())
        for name in art_order(list(txd.keys())):
            w, h, rgba = txd[name]
            im = Image.frombytes("RGBA", (w, h), bytes(rgba))
            if (w, h) != (SIZE, SIZE):
                im = im.resize((SIZE, SIZE), Image.LANCZOS)
            arts.append(im.tobytes())
            print("  %-14s %-12s %dx%d%s" % (os.path.basename(path), name, w, h,
                                             "  <- title card" if len(arts) == 1 else ""))

    if not arts:
        raise SystemExit("LOADS*.txd found but held no textures")

    blob = bytearray(b"LDSC" + struct.pack("<III", len(arts), SIZE, SIZE))
    for texels in arts:
        assert len(texels) == SIZE * SIZE * 4
        blob += texels
    open(out, "wb").write(blob)
    print("wrote %s : %d loading arts, %.1f MB" % (out, len(arts), len(blob) / 1048576.0))


if __name__ == "__main__":
    main()
