#!/usr/bin/env python3
"""grass_tex_bake - the grass blade textures -> data/effects/grass.bin.

PlantMgr draws its tufts as alpha-tested cutout crosses and wants exactly two
blade sets. On a PS2 disc they live in models/particle.txd as txgrassbig0 (lush)
and txgrassbig1 (dry), 64x256 each - the PC build kept them in plant1.txd under
different names, which is why the old reader found nothing here.

grass.bin (little-endian), the same container shape CarFx uses so the runtime
loader stays one proven piece of code:
 'GRS1' u16 count u16 pad
 per texture: u16 w, h, levels|amode<<8 (+0x8000 when T4), clutEntries,
 u32 texelLen, clutLen, then swizzled texels and an RGBA8888 clut

Usage: grass_tex_bake.py <particle.txd> <out grass.bin>
"""
import os
import struct
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd                      # PS2-native TXD codec (device id 6)
from pmap_tex_ps2native import encode_indexed

NAMES = ("txgrassbig0", "txgrassbig1")
AMODE_CUTOUT = 1                                # alpha-tested blades, not blended
BLADE = 64                                      # one blade cell; PlantMgr maps UV 0..1 over it


def _first_blade(w, h, rgba):
    """The PS2 sheets stack four blade variants in one 64x256 column (measured:
 four 64-row bands of clearly different alpha coverage). PlantMgr wants a single
 blade per set and maps UV across the whole texture, so hand it the top cell - otherwise every tuft renders all four variants squashed into one quad."""
    if h <= BLADE or h % BLADE:
        return w, h, rgba
    return w, BLADE, rgba[: w * BLADE * 4]


def _encode(name, w, h, rgba):
    arr = np.frombuffer(bytes(rgba), np.uint8).reshape(h, w, 4).copy()
    enc = encode_indexed(arr.copy(), force_t8=True)
    if enc is None:
        # Cutout art with too many colours for a direct palette. The runtime
        # alpha-tests at 0x40, so binarise alpha, collapse the transparent field
        # to one entry and spend the palette on the visible blade.
        from PIL import Image
        keep = arr[..., 3] >= 128
        im = Image.frombytes("RGBA", (w, h), arr.tobytes())
        pal = im.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
        q = np.frombuffer(pal.convert("RGB").tobytes(), np.uint8).reshape(h, w, 3).copy()
        q[~keep] = 0
        arr2 = np.dstack([q, np.where(keep, 255, 0).astype(np.uint8)])
        enc = encode_indexed(arr2.copy(), force_t8=True)
    if enc is None:
        raise SystemExit("%s: could not encode even after quantising" % name)
    return enc


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    txd = {k.lower(): v for k, v in sa_txd.decode(open(src, "rb").read()).items()}

    recs = []
    for nm in NAMES:
        if nm not in txd:
            print("grass_tex_bake: %s not in %s - grass skipped"
                  % (nm, os.path.basename(src)))
            return 1
        w, h, rgba = _first_blade(*txd[nm])
        fmt, texels, clut, _bufw, ce = _encode(nm, w, h, rgba)
        assert fmt == 5, "grass.bin is T8-only: PlantMgr binds GU_PSM_T8 unconditionally"
        recs.append((w, h, 1 | (AMODE_CUTOUT << 8), ce, texels, clut, fmt))
        print("  %s: %dx%d %s texels=%d clut=%d"
              % (nm, w, h, "T4" if fmt == 4 else "T8", len(texels), ce))

    blob = bytearray(struct.pack("<4sHH", b"GRS1", len(recs), 0))
    for (w, h, la, ce, texels, clut, fmt) in recs:
        blob += struct.pack("<HHHHII", w, h, la, ce, len(texels), len(clut))
        blob += texels
        blob += clut
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    open(dst, "wb").write(bytes(blob))
    print("grass.bin: %d bytes -> %s" % (len(blob), dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
