#!/usr/bin/env python3
"""fix_translucent_class - move GRADIENT-alpha cutout textures to the blend class.

amode 1 (alpha-test cutout, GU_GREATER 0x40, no blend) is correct for SA's
bimodal-alpha art (fences, foliage: alpha 0 or 255). But soft decals - dust,
dirt, baked shadows - carry an alpha GRADIENT; under alpha-test their dense
texels draw as solid patches (the interior dust i24_m17_s69 report). Reclass
any amode-1 texture whose alpha is substantially MID-BAND to amode 2
(translucent blend pass): renders soft, exactly like the Blender preview.

Only the texture-table class byte changes; pools untouched.

Usage: fix_translucent_class.py <file-or-dir> [more...] [--dry]
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from pmap_tex_t4from128 import HDR, TEX, load_v2
from pmap_tex_ps2native import decode_deployed


def main():
    argv = sys.argv[1:]
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    files = []
    for a in argv:
        if os.path.isdir(a):
            files += sorted(os.path.join(a, f) for f in os.listdir(a)
                            if f.lower().endswith(".pmap"))
        else:
            files.append(a)

    tot = nfiles = 0
    for path in files:
      try:
        prod, ver = load_v2(path)
        hp = HDR.unpack_from(prod, 0)
        tc, toff = hp[7], hp[8]
        changed = []
        out = bytearray(prod)
        for ti in range(tc):
            t = TEX.unpack_from(prod, toff + 32 * ti)
            nlev = t[8]
            if ((nlev >> 8) & 3) != 1:
                continue
            rgba = decode_deployed(prod, hp, t)
            if rgba is None:
                continue
            a = rgba[..., 3].astype(np.int32)
            mid = float(((a >= 16) & (a <= 239)).mean())
            if mid <= 0.35:
                continue                       # bimodal cutout art: keep class 1
            new_nlev = (nlev & 0xFFFF00FF) | (2 << 8)
            TEX.pack_into(out, toff + 32 * ti,
                          t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], new_nlev)
            changed.append((ti, round(mid, 2)))
        if not changed:
            continue
        nfiles += 1; tot += len(changed)
        print(f"  {os.path.basename(path)}: {len(changed)} cutout->translucent "
              f"{changed[:6]}", flush=True)
        if dry:
            continue
        if ver == 3:
            tmp = tempfile.mktemp(suffix='.pmap')
            open(tmp, 'wb').write(bytes(out))
            subprocess.check_call([sys.executable,
                                   os.path.join(TOOLS, 'pmap_lz4.py'),
                                   tmp, path], stdout=subprocess.DEVNULL)
            os.remove(tmp)
        else:
            open(path, 'wb').write(bytes(out))
      except Exception as ex:
        print(f"  {os.path.basename(path)}: ERROR {ex}", flush=True)
    print(f"DONE: {tot} textures reclassed in {nfiles} files")


if __name__ == "__main__":
    main()
