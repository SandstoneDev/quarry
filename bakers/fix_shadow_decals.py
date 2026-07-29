#!/usr/bin/env python3
"""fix_shadow_decals - restore ALPHA on SA shadow-decal textures.

SA bakes soft shadows as black textures with an alpha GRADIENT (treeshad.txd:
railshadowdif etc.), drawn blended. Our bake lost the alpha (opaque 255) and
classified them opaque -> solid black blobs on interior floors and black
silhouettes under world trees/props.

For every texture in a pmap whose decoded content is near-black (lum < 40):
fingerprint-match it to the PC library; if the match is a shadow texture
(txd or name contains 'shad'), re-encode it from the PC original WITH its
true alpha (T4/T8, palette keeps RGBA) and set the texture's alpha_mode byte
(num_levels byte 1) to 1 so the renderer draws it in the blended alpha pass.

Usage:
 fix_shadow_decals.py <file-or-dir> [more...] --db <fp_prefix> [--dry]
v3 handled; in-place rewrite (make backups upstream if wanted).
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
GVCS = os.environ.get("GVCS_ROOT", "")
SAW = os.environ.get("SAW_ROOT", "")
for p in (GVCS, SAW):
    if p not in sys.path:
        sys.path.insert(0, p)
from gvcslib import sa_txd_d3d9
from core.imgarchive import ImgArchive

import tex_fingerprint_db as fdb
from pmap_tex_t4from128 import HDR, TEX, load_v2
from pmap_tex_ps2native import decode_deployed, encode_indexed

PC_IMGS = ["",
           ""]


def main():
    argv = sys.argv[1:]
    def opt(name, default=None):
        if name in argv:
            k = argv.index(name); v = argv[k + 1]; del argv[k:k + 2]
            return v
        return default
    db_prefix = opt("--db")
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]

    files = []
    for a in argv:
        if os.path.isdir(a):
            files += sorted(os.path.join(a, f) for f in os.listdir(a)
                            if f.lower().endswith(".pmap"))
        else:
            files.append(a)

    z = np.load(db_prefix + ".npz", allow_pickle=True)
    dbh = z["hashes"]; meta = z["meta"]

    imgs = [ImgArchive.open(p) for p in PC_IMGS]
    txd_cache = {}
    def pc_tex(txd_name, tex_name):
        if txd_name not in txd_cache:
            txd_cache[txd_name] = {}
            for img in imgs:
                for e in img.entries:
                    if e.name.lower() == txd_name + ".txd":
                        try:
                            txd_cache[txd_name] = sa_txd_d3d9.decode(img.extract(e))
                        except Exception:
                            pass
                        break
                if txd_cache[txd_name]:
                    break
        return txd_cache[txd_name].get(tex_name)

    tot_fixed = tot_files = 0
    for path in files:
      try:
        prod, ver = load_v2(path)
        hp = HDR.unpack_from(prod, 0)
        tc, tex_off = hp[7], hp[8]
        tp = [TEX.unpack_from(prod, tex_off + 32 * i) for i in range(tc)]
        fixes = {}
        for ti in range(tc):
            rgba = decode_deployed(prod, hp, tp[ti])
            if rgba is None or float(rgba[..., :3].mean()) >= 40.0:
                continue
            key, _ = fdb.hashes(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
            q = np.frombuffer(bytes.fromhex(key), np.uint8)
            d = np.unpackbits(dbh ^ q, axis=1).sum(1)
            j = int(d.argmin())
            if d[j] > 4:
                continue
            txd, nm = str(meta[j][0]), str(meta[j][1])
            if "shad" not in txd and "shad" not in nm:
                continue
            got = pc_tex(txd, nm)
            if got is None:
                continue
            w, h, prgba = got
            while max(w, h) > 128:               # keep the deployed size class
                from PIL import Image
                im = Image.frombytes("RGBA", (w, h), bytes(prgba))
                w //= 2; h //= 2
                prgba = im.resize((w, h), Image.LANCZOS).tobytes()
            arr = np.frombuffer(bytes(prgba), np.uint8).reshape(h, w, 4)
            if arr[..., 3].max() >= 250:         # not actually a soft decal
                continue
            enc = encode_indexed(arr)
            if enc is None:
                continue
            fixes[ti] = (w, h) + enc            # (w,h,fmt,texels,clut,bufw,ce)
            if dry:
                print(f"    {os.path.basename(path)} tex{ti}: {txd}/{nm} "
                      f"alpha_mean={arr[..., 3].mean():.0f}")
        if not fixes:
            continue
        if dry:
            tot_files += 1; tot_fixed += len(fixes)
            continue
        # splice texel+clut pools (last two sections)
        texel_pool = bytearray(); clut_pool = bytearray(); new_tex = []
        for i in range(tc):
            (w, h, fmt, texel_first, texel_bytes, bufw, clut_first,
             clut_entries, nlev) = tp[i]
            if i in fixes:
                nw, nh, nfmt, texels, clut, bufw_tex, ce = fixes[i]
                nt = (nw, nh, nfmt, len(texel_pool), len(texels), bufw_tex,
                      len(clut_pool), ce, (nlev & 0xFFFF00FF) | (2 << 8))
                # class 2 = TRANSLUCENT (blend pass). Class 1 is alpha-TEST
                # cutout (GU_GREATER 0x40, no blend) - a soft shadow there
                # draws its dense core as solid black and drops its edges.
                texel_pool += texels; clut_pool += clut
            else:
                nt = (w, h, fmt, len(texel_pool), texel_bytes, bufw,
                      len(clut_pool), clut_entries, nlev)
                texel_pool += prod[hp[16] + texel_first:
                                   hp[16] + texel_first + texel_bytes]
                clut_pool += prod[hp[18] + clut_first:
                                  hp[18] + clut_first + clut_entries * 4]
            new_tex.append(nt)
            while len(texel_pool) % 16: texel_pool.append(0)
            while len(clut_pool) % 16: clut_pool.append(0)
        out = bytearray(prod[:hp[16]])
        out += texel_pool; out += clut_pool
        hl = list(hp)
        hl[2] = len(out); hl[17] = len(texel_pool)
        hl[18] = hp[16] + len(texel_pool); hl[19] = len(clut_pool)
        HDR.pack_into(out, 0, *hl)
        for i, nt in enumerate(new_tex):
            TEX.pack_into(out, tex_off + 32 * i, *nt)
        if ver == 3:
            tmp = tempfile.mktemp(suffix='.pmap')
            open(tmp, 'wb').write(bytes(out))
            subprocess.check_call([sys.executable,
                                   os.path.join(TOOLS, 'pmap_lz4.py'),
                                   tmp, path], stdout=subprocess.DEVNULL)
            os.remove(tmp)
        else:
            open(path, 'wb').write(bytes(out))
        tot_files += 1; tot_fixed += len(fixes)
        print(f"  {os.path.basename(path)}: {len(fixes)} shadow textures "
              f"re-alphaed", flush=True)
      except Exception as ex:
        print(f"  {os.path.basename(path)}: ERROR {ex}", flush=True)
    print(f"DONE: {tot_fixed} textures in {tot_files} files")


if __name__ == "__main__":
    main()
