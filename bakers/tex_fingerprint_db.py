#!/usr/bin/env python3
"""tex_fingerprint_db - perceptual-hash DB of every PC SA world texture.

Purpose (PS2-native retexture, session 2026-07-21): deployed pmaps carry no
texture names; the bake sources are rotten for a name-ordered re-run. But every
deployed texel ultimately came from a PC TXD via a known LANCZOS downscale, so
CONTENT identifies the (txd, name) pair: hash every PC texture at 64px and
match deployed textures by perceptual hash. The name then keys the PS2-native
lookup (PS2 TXDs use the same txd/texture names).

Builds: {ahash64+dhash64 hex: [(txd, name, w, h), ...]} -> tex_fp_db.json
        + packed hashes tex_fp_db.npz for fast Hamming fallback.

Usage: python tex_fingerprint_db.py <img> [<img2> ...] <out_prefix> [--limit N]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

GVCS = os.environ.get("GVCS_ROOT", "")
SAW = os.environ.get("SAW_ROOT", "")
for p in (GVCS, SAW):
    if p not in sys.path:
        sys.path.insert(0, p)
from gvcslib import sa_txd_d3d9
from core.imgarchive import ImgArchive


def hashes(rgba, w, h):
    # ALPHA-AWARE: composite over mid-gray first. Mostly-transparent art
    # (wires, foliage) has garbage RGB under alpha=0 (DXT blocks) - raw-RGB
    # hashing matched wires to unrelated pale textures and the PS2 retex
    # see the SHAPE the player sees.
    a = np.frombuffer(bytes(rgba), np.uint8).reshape(-1, 4).astype(np.uint16)
    comp = ((a[:, :3] * a[:, 3:4] + 128 * (255 - a[:, 3:4])) // 255).astype(np.uint8)
    im = Image.frombytes("RGB", (w, h), comp.tobytes()).convert("L")
    a = np.asarray(im.resize((8, 8), Image.LANCZOS), np.float32)
    ah = (a > a.mean()).flatten()
    d = np.asarray(im.resize((9, 8), Image.LANCZOS), np.float32)
    dh = (d[:, 1:] > d[:, :-1]).flatten()
    bits = np.concatenate([ah, dh])                      # 128 bits
    v = np.packbits(bits.astype(np.uint8))
    return v.tobytes().hex(), bits


def main():
    args = [a for a in sys.argv[1:]]
    limit = 0
    if "--limit" in args:
        k = args.index("--limit")
        limit = int(args[k + 1])
        del args[k:k + 2]
    img_paths, outp = args[:-1], args[-1]
    db = {}
    packed = []
    meta = []
    n_txd = n_tex = n_fail = 0
    for img_path in img_paths:
        img = ImgArchive.open(img_path)
        for e in img.entries:
            if not e.name.lower().endswith(".txd"):
                continue
            n_txd += 1
            if limit and n_txd > limit:
                break
            txd_name = e.name[:-4].lower()
            try:
                texd = sa_txd_d3d9.decode(img.extract(e))
            except Exception:
                n_fail += 1
                continue
            for nm, (w, h, rgba) in texd.items():
                try:
                    key, bits = hashes(rgba, w, h)
                except Exception:
                    n_fail += 1
                    continue
                db.setdefault(key, []).append((txd_name, nm.lower(), w, h))
                packed.append(np.packbits(bits.astype(np.uint8)))
                meta.append((txd_name, nm.lower(), w, h))
                n_tex += 1
            if n_txd % 500 == 0:
                print(f"  {n_txd} txd, {n_tex} tex...", flush=True)
    json.dump(db, open(outp + ".json", "w"))
    np.savez_compressed(outp + ".npz",
                        hashes=np.stack(packed) if packed else np.zeros((0, 16), np.uint8),
                        meta=np.array(meta, dtype=object))
    print(f"DONE: {n_txd} txd, {n_tex} textures, {n_fail} failures -> {outp}.json/.npz")


if __name__ == "__main__":
    main()
