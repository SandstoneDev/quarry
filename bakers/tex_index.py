#!/usr/bin/env python3
"""Global texture-name index across all TXDs in an IMG archive.

SA models often reference textures that live in a DIFFERENT (parent / generic)
TXD than the model's own - e.g. scumwires1_las2 uses `telewires_law`, which is
NOT in wiresetc_las2.txd. The exporter only loads a model's own TXD, so those
textures resolve to -1 -> the surface renders white (the untextured wires).

This builds {texture_name_lower: txd_filename} across an IMG's TXDs (first
writer wins), cached to a pickle so it is built once. The exporter's texture
resolver falls back to it when a name is missing from the model's own TXD.
"""
import os
import pickle
import sys

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import sa_txd
from gvcslib.sa_img import SaImg


def build_index(img_path, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    img = SaImg(img_path)
    index = {}
    ntxd = 0
    for n in img.names():
        if not n.lower().endswith(".txd"):
            continue
        try:
            d = sa_txd.decode(img.extract(n))
        except Exception:
            continue
        ntxd += 1
        for name in d:
            k = name.lower()
            if k not in index:              # first writer wins
                index[k] = n
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)
    print("tex index: %d names across %d TXDs -> %s" % (len(index), ntxd, cache_path or "(no cache)"))
    return index


if __name__ == "__main__":
    root = sys.argv[1]
    img = root + "/MODELS/GTA3.IMG"
    cache = sys.argv[2] if len(sys.argv) > 2 else None
    idx = build_index(img, cache)
    for q in sys.argv[3:]:
        print("  %-22s -> %s" % (q, idx.get(q.lower(), "NOT FOUND")))
