#!/usr/bin/env python3
"""fix_graft_texnames - NAME-driven texture repair for grafted models.

Fingerprint matching cannot recover art the old bake DESTROYED (the long
telephone wires: the thin alpha line averaged away by the 64px downscale into
a flat ~65-alpha black sheet -> invisible when drawn honestly). But for every
census-grafted model we KNOW the source DFF, and its material split order
names each texture: walking the prod model's distinct-texture sequence against
the DFF's split texture-name sequence assigns names deterministically.

For each named texture that is ALPHA-CLASS art (cutout/translucent) and whose
deployed version is smaller than the PC original or alpha-flat, re-encode from
the PC original at native size (cap 256) with true alpha, and set the class
from the psp_tex rule (bimodal -> cutout, gradient -> translucent).

Usage: fix_graft_texnames.py <chunks_dir> <census.json> [--dry]
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "map_export"))
import sa_source

SAW = os.environ.get("SAW_ROOT", "")
GVCS = os.environ.get("GVCS_ROOT", "")
for _p in (SAW, GVCS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from formats.dff import parse_dff
from gvcslib import sa_txd_d3d9

from pmap_tex_t4from128 import HDR, TEX, load_v2
from pmap_tex_ps2native import decode_deployed, encode_indexed, _alpha_class


def encode_indexed_q(arr):
    """encode_indexed with a median-cut quantise fallback: photo art with >256
 unique colours (encode_indexed returns None) gets palettised to 256 first."""
    enc = encode_indexed(arr.copy())
    if enc is not None:
        return enc
    h, w = arr.shape[0], arr.shape[1]
    im = Image.frombytes("RGBA", (w, h), arr.tobytes())
    pal = im.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT)
    q = np.frombuffer(pal.convert("RGB").tobytes(), np.uint8).reshape(h, w, 3)
    out = np.dstack([q, arr[..., 3]])
    return encode_indexed(out.copy())

FMT_T4, FMT_T8 = 4, 5


def main():
    argv = sys.argv[1:]
    dry = "--dry" in argv
    force = "--force" in argv   # re-encode EVERY named texture from PC art,
                                # opaque included, even if deployed looks native
                                # (repairs corrupted content the gates would skip)
    argv = [a for a in argv if a not in ("--dry", "--force")]
    chunks_dir, census_path = argv
    census = json.load(open(census_path))

    defs = sa_source.load_defs()
    img = sa_source.open_img()
    name2def = {}
    for mid, d in defs.items():
        name2def.setdefault(d["dff"], d)
        name2def.setdefault(d["dff"].lower(), d)   # census names may be lowercased

    dff_cache = {}
    def split_texnames(dff_name):
        """texture-name sequence over the DFF's material splits (in order)."""
        if dff_name in dff_cache:
            return dff_cache[dff_name]
        seq = None
        blob = sa_source.img_read(img, dff_name + ".dff")
        if blob:
            try:
                dff = parse_dff(blob)
                seq = []
                for a in dff.atomics:
                    geo = dff.geometries[a.geometry_index]
                    for sp in geo.splits:
                        mi = sp["mat_index"]
                        mat = geo.materials[mi] if 0 <= mi < len(geo.materials) else None
                        nm = (getattr(mat, "texture_name", "") or "").lower()
                        seq.append(nm)
            except Exception:
                seq = None
        dff_cache[dff_name] = seq
        return seq

    txd_cache = {}
    def pc_tex(txd_name, tex_name):
        if txd_name not in txd_cache:
            blob = sa_source.img_read(img, txd_name + ".txd")
            try:
                txd_cache[txd_name] = ({k.lower(): v for k, v in
                                        sa_txd_d3d9.decode(blob).items()}
                                       if blob else {})
            except Exception:
                txd_cache[txd_name] = {}
        return txd_cache[txd_name].get(tex_name)

    tot = files = 0
    for fn, entries in sorted(census.items()):
        path = os.path.join(chunks_dir, fn)
        if not os.path.exists(path):
            continue
      # per-file guard
        try:
            prod, ver = load_v2(path)
            hp = HDR.unpack_from(prod, 0)
            mc, moff, sc, soff = hp[3], hp[4], hp[5], hp[6]
            tc, tex_off = hp[7], hp[8]
            models = [struct.unpack_from('<2I6f', prod, moff + 32*i) for i in range(mc)]
            subs = [struct.unpack_from('<i4I', prod, soff + 20*i) for i in range(sc)]
            tp = [list(TEX.unpack_from(prod, tex_off + 32*i)) for i in range(tc)]

            fixes = {}
            class_fixes = {}   # tex idx -> alpha class ONLY (pixels already native)
            for e in entries:
                mi, dff_name = e["model"], e["name"]
                if mi >= mc:
                    continue
                d = name2def.get(dff_name) or name2def.get(dff_name.lower())
                seq = split_texnames(d["dff"]) if d else None   # IMG lookup wants the IDE-cased name
                if d is None or seq is None:
                    continue
                m = models[mi]
                # distinct-in-order texture indices of the prod model
                t_seq = []
                for s in range(m[0], m[0] + m[1]):
                    ti = subs[s][0]
                    if ti >= 0 and ti not in t_seq:
                        t_seq.append(ti)
                # distinct-in-order source names (skip empty)
                n_seq = []
                for nm in seq:
                    if nm and nm not in n_seq:
                        n_seq.append(nm)
                if len(t_seq) != len(n_seq):
                    continue          # ambiguous mapping: leave alone
                for ti, nm in zip(t_seq, n_seq):
                    if ti in fixes:
                        continue
                    got = pc_tex(d["txd"], nm)
                    if got is None:
                        continue
                    w, h, prgba = got
                    arr = np.frombuffer(bytes(prgba), np.uint8).reshape(h, w, 4)
                    cls = _alpha_class(arr.tobytes())
                    if cls == 0 and not force:
                        continue      # opaque art: fingerprint path handles it
                    dep = tp[ti]
                    dep_rgba = decode_deployed(prod, hp, dep)
                    flat = (dep_rgba is not None and
                            int(dep_rgba[..., 3].max()) - int(dep_rgba[..., 3].min()) < 12)
                    if not force and not flat and dep[0] >= w and dep[1] >= h:
                        # deployed pixels are fine - but the CLASS byte may have
                        # been reset to opaque by a later fingerprint re-pass
                        # (alpha data present, renderer draws it opaque = invisible
                        # wires). Patch just the class.
                        cur = (dep[8] >> 8) & 0xFF
                        if cur != cls:
                            class_fixes[ti] = cls
                        continue
                    while max(w, h) > 256:
                        im = Image.frombytes("RGBA", (w, h), arr.tobytes())
                        w //= 2; h //= 2
                        arr = np.frombuffer(im.resize((w, h), Image.LANCZOS).tobytes(),
                                            np.uint8).reshape(h, w, 4)
                    if w < 8 or h < 8 or (w & (w-1)) or (h & (h-1)):
                        continue
                    enc = encode_indexed_q(arr)
                    if enc is None:
                        continue
                    fmt, texels, clut, bufw_tex, ce = enc
                    fixes[ti] = (w, h, fmt, texels, clut, bufw_tex, ce, cls, nm)
            if not fixes and not class_fixes:
                continue
            files += 1; tot += len(fixes) + len(class_fixes)
            print(f"  {fn}: {len(fixes)} repaired "
                  f"{[v[8] for v in list(fixes.values())[:5]]}"
                  f" + {len(class_fixes)} class-only", flush=True)
            if dry:
                continue
            # splice texel/clut pools (last two sections)
            texel_pool = bytearray(); clut_pool = bytearray(); new_tex = []
            for i in range(tc):
                (w, h, fmt, texel_first, texel_bytes, bufw, clut_first,
                 clut_entries, nlev) = tp[i]
                if i in fixes:
                    nw, nh, nfmt, texels, clut, bufw_tex, ce, cls, _nm = fixes[i]
                    nt = (nw, nh, nfmt, len(texel_pool), len(texels), bufw_tex,
                          len(clut_pool), ce, (nlev & 0xFFFF00FF) | (cls << 8))
                    texel_pool += texels; clut_pool += clut
                else:
                    if i in class_fixes:
                        nlev = (nlev & 0xFFFF00FF) | (class_fixes[i] << 8)
                    nt = (w, h, fmt, len(texel_pool), texel_bytes, bufw,
                          len(clut_pool), clut_entries, nlev)
                    texel_pool += prod[hp[16]+texel_first: hp[16]+texel_first+texel_bytes]
                    clut_pool += prod[hp[18]+clut_first: hp[18]+clut_first+clut_entries*4]
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
                TEX.pack_into(out, tex_off + 32*i, *nt)
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
            print(f"  {fn}: ERROR {ex}", flush=True)
    print(f"DONE: {tot} textures repaired in {files} files")


if __name__ == "__main__":
    main()
