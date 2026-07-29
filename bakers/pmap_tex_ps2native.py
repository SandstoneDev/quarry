#!/usr/bin/env python3
"""pmap_tex_ps2native - replace deployed world textures with PS2-NATIVE ones.

 Deployed pmaps
carry nameless 64px T8 textures downscaled from PC TXDs; the PS2 build ships
the same art at its native tier (93.6% are <=128px 4-bit CLUT = PSP T4 1:1,
authored 16-colour palettes, no quantization needed).

Per region:
 1. decode every deployed texture (T4/T8 level0) -> perceptual hash
 2. match against the PC fingerprint DB (tex_fingerprint_db.py) -> (txd, name)
 3. pull the SAME (txd, name) from the PS2 GTA3.IMG / GTA_INT.IMG via
 gvcslib.sa_txd (validated librw transfer-stream deswizzle; every PSMT4/8
 texture in the archive is GS-swizzled)
 4. content-gate (corr + total-variation vs the deployed motif - guards
 against wrong-NAME matches; the decode itself is trusted)
 5. rebuild the exact indexed form from unique colours (<=16 -> T4,
 <=256 -> T8), swizzle for the PSP GE, splice at NATIVE size (cap 256)
 6. unmatched / rejected / oversized textures keep the deployed version

UVs are untouched (resolution-independent). alpha_mode byte (num_levels byte 1)
is preserved from the deployed entry. Splice machinery = pmap_tex_t4from128.

Usage (single):
 pmap_tex_ps2native.py <region.pmap> <out.pmap> --db <fp_prefix>
 --ps2img <GTA3.IMG> [--ps2int <GTA_INT.IMG>] [--cap 256] [--hamming 12]
 [--report]
Usage (batch, in-place, shared decode caches across files):
 pmap_tex_ps2native.py <file-or-dir> [more ...] --inplace --backup <DIR>
 --db <fp_prefix> --ps2img <GTA3.IMG> [--ps2int <GTA_INT.IMG>]
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
GVCS = os.environ.get("GVCS_ROOT", "")
SAW = os.environ.get("SAW_ROOT", "")
for p in (GVCS, SAW):
    if p not in sys.path:
        sys.path.insert(0, p)
from gvcslib import sa_txd
from core.imgarchive import ImgArchive

import tex_fingerprint_db as fdb
from pmap_tex_t4from128 import HDR, TEX, load_v2, unswizzle, swizzle

FMT_T4, FMT_T8 = 4, 5


def decode_deployed(blob, h20, t):
    """Deployed texture level0 -> RGBA numpy (h,w,4). T4/T8 only."""
    (w, hh, fmt, texel_first, texel_bytes, bufw, clut_first, clut_entries,
     nlev) = t
    if fmt not in (FMT_T4, FMT_T8):
        return None
    texel_off, clut_off = h20[16], h20[18]
    wb = bufw if fmt == FMT_T8 else bufw // 2
    wb = max(wb, 16)
    lvl0 = blob[texel_off + texel_first: texel_off + texel_first + wb * hh]
    if len(lvl0) < wb * hh:
        return None
    lin = unswizzle(lvl0, wb, hh)
    clut = blob[clut_off + clut_first: clut_off + clut_first + clut_entries * 4]
    pal = np.frombuffer(clut, np.uint8).reshape(-1, 4)
    if fmt == FMT_T8:
        idx = np.frombuffer(lin, np.uint8).reshape(hh, wb)[:, :w]
    else:
        b = np.frombuffer(lin, np.uint8).reshape(hh, wb)
        lo = b & 0xF; hi = b >> 4
        idx = np.empty((hh, wb * 2), np.uint8)
        idx[:, 0::2] = lo; idx[:, 1::2] = hi
        idx = idx[:, :w]
    if idx.max(initial=0) >= len(pal):
        return None
    return pal[idx]


def encode_indexed(rgba, force_t8=False):
    """RGBA (h,w,4) with <=256 unique colours -> (fmt, swizzled, clut, bufw, ce).

 force_t8 keeps a <=16-colour image out of T4 for the runtime loaders that bind
 GU_PSM_T8 unconditionally and would read 4bpp texels as 8bpp (grass.bin).
 """
    h, w = rgba.shape[:2]
    flat = rgba.reshape(-1, 4)
    colors, inv = np.unique(flat.view(np.uint32).reshape(-1), return_inverse=True)
    n = len(colors)
    pal = colors.view(np.uint8).reshape(-1, 4)
    idx = inv.astype(np.uint8).reshape(h, w)
    if n <= 16 and not force_t8:
        fmt = FMT_T4
        clut = pal.tobytes() + bytes((16 - n) * 4)
        bufw_tex = max(w, 32)                      # 4bpp bufw mult-of-32 texels
        wb = bufw_tex // 2
        row = np.zeros((h, wb), np.uint8)
        packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
        row[:, :w // 2] = packed
        lin = row.tobytes()
        ce = 16
    elif n <= 256:
        fmt = FMT_T8
        clut = pal.tobytes() + bytes((256 - n) * 4)
        bufw_tex = max(w, 16)                      # 8bpp bufw mult-of-16 texels
        wb = bufw_tex
        row = np.zeros((h, wb), np.uint8)
        row[:, :w] = idx
        lin = row.tobytes()
        ce = 256
    else:
        return None                                # truecolor: caller keeps deployed
    return fmt, swizzle(lin, wb, h), clut, bufw_tex, ce


def _composite(rgba_bytes, cw, ch):
    """RGBA -> RGB composited over mid-gray (alpha-aware, like the hash)."""
    a = np.frombuffer(bytes(rgba_bytes), np.uint8).reshape(-1, 4).astype(np.uint16)
    comp = ((a[:, :3] * a[:, 3:4] + 128 * (255 - a[:, 3:4])) // 255).astype(np.uint8)
    return Image.frombytes("RGB", (cw, ch), comp.tobytes())


def _gray32(rgba_bytes, cw, ch):
    im = _composite(rgba_bytes, cw, ch)
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS),
                   np.float32).flatten()
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-6 else g


def _tv(rgba_bytes, cw, ch):
    """Mean per-pixel gradient at NATIVE res: a scrambled decode has several
 times the deployed texture's energy; legit finer art ~1-2x."""
    im = _composite(rgba_bytes, cw, ch)
    g = np.asarray(im.convert("L"), np.float32)
    return float((np.abs(np.diff(g, axis=0)).mean()
                  + np.abs(np.diff(g, axis=1)).mean()) * 0.5)


def _alpha_class(rgba_bytes):
    """0 opaque / 1 cutout (bimodal) / 2 translucent (gradient) - psp_tex rule."""
    a = np.frombuffer(bytes(rgba_bytes), np.uint8)[3::4]
    n = a.size
    transp = int((a < 96).sum())
    mid = int(((a >= 96) & (a < 200)).sum())
    if transp > n * 0.02:
        return 1 if mid <= n * 0.20 else 2
    return 2 if mid > n * 0.20 else 0


GATE = 0.35   # decode is trusted (validated deswizzle); the gate only guards
              # against wrong-NAME matches. Different masters run 0.4-0.7
              # (dt_road 0.44 verified correct).
TV_K = 3.0    # loose scramble safety net


class Ctx:
    """Shared expensive state: fingerprint DB + PS2 archive decode cache."""

    def __init__(self, db_prefix, ps2img_path, ps2int_path):
        z = np.load(db_prefix + ".npz", allow_pickle=True)
        self.db_hashes = z["hashes"]               # (N,16) uint8
        self.db_meta = z["meta"]                   # (N,4) object
        self.exact = {}
        for i in range(len(self.db_meta)):
            self.exact.setdefault(self.db_hashes[i].tobytes(), i)
        ps2_imgs = [ImgArchive.open(ps2img_path)]
        if ps2int_path:
            ps2_imgs.append(ImgArchive.open(ps2int_path))
        self.ps2_index = {}
        for im in ps2_imgs:
            for e in im.entries:
                if e.name.lower().endswith(".txd"):
                    self.ps2_index.setdefault(e.name[:-4].lower(), (im, e))
        self.txd_cache = {}

    def ps2_lookup(self, txd_name, tex_name):
        if txd_name not in self.txd_cache:
            ent = self.ps2_index.get(txd_name)
            if ent is None:
                self.txd_cache[txd_name] = {}
            else:
                im, e = ent
                try:
                    self.txd_cache[txd_name] = sa_txd.decode(im.extract(e))
                except Exception:
                    self.txd_cache[txd_name] = {}
        d = self.txd_cache[txd_name]
        return d.get(tex_name)


def process_one(ctx, in_path, out_path, cap, ham_max, report):
    prod, prod_ver = load_v2(in_path)
    hp = HDR.unpack_from(prod, 0)
    tc = hp[7]; tex_off = hp[8]
    tp = [TEX.unpack_from(prod, tex_off + 32 * i) for i in range(tc)]

    stats = dict(matched=0, exact=0, ham=0, nomatch=0, ps2miss=0,
                 upgraded=0, kept=0, t4=0, t8=0, reject=0)
    upgrades = {}
    names = {}
    for i in range(tc):
        rgba = decode_deployed(prod, hp, tp[i])
        if rgba is None:
            stats['kept'] += 1
            continue
        key, _bits = fdb.hashes(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
        kb = bytes.fromhex(key)
        di = ctx.exact.get(kb)
        kind = 'exact'
        if di is None:
            q = np.frombuffer(kb, np.uint8)
            d = np.unpackbits(ctx.db_hashes ^ q, axis=1).sum(1)
            j = int(d.argmin())
            if d[j] <= ham_max:
                di = j; stats['ham'] += 1; kind = 'ham%d' % int(d[j])
            else:
                stats['nomatch'] += 1
                continue
        else:
            stats['exact'] += 1
        stats['matched'] += 1
        txd_name, tex_name = str(ctx.db_meta[di][0]), str(ctx.db_meta[di][1])
        got = ctx.ps2_lookup(txd_name, tex_name)
        if got is None:
            stats['ps2miss'] += 1
            continue
        cw, ch, crgba = got
        # CONTENT GATE vs the deployed motif (wrong-name safety net)
        dep_g = _gray32(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
        dep_tv = _tv(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
        corr = float(np.dot(dep_g, _gray32(crgba, cw, ch)))
        if corr < GATE or _tv(crgba, cw, ch) > TV_K * dep_tv + 4.0:
            stats['reject'] += 1
            continue
        # ALPHA-CLASS GUARD: never swap a transparent texture for an opaque
        # one or vice versa (an opaque "match" on a wire/foliage texture is a
        if _alpha_class(rgba.tobytes()) != _alpha_class(crgba):
            stats['reject'] += 1
            continue
        w, h, rgba_ps2 = cw, ch, crgba
        while max(w, h) > cap:                     # halve oversized (rare 256/512)
            im = Image.frombytes("RGBA", (w, h), bytes(rgba_ps2))
            w //= 2; h //= 2
            rgba_ps2 = im.resize((w, h), Image.LANCZOS).tobytes()
        if w < 8 or h < 8 or (w & (w - 1)) or (h & (h - 1)):
            stats['kept'] += 1
            continue
        arr = np.frombuffer(bytes(rgba_ps2), np.uint8).reshape(h, w, 4)
        enc = encode_indexed(arr)
        if enc is None:
            stats['kept'] += 1
            continue
        fmt, texels, clut, bufw_tex, ce = enc
        if w * h < tp[i][0] * tp[i][1]:            # never downgrade
            stats['kept'] += 1
            continue
        upgrades[i] = (w, h, fmt, texels, clut, bufw_tex, ce)
        names[i] = (txd_name, tex_name, tp[i][0], tp[i][1], w, h,
                    'T4' if fmt == FMT_T4 else 'T8',
                    "%s c%.2f" % (kind, corr))
        stats['upgraded'] += 1
        stats['t4' if fmt == FMT_T4 else 't8'] += 1
        if report and stats['upgraded'] <= 12:
            print(f"    tex{i}: {tp[i][0]}x{tp[i][1]} -> {w}x{h} "
                  f"{'T4' if fmt == FMT_T4 else 'T8'}  ({txd_name}/{tex_name})")

    # splice (texel + clut pools are the file's last two sections)
    texel_pool = bytearray(); clut_pool = bytearray(); new_tex = []
    for i in range(tc):
        (w, h, fmt, texel_first, texel_bytes, bufw, clut_first, clut_entries,
         nlev) = tp[i]
        if i in upgrades:
            nw, nh, nfmt, texels, clut, bufw_tex, ce = upgrades[i]
            nt = (nw, nh, nfmt, len(texel_pool), len(texels), bufw_tex,
                  len(clut_pool), ce, (nlev & 0xFFFFFF00) | 1)
            texel_pool += texels; clut_pool += clut
        else:
            nt = (w, h, fmt, len(texel_pool), texel_bytes, bufw,
                  len(clut_pool), clut_entries, nlev)
            texel_pool += prod[hp[16] + texel_first: hp[16] + texel_first + texel_bytes]
            clut_pool += prod[hp[18] + clut_first: hp[18] + clut_first + clut_entries * 4]
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

    if prod_ver == 3:
        tmp = tempfile.mktemp(suffix='.pmap')
        open(tmp, 'wb').write(bytes(out))
        subprocess.check_call([sys.executable, os.path.join(TOOLS, 'pmap_lz4.py'),
                               tmp, out_path], stdout=subprocess.DEVNULL)
        os.remove(tmp)
    else:
        open(out_path, 'wb').write(bytes(out))
    json.dump({str(k): v for k, v in names.items()},
              open(out_path + ".names.json", "w"), indent=0)
    print(f"  {os.path.basename(out_path)}: {stats['upgraded']}/{tc} upgraded "
          f"(T4 {stats['t4']} / T8 {stats['t8']}), exact={stats['exact']} "
          f"ham={stats['ham']} nomatch={stats['nomatch']} "
          f"ps2miss={stats['ps2miss']} reject={stats['reject']}, "
          f"texel_pool {hp[17]}->{len(texel_pool)}B", flush=True)
    return stats


def main():
    argv = sys.argv[1:]
    def opt(name, default=None):
        if name in argv:
            k = argv.index(name); v = argv[k + 1]
            del argv[k:k + 2]
            return v
        return default
    db_prefix = opt("--db")
    ps2img_path = opt("--ps2img")
    ps2int_path = opt("--ps2int")
    backup_dir = opt("--backup")
    cap = int(opt("--cap", "256"))
    ham_max = int(opt("--hamming", "12"))
    report = "--report" in argv
    inplace = "--inplace" in argv
    argv = [a for a in argv if a not in ("--report", "--inplace")]

    if inplace:
        files = []
        for a in argv:
            if os.path.isdir(a):
                files += sorted(os.path.join(a, f) for f in os.listdir(a)
                                if f.lower().endswith(".pmap"))
            else:
                files.append(a)
        jobs = [(f, f) for f in files]
    else:
        in_path, out_path = argv
        jobs = [(in_path, out_path)]

    ctx = Ctx(db_prefix, ps2img_path, ps2int_path)
    total_files = total_up = errors = 0
    for in_path, out_path in jobs:
        try:
            if inplace and backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                bp = os.path.join(backup_dir, os.path.basename(in_path))
                if not os.path.exists(bp):
                    shutil.copyfile(in_path, bp)
            st = process_one(ctx, in_path, out_path, cap, ham_max, report)
            total_files += 1
            total_up += st['upgraded']
        except Exception as ex:
            errors += 1
            print(f"  {os.path.basename(in_path)}: ERROR {ex}", flush=True)
    print(f"DONE: {total_files}/{len(jobs)} files, {total_up} textures "
          f"upgraded, {errors} errors")


if __name__ == "__main__":
    main()
