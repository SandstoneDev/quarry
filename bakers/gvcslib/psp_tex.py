"""Author a PSP-GE-ready swizzled paletted texture from a decoded RGBA image.

This is the *encode* side of the texture pipeline for the custom pspsdk
homebrew engine: it takes a decoded RGBA8888 image (e.g. the ``(w, h, rgba)``
tuple a :mod:`gvcslib.sa_txd` texture decodes to) and produces the exact byte
planes the engine uploads to the GE - a swizzled paletted texel plane plus a
CLUT - together with the ``sceGu`` pixel-format constants needed to draw it.

It REUSES the PSP 16x8 GS swizzle from :mod:`gvcslib.rwtex` (``rwtex.swizzle`` /
``rwtex.byte_width``) - the very same swizzle the GE wants for a ``GU_TRUE``
swizzled upload - so quantised T8/T4 index planes are laid out identically to
the game's own RW-PSP rasters.

Authored blob layout
---------------------
``author_psp_texture`` returns a dict (NOT a packed file - the engine owns the
container) with:

    gu_pixfmt   int   GU_PSM_T8 (5) or GU_PSM_T4 (4): the texture pixel format
                      passed to ``sceGuTexMode``.
    gu_clutfmt  int   GU_PSM_8888 (3): CLUT entry format for ``sceGuClutMode``.
    width       int   pixel width  (== input w).
    height      int   pixel height (== input h).
    swizzle     int   1 - texel_bytes is swizzled (upload with GU_TRUE).
    clut_entries int  256 (T8) or 16 (T4).
    texel_bytes bytes PSP-swizzled index plane.  Row stride in bytes is
                      ``rwtex.byte_width(fmt, w)`` (== w for T8; max(w//2,16) for
                      T4).  Length = byte_width * h.
    clut_bytes  bytes CLUT: clut_entries * RGBA8888, alpha LAST byte, stored
                      LINEARLY (index i -> clut_bytes[i*4:i*4+4]), matching the
                      RW-PSP convention used by gvcslib.rwtex.decode_rgba.

Quantisation
------------
Colours are reduced with PIL's MEDIANCUT quantiser to 256 (T8) or 16 (T4)
entries.  Alpha is quantised too: the RGBA image is fed to PIL in RGBA mode so
the palette carries per-entry alpha (alpha LAST byte in the emitted CLUT).

Decode-back
-----------
:func:`decode_psp_texture` is the engine-side reference: it de-swizzles the
texel plane (``rwtex.unswizzle``), reads indices at the stored row stride, and
applies the linear CLUT - returning RGBA8888, the exact inverse path used to
prove the round-trip in the test.
"""

from PIL import Image

from . import rwtex

# sceGu pixel-format constants (psp pspgu.h values).
GU_PSM_5650 = 0
GU_PSM_5551 = 1
GU_PSM_4444 = 2
GU_PSM_8888 = 3
GU_PSM_T4 = 4
GU_PSM_T8 = 5

# rwtex TPSM ids for the swizzle byte-width helper (T4=4, T8=5).
_RWTEX_FMT = {"T4": rwtex.FMT_T4, "T8": rwtex.FMT_T8}

_FMT_COLORS = {"T8": 256, "T4": 16}
_FMT_GU = {"T8": GU_PSM_T8, "T4": GU_PSM_T4}

CLUT_ENTRY_BYTES = 4  # RGBA8888, alpha last


def _quantize(rgba, w, h, ncolors):
    """Quantise an RGBA8888 byte image to <=ncolors.

    Returns (indices: bytes length w*h, clut: bytes ncolors*4 RGBA8888 alpha
    last, pal_img: PIL P-image, alpha_varies: bool).  The CLUT is padded out to
    exactly ``ncolors`` entries (trailing entries zero).

    If the alpha channel carries SHAPE (transparent / semi-transparent regions:
    cutout foliage, ground cracks, wall graffiti), quantise in RGBA space with
    FASTOCTREE - the only PIL method that clusters alpha - so texels that
    differ ONLY in alpha get DISTINCT palette entries.  The old RGB-only
    MEDIANCUT + per-entry alpha AVERAGING crushed such textures: a black-on-alpha
    crack (one RGB colour, whole shape in the alpha channel) collapsed to a
    single palette entry whose alpha was the flat mean -> the crack read as a
    uniform near-transparent quad (invisible) and soft foliage edges hardened.

    Opaque / uniform-alpha textures keep MEDIANCUT on RGB (better colour
    fidelity) with the (uniform) alpha recovered per entry - unchanged.
    """
    a_vals = rgba[3::4]
    alpha_varies = bool(a_vals) and (max(a_vals) - min(a_vals)) > 16

    img = Image.frombytes("RGBA", (w, h), bytes(rgba))

    if alpha_varies:
        # RGBA-aware quantise: distinct alpha levels of the same colour become
        # distinct CLUT entries (a pure-alpha decal -> a 16-step alpha ramp).
        pal_img = img.quantize(colors=ncolors, method=Image.FASTOCTREE)
        indices = bytearray(pal_img.tobytes())
        assert len(indices) == w * h
        raw = pal_img.getpalette("RGBA") or []  # flat RGBA list, alpha last
        used = len(raw) // 4
        clut = bytearray(ncolors * CLUT_ENTRY_BYTES)
        for i in range(min(used, ncolors)):
            clut[i * 4:i * 4 + 4] = bytes(raw[i * 4:i * 4 + 4])
        return bytes(indices), bytes(clut), pal_img, True

    # PIL's MEDIANCUT only accepts RGB input, so quantise the RGB channels for
    # the colour palette, then recover per-entry alpha by averaging the source
    # alpha of every pixel that maps to each index (uniform here, so exact).
    rgb = img.convert("RGB")
    pal_img = rgb.quantize(colors=ncolors, method=Image.MEDIANCUT)

    indices = bytearray(pal_img.tobytes())  # one byte per pixel, row-major
    assert len(indices) == w * h

    raw_pal = pal_img.getpalette()  # flat RGB list
    used = len(raw_pal) // 3

    # accumulate source alpha per palette index.
    a_sum = [0] * ncolors
    a_cnt = [0] * ncolors
    for p in range(w * h):
        ci = indices[p]
        if ci < ncolors:
            a_sum[ci] += rgba[p * 4 + 3]
            a_cnt[ci] += 1

    clut = bytearray(ncolors * CLUT_ENTRY_BYTES)
    for i in range(min(used, ncolors)):
        r = raw_pal[i * 3 + 0]
        g = raw_pal[i * 3 + 1]
        b = raw_pal[i * 3 + 2]
        a = (a_sum[i] // a_cnt[i]) if a_cnt[i] else 0xFF
        clut[i * 4:i * 4 + 4] = bytes((r, g, b, a))
    # pal_img (P-mode) is returned so mip levels can be remapped to the SAME
    # palette (Image.quantize(palette=pal_img)) -> consistent CLUT across levels.
    return bytes(indices), bytes(clut), pal_img, False


def _remap_rgba_to_clut(rgba, n, clut, count):
    """Nearest-entry map RGBA texels to an RGBA CLUT (colour+alpha distance).

    Used to build mip levels for alpha-carrying textures WITHOUT dropping alpha
    (PIL's palette-remap resizes in RGB and would flatten cutout edges).  Alpha
    is weighted equally with the colour channels so a resized cutout edge snaps
    to the right transparency step instead of a wrong opaque/clear entry.  Only
    the first ``count`` (actually-used) entries are candidates so trailing
    zero-padding cannot steal texels.
    """
    ent = [(clut[i * 4], clut[i * 4 + 1], clut[i * 4 + 2], clut[i * 4 + 3])
           for i in range(max(count, 1))]
    out = bytearray(n)
    for p in range(n):
        o = p * 4
        pr, pg, pb, pa = rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]
        best_i, best_d = 0, 1 << 30
        for i, (er, eg, eb, ea) in enumerate(ent):
            dr = pr - er; dg = pg - eg; db = pb - eb; da = pa - ea
            d = dr * dr + dg * dg + db * db + da * da
            if d < best_d:
                best_d, best_i = d, i
        out[p] = best_i
    return bytes(out)


def _swizzle_level(indices, lw, lh, fmt):
    """Pack one-byte-per-pixel indices for a single mip level and swizzle it."""
    rfmt = _RWTEX_FMT[fmt]
    plane = _pack_t8(indices, lw, lh) if fmt == "T8" else _pack_t4(indices, lw, lh)
    wb = rwtex.byte_width(rfmt, lw)
    if wb % rwtex.BLOCK_W:
        raise ValueError("byte stride %d not mult of %d (w=%d %s)"
                         % (wb, rwtex.BLOCK_W, lw, fmt))
    if lh % rwtex.BLOCK_H:
        raise ValueError("height %d not mult of %d" % (lh, rwtex.BLOCK_H))
    return rwtex.swizzle(plane, wb, lh)


def _pack_t4(indices, w, h):
    """Pack one-byte-per-pixel indices into a T4 byte plane at the PSP stride.

    Row stride is ``byte_width(T4, w) = max(w//2, 16)`` bytes; each row holds
    w//2 real index bytes (low nibble = even pixel, high nibble = odd) followed
    by zero padding up to the stride.  Mirrors rwtex's strided T4 layout so the
    plane can be swizzled with the shared helper and de-swizzled identically.
    """
    wb = rwtex.byte_width(rwtex.FMT_T4, w)
    plane = bytearray(wb * h)
    for y in range(h):
        row = y * wb
        for x in range(w):
            v = indices[y * w + x] & 0x0F
            bi = row + (x >> 1)
            if (x & 1) == 0:
                plane[bi] = (plane[bi] & 0xF0) | v
            else:
                plane[bi] = (plane[bi] & 0x0F) | (v << 4)
    return bytes(plane)


def _pack_t8(indices, w, h):
    """Lay out one-byte-per-pixel indices at the PSP T8 stride (== w bytes)."""
    wb = rwtex.byte_width(rwtex.FMT_T8, w)  # == w
    if wb == w:
        return bytes(indices)
    plane = bytearray(wb * h)
    for y in range(h):
        plane[y * wb:y * wb + w] = indices[y * w:y * w + w]
    return bytes(plane)


def author_psp_texture(rgba, w, h, *, fmt="T8", mipmaps=False, max_levels=5):
    """Convert a decoded RGBA8888 image into a PSP-GE swizzled paletted texture,
    optionally with a mip chain (kills moiré/aliasing on tiled textures).

    Args:
        rgba: RGBA8888 bytes, row-major, alpha last, length w*h*4.
        w, h: pixel dimensions (power-of-two; >= 16x8).
        fmt:  "T8" (256-colour) or "T4" (16-colour).
        mipmaps: generate a mip chain (each level remapped to the base palette).
        max_levels: cap on total levels (incl. level 0).

    Returns: dict with gu_pixfmt, gu_clutfmt, width, height, swizzle,
    clut_entries, num_levels, texel_bytes (ALL levels concatenated, level 0
    first), clut_bytes.  Per-level offsets/dims are derivable: level k is
    (w>>k, h>>k), byte_width(fmt, w>>k) * (h>>k) bytes.
    """
    if fmt not in _FMT_COLORS:
        raise ValueError("fmt must be 'T8' or 'T4', got %r" % (fmt,))
    if len(rgba) != w * h * 4:
        raise ValueError("rgba length %d != w*h*4 (%d)" % (len(rgba), w * h * 4))

    ncolors = _FMT_COLORS[fmt]
    indices, clut, pal_img, alpha_varies = _quantize(rgba, w, h, ncolors)
    used = (max(indices) + 1) if indices else 1

    levels = [_swizzle_level(indices, w, h, fmt)]
    if mipmaps:
        base_rgba = Image.frombytes("RGBA", (w, h), bytes(rgba))
        rgb_img = base_rgba.convert("RGB")
        k = 1
        while len(levels) < max_levels:
            lw, lh = w >> k, h >> k
            if lw < 16 or lh < 8:
                break
            if alpha_varies:
                # keep alpha: resize RGBA then nearest-map to the RGBA CLUT so
                # cutout/decal transparency survives into the mip chain (PIL's
                # palette-remap resizes in RGB and would flatten the alpha).
                small = base_rgba.resize((lw, lh), Image.BILINEAR)
                pidx = _remap_rgba_to_clut(small.tobytes(), lw * lh, clut, used)
            else:
                small = rgb_img.resize((lw, lh), Image.BILINEAR)
                # remap to the SAME base palette so all levels share one CLUT.
                pidx = small.quantize(palette=pal_img,
                                      dither=Image.Dither.NONE).tobytes()
            levels.append(_swizzle_level(pidx, lw, lh, fmt))
            k += 1

    texel_bytes = b"".join(levels)

    # alpha mode (drives the engine's per-texture blend/alpha-test):
    #   0 = opaque (no alpha at all)
    #   1 = cutout (hard 0/1 alpha: foliage/fence -> alpha-test, NO blend)
    #   2 = translucent (partial alpha: glass/water -> blend)
    ah = Image.frombytes("RGBA", (w, h), bytes(rgba)).getchannel("A").histogram()
    n = w * h
    transparent = sum(ah[0:32])      # fully-transparent (cutout background)
    midalpha = sum(ah[96:200])       # genuinely semi-transparent (glass)
    # CUTOUT first: any real transparent region = foliage/fence/tree -> alpha-test
    # (AA edges add some mid-alpha but it's still a cutout, not glass).
    if transparent > n * 0.015:
        alpha_mode = 1
    elif midalpha > n * 0.20:
        alpha_mode = 2               # mostly semi-transparent = glass/water
    else:
        alpha_mode = 0

    return {
        "gu_pixfmt": _FMT_GU[fmt],
        "gu_clutfmt": GU_PSM_8888,
        "width": w,
        "height": h,
        "swizzle": 1,
        "clut_entries": ncolors,
        "num_levels": len(levels),
        "alpha_mode": alpha_mode,
        "texel_bytes": texel_bytes,
        "clut_bytes": clut,
    }


def decode_psp_texture(tex):
    """Engine-side reference decode of an :func:`author_psp_texture` dict.

    De-swizzles the texel plane, reads indices at the stored row stride, and
    applies the linear CLUT.  Returns RGBA8888 bytes (row-major, alpha last),
    length width*height*4 - the exact inverse path used by the round-trip test.
    """
    w = tex["width"]
    h = tex["height"]
    fmt = "T4" if tex["gu_pixfmt"] == GU_PSM_T4 else "T8"
    rfmt = _RWTEX_FMT[fmt]
    wb = rwtex.byte_width(rfmt, w)

    lin = rwtex.unswizzle(tex["texel_bytes"], wb, h)

    idx = bytearray(w * h)
    if fmt == "T8":
        for y in range(h):
            base = y * wb
            for x in range(w):
                idx[y * w + x] = lin[base + x]
    else:  # T4
        for y in range(h):
            base = y * wb
            for x in range(w):
                byte = lin[base + (x >> 1)]
                idx[y * w + x] = byte & 0x0F if (x & 1) == 0 else (byte >> 4) & 0x0F

    clut = tex["clut_bytes"]
    out = bytearray(w * h * 4)
    for p in range(w * h):
        s = idx[p] * CLUT_ENTRY_BYTES
        out[p * 4:p * 4 + 4] = clut[s:s + 4]
    return bytes(out)
