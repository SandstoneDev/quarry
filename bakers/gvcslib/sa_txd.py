"""the source game PS2-native TXD (TexDictionary) decoder.

Decodes RenderWare TXD blobs extracted from the source game' PS2 GTA3.IMG archive
into per-texture RGBA8888 images.

Chunk layout
------------
TexDictionary (0x16)
  STRUCT (0x01)  - u16 numTextures, u16 deviceId (6 = PS2)
  [TextureNative (0x15)] * numTextures
    STRUCT (0x01)  - u32 platform (1), u32 filterAddr
    STRING (0x02)  - texture name (null-padded)
    STRING (0x02)  - alpha mask name (null-padded)
    STRUCT (0x01)  - PS2 raster + GIF data (the large one)
      STRUCT (0x01)  - raster header (w, h, depth, rasterFmt, GsTex0, ...)
      STRUCT (0x01)  - GIF packet stream (mip levels + CLUT)
    EXTENSION (0x03)

Pixel formats
-------------
PSM 0x13  PSMT8   8bpp, 256-entry CLUT
PSM 0x14  PSMT4   4bpp, 16-entry CLUT
PSM 0x00  PSMCT32 32bpp direct (uncommon in SA)

The GIF packet stream stores:
  [mip level 0 IMAGE] [mip level 1 IMAGE] ... [CLUT IMAGE]
The LAST IMAGE packet is always the CLUT.

PS2 swizzle
-----------
RenderWare does NOT store a GS VRAM dump - it stores the host->local
*transfer stream* that gets DMA'd to the GS.  Swizzled rasters are uploaded
in a wider pixel format with halved dimensions (the GIF TRXREG on disk says
w/2 x h/2):

  PSMT8 (8bpp)  is transferred as PSMCT32, w/2 x h/2 (byte-unit swizzle)
  PSMT4 (4bpp)  is transferred as PSMCT16, w/2 x h/2 (nibble-unit swizzle)

The net texel scramble is the GS block/column CT32<->T8 (CT16<->T4)
relationship with all page bookkeeping cancelling out; the closed form is
librw ps2raster.cpp swizzle() (equivalent to the classic "Sparky"
unswizzle8).  Rows repeat in strips of 4; within a strip the CT row is
2*(y//4)+(y&1), the byte/nibble lane is ((y>>1)&1) | x-bit-3, and rows with
(y>>1 ^ y>>2) odd have their 8-texel word halves swapped (x ^= 4).
Transfers have a minimum size (RW transferMinSize): swizzled PSMT8 is at
least 16x4, swizzled PSMT4 at least 32x4, so the stored row stride uses
max(w, 16) / max(w, 32) texels.  Empirically every 4/8-bit texture in the
SA PS2 GTA3.IMG (26k textures incl. SAR-mod repacks) is stored swizzled
(raster version field == 2), and mip0 size == align16(max(w,minw) *
max(h,4) * bpp/8) for all of them.

CLUT deswizzle (PSMT8 only)
---------------------------
For PSMT8, the 256-entry CLUT is stored in GS VRAM with a column-interleave:
within each 32-entry group the two 8-entry columns are swapped, i.e.
index i maps to storage position (i & ~0x18) | ((i & 8) << 1) | ((i & 16) >> 1).
For PSMT4 (16 entries) no deswizzle is needed.

Alpha scaling
-------------
PS2 stores alpha as 0-128 (0x80 = fully opaque). We scale to 0-255 by
multiplying by 2 and clamping to 255.
"""
import struct

# ---------------------------------------------------------------------------
# RW chunk types used here
# ---------------------------------------------------------------------------
_TEXDICTIONARY  = 0x16
_TEXTURENATIVE  = 0x15
_STRUCT         = 0x01
_STRING         = 0x02
_EXTENSION      = 0x03

# PS2 GS pixel format codes (PSM field in GsTex0 register)
_PSM_PSMCT32 = 0x00
_PSM_PSMT8   = 0x13  # 8bpp, 256-entry CLUT
_PSM_PSMT4   = 0x14  # 4bpp, 16-entry CLUT

# ---------------------------------------------------------------------------
# GS swizzled-transfer address mapping (librw ps2raster.cpp swizzle())
# ---------------------------------------------------------------------------
def _gs_transfer_addr(x: int, y: int, w: int) -> int:
    """Return the unit index in the swizzled transfer stream for texel (x, y).

    Units are bytes for PSMT8 (PSMCT32 transfer) and nibbles for PSMT4
    (PSMCT16 transfer).  w is the stored row stride in texel units --
    max(width, 16) for PSMT8 / max(width, 32) for PSMT4 (RW min transfer).
    Validated against librw's swizzle() and the classic Sparky unswizzle8
    (both give the identical mapping).
    """
    xx = x ^ ((((y >> 1) ^ (y >> 2)) & 1) << 2)   # half-word swap rows
    nx = (xx & 7) | ((x >> 1) & ~7)               # CT word/unit x
    ny = (y & 1) | ((y >> 1) & ~1)                # CT row (2 per 4-texel strip)
    lane = ((y >> 1) & 1) | (((x >> 3) & 1) << 1)  # texel lane within CT unit
    return lane | (nx << 2) | ny * 2 * w


def _unswizzle_psmt8(data: bytes, w: int, h: int) -> bytes:
    """De-swizzle GS PSMT8 raw bytes → linear palette-index byte array (w*h).

    The stream is a PSMCT32 transfer of max(w,16)/2 x max(h,4)/2.
    """
    ww = max(w, 16)
    out = bytearray(w * h)
    n = len(data)
    for y in range(h):
        row = y * w
        for x in range(w):
            a = _gs_transfer_addr(x, y, ww)
            if a < n:
                out[row + x] = data[a]
    return bytes(out)


def _unswizzle_psmt4(data: bytes, w: int, h: int) -> bytes:
    """De-swizzle GS PSMT4 raw bytes → linear palette-index byte array (w*h).

    The stream is a PSMCT16 transfer of max(w,32)/2 x max(h,4)/2; the same
    address mapping as PSMT8 applies but in nibble units (low nibble first).
    Each output byte contains one palette index in [0, 15].
    """
    ww = max(w, 32)
    out = bytearray(w * h)
    n = len(data)
    for y in range(h):
        row = y * w
        for x in range(w):
            a = _gs_transfer_addr(x, y, ww)   # nibble index
            byte_addr = a >> 1
            if byte_addr < n:
                b = data[byte_addr]
                out[row + x] = (b >> 4) & 0xF if (a & 1) else b & 0xF
    return bytes(out)


# ---------------------------------------------------------------------------
# PSMT8 CLUT deswizzle
# ---------------------------------------------------------------------------
def _deswizzle_clut8(raw: bytes) -> bytes:
    """Deswizzle a 256-entry PSMT8 CLUT (1024 bytes RGBA8888).

    Within each 32-entry group of the CLUT the two 8-entry sub-blocks are
    stored swapped (column-interleave in PSMCT32 VRAM).  This restores the
    linear 0-255 ordering.
    """
    out = bytearray(256 * 4)
    for i in range(256):
        j = (i & ~0x18) | ((i & 0x08) << 1) | ((i & 0x10) >> 1)
        out[j * 4: j * 4 + 4] = raw[i * 4: i * 4 + 4]
    return bytes(out)


# ---------------------------------------------------------------------------
# RW chunk tree parser (minimal, self-contained)
# ---------------------------------------------------------------------------
def _parse_chunks(data: bytes, o: int, end: int):
    """Return a flat list of (type, size, data_off, child_off) tuples."""
    result = []
    while o + 12 <= end:
        typ, sz, _lib = struct.unpack_from("<III", data, o)
        result.append((typ, sz, o + 12, o))
        o = o + 12 + sz
    return result


# ---------------------------------------------------------------------------
# GIF packet parser
# ---------------------------------------------------------------------------
def _parse_gif_images(d: bytes):
    """Return a list of raw bytes for every IMAGE-mode GIF packet in d.

    Each IMAGE GIF tag (flg=2) is followed by nloop*16 bytes of data.
    PACKED/REGLIST tags (BITBLTBUF / TRXREG / TRXDIR setup) are skipped.
    """
    images = []
    off = 0
    while off + 16 <= len(d):
        qw = struct.unpack_from("<Q", d, off)[0]
        nloop = qw & 0x3FFF
        flg = (qw >> 58) & 0x3
        nreg = (qw >> 60) & 0xF
        if flg == 2:  # IMAGE mode
            # NLOOP is 14 bits, so a transfer of exactly 16384 quadwords stores as
            # 0. That is precisely a 512x512 8bpp level (262144 bytes), which is how
            # the loading screens ship. Reading 0 literally consumed no pixels and
            # left the walk inside the pixel data, so the following "tag" was noise
            # and the CLUT was never found: every such texture decoded to a single
            # flat colour. A zero-length image transfer is meaningless, so 0 here
            # can only be the wrap.
            qwords = nloop if nloop else 0x4000
            data_bytes = qwords * 16
            images.append(d[off + 16: off + 16 + data_bytes])
            off += 16 + data_bytes
        else:
            # PACKED or REGLIST: nloop * max(nreg, 1) register-data entries
            entries = nloop * max(nreg, 1)
            off += 16 + entries * 16
    return images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def decode(blob, linear=False) -> "dict[str, tuple[int, int, bytes]]":
    """Decode a the source game PS2-native TXD blob.

    linear=True reads PSMT4/PSMT8 texel data as plain row-major bytes instead
    of GS-deswizzling. Modded repacks (e.g. Magic.TXD output, SAR Mod) store
    texels linear; vanilla discs store them GS-swizzled. Callers that don't
    know the provenance should decode both ways and pick by content.

    Parameters
    ----------
    blob : bytes-like
        Raw TXD chunk bytes (starts with TexDictionary chunk 0x16).

    Returns
    -------
    dict mapping lower-cased texture name → (width, height, rgba_bytes)
        rgba_bytes is len w*h*4, channel order R G B A (alpha last),
        alpha in 0-255 range (PS2's 0-128 scale is doubled and clamped).

    Raises
    ------
    ValueError
        If ``blob`` does not start with a TexDictionary chunk.
    """
    data = bytes(blob)

    # Validate top-level chunk
    if len(data) < 12:
        raise ValueError("blob too short for RW chunk header")
    top_type, top_size, _lib = struct.unpack_from("<III", data, 0)
    if top_type != _TEXDICTIONARY:
        raise ValueError(
            f"expected TexDictionary chunk 0x{_TEXDICTIONARY:x}, got 0x{top_type:x}"
        )

    txd_body = 12
    txd_end = txd_body + top_size

    children = _parse_chunks(data, txd_body, min(txd_end, len(data)))

    result = {}

    for chunk_type, chunk_size, chunk_data_off, chunk_off in children:
        if chunk_type != _TEXTURENATIVE:
            continue

        tn_end = chunk_data_off + chunk_size
        tn_children = _parse_chunks(data, chunk_data_off, min(tn_end, len(data)))

        # --- extract texture name from first STRING child ---
        tex_name = ""
        string_chunks = [c for c in tn_children if c[0] == _STRING]
        if string_chunks:
            s_off = string_chunks[0][2]
            s_sz = string_chunks[0][1]
            raw = data[s_off: s_off + s_sz]
            tex_name = raw.split(b"\x00", 1)[0].decode("latin1", errors="replace")

        # --- find the large STRUCT (raster data) ---
        struct_chunks = [c for c in tn_children if c[0] == _STRUCT]
        if not struct_chunks:
            continue
        # There are typically two STRUCTs: a tiny one (8 bytes) with platform/filter,
        # and a large one containing nested chunks.  Take the largest.
        big = max(struct_chunks, key=lambda c: c[1])
        big_body = big[2]
        big_size = big[1]

        # --- inner chunks: STRUCT(raster header) + STRUCT(GIF data) ---
        inner = _parse_chunks(data, big_body, big_body + big_size)
        inner_structs = [c for c in inner if c[0] == _STRUCT]
        if len(inner_structs) < 2:
            # Fallback: try to find at least the header
            if not inner_structs:
                continue
            # single struct = just the header, no data block
            continue

        hdr = inner_structs[0]
        data_blk = inner_structs[1]

        # --- parse raster header: 14 u32 fields ---
        if hdr[1] < 56:  # need at least 14 * 4 = 56 bytes
            continue
        try:
            fields = struct.unpack_from("<14I", data, hdr[2])
        except struct.error:
            continue

        w      = fields[0]
        h      = fields[1]
        depth  = fields[2]          # 4 / 8 / 32 - authoritative (mod repacks
                                    # write a correct depth but garbage tex0)
        # fields[3] = rasterFormat flags
        tex0_lo = fields[4]
        tex0_hi = fields[5]
        # fields[6..11] = tex1, miptbp1, miptbp2 (64-bit each)
        # fields[12] = texelDataSize
        # fields[13] = paletteSize

        # PSM is in GsTex0 bits [25:20]
        tex0 = (tex0_hi << 32) | tex0_lo
        psm  = (tex0 >> 20) & 0x3F
        # depth wins over tex0 when they disagree (SAR-mod / Magic.TXD output)
        if depth == 8 and psm != _PSM_PSMT8:
            psm = _PSM_PSMT8
        elif depth == 4 and psm != _PSM_PSMT4:
            psm = _PSM_PSMT4

        if w == 0 or h == 0:
            continue
        if (w & (w - 1)) != 0 or (h & (h - 1)) != 0:
            # Dimensions are not powers of two; skip (shouldn't happen in valid SA TXDs)
            continue

        # --- parse GIF packet stream ---
        gif_data = data[data_blk[2]: data_blk[2] + data_blk[1]]
        image_packets = _parse_gif_images(gif_data)

        # Indexed formats need mip-0 + CLUT (2 packets); 32-bit DIRECT colour
        # (PSMCT32, e.g. the ryd_holes decal) has NO CLUT -> only mip-0 (1 packet).
        # The old flat "< 2" test silently dropped every 32-bit texture map-wide.
        need_packets = 1 if psm == _PSM_PSMCT32 else 2
        if len(image_packets) < need_packets:
            continue

        mip0_raw = image_packets[0]   # first IMAGE = mip level 0 pixels
        clut_raw = image_packets[-1]  # last  IMAGE = CLUT (unused for PSMCT32)

        # --- decode by pixel format ---
        if psm == _PSM_PSMT8:
            # 8bpp, 256-entry CLUT
            if len(clut_raw) < 256 * 4:
                continue
            if linear:
                need = w * h
                indices = (mip0_raw + bytes(need))[:need]
                clut = clut_raw[:256 * 4]      # mod repacks store the CLUT linear too
            else:
                indices = _unswizzle_psmt8(mip0_raw, w, h)
                clut    = _deswizzle_clut8(clut_raw[:256 * 4])
            n_clut  = 256

        elif psm == _PSM_PSMT4:
            # 4bpp, 16-entry CLUT (stored as first 16 entries of the palette packet)
            if len(clut_raw) < 16 * 4:
                continue
            if linear:
                need = w * h
                packed = (mip0_raw + bytes((need + 1) // 2))[:(need + 1) // 2]
                indices = bytearray(need)
                for pix in range(need):
                    b = packed[pix >> 1]
                    indices[pix] = (b >> 4) & 0xF if (pix & 1) else b & 0xF
                indices = bytes(indices)
            else:
                indices = _unswizzle_psmt4(mip0_raw, w, h)
            clut    = clut_raw[:16 * 4]  # no column-swizzle needed for 16 entries
            n_clut  = 16

        elif psm == _PSM_PSMCT32:
            # 32bpp direct: mip0_raw is already RGBA8888 linear (no swizzle in SA data).
            # Scale alpha: PS2 0-128 → 0-255.
            expected = w * h * 4
            raw32 = (mip0_raw + bytes(expected))[:expected]
            rgba_out = bytearray(expected)
            for i in range(0, expected, 4):
                r, g, b, a = raw32[i: i + 4]
                rgba_out[i: i + 4] = bytes([r, g, b, min(255, a * 2)])
            name_key = tex_name.lower() if tex_name else f"tex_{len(result)}"
            result[name_key] = (w, h, bytes(rgba_out))
            continue

        else:
            # Unknown / unsupported PSM - skip this texture
            continue

        # --- map indices → RGBA8888 ---
        n_pixels = w * h
        rgba = bytearray(n_pixels * 4)
        mask = n_clut - 1
        for p in range(n_pixels):
            idx = indices[p] & mask
            r, g, b, a = clut[idx * 4: idx * 4 + 4]
            rgba[p * 4: p * 4 + 4] = bytes([r, g, b, min(255, a * 2)])

        name_key = tex_name.lower() if tex_name else f"tex_{len(result)}"
        result[name_key] = (w, h, bytes(rgba))

    return result
