"""the source game TXD (RenderWare D3D9 texture dictionary) decoder.

Decodes every TextureNative raster - DXT1/3/5, raw 565/1555/4444/555/8888/888/LUM8,
PAL8/PAL4 - to RGBA8888 for PNG preview. PC D3D9 native layout (platformId 9).

 (confirmed, with a radardisc worked example)
Reference: librw-master/src/d3d/d3d9.cpp, image.cpp (DXT ramps)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

from core import rwstream as rw
from formats import txd_ps2

# d3dFormat FourCC sentinels
_DXT1 = 0x31545844
_DXT3 = 0x33545844
_DXT5 = 0x35545844

# rasterFormat flag bits
_PAL8 = 0x2000
_PAL4 = 0x4000
_MIPS = 0x8000

_FMT_NIBBLE = {
    0x1: "ARGB1555", 0x2: "RGB565", 0x3: "ARGB4444",
    0x4: "LUM8", 0x5: "RGBA8888", 0x6: "RGB888", 0xA: "RGB555",  # rwRASTERFORMAT555=0x0A00, 5/5/5 no alpha (the platform SDK headers)
}


def _cstr(buf, off, n) -> str:
    return buf[off:off + n].split(b"\x00", 1)[0].decode("latin-1")


def _565(c: int):
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    return (r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)


@dataclass
class Texture:
    name: str
    mask: str
    width: int
    height: int
    depth: int
    num_levels: int
    raster_format: int
    d3d_format: int
    fmt: str
    has_alpha: bool
    palette: Optional[bytes]          # BGRA entries (1024 PAL8 / 128 PAL4) or None
    levels: List[bytes] = field(default_factory=list)  # raw pixel bytes per mip
    filter_addr: int = 0x1102         # Block A filterAndAddressing (preserved for re-emit)
    raster_type: int = 4              # Block B rasterType (4 = TEXTURE)
    comp_flags: int = 0               # Block B compressionFlags byte (preserved for re-emit)
    rgba_direct: Optional[bytes] = None  # already-decoded RGBA8888 (PS2 native path; level 0 only)

    def rgba(self, level: int = 0) -> bytes:
        if self.rgba_direct is not None:  # PS2 native: decoded up-front (no per-mip re-decode)
            return self.rgba_direct
        w = max(self.width >> level, 1)
        h = max(self.height >> level, 1)
        return _decode_level(self, self.levels[level], w, h)


@dataclass
class Txd:
    device_id: int
    textures: List[Texture]

    def find(self, name: str) -> Optional[Texture]:
        nl = name.lower()
        for t in self.textures:
            if t.name.lower() == nl:
                return t
        return None


def parse_txd(data: bytes) -> Txd:
    root = rw.read_header(data, 0)
    if root.type != rw.TEXTURE_DICTIONARY:
        # tolerate a stray leading chunk: scan for the dictionary
        found = rw.find_chunk(data, rw.TEXTURE_DICTIONARY, 0, len(data))
        if not found:
            raise ValueError("no TextureDictionary chunk")
        root = found

    body, end = root.body_offset, root.end
    st = rw.find_chunk(data, rw.STRUCT, body, end)
    num_tex, device_id = struct.unpack_from("<HH", data, st.body_offset)

    textures: List[Texture] = []
    for ch in rw.iter_chunks(data, st.end, end):
        if ch.type != rw.TEXTURE_NATIVE:
            continue
        inner = rw.find_chunk(data, rw.STRUCT, ch.body_offset, ch.end)
        if not inner:
            continue
        # Try the PC D3D8/D3D9 Block-A/B reader first (it validates platformId 8/9 and
        # raises on anything else); fall back to the PS2 GS native reader. This is robust
        # to the deviceId/platformId ambiguity - SA-PS2 TXDs report a non-D3D platform.
        try:
            textures.append(_parse_native(data, inner.body_offset))
        except Exception:
            try:
                textures.append(_native_ps2(data, ch))
            except Exception as e:  # one bad texture must not kill the dict
                textures.append(Texture(f"<error:{e}>", "", 0, 0, 0, 0, 0, 0, "ERROR", False, None))
    return Txd(device_id, textures)


def _native_ps2(data: bytes, tn: rw.ChunkHeader) -> Texture:
    """Build a Texture from a PS2-native TextureNative (platformId 4)."""
    r = txd_ps2.decode_texturenative(data, tn)
    if r is None:
        return Texture("<ps2:undecoded>", "", 0, 0, 0, 0, 0, 0, "PS2_UNKNOWN", False, None)
    return Texture(
        r["name"], r["mask"], r["width"], r["height"], r["depth"], 1,
        r["raster_format"], 0, r["fmt"], r["has_alpha"], None,
        rgba_direct=r["rgba"],
    )


def _parse_native(buf, so: int) -> Texture:
    # Block A (72 bytes)
    platform_id = struct.unpack_from("<I", buf, so)[0]
    # platform_id is RwPlatformID (the platform SDK headers): PS2=4, XBOX=5, GAMECUBE=6, PCD3D8=8, PCD3D9=9.
    # This path decodes the PC D3D8/D3D9 native Block-A/B layout only. PS2/Xbox/GameCube native
    # rasters (swizzled PSMT8/PSMT4 + GS CLUT for PS2) have a different STRUCT tree - decoding them
    # here reads garbage. Reject clearly instead. ✅ SDK-verification.
    if platform_id not in (8, 9):
        _plat = {0: "unknown", 1: "PCD3D7", 2: "PCOGL", 3: "MAC", 4: "PS2",
                 5: "XBOX", 6: "GAMECUBE", 7: "SOFTRAS"}.get(platform_id, f"0x{platform_id:X}")
        raise ValueError(
            f"non-D3D native raster (platformId={platform_id}={_plat}) not supported by the D3D path"
        )
    filter_addr = struct.unpack_from("<I", buf, so + 4)[0]
    name = _cstr(buf, so + 8, 32)
    mask = _cstr(buf, so + 40, 32)
    # Block B (16 bytes) at so+0x48
    b = so + 0x48
    raster_format, d3d_format = struct.unpack_from("<II", buf, b)
    width, height = struct.unpack_from("<HH", buf, b + 8)
    depth = buf[b + 12]
    num_levels = buf[b + 13]
    raster_type = buf[b + 14]
    comp_flags = buf[b + 15]

    if d3d_format == _DXT1:
        fmt = "DXT1"
    elif d3d_format == _DXT3:
        fmt = "DXT3"
    elif d3d_format == _DXT5:
        fmt = "DXT5"
    elif raster_format & _PAL8:
        fmt = "PAL8"
    elif raster_format & _PAL4:
        fmt = "PAL4"
    else:
        fmt = _FMT_NIBBLE.get((raster_format >> 8) & 0xF, "UNKNOWN")
    has_alpha = bool(comp_flags & 0x01)

    p = b + 16
    palette = None
    if raster_format & _PAL8:
        palette = buf[p:p + 1024]; p += 1024
    elif raster_format & _PAL4:
        palette = buf[p:p + 128]; p += 128

    levels: List[bytes] = []
    for _ in range(max(num_levels, 1)):
        if p + 4 > len(buf):
            break
        size = struct.unpack_from("<I", buf, p)[0]; p += 4
        levels.append(buf[p:p + size]); p += size

    return Texture(name, mask, width, height, depth, num_levels,
                   raster_format, d3d_format, fmt, has_alpha, palette, levels,
                   filter_addr, raster_type, comp_flags)


# --------------------------- pixel decode ---------------------------

def _decode_level(t: Texture, data: bytes, w: int, h: int) -> bytes:
    if t.fmt in ("DXT1", "DXT3", "DXT5"):
        return _decode_dxt(t.fmt, data, w, h)
    if t.fmt == "PAL8":
        return _decode_pal(data, w, h, t.palette, 8, t.has_alpha)
    if t.fmt == "PAL4":
        return _decode_pal(data, w, h, t.palette, 4, t.has_alpha)
    return _decode_raw(t.fmt, data, w, h, t.has_alpha)


def _decode_dxt(fmt: str, data: bytes, w: int, h: int) -> bytes:
    out = bytearray(w * h * 4)
    bw = (w + 3) >> 2
    bh = (h + 3) >> 2
    bsize = 8 if fmt == "DXT1" else 16
    pos = 0
    for by in range(bh):
        for bx in range(bw):
            block = data[pos:pos + bsize]
            pos += bsize
            if len(block) < bsize:
                continue
            _emit_dxt_block(fmt, block, out, bx * 4, by * 4, w, h)
    return bytes(out)


def _emit_dxt_block(fmt, block, out, ox, oy, w, h):
    # alpha plane
    alpha = [255] * 16
    if fmt == "DXT3":
        abits = int.from_bytes(block[0:8], "little")
        for i in range(16):
            alpha[i] = ((abits >> (i * 4)) & 0xF) * 17
        cofs = 8
    elif fmt == "DXT5":
        a0, a1 = block[0], block[1]
        abits = int.from_bytes(block[2:8], "little")
        ramp = [a0, a1, 0, 0, 0, 0, 0, 0]
        if a0 > a1:
            for k in range(1, 7):
                ramp[k + 1] = ((7 - k) * a0 + k * a1) // 7
        else:
            for k in range(1, 5):
                ramp[k + 1] = ((5 - k) * a0 + k * a1) // 5
            ramp[6] = 0
            ramp[7] = 255
        for i in range(16):
            alpha[i] = ramp[(abits >> (i * 3)) & 0x7]
        cofs = 8
    else:
        cofs = 0

    c0 = block[cofs] | (block[cofs + 1] << 8)
    c1 = block[cofs + 2] | (block[cofs + 3] << 8)
    idx = int.from_bytes(block[cofs + 4:cofs + 8], "little")
    r0, g0, b0 = _565(c0)
    r1, g1, b1 = _565(c1)
    cols = [(r0, g0, b0, 255), (r1, g1, b1, 255), None, None]
    if fmt == "DXT1" and c0 <= c1:
        cols[2] = ((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255)
        cols[3] = (0, 0, 0, 0)
    else:
        cols[2] = ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3, 255)
        cols[3] = ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3, 255)

    for i in range(16):
        px = ox + (i & 3)
        py = oy + (i >> 2)
        if px >= w or py >= h:
            continue
        cr, cg, cb, ca = cols[(idx >> (i * 2)) & 3]
        a = alpha[i] if fmt != "DXT1" else ca
        o = (py * w + px) * 4
        out[o] = cr; out[o + 1] = cg; out[o + 2] = cb; out[o + 3] = a


def _decode_raw(fmt: str, data: bytes, w: int, h: int, has_alpha: bool) -> bytes:
    out = bytearray(w * h * 4)
    n = w * h
    if fmt in ("RGBA8888", "RGB888"):
        for i in range(min(n, len(data) // 4)):
            b, g, r, a = data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]
            o = i * 4
            out[o] = r; out[o + 1] = g; out[o + 2] = b
            out[o + 3] = (a if has_alpha else 255) if fmt == "RGBA8888" else 255
    elif fmt in ("RGB565", "ARGB1555", "ARGB4444", "RGB555"):
        for i in range(min(n, len(data) // 2)):
            v = data[i * 2] | (data[i * 2 + 1] << 8)
            o = i * 4
            if fmt == "RGB565":
                r, g, b = _565(v); a = 255
            elif fmt in ("ARGB1555", "RGB555"):
                a = 255 if (fmt == "RGB555" or not has_alpha) else (255 if v & 0x8000 else 0)
                r5 = (v >> 10) & 0x1F; g5 = (v >> 5) & 0x1F; b5 = v & 0x1F
                r = (r5 << 3) | (r5 >> 2); g = (g5 << 3) | (g5 >> 2); b = (b5 << 3) | (b5 >> 2)
            else:  # ARGB4444
                a4 = (v >> 12) & 0xF; r4 = (v >> 8) & 0xF; g4 = (v >> 4) & 0xF; b4 = v & 0xF
                r = (r4 << 4) | r4; g = (g4 << 4) | g4; b = (b4 << 4) | b4
                a = ((a4 << 4) | a4) if has_alpha else 255
            out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = a
    elif fmt == "LUM8":
        for i in range(min(n, len(data))):
            o = i * 4; L = data[i]
            out[o] = L; out[o + 1] = L; out[o + 2] = L; out[o + 3] = 255
    return bytes(out)


def _decode_pal(data: bytes, w: int, h: int, palette: Optional[bytes], bits: int, has_alpha: bool) -> bytes:
    out = bytearray(w * h * 4)
    if not palette:
        return bytes(out)
    n = w * h
    if bits == 8:
        for i in range(min(n, len(data))):
            e = data[i] * 4
            _put_pal(out, i, palette, e, has_alpha)
    else:  # PAL4: two indices per byte, low nibble = even x
        for i in range(n):
            byte_i = i >> 1
            if byte_i >= len(data):
                break
            idx = (data[byte_i] & 0xF) if (i & 1) == 0 else (data[byte_i] >> 4)
            _put_pal(out, i, palette, idx * 4, has_alpha)
    return bytes(out)


def _put_pal(out, i, palette, e, has_alpha):
    if e + 3 >= len(palette):
        return
    b, g, r, a = palette[e], palette[e + 1], palette[e + 2], palette[e + 3]
    o = i * 4
    out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = a if has_alpha else 255


# =========================== encode / import path ===========================
# Mirrors librw d3d9::writeNativeTexture / getSizeNativeTexture and
# texture.cpp TexDictionary::streamWrite. Re-emits the exact RW chunk tree:
# TEXDICTIONARY { STRUCT{numTex,deviceId} N*TEXTURENATIVE EXTENSION(0) }
# TEXTURENATIVE { STRUCT{BlockA BlockB palette mips} EXTENSION(0) }
# A passthrough texture is byte-identical (all stored fields + raw level/palette
# bytes verbatim); encode_texture_raw8888 is the new-texture (raw 8888) path.

_RW_LIBID = 0x1803FFFF  # RW 3.6.0.3 (SA PC retail) - written on every chunk header


def _chunk(type_: int, body: bytes) -> bytes:
    """Wrap a body in a 12-byte RW chunk header (libid 0x1803FFFF)."""
    return struct.pack("<III", type_, len(body), _RW_LIBID) + body


def _native_struct_body(tex: "Texture") -> bytes:
    """The inner STRUCT body of one TextureNative: Block A + Block B + palette + mips.

 Serializes any Texture (DXT / raw / palette) from its stored fields + raw
 `levels` (and `palette`) bytes verbatim - this is the passthrough path.
 """
    name = tex.name.encode("latin-1")[:31]
    mask = tex.mask.encode("latin-1")[:31]
    # Block A (72 bytes)
    a = struct.pack("<I", 9)                       # platformId
    a += struct.pack("<I", tex.filter_addr)        # filterAndAddressing
    a += name + b"\x00" * (32 - len(name))         # name[32]
    a += mask + b"\x00" * (32 - len(mask))         # maskName[32]
    # Block B (16 bytes)
    b = struct.pack("<i", tex.raster_format)       # rasterFormat
    b += struct.pack("<I", tex.d3d_format)         # d3dFormat
    b += struct.pack("<HH", tex.width, tex.height)
    b += bytes((tex.depth & 0xFF, len(tex.levels) & 0xFF,
                tex.raster_type & 0xFF, tex.comp_flags & 0xFF))
    body = a + b
    if tex.palette:
        body += bytes(tex.palette)                 # 1024 (PAL8) / 128 (PAL4), verbatim
    for lvl in tex.levels:
        body += struct.pack("<I", len(lvl)) + bytes(lvl)
    return body


def _texture_native_bytes(tex: "Texture") -> bytes:
    """Serialize one Texture as a complete TEXTURENATIVE chunk (+ empty EXTENSION).

 Byte-identical for a passthrough (unchanged) Texture: re-emits the original
 Block A/B fields and the raw `levels` / `palette` bytes verbatim.
 """
    inner = _chunk(rw.STRUCT, _native_struct_body(tex))
    ext = _chunk(rw.EXTENSION, b"")                 # SA writes an empty extension
    return _chunk(rw.TEXTURE_NATIVE, inner + ext)


def encode_texture_raw8888(name: str, rgba: bytes, w: int, h: int,
                           has_alpha: bool = True, mask: str = "") -> bytes:
    """Encode an RGBA8888 buffer as one raw-8888 TEXTURENATIVE chunk (lossless).

 Block B: rasterFormat nibble 0x5 (0x0500), d3dFormat 0, depth 32, numLevels 1,
 rasterType 4, compressionFlags = has_alpha?1:0. On-disk pixel order is B,G,R,A.
 """
    if len(rgba) < w * h * 4:
        raise ValueError(f"rgba too short: {len(rgba)} < {w * h * 4}")
    # RGBA -> on-disk BGRA
    src = memoryview(rgba)
    bgra = bytearray(w * h * 4)
    for i in range(w * h):
        o = i * 4
        bgra[o] = src[o + 2]      # B
        bgra[o + 1] = src[o + 1]  # G
        bgra[o + 2] = src[o]      # R
        bgra[o + 3] = src[o + 3]  # A
    tex = Texture(
        name=name, mask=mask, width=w, height=h, depth=32, num_levels=1,
        raster_format=0x0500, d3d_format=0, fmt="RGBA8888",
        has_alpha=has_alpha, palette=None, levels=[bytes(bgra)],
        filter_addr=0x1102, raster_type=4, comp_flags=(1 if has_alpha else 0),
    )
    return _texture_native_bytes(tex)


def build_txd(textures: List["Texture"], device_id: int = 2) -> bytes:
    """Build a complete TEXDICTIONARY from a list of Textures.

 STRUCT{u16 numTextures, u16 deviceId} + each texture re-emitted (passthrough
 byte-identical for unchanged Textures) + dictionary-level empty EXTENSION.
 """
    body = _chunk(rw.STRUCT, struct.pack("<HH", len(textures), device_id))
    for tex in textures:
        body += _texture_native_bytes(tex)
    body += _chunk(rw.EXTENSION, b"")              # dictionary-level empty extension
    return _chunk(rw.TEXTURE_DICTIONARY, body)


def replace_texture(txd_bytes: bytes, tex_name: str, rgba: bytes,
                    w: int, h: int, has_alpha: bool = True) -> bytes:
    """Swap the named texture for a new raw-8888 TextureNative; rebuild the TXD.

 Every other texture is re-emitted byte-identically (passthrough). The named
 texture is replaced by a fresh raw 8888 raster from `rgba` (w x h).
 """
    txd = parse_txd(txd_bytes)
    target = tex_name.lower()
    new_native = encode_texture_raw8888(tex_name, rgba, w, h, has_alpha,
                                        mask=(txd.find(tex_name).mask if txd.find(tex_name) else ""))
    out = _chunk(rw.STRUCT, struct.pack("<HH", len(txd.textures), txd.device_id))
    replaced = False
    for tex in txd.textures:
        if not replaced and tex.name.lower() == target:
            out += new_native
            replaced = True
        else:
            out += _texture_native_bytes(tex)
    if not replaced:
        raise KeyError(f"texture {tex_name!r} not found in TXD")
    out += _chunk(rw.EXTENSION, b"")
    return _chunk(rw.TEXTURE_DICTIONARY, out)


def png_to_rgba(png_bytes: bytes):
    """Decode a PNG to (rgba_bytes, w, h) via Pillow (server import path)."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return im.tobytes(), im.width, im.height
