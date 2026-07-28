"""Convert a decoded SA DFF (sa_dff.SaModel) into PSP GE (sceGu) ready geometry.

Target: a custom PSP homebrew engine (pspsdk / sceGu). Map geometry wants the
most compact vertex the GE can consume so the EDRAM/display-list budget lasts:

 vertex format (GE component order is FIXED: weights, texcoord, color, normal, position):

 struct PspMapVertex { // 12 bytes, 2-byte aligned
 s16 u, v; // GU_TEXTURE_16BIT (fixed-point, /UV_FIXED_ONE)
 u16 color; // GU_COLOR_5551 (from RGBA8888 vertex color)
 s16 x, y, z; // GU_VERTEX_16BIT (quantized about model AABB)
 };

 vfmt flag word (what you pass to sceGuDrawArray / store in the model header):

 GU_TEXTURE_16BIT | GU_COLOR_5551 | GU_VERTEX_16BIT | GU_TRANSFORM_3D

Position scale convention
-------------------------
Positions are quantized to int16 about the model AABB center:

 center = (aabb_min + aabb_max) / 2
 half = max over axes of (aabb_max - aabb_min) / 2 (single uniform half-extent)
 scale = half / 32767.0 (one float per model)

 qi = round((p_i - center_i) / scale) clamped to [-32767, 32767] (int16)

On the engine the original position is recovered with a uniform sceGumScale +
translate:

 sceGumTranslate(center); sceGumScale(scale, scale, scale);
 p_i = center_i + qi * scale

A single uniform scale (one float) keeps the int16 lattice isotropic and matches
how SA/the console title already drive sceGumScale per object. `center` is returned so the
engine can translate as well (degenerate / zero-extent models get scale=1.0).

UV scale convention
-------------------
UVs are stored as GU_TEXTURE_16BIT fixed-point: raw = round(uv * UV_FIXED_ONE).
THE GE READS THE 16 BITS AS UNSIGNED u1.15 (Sony GE-UM 6.1/6.5, GE-CR p13) - there is no signed texcoord path in the hardware. With the engine's global
sceGuTexScale(8,8) the sampling window is [0,16) tiles, wrapping mod 16, so
the honest packing is raw mod 65536 (bit-identical to the old signed value
for |uv| < 8; correct instead of destroyed for larger magnitudes). Bakers
must REBASE each submesh's UVs into [0, span<=15.75] (integer-tile shift,
GU_REPEAT-invariant) BEFORE packing: any triangle whose raw range crosses a
window seam interpolates the long way round (the striped-roads root cause,
see GTASA_PSP research/striped_textures_rootcause_and_fix.md).

This module is pure-Python and stdlib-only. Pack/unpack are byte-exact round
trips (see tests/test_psp_mesh.py).
"""
import struct
from typing import List, Dict

# --- GU vertex-type flag bits (from pspgu.h) ----------------------------------
GU_TEXTURE_16BIT = 0x00000001   # texcoord component = 2x s16
GU_COLOR_5551    = 0x00600000   # color component = 1x u16 (5551)
GU_VERTEX_16BIT  = 0x00000080   # position component = 3x s16
GU_TRANSFORM_3D  = 0x00000000   # transformed by the GE matrices (vs 2D)
GU_INDEX_16BIT   = 0x00000800   # index list = u16

# The packed vertex-type word the engine passes to sceGuDrawArray for this format.
VFMT = GU_TEXTURE_16BIT | GU_COLOR_5551 | GU_VERTEX_16BIT | GU_TRANSFORM_3D

# Fixed-point UV unit: s16 = uv * UV_FIXED_ONE.
UV_FIXED_ONE = 4096.0

# Interleaved struct: u(s16) v(s16) color(u16) x(s16) y(s16) z(s16) = 12 bytes.
_VTX = struct.Struct("<hhHhhh")
VERTEX_SIZE = _VTX.size   # 12
_IDX = struct.Struct("<H")

_S16_MAX = 32767
_S16_MIN = -32768


def _clamp_s16(v: int) -> int:
    if v > _S16_MAX:
        return _S16_MAX
    if v < _S16_MIN:
        return _S16_MIN
    return v


def _pack_uv16(v: float) -> int:
    """Quantize one UV into the GE's UNSIGNED u16 window (mod 16 tiles at the
 engine's TexScale 8) and return it as the equivalent s16 bit pattern for
 the '<h' struct slot. Identical bits to the legacy signed value while
 |uv| < 8; correct window aliasing (instead of a destructive clamp) beyond."""
    q = int(round(v * UV_FIXED_ONE)) % 0x10000
    return q - 0x10000 if q > _S16_MAX else q


def rgba8888_to_5551(rgba: int) -> int:
    """Pack an RGBA8888 int (R high byte) into a GU_COLOR_5551 u16.

 GE 5551 little-endian bit layout: R[0:5] G[5:10] B[10:15] A[15].
 """
    r = (rgba >> 24) & 0xFF
    g = (rgba >> 16) & 0xFF
    b = (rgba >> 8) & 0xFF
    a = rgba & 0xFF
    r5 = r >> 3
    g5 = g >> 3
    b5 = b >> 3
    a1 = 1 if a >= 128 else 0
    return r5 | (g5 << 5) | (b5 << 10) | (a1 << 15)


def color5551_to_rgba8888(c: int) -> int:
    """Inverse of rgba8888_to_5551 (5-bit channels expanded back to 8-bit)."""
    r5 = c & 0x1F
    g5 = (c >> 5) & 0x1F
    b5 = (c >> 10) & 0x1F
    a1 = (c >> 15) & 1
    r = (r5 << 3) | (r5 >> 2)
    g = (g5 << 3) | (g5 >> 2)
    b = (b5 << 3) | (b5 >> 2)
    a = 0xFF if a1 else 0x00
    return (r << 24) | (g << 16) | (b << 8) | a


def model_aabb(model) -> tuple:
    """Return (min_xyz, max_xyz) over every vertex of every mesh."""
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    any_v = False
    for mesh in model.meshes:
        for (x, y, z) in mesh.positions:
            any_v = True
            if x < mn[0]: mn[0] = x
            if y < mn[1]: mn[1] = y
            if z < mn[2]: mn[2] = z
            if x > mx[0]: mx[0] = x
            if y > mx[1]: mx[1] = y
            if z > mx[2]: mx[2] = z
    if not any_v:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])


def compute_quant(model) -> tuple:
    """Return (center(x,y,z), scale) for the uniform int16 position quantization."""
    (mnx, mny, mnz), (mxx, mxy, mxz) = model_aabb(model)
    cx = (mnx + mxx) * 0.5
    cy = (mny + mxy) * 0.5
    cz = (mnz + mxz) * 0.5
    half = max((mxx - mnx), (mxy - mny), (mxz - mnz)) * 0.5
    if half <= 0.0:
        scale = 1.0
    else:
        scale = half / float(_S16_MAX)
    return (cx, cy, cz), scale


def pack_model(model) -> Dict:
    """Pack a sa_dff.SaModel into PSP-GE-ready geometry.

 Returns:
 {
 "vfmt": int, # GU vertex-type flag word (VFMT)
 "scale": float, # uniform position dequant scale
 "center":(x,y,z), # AABB center the engine translates by
 "uv_fixed_one": float, # UV fixed-point unit
 "prims": [ # one entry per SaMesh (indexed GU_PRIM_TRIANGLES)
 {
 "material_index": int,
 "vertex_bytes": bytes, # vcount * VERTEX_SIZE
 "index_bytes": bytes, # icount * 2 (GU_INDEX_16BIT)
 "vcount": int,
 "icount": int,
 }, ...
 ],
 }
 """
    (cx, cy, cz), scale = compute_quant(model)
    inv = (1.0 / scale) if scale != 0.0 else 0.0

    prims: List[Dict] = []
    for mesh in model.meshes:
        n = len(mesh.positions)
        if n == 0 or not mesh.triangles:
            continue

        vbuf = bytearray()
        for i in range(n):
            x, y, z = mesh.positions[i]
            qx = _clamp_s16(int(round((x - cx) * inv)))
            qy = _clamp_s16(int(round((y - cy) * inv)))
            qz = _clamp_s16(int(round((z - cz) * inv)))

            if i < len(mesh.uv):
                u, v = mesh.uv[i]
            else:
                u = v = 0.0
            su = _pack_uv16(u)
            sv = _pack_uv16(v)

            col = mesh.colors[i] if i < len(mesh.colors) else 0xFFFFFFFF
            c16 = rgba8888_to_5551(col)

            vbuf += _VTX.pack(su, sv, c16, qx, qy, qz)

        ibuf = bytearray()
        for (a, b, c) in mesh.triangles:
            # indices are mesh-local; guard against any stray out-of-range
            if a >= n or b >= n or c >= n:
                continue
            ibuf += _IDX.pack(a)
            ibuf += _IDX.pack(b)
            ibuf += _IDX.pack(c)

        prims.append({
            "material_index": mesh.material_index,
            "vertex_bytes": bytes(vbuf),
            "index_bytes": bytes(ibuf),
            "vcount": n,
            "icount": len(ibuf) // 2,
        })

    return {
        "vfmt": VFMT,
        "scale": scale,
        "center": (cx, cy, cz),
        "uv_fixed_one": UV_FIXED_ONE,
        "prims": prims,
    }


def unpack_positions(vertex_bytes: bytes, scale: float, center: tuple) -> List[tuple]:
    """Read back dequantized positions from a packed vertex buffer.

 Reverses pack_model's quantization: p = center + qi * scale.
 """
    cx, cy, cz = center
    out = []
    n = len(vertex_bytes) // VERTEX_SIZE
    for i in range(n):
        _su, _sv, _c, qx, qy, qz = _VTX.unpack_from(vertex_bytes, i * VERTEX_SIZE)
        out.append((cx + qx * scale, cy + qy * scale, cz + qz * scale))
    return out


def unpack_vertex(vertex_bytes: bytes, index: int, scale: float, center: tuple) -> Dict:
    """Read back one full vertex (position dequantized, uv, rgba8888 color)."""
    su, sv, c16, qx, qy, qz = _VTX.unpack_from(vertex_bytes, index * VERTEX_SIZE)
    cx, cy, cz = center
    return {
        "pos": (cx + qx * scale, cy + qy * scale, cz + qz * scale),
        "uv": (su / UV_FIXED_ONE, sv / UV_FIXED_ONE),
        "rgba8888": color5551_to_rgba8888(c16),
    }


def unpack_indices(index_bytes: bytes) -> List[int]:
    """Read back the GU_INDEX_16BIT index list as a flat list of ints."""
    n = len(index_bytes) // 2
    return [_IDX.unpack_from(index_bytes, i * 2)[0] for i in range(n)]
