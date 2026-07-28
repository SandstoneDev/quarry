"""Per-model sub-mesh extraction from the console title PSP geometry bundles.

CRACKED 2026-06-14 (workflow wr9oq9q9d, 2 converging angles, full-corpus verified:
1260 bundles / 34254 mesh models, 0 OOB, 0 span overlaps; same model_id extracted from
different bundles yields byte-identical vertices). See [[model-geometry-map]] / map-placement.

A zone IMG geometry **bundle** packs many models. The ``model_id -> in_blob_offset``
directory (see :func:`gvcslib.model_map.parse_bundle_directory`) points **0x20 bytes past**
a per-model *mesh-descriptor* that the engine geometry compiler ``FUN_00156a98`` walks at
first draw. ``header_off = in_blob_offset - 0x20``::

    header_off +0x00  u32  N          draw-record count
    header_off +0x04  u32  pad = 0    (non-zero => material/special blob, NOT a mesh)
    header_off +0x08  draw_record[N] (each 0x18 = 24 bytes)
    header_off +0x08 + N*0x18  ->  VTYPE-0x115 vertex stream (stride 10), consumed
                                   sequentially by the records.

draw_record (24 bytes)::

    +0x00 u16   material/texture id  (0xffff = none)
    +0x02 u16   low 15 bits = strip vertex count; bit 0x8000 = z-bias/decal (alpha) flag
    +0x04 fp16  tex-env scale U  (GE cmd 0x48; UV scale, NOT position scale)
    +0x06 fp16  tex-env scale V  (GE cmd 0x49)
    +0x08 fp16  POSITION SCALE   (uniform, PER-RECORD; all 3 axes use this scalar)
    +0x0a fp16  unused (== 0)
    +0x0c s16[6] packed cull AABB (frustum cull / record vert bounds; not geometry)

Each record is one GE PRIM type 4 (TRIANGLE_STRIP). Positions dequantize as
``s16/32768 * posScale`` where ``posScale`` is the PER-RECORD uniform fp16 at
**draw_record +0x08** (NOT the bundle-primary 3x fp16 at bundle header +0x40, which
is wrong for secondary models). Final world placement = these local positions x the
DTZ instance matrix (out of scope).

Model-type discriminator:
* ``in_blob_offset == 0`` -> the bundle's PRIMARY pooled mesh (decode the whole bundle with
  :func:`gvcslib.geometry.decode`); :func:`extract_model` returns ``None`` for it.
* ``pad != 0`` or implausible ``N`` -> shared/pooled sub-object or material blob -> ``None``.
* otherwise -> a standalone mesh (returned).
"""
import struct

from . import geometry, model_map

VSTRIDE = 10            # VTYPE-0x115 vertex stride
RECORD = 0x18           # draw-record size
HDR_BACK = 0x20         # directory offset is this many bytes past the descriptor header
MAX_RECORDS = 2048      # sanity bound on N (rejects shared/pooled garbage counts)


def _read_scale(blob):
    return geometry._read_scale(blob, geometry.SCALE_OFF)


def extract_model(blob, in_blob_off, dequantize=True):
    """Extract ONE model's sub-mesh from a bundle, or ``None`` for primary/shared entries.

    Returns ``{header_off, N, vstream_off, vstream_end, scale, positions, uv, colors,
    triangles, prims}``. With ``dequantize`` (default), ``positions`` are local floats
    (``s16/32768 * pos_scale``, the PER-RECORD uniform scalar @draw_record+0x08) and
    ``uv`` are 0..1; otherwise ``positions`` are raw s16 tuples and ``uv`` are raw u8.
    ``colors`` are RGBA5551 ints. ``scale`` is the first record's ``pos_scale`` (as a
    3-tuple, for back-compat). ``prims`` is the per-record list
    ``{material, vert_start, vert_count, prim_type, zbias, tex_scale, pos_scale}``.
    """
    blob = bytes(blob)
    hdr = in_blob_off - HDR_BACK
    if hdr < 0 or hdr + 8 > len(blob):
        return None                                   # primary whole-bundle mesh (off 0)
    n, pad = struct.unpack_from("<II", blob, hdr)
    if pad != 0 or not (1 <= n <= MAX_RECORDS):
        return None                                   # shared/pooled sub-object or material blob
    recbase = hdr + 8
    stream = recbase + n * RECORD
    if stream > len(blob):
        return None

    positions, uv, colors, tris, prims = [], [], [], [], []
    pos = stream
    vbase = 0
    first_pscale = None
    for i in range(n):
        rb = recbase + i * RECORD
        material = struct.unpack_from("<H", blob, rb)[0]
        f = struct.unpack_from("<H", blob, rb + 2)[0]
        vc = f & 0x7FFF
        zbias = (f >> 15) & 1
        tex_scale = struct.unpack_from("<2e", blob, rb + 4)
        # PER-RECORD uniform position scale: fp16 @ draw_record+0x08 (NOT bundle+0x40).
        pscale = geometry._half_to_float(struct.unpack_from("<H", blob, rb + 8)[0])
        if first_pscale is None:
            first_pscale = pscale
        if pos + vc * VSTRIDE > len(blob):
            return None
        for v in range(vc):
            o = pos + v * VSTRIDE
            u8, v8 = blob[o], blob[o + 1]
            color = blob[o + 2] | (blob[o + 3] << 8)        # RGBA5551
            x, y, z = struct.unpack_from("<3h", blob, o + 4)
            if dequantize:
                positions.append((x / 32768.0 * pscale, y / 32768.0 * pscale,
                                  z / 32768.0 * pscale))
                uv.append((u8 / 255.0, v8 / 255.0))
            else:
                positions.append((x, y, z))
                uv.append((u8, v8))
            colors.append(color)
        # GE PRIM type 4 = TRIANGLE_STRIP: expand, alternate winding, drop degenerate joints
        for k in range(2, vc):
            a, b, c = vbase + k - 2, vbase + k - 1, vbase + k
            if positions[a] == positions[b] or positions[b] == positions[c] \
                    or positions[a] == positions[c]:
                continue
            tris.append((a, b, c) if (k % 2 == 0) else (b, a, c))
        prims.append({"material": material, "vert_start": vbase, "vert_count": vc,
                      "prim_type": 4, "zbias": zbias, "tex_scale": tex_scale,
                      "pos_scale": pscale})
        vbase += vc
        pos += vc * VSTRIDE
    s = first_pscale if first_pscale is not None else 0.0
    return {"header_off": hdr, "N": n, "vstream_off": stream, "vstream_end": pos,
            "scale": (s, s, s), "positions": positions, "uv": uv, "colors": colors,
            "triangles": tris, "prims": prims}


def classify(blob, in_blob_off):
    """'primary' (off 0), 'mesh' (standalone, extractable), or 'shared' (pooled/material)."""
    if in_blob_off == 0:
        return "primary"
    return "mesh" if extract_model(blob, in_blob_off) is not None else "shared"


def extract_all(blob):
    """``{model_id: extract_model(...) }`` for every standalone-mesh entry in the bundle.

    Skips primary (off 0) and shared/pooled entries (which reference the primary pool).
    """
    blob = bytes(blob)
    out = {}
    for mid, off in model_map.parse_bundle_directory(blob):
        m = extract_model(blob, off)
        if m is not None:
            out[mid] = m
    return out
