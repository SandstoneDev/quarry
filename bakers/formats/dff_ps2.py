"""PS2/console-native DFF geometry decoder (nativeData plugin 0x0510).

the source game-PS2 (and VCS console) DFFs store geometry with the rpGEOMETRYNATIVE flag
(0x01000000) set: no generic vertex/uv/triangle arrays, only a nativeData 0x0510
packet holding VU-ready DMA/VIF clusters. This module walks that packet statically
(no VU execution) and rebuilds positions / uv / colours / triangles.

On-disk layout is documented in 
The layout was derived in gvcslib (validated against real SA-PS2 models);
this is an independent re-implementation on top of core.rwstream.

Quantization: position s16/128, uv s16/4096, colour u16 per channel -> RGBA8888.
Cluster map (VIF UNPACK vu_dst = imm & 0x3FF): 0=POS(V4_16), 1=UV(V2_16), 2=COL(V4_16).
Tri-strip: POS.w is the ADC bit (w != 0 => strip restart, no triangle ends here);
winding alternates by absolute strip position.
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple

from core import rwstream as rw

_STCYCL_TAG = 0x01000103        # VIFcode prefixing every native sub-chain
_POS_SCALE = 1.0 / 128.0
_UV_SCALE = 1.0 / 4096.0
_POS_MAX = 32767                # full s16 range; the W-bit is the primary validity signal

_4I = struct.Struct("<4I")


def _is_unpack(vifcode: int) -> bool:
    return ((vifcode >> 24) & 0x7F) & 0x60 == 0x60


def _walk_dma_chain(buf: bytes, chain_start: int, chunk_end: int) -> Tuple[List[int], int]:
    """Walk one DMA source packet. Return (split_pos_nums, data_block_start).

 split_pos_nums = per-split POS vertex count (POS UNPACK `num`, vu_dst==0).
 data_block_start = byte offset of the first POS entry in the flat attribute block.
 See ps2_native_geometry.md § "Chain walk".
 """
    o = chain_start
    split_pos_nums: List[int] = []
    current_pos_num = 0
    data_block_start = chunk_end  # fallback

    while o + 16 <= chunk_end:
        taglo, _addr, v0, v1 = _4I.unpack_from(buf, o)
        idv = (taglo >> 28) & 0x7
        qwc = taglo & 0xFFFF

        for vifcode in (v1, v0):
            if not _is_unpack(vifcode):
                continue
            num = (vifcode >> 16) & 0xFF
            if (vifcode & 0xFFFF) & 0x3FF == 0:   # vu_dst 0 == POS
                current_pos_num = num

        if idv == 1:                       # cnt: advance past inline data
            if qwc == 1:                   # split separator
                split_pos_nums.append(current_pos_num)
                current_pos_num = 0
            o += 16 + qwc * 16
        elif idv in (3, 4):                # ref/refs: data external, ptr advances 16
            o += 16
        else:                              # ret(6)/refe(0)/end(7): packet terminator
            split_pos_nums.append(current_pos_num)
            data_block_start = o + 16 + qwc * 16
            break

    return split_pos_nums, data_block_start


def _is_valid_pos(x: int, y: int, z: int, w: int) -> bool:
    return w in (0, -32768) and abs(x) <= _POS_MAX and abs(y) <= _POS_MAX and abs(z) <= _POS_MAX


def _align16(base: int, off: int) -> int:
    rem = (off - base) % 16
    return off + (16 - rem) % 16


def _decode_pos_uv_col(buf: bytes, db: int, chunk_end: int, split_pos_nums: List[int]):
    """Read the flat POS/UV/COL arrays at data_block_start (db).

 Layout (all splits concatenated, each block qword-aligned from db):
 [n_max*8 POS V4_16][pad][n_max*4 UV V2_16][pad][n_max*8 COL V4_16]
 where n_max = sum(split_pos_nums). Trailing DMA padding is trimmed by scanning
 POS until the first entry failing _is_valid_pos -> n_valid.
 """
    n_max = sum(split_pos_nums)
    if n_max == 0 or db >= chunk_end:
        return [], [], [], []

    n_valid = 0
    for i in range(n_max):
        off = db + i * 8
        if off + 8 > chunk_end:
            break
        x, y, z, w = struct.unpack_from("<4h", buf, off)
        if not _is_valid_pos(x, y, z, w):
            break
        n_valid += 1
    if n_valid == 0:
        return [], [], [], []

    # true per-split counts: distribute n_valid across splits in order
    true_vcounts: List[int] = []
    remaining = n_valid
    for cnt in split_pos_nums:
        alloc = min(cnt, remaining)
        true_vcounts.append(alloc)
        remaining -= alloc
        if remaining <= 0:
            break
    while len(true_vcounts) < len(split_pos_nums):
        true_vcounts.append(0)

    positions = []
    for i in range(n_valid):
        x, y, z, w = struct.unpack_from("<4h", buf, db + i * 8)
        positions.append((x * _POS_SCALE, y * _POS_SCALE, z * _POS_SCALE, w))

    # UV/COL start after the FULL (n_max) prior block, qword-aligned from db
    uv_start = _align16(db, db + n_max * 8)
    uvs = []
    for i in range(n_valid):
        off = uv_start + i * 4
        if off + 4 > chunk_end:
            uvs.append((0.0, 0.0))
            continue
        u, v = struct.unpack_from("<2h", buf, off)
        uvs.append((u * _UV_SCALE, v * _UV_SCALE))

    col_start = _align16(db, uv_start + n_max * 4)
    colors = []
    for i in range(n_valid):
        off = col_start + i * 8
        if off + 8 > chunk_end:
            colors.append((255, 255, 255, 255))
            continue
        r, g, b, a = struct.unpack_from("<4H", buf, off)
        colors.append((r >> 8, g >> 8, b >> 8, a >> 8))

    return positions, uvs, colors, true_vcounts


def _data_block_end(db: int, n_max: int) -> int:
    uv_start = _align16(db, db + n_max * 8)
    col_start = _align16(db, uv_start + n_max * 4)
    return col_start + n_max * 8


def _strip_to_tris(vcnt: int, positions: list, vtx_base: int) -> List[Tuple[int, int, int]]:
    """One split's tri-strip -> mesh-local triangles, using the ADC W-bit."""
    tris = []
    for k in range(2, vcnt):
        if positions[vtx_base + k][3] != 0:   # ADC restart at this vertex
            continue
        a, b, c = k - 2, k - 1, k
        if a == b or b == c or a == c:
            continue
        tris.append((a, c, b) if (k & 1) else (a, b, c))
    return tris


def _is_chain_start(buf: bytes, o: int, end: int) -> bool:
    if o + 16 > end:
        return False
    taglo, _addr, v0, v1 = _4I.unpack_from(buf, o)
    idv = (taglo >> 28) & 0x7
    return idv == 3 and v0 == _STCYCL_TAG and _is_unpack(v1) and (v1 & 0x3FF) == 0


def _scan_chain_start(buf: bytes, o: int, end: int, origin: int) -> Optional[int]:
    rem = (o - origin) % 16
    if rem:
        o += 16 - rem
    while o + 16 <= end:
        if _is_chain_start(buf, o, end):
            return o
        o += 16
    return None


def _binmesh_mats(data: bytes, ext: rw.ChunkHeader) -> List[int]:
    """Per-sub-chain material index (draw order) from binMesh 0x050E. PS2 native
 carries no index payload: header {u32 flags,u32 numMeshes,u32 total} then
 per mesh {u32 numIndices, i32 matIndex}."""
    bm = rw.find_chunk(data, rw.BIN_MESH_PLG, ext.body_offset, ext.end)
    if bm is None or bm.size < 12:
        return []
    o = bm.body_offset
    _flags, num, _total = struct.unpack_from("<3I", data, o)
    o += 12
    mats = []
    for _ in range(num):
        if o + 8 > bm.end:
            break
        _numidx, mat = struct.unpack_from("<2i", data, o)
        o += 8
        mats.append(mat)
    return mats


def parse_native_geometry(data: bytes, ext: rw.ChunkHeader):
    """Decode a native geometry EXTENSION into flat render data.

 Returns (vertices, uvs, colors, triangles, splits) where:
 vertices : [(x,y,z)] global vertex pool
 uvs : [(u,v)] parallel to vertices (uv set 0)
 colors : [(r,g,b,a)] parallel to vertices
 triangles: [(a,b,c,matId)] global indices + material-list index
 splits : [{"mat_index", "indices", "strip": False}] render batches
 Returns None if no nativeData 0x0510 packet is present.
 """
    nat = rw.find_chunk(data, rw.NATIVE_DATA_PLG, ext.body_offset, ext.end)
    if nat is None:
        return None

    chunk_end = nat.end
    origin = nat.body_offset + 0x18   # skip the 24-byte nativeData sub-header
    mesh_mats = _binmesh_mats(data, ext)
    n_meshes = len(mesh_mats) if mesh_mats else 1

    vertices: List[tuple] = []
    uvs: List[tuple] = []
    colors: List[tuple] = []
    triangles: List[tuple] = []
    splits: List[dict] = []

    o = origin
    for mesh_i in range(n_meshes):
        chain_start = origin if mesh_i == 0 else _scan_chain_start(data, o, chunk_end, origin)
        if chain_start is None:
            break
        split_pos_nums, db = _walk_dma_chain(data, chain_start, chunk_end)
        if not split_pos_nums:
            break
        positions, muv, mcol, true_vcounts = _decode_pos_uv_col(data, db, chunk_end, split_pos_nums)
        if not positions:
            break

        mat = mesh_mats[mesh_i] if mesh_i < len(mesh_mats) else 0

        vtx_base = 0
        for si, vcnt in enumerate(true_vcounts):
            if vcnt == 0:
                vtx_base += split_pos_nums[si]
                continue
            base = len(vertices)   # global index base for this split
            for gi in range(vtx_base, vtx_base + vcnt):
                px, py, pz, _w = positions[gi]
                vertices.append((px, py, pz))
                uvs.append(muv[gi] if gi < len(muv) else (0.0, 0.0))
                colors.append(mcol[gi] if gi < len(mcol) else (255, 255, 255, 255))
            local = _strip_to_tris(vcnt, positions, vtx_base)
            flat_idx: List[int] = []
            for a, b, c in local:
                triangles.append((base + a, base + b, base + c, mat))
                flat_idx.extend((base + a, base + b, base + c))
            splits.append({"mat_index": mat, "indices": flat_idx, "strip": False})
            vtx_base += vcnt

        o = _data_block_end(db, sum(split_pos_nums))

    return vertices, uvs, colors, triangles, splits
