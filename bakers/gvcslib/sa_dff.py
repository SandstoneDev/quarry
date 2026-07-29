"""the source game PS2-native DFF decoder (read-only). Part 1: RW chunk tree.
Part 2: PS2-native geometry decode (DMA chain + ADC tri-strip).
"""
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

STRUCT=0x01; STRING=0x02; EXTENSION=0x03; TEXTURE=0x06; MATERIAL=0x07
MATERIALLIST=0x08; FRAMELIST=0x0E; GEOMETRY=0x0F; CLUMP=0x10; GEOMETRYLIST=0x1A
PS2_NATIVE=0x0510; BINMESH=0x0253F2F9; ADCPLG=0x0253F2FD
BINMESH_PLG=0x050E   # real RW Bin-Mesh PLG (per-mesh material+index map).
# NB: the 0x0253F2F9 constant above is a mislabel kept for back-compat; the
# split->material mapping lives in BINMESH_PLG (0x050E).
_STCYCL_TAG=0x01000103   # VIFcode that prefixes every native geometry sub-chain.
_CONTAINER = {CLUMP, FRAMELIST, GEOMETRYLIST, GEOMETRY, MATERIALLIST, MATERIAL, TEXTURE, EXTENSION}

_POS_SCALE = 1.0 / 128.0
_UV_SCALE  = 1.0 / 4096.0
_POS_MAX   = 32767  # full s16 range; rely on W-bit as primary validity signal


class Chunk:
    __slots__ = ("type", "size", "off", "data_off", "children")
    def __init__(self, type, size, off, data_off):
        self.type, self.size, self.off, self.data_off, self.children = type, size, off, data_off, []
    def find(self, t):
        for c in self.children:
            if c.type == t:
                return c
        return None
    def find_all(self, t):
        return [c for c in self.children if c.type == t]


def _parse(data, o, end):
    nodes = []
    while o + 12 <= end:
        typ, sz, _lib = struct.unpack_from("<III", data, o)
        body = o + 12
        c = Chunk(typ, sz, o, body)
        if typ in _CONTAINER and sz >= 4:
            c.children = _parse(data, body, min(body + sz, end))
        nodes.append(c)
        o = body + sz
    return nodes


def parse_chunks(blob):
    """Parse a DFF into its RW chunk tree; returns the root Clump Chunk."""
    blob = bytes(blob)
    top = _parse(blob, 0, len(blob))
    # The CLUMP is usually top[0], but some DFFs lead with a non-clump header
    # (e.g. a 0x2B UV-anim dictionary on scrolling-sign/waterfall materials);
    # scan for the first CLUMP rather than requiring index 0.
    for c in top:
        if c.type == CLUMP:
            return c
    raise ValueError("not a DFF clump")


# ---------------------------------------------------------------------------
# Part 2: PS2-native geometry decode
# ---------------------------------------------------------------------------

@dataclass
class SaMesh:
    """One material split decoded from PS2-native tri-strip data."""
    material_index: int
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    uv:         List[Tuple[float, float]]        = field(default_factory=list)
    colors:     List[int]                         = field(default_factory=list)  # RGBA8888 packed int
    triangles:  List[Tuple[int, int, int]]        = field(default_factory=list)


@dataclass
class SaModel:
    """Decoded SA PS2-native DFF model."""
    meshes:    List[SaMesh]   = field(default_factory=list)
    materials: List[dict]     = field(default_factory=list)  # {"texture_name": str, "color": int}


def _walk_dma_chain(b: bytes, chain_start: int, chunk_end: int):
    """Walk the DMA source chain in a single PS2_NATIVE packet.

 Returns (split_pos_nums, data_block_start) where:
 split_pos_nums = per-split POS vertex count (from the V4_16 VU-dst-0 UNPACK num)
 data_block_start = byte offset of the first POS entry in the flat attribute arrays

 Chain layout (per recon docs):
 - ref tags (id=3): carry UNPACK VIFcode in v1; advance o by 16 (data is external).
 - cnt qwc=0 (id=1): NOP padding; advance by 16.
 - cnt qwc=1 (id=1): intra-chain split separator; inline data = 1 GIFtag qword;
 advance by 32; signals end of the PREVIOUS split's three ref tags.
 - ret qwc=1 (id=6): final terminator for single-packet chains; inline data = 1
 GIFtag qword at (tag_off + 16). data_block_start = tag_off + 16 + qwc*16.

 For multi-packet chains (separated by ret tags) each packet has its own data block.
 This function handles one packet and returns at the first ret encountered.

 NOTE - contiguous-block assumption: this parser treats the attribute data as a single
 flat block starting at data_block_start and ignores the `addr` ADDR fields carried in
 the ref tags (those are PS2 VU-memory addresses, not host offsets). This holds for
 all SA single-packet and multi-packet map models seen so far (bridge_1, statue, etc.).

 UNPACK vl/vn disambiguation:
 VU dst 0 (imm & 0x3FF == 0) = POS (V4_16, usn=0)
 VU dst 1 (imm & 0x3FF == 1) = UV (V2_16, usn=0)
 VU dst 2 (imm & 0x3FF == 2) = COL (V4_16, usn=1)
 We collect the POS UNPACK num for each split.
 """
    o = chain_start
    split_pos_nums: List[int] = []
    current_pos_num: int = 0
    data_block_start: int = chunk_end  # fallback

    while o + 16 <= chunk_end:
        taglo, addr, v0, v1 = struct.unpack_from("<4I", b, o)
        idv  = (taglo >> 28) & 0x7
        qwc  = taglo & 0xFFFF

        # Check VIF code in v1 for UNPACK (and v0 secondarily)
        for vifcode in (v1, v0):
            cmd = (vifcode >> 24) & 0x7F
            if (cmd & 0x60) != 0x60:
                continue
            num = (vifcode >> 16) & 0xFF
            imm = vifcode & 0xFFFF
            vu_dst = imm & 0x3FF
            if vu_dst == 0:   # POS UNPACK
                current_pos_num = num

        if idv == 1:      # cnt: advance past inline data
            inline_end = o + 16 + qwc * 16
            if qwc == 1:
                # split separator: current_pos_num is for the split just ended
                split_pos_nums.append(current_pos_num)
                current_pos_num = 0
            o = inline_end

        elif idv in (3, 4):  # ref/refs: data elsewhere, chain ptr advances by 16
            o = o + 16

        elif idv == 6:    # ret: end of this packet
            # The inline qword (qwc=1) is the final GIFtag before the data block.
            # SA invariant: qwc == 1 here. If qwc == 0 data_block_start would be
            # wrong (points to the GIFtag itself); treat as unrecognised layout.
            split_pos_nums.append(current_pos_num)
            data_block_start = o + 16 + qwc * 16
            break

        else:             # refe(0) / end(7): true terminator (multi-packet)
            split_pos_nums.append(current_pos_num)
            data_block_start = o + 16 + qwc * 16
            break

    return split_pos_nums, data_block_start


def _is_valid_pos(x: int, y: int, z: int, w: int) -> bool:
    """True if this V4_16 entry looks like real position data (not DMA padding)."""
    return w in (0, -32768) and abs(x) <= _POS_MAX and abs(y) <= _POS_MAX and abs(z) <= _POS_MAX


def _decode_pos_uv_col(b: bytes, db: int, chunk_end: int, split_pos_nums: List[int]):
    """Read the flat POS / UV / COL arrays that begin at data_block_start (db).

 Layout (all splits concatenated):
 [n_total * V4_16] POS (8 bytes/vert) <- starts at db
 [pad to qword from db]
 [n_total * V2_16] UV (4 bytes/vert)
 [pad to qword from db]
 [n_total * V4_16] COL (8 bytes/vert)

 The POS UNPACK num fields may include a few DMA-buffer padding entries at the end
 of the last split. We scan forward from db and stop at the first entry that fails
 the _is_valid_pos check, giving the true n_total.

 Returns:
 positions : list of (x_f, y_f, z_f, w_raw) w_raw = 0 (continue) or -32768 (restart)
 uvs : list of (u_f, v_f)
 colors : list of packed RGBA8888 ints
 split_vcounts : per-split true vertex counts (prefix-sum over valid pos run)
 """
    n_max = sum(split_pos_nums)
    if n_max == 0 or db >= chunk_end:
        return [], [], [], []

    # --- scan POS to find the true vertex count ---
    n_valid = 0
    for i in range(n_max):
        off = db + i * 8
        if off + 8 > chunk_end:
            break
        x, y, z, w = struct.unpack_from("<4h", b, off)
        if not _is_valid_pos(x, y, z, w):
            break
        n_valid += 1

    if n_valid == 0:
        return [], [], [], []

    # --- build true per-split vertex counts ---
    # Distribute n_valid across splits in order; the last split may be smaller.
    true_vcounts: List[int] = []
    remaining = n_valid
    for cnt in split_pos_nums:
        alloc = min(cnt, remaining)
        true_vcounts.append(alloc)
        remaining -= alloc
        if remaining <= 0:
            break

    # Pad to account for any splits beyond the valid data (edge case: multi-geometry)
    while len(true_vcounts) < len(split_pos_nums):
        true_vcounts.append(0)

    # --- read POS ---
    positions = []
    for i in range(n_valid):
        x, y, z, w = struct.unpack_from("<4h", b, db + i * 8)
        positions.append((x * _POS_SCALE, y * _POS_SCALE, z * _POS_SCALE, w))

    # --- UV array start: after the FULL POS block (n_max entries), qword-aligned from db ---
    # IMPORTANT: UV/COL are laid out after the *entire* POS buffer (n_max * 8 bytes),
    # including any DMA-padding entries beyond n_valid. Using n_valid here would give
    # wrong offsets whenever the last split has padding vertices.
    pos_bytes = n_max * 8
    uv_start = db + pos_bytes
    rem = (uv_start - db) % 16
    if rem:
        uv_start += 16 - rem

    # Read only the first n_valid UV entries (the valid verts), but the full block is n_max * 4.
    uvs = []
    for i in range(n_valid):
        off = uv_start + i * 4
        if off + 4 > chunk_end:
            uvs.append((0.0, 0.0))
            continue
        u, v = struct.unpack_from("<2h", b, off)
        uvs.append((u * _UV_SCALE, v * _UV_SCALE))

    # --- COL array start: after the FULL UV block (n_max * 4 bytes), qword-aligned from db ---
    col_start = uv_start + n_max * 4
    rem = (col_start - db) % 16
    if rem:
        col_start += 16 - rem

    colors = []
    for i in range(n_valid):
        off = col_start + i * 8
        if off + 8 > chunk_end:
            colors.append(0xFFFFFFFF)
            continue
        r, g, bl, a = struct.unpack_from("<4H", b, off)
        packed = ((r >> 8) << 24) | ((g >> 8) << 16) | ((bl >> 8) << 8) | (a >> 8)
        colors.append(packed)

    return positions, uvs, colors, true_vcounts


def _strip_to_tris_local(vcnt: int, positions_global: list, vtx_base: int):
    """Convert one split's tri-strip to mesh-local triangle indices.

 Vertex range in positions_global is [vtx_base, vtx_base + vcnt).
 Uses ADC W-bit (w != 0 ⟹ strip restart at current vertex; skip triangle).
 Winding alternates by absolute strip position k (not reset on restart).
 Degenerate triangles (repeated index) are skipped.

 This mirrors librw unconvertADC / the PoC strip_to_tris:
 for k in range(2, vcnt):
 if ADC[k]: skip
 emit (k-2, k-1, k) with alternating winding by k parity.
 """
    tris = []
    for k in range(2, vcnt):
        gi = vtx_base + k
        w  = positions_global[gi][3]
        if w != 0:  # ADC restart at this vertex -> no triangle ends here
            continue
        a  = k - 2
        bb = k - 1
        c  = k
        if a == bb or bb == c or a == c:
            continue
        if k & 1:
            tris.append((a, c, bb))
        else:
            tris.append((a, bb, c))
    return tris


def _get_material_names(blob: bytes, geo: Chunk):
    """Extract material info from MaterialList -> list of {"texture_name":str, "color":int}."""
    materials = []
    mlist = geo.find(MATERIALLIST)
    if not mlist:
        return materials
    for mat in mlist.find_all(MATERIAL):
        tex_name = ""
        color = 0  # default: no colour read
        # Read the real material colour from the Material STRUCT child.
        # The STRUCT body starts with 4 bytes of flags, then 4 bytes RGBA colour.
        mat_struct = mat.find(STRUCT)
        if mat_struct and mat_struct.size >= 8:
            r, g, b_, a = struct.unpack_from("<4B", blob, mat_struct.data_off + 4)
            color = (r << 24) | (g << 16) | (b_ << 8) | a
        tex = mat.find(TEXTURE)
        if tex:
            strs = tex.find_all(STRING)
            if strs:
                raw = blob[strs[0].data_off: strs[0].data_off + strs[0].size]
                tex_name = raw.split(b"\x00", 1)[0].decode("latin1")
        materials.append({"texture_name": tex_name, "color": color})
    return materials


def _parse_binmesh_mats(blob: bytes, ext) -> List[int]:
    """Return the per-mesh material index list from the Bin-Mesh PLG (0x050E),
 in draw order. numMeshes == number of materials with geometry; each mesh is
 one PS2-native sub-chain. Returns [] if absent/short.

 binMesh layout: u32 flags, u32 numMeshes, u32 totalIndices, then per mesh
 { u32 numIndices, i32 matIndex }. PS2 native carries no index payload."""
    if not ext:
        return []
    bm = ext.find(BINMESH_PLG)
    if not bm or bm.size < 12:
        return []
    o = bm.data_off
    try:
        _flags, num, _total = struct.unpack_from("<3I", blob, o)
    except struct.error:
        return []
    o += 12
    mats = []
    for _ in range(num):
        if o + 8 > bm.data_off + bm.size or o + 8 > len(blob):
            break
        _numidx, mat = struct.unpack_from("<2i", blob, o)
        o += 8
        mats.append(mat)
    return mats


def _is_chain_start(blob: bytes, o: int, end: int) -> bool:
    """True if a 16-byte tag at o begins a native geometry sub-chain
 (a 'ref' DMA tag carrying STCYCL + an UNPACK to VU dst 0 == POS)."""
    if o + 16 > end:
        return False
    taglo, _addr, v0, v1 = struct.unpack_from("<4I", blob, o)
    idv = (taglo >> 28) & 0x7
    cmd = (v1 >> 24) & 0x7F
    return idv == 3 and v0 == _STCYCL_TAG and (cmd & 0x60) == 0x60 and (v1 & 0x3FF) == 0


def _scan_chain_start(blob: bytes, o: int, end: int, origin: int):
    """Find the next sub-chain start at/after o (tags are 16-aligned RELATIVE TO
 the chain origin, not the file). Returns offset or None."""
    rel = (o - origin) % 16
    if rel:
        o += 16 - rel
    while o + 16 <= end:
        if _is_chain_start(blob, o, end):
            return o
        o += 16
    return None


def _data_block_end(db: int, n_max: int) -> int:
    """Byte after a sub-chain's POS+UV+COL data block (mirrors the offset math in
 _decode_pos_uv_col), i.e. where the next mesh's control/chain region begins.
 Note: vertex-lit (foliage) variants append extra attribute arrays beyond
 this; for those the scan simply fails to resync and that mesh is skipped."""
    uv_start = db + n_max * 8
    rem = (uv_start - db) % 16
    if rem:
        uv_start += 16 - rem
    col_start = uv_start + n_max * 4
    rem = (col_start - db) % 16
    if rem:
        col_start += 16 - rem
    return col_start + n_max * 8


def decode(blob) -> SaModel:
    """Decode a the source game PS2-native DFF blob into a SaModel.

 Walks all Geometry chunks → PS2_NATIVE (0x0510) extension → DMA source chain →
 flat POS/UV/COL attribute arrays → per-split tri-strip → SaMesh list.

 Chain walk: stops at the first ret (id=6) tag, which also encodes the GIFtag
 that precedes the data block. data_block_start = ret_tag_off + 16 + qwc*16.

 Attribute layout: [POS n_total×8B][pad][UV n_total×4B][pad][COL n_total×8B],
 all starting from data_block_start. Splits share one flat array in order.

 Vertex validity: trailing DMA-buffer padding entries are trimmed by scanning
 the POS array until the first entry with w not in {0,-32768} or |coord|>32767
 (full s16 range; W-bit is the primary signal).

 Triangle generation: ADC W-bit (w!=0) marks strip restart; standard alternating-
 winding tri-strip emission, skipping degenerate triangles.

 Supports static map geometry (format 0x0101002F). Multi-packet (multi-ret) models
 are partially supported: each packet is decoded independently and meshes/materials
 are concatenated (sufficient for the bridge_1 and statue test cases).
 """
    blob = bytes(blob)
    root = parse_chunks(blob)
    model = SaModel()

    geo_list = root.find(GEOMETRYLIST)
    if not geo_list:
        raise ValueError("No GeometryList in DFF")

    for geo in geo_list.find_all(GEOMETRY):
        mat_base = len(model.materials)   # global offset (multi-geometry models)
        model.materials.extend(_get_material_names(blob, geo))

        ext = geo.find(EXTENSION)
        if not ext:
            continue
        nat = ext.find(PS2_NATIVE)
        if not nat:
            continue

        chunk_end = nat.data_off + nat.size
        # The DMA source chain starts at data_off + 0x18 (skip the 24-byte sub-header).
        origin = nat.data_off + 0x18

        # PS2 native packs ONE sub-chain per Bin-Mesh mesh (== one material), each
        # ending in a ret. Decode every mesh, not just the first: the material for
        # a whole sub-chain is binMesh[mesh].matIndex (NOT the split index - splits
        # are VU-sized batches of a SINGLE material).
        mesh_mats = _parse_binmesh_mats(blob, ext)
        n_meshes = len(mesh_mats) if mesh_mats else 1

        o = origin
        for mesh_i in range(n_meshes):
            # mesh 0 starts at the chunk's chain origin (legacy path); later meshes
            # are found by scanning past the previous data block + control region.
            chain_start = origin if mesh_i == 0 else _scan_chain_start(blob, o, chunk_end, origin)
            if chain_start is None:
                break

            split_pos_nums, data_block_start = _walk_dma_chain(blob, chain_start, chunk_end)
            if not split_pos_nums:
                break

            positions, uvs, colors, true_vcounts = _decode_pos_uv_col(
                blob, data_block_start, chunk_end, split_pos_nums
            )
            if not positions:
                break   # foliage-lit variant / resync miss: stop, keep what we have

            # One material for this whole sub-chain.
            if mesh_mats and mesh_i < len(mesh_mats):
                mat = mat_base + mesh_mats[mesh_i]
            else:
                mat = mat_base
            if mat < 0 or mat >= len(model.materials):
                mat = mat_base

            # Build one SaMesh per VU split, all sharing this mesh's material.
            vtx_base = 0
            for si, vcnt in enumerate(true_vcounts):
                if vcnt == 0:
                    vtx_base += split_pos_nums[si]
                    continue

                mesh = SaMesh(material_index=mat)
                for gi in range(vtx_base, vtx_base + vcnt):
                    px, py, pz, _w = positions[gi]
                    mesh.positions.append((px, py, pz))
                    mesh.uv.append(uvs[gi] if gi < len(uvs) else (0.0, 0.0))
                    mesh.colors.append(colors[gi] if gi < len(colors) else 0xFFFFFFFF)

                mesh.triangles = _strip_to_tris_local(vcnt, positions, vtx_base)
                model.meshes.append(mesh)
                vtx_base += vcnt

            # Advance past this mesh's data block to look for the next sub-chain.
            o = _data_block_end(data_block_start, sum(split_pos_nums))

    return model
