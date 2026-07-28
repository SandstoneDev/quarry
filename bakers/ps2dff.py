#!/usr/bin/env python3
"""ps2dff - PS2 (SKY) DFF world-geometry codec for the Quarry converter.

Decodes the PS2-native RenderWare DFF layout straight from the user's disc
into plain triangles + materials, INCLUDING both prelight sets: on PS2 the
vertex stream carries day AND night colours interleaved (V4_16 colour: low
byte = day, high byte = night) - see docs/gta_sa_psp/research/ps2_dff_format.md
and tools/ps2_dff_probe.py for the discovery trail (verified byte-exact against
the PC twin of ganghous01_lax.dff; welded vertex count matches the authored
count exactly).

Scope: STATIC world geometry (the SA world pipeline: DMAref broken-out
attributes, slots XYZ/UV/RGBA). Skinned peds/vehicles use different pipelines
and land with their own phase.

API:
    model = load_dff(open(p,'rb').read())
    model.geometries[i] -> Geo(verts, uvs, day, night, tris, materials)
    model.frames[i]     -> Frame(name, rot, pos, parent, ltm_rot, ltm_pos)
    tris entries: (a, b, c, matIndex)

A clump is a FRAME TREE with atomics (geometry) hung off its nodes, and an
atomic's geometry is stored in ITS FRAME's space.  Whether that frame matters
depends on how SA loads the model, and the two paths DISAGREE:

  * ATOMIC models (IDE objs/tobj) - CFileLoader::SetRelatedModelInfoCB
    (0x537150) hands the atomic to the model info and then does
    `RpAtomicSetFrame(atomic, RwFrameCreate())`: the authored frame is THROWN
    AWAY and replaced by an identity one.  The offset an artist left on
    aw_streettree1 / lhouse_barrier* / grassplant is therefore NOT part of the
    in-game model - baking it would move those props metres off.
  * CLUMP models (IDE anim -> CClumpModelInfo + bHasAnimBlend ->
    CAnimatedBuilding) - CFileLoader::LoadClumpFile keeps the whole clump,
    frame tree included, so the frame offsets ARE real (vrocksign03 sits
    +21.4m up the vrockpole).

load_dff stays faithful to the file: vertices come out FRAME-LOCAL and each Geo
carries its frame index/name plus the composed local-to-model matrix, so the
caller applies the SA rule for its own model class (see sa_export_pmap, which
bakes/splits by IDE section).
"""
import os
import struct
import sys
from dataclasses import dataclass, field

# PS2DFF_DIAG=1 reports VIF streams that yield implausibly few vertices for their
# size - the signature of batches being walked past instead of decoded.
_DIAG = os.environ.get("PS2DFF_DIAG") == "1"

# RW chunk ids
C_STRUCT, C_STRING, C_EXT = 0x01, 0x02, 0x03
C_MATERIAL, C_MATLIST = 0x07, 0x08
C_FRAMELIST, C_GEOMETRY, C_CLUMP, C_ATOMIC, C_GEOMLIST = 0x0E, 0x0F, 0x10, 0x14, 0x1A
C_TEXTURE = 0x06
C_UVANIMDICT, C_ANIMANIM, C_UVANIMEXT = 0x2B, 0x1B, 0x135   # RpUVAnim (animated textures)
C_BINMESH, C_NATIVE = 0x50E, 0x510
C_SKIN = 0x116                 # RpSkin native-skin plugin (bone weights/indices ride the VIF stream)
C_NIGHT = 0x0253F2F9
C_FRAMENAME = 0x0253F2FE       # per-frame node-name extension (RW "frame" plugin)

IDENT_ROT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

FMT_TRISTRIP = 0x01
FMT_NATIVE = 0x01000000

# 16-bit position clusters are fixed-point, and the number of fractional bits is a
# property of the VU1 pipeline the atomic was authored for - it is NOT carried in
# the VIF UNPACK code, which only gives the element WIDTH. The world pipeline packs
# s16 9.7 (divide by 128); the vehicle pipeline packs s16 6.10 (divide by 1024).
#
# Measured, not guessed: decoding a vehicle with the world scale gives a wheel mesh
# 5.92 units across, while the engine builds its tyre from handling's wheelScale
# 0.74 (Vehicle.c: radius = wheelScale * 0.5). 5.92 / 0.74 = 8.000 exactly, and
# 8 = 1024/128. Applying the vehicle scale puts a Bravura chassis at 2.27 x 5.16 x
# 1.32 m and the tyre at 0.74 m, which are the real dimensions.
#
# Passing the wrong one does not fail loudly: the model simply comes out a uniform
# 8x too large, and because the engine fit-scales wheels from handling the tyres
# still look right, which is what hid this for so long.
POS_SCALE_WORLD = 1.0 / 128.0      # world + interiors: s16 9.7 fixed
POS_SCALE_VEHICLE = 1.0 / 1024.0   # vehicle pipeline:  s16 6.10 fixed
POS_SCALE = POS_SCALE_WORLD        # default for the world decode path
UV_SCALE = 1.0 / 4096.0        # uv: s16 12.4 fixed


def _chunks(b, off, end):
    while off + 12 <= end:
        cid, size, _ = struct.unpack_from("<III", b, off)
        if size > end - off - 12:
            return
        yield cid, off + 12, size
        off += 12 + size


def _find(b, off, end, cid):
    for c, o, s in _chunks(b, off, end):
        if c == cid:
            return o, s
    return None, 0


def _find_all(b, off, end, cid):
    return [(o, s) for c, o, s in _chunks(b, off, end) if c == cid]


def parse_uvanimdict(b):
    """RpUVAnim dict prepended at offset 0 of an ANIMATED DFF ('UVANIMDICT' 0x2B,
    before the CLUMP). Returns {anim_name.lower(): (du_dt, dv_dt)} = net UV-scroll
    rate (translation delta / duration, UV-units/sec). Empty when there is no dict
    (the common case) or the anim is a rotate/pulse (net translation ~0)."""
    if len(b) < 28 or struct.unpack_from("<I", b, 0)[0] != C_UVANIMDICT:
        return {}
    out = {}
    numAnims = struct.unpack_from("<I", b, 24)[0]        # UVANIMDICT hdr(12)+STRUCT hdr(12)+numAnims
    off = 28
    for _ in range(numAnims):
        if off + 12 > len(b):
            break
        cid, csize, _lib = struct.unpack_from("<III", b, off)
        if cid != C_ANIMANIM:
            break
        body = off + 12                                  # version,typeID,numFrames,flags,duration,pad
        numFrames = struct.unpack_from("<i", b, body + 8)[0]
        duration = struct.unpack_from("<f", b, body + 16)[0]
        name = b[body + 24:body + 56].split(b"\0")[0].decode("ascii", "replace").lower()
        kf = body + 88                                   # after name[32] + nodeToUVChannel[8]
        if numFrames >= 2 and duration > 1e-6 and kf + (numFrames - 1) * 32 + 28 <= len(b):
            x0, y0 = struct.unpack_from("<2f", b, kf + 20)                 # uv[4],uv[5] of frame 0
            xL, yL = struct.unpack_from("<2f", b, kf + (numFrames - 1) * 32 + 20)
            du = (xL - x0) / duration
            dv = (yL - y0) / duration
            if abs(du) > 1e-4 or abs(dv) > 1e-4:         # a real scroll (skip rotate/pulse)
                out[name] = (du, dv)
        off += 12 + csize
    return out


@dataclass
class Mat:
    color: tuple = (255, 255, 255, 255)
    texture: str = ""
    mask: str = ""
    uvanim_name: str = ""     # RpUVAnim (0x135) referenced anim name (dict key), or ""
    uvscroll: tuple = None    # (du_dt, dv_dt) UV-units/sec, resolved from the dict, or None


@dataclass
class Frame:
    """One node of the clump's frame tree (RW FrameList stream record).

    `rot`/`pos` are the AUTHORED local matrix (row-vector convention: a point is
    transformed v*rot + pos, exactly librw's right/up/at rows); `ltm_*` is the
    same composed down the parent chain = local-to-MODEL.  `parent` is -1 at the
    root."""
    name: str = ""
    rot: tuple = IDENT_ROT
    pos: tuple = (0.0, 0.0, 0.0)
    parent: int = -1
    ltm_rot: tuple = IDENT_ROT
    ltm_pos: tuple = (0.0, 0.0, 0.0)

    @property
    def ltm_is_identity(self):
        return (self.ltm_rot == IDENT_ROT and self.ltm_pos == (0.0, 0.0, 0.0))


@dataclass
class Geo:
    verts: list = field(default_factory=list)    # (x,y,z) floats, FRAME-local
    uvs: list = field(default_factory=list)      # (u,v) floats
    day: list = field(default_factory=list)      # (r,g,b,a)
    night: list = field(default_factory=list)    # (r,g,b,a)
    tris: list = field(default_factory=list)     # (a,b,c,matIndex)
    materials: list = field(default_factory=list)
    declared_verts: int = 0
    frame_index: int = -1        # atomic's frame, -1 = geometry with no atomic
    frame_name: str = ""         # frame node name (== the IFP sequence name when animated)
    frame_ltm: tuple = None      # (rot9, pos3) local-to-model, or None when identity
    boneIdx: list = field(default_factory=list)  # skinned peds: (b0..b3) per vert, parallel to verts
    boneW: list = field(default_factory=list)    # skinned peds: (w0..w3) per vert, parallel to verts
    skin: dict = None            # native-skin plugin: {numBones,numUsed,numWeights,used,invBind}


@dataclass
class Model:
    geometries: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # Frame[], index-parallel to the RW FrameList


def _unpack_info(cmd):
    if (cmd & 0x60) != 0x60:
        return None
    vl = cmd & 3
    if vl == 3:
        return None
    return ((cmd >> 2) & 3) + 1, 4 >> vl


def _decode_weight_word(w):
    """Decode ONE of a skinned vertex's four V4_32 skin words (librw
    genericUninstanceCB / skinUninstanceCB, ps2.cpp):

        weight = float32(w & ~0x3FF)          # low 10 bits stolen for the index
        boneIndex = ((w & 0x3FF) >> 2) - 1     # 1-based on disc; 0 => unused

    Returns (weight_float, boneIndex) with boneIndex 0-based into the geometry's
    bone-matrix array (== the sa_skin PC index space).  A zero weight means the
    slot is unused -> index 0 (a valid bone, contributes nothing at weight 0)."""
    wf = struct.unpack("<f", struct.pack("<I", w & 0xFFFFFC00))[0]
    rawidx = (w & 0x3FF) >> 2
    bi = rawidx - 1 if rawidx else 0
    if wf == 0.0:
        bi = 0
    return wf, bi


class _MeshStream:
    """Decoded per-mesh vertex stream (with duplicates, strip order)."""
    def __init__(self):
        self.pos = []        # (x,y,z) float
        self.adc = []        # bool: strip-restart flag
        self.uv = []
        self.day = []
        self.night = []
        self.bidx = []       # skinned peds only: (b0,b1,b2,b3) bone indices
        self.bw = []         # skinned peds only: (w0,w1,w2,w3) bone weights


def _parse_chain(raw, tristrip, pos_scale=POS_SCALE):
    ms = _MeshStream()
    o = 0
    first = True
    pend = []
    n = len(raw)

    def flush(nverts):
        """Emit one VU1 batch: the attributes gathered since the previous ITOP.

        `pend` is NOT cleared unless the batch actually emits - broken-out DMAref
        attributes accumulate across several DMA tags before the DMAcnt carrying
        their ITOP arrives, and clearing early would throw them away."""
        nonlocal first, pend
        # VU1 holds at most ~256 verts per batch; a larger ITOP is a desync
        # (data misread as a command) -> reject rather than read past payload.
        if nverts is None or nverts > 256 or not pend:
            return
        skip = 0 if first else (2 if tristrip else 0)
        for slot, ncomp, csz, src in pend:
            pay = raw[src[1]: src[1] + src[2]]
            asz = ncomp * csz
            for i in range(skip, nverts):
                el = pay[i * asz:(i + 1) * asz]
                if len(el) < asz:
                    break                               # truncated-tail guard
                if slot == 0:
                    if csz == 2 and ncomp >= 3:         # V3/V4_16 s16 9.7
                        vals = struct.unpack("<%dh" % ncomp, el)
                        ms.pos.append((vals[0] * pos_scale,
                                       vals[1] * pos_scale,
                                       vals[2] * pos_scale))
                        ms.adc.append(ncomp > 3 and vals[3] != 0)
                    elif csz == 4 and ncomp >= 3:       # V3/V4_32 float
                        vals = struct.unpack("<%df" % ncomp, el)
                        ms.pos.append(vals[:3])
                        ms.adc.append(ncomp > 3 and vals[3] != 0)
                    # unknown position format: emit nothing - the stream-length
                    # check downstream drops this mesh cleanly
                elif slot == 1:
                    if csz == 2 and ncomp >= 2:         # V2_16 (V4_16 = uv2 pair)
                        u, v = struct.unpack_from("<2h", el, 0)
                        ms.uv.append((u * UV_SCALE, v * UV_SCALE))
                    elif csz == 4 and ncomp >= 2:       # V2/V4_32 float
                        ms.uv.append(struct.unpack_from("<2f", el, 0))
                    else:
                        ms.uv.append((0.0, 0.0))        # keep arrays in step
                elif slot == 2:
                    if csz == 2 and ncomp == 4:         # V4_16: lo=day hi=night
                        c = struct.unpack("<4H", el)
                        ms.day.append(tuple(x & 0xFF for x in c))
                        ms.night.append(tuple(x >> 8 for x in c))
                    elif csz == 1 and ncomp == 4:       # V4_8 plain RGBA
                        ms.day.append(tuple(el[:4]))
                        ms.night.append(tuple(el[:4]))
                    else:
                        ms.day.append((255, 255, 255, 255))
                        ms.night.append((255, 255, 255, 255))
                elif slot >= 3 and ncomp == 4 and csz == 4:
                    # SKIN weights (AT_NORMAL+1): V4_32, one word per bone slot.
                    # Retail packs it at slot 3 for peds with no streamed normal
                    # (cutscene actors) or slot 4 when a V3_8 normal precedes it.
                    ww = struct.unpack("<4I", el)
                    bi = [0, 0, 0, 0]
                    bw = [0.0, 0.0, 0.0, 0.0]
                    for kk in range(4):
                        bw[kk], bi[kk] = _decode_weight_word(ww[kk])
                    ms.bidx.append(tuple(bi))
                    ms.bw.append(tuple(bw))
        first = False
        pend = []

    while o + 16 <= n:
        w0, addr, v0, v1 = struct.unpack_from("<4I", raw, o)
        did = (w0 >> 28) & 7
        qwc = w0 & 0xFFFF
        if did == 3:                                   # DMAref: one attribute
            info = _unpack_info((v1 >> 24) & 0x7F)
            if info:
                pend.append((v1 & 0xFF, info[0], info[1],
                             ("ref", addr * 16, qwc * 16)))
            o += 16
            continue
        if did in (1, 6):                              # DMAcnt/DMAret inline
            vbefore = len(ms.pos)
            inline_base = o + 16
            inline = raw[inline_base: inline_base + qwc * 16]
            # walk the inline VIF stream: small models (generic pipeline) carry
            # attribute UNPACKs INLINE here instead of broken-out DMArefs; the
            # ITOP closes the batch either way.
            #
            # One tag can carry MANY batches back to back. This used to stop at the
            # first ITOP, which silently threw away everything after it: the boxing
            # gym's room shell packs its whole 87 KB stream under a single DMAcnt and
            # decoded to 56 vertices, so the floor, ceiling and walls came out as
            # holes. Flush at every ITOP and keep walking the same buffer instead.
            k = 0
            while k + 4 <= len(inline):
                t = struct.unpack_from("<I", inline, k)[0]
                cmd = (t >> 24) & 0x7F
                if cmd == 0x04:                        # VIF_ITOP closes a batch
                    flush(t & 0xFFFF)
                    k += 4
                    continue
                if cmd in (0x30, 0x31):                # STROW/STCOL + 4 data words
                    k += 4 + 16
                    continue
                if cmd == 0x20:                        # STMASK + 1 data word (32-bit mask)
                    k += 4 + 4
                    continue
                info = _unpack_info(cmd)
                if info:
                    ncomp, csz = info
                    cnt = (t >> 16) & 0xFF
                    asz = ncomp * csz
                    pend.append((t & 0xFF, ncomp, csz,
                                 ("inline", inline_base + k + 4, asz * cnt)))
                    k += 4 + ((asz * cnt + 3) // 4) * 4   # word-aligned inline data
                    continue
                k += 4
            o += 16 + qwc * 16
            if _DIAG and qwc:
                # Starved-stream watch. A VIF stream costs ~18-40 bytes per vertex; a
                # ratio far above that means batches are being walked past, which is
                # exactly how the first-ITOP-only bug hid (87 KB -> 56 vertices, an
                # interior's floor and ceiling gone). Set PS2DFF_DIAG=1 to see it.
                # A stream with NO vertices at all is normal (VU program uploads and
                # GIF blocks carry none), so only judge streams that did produce some.
                got = len(ms.pos) - vbefore
                if got and (qwc * 16.0) / got > 120.0:
                    sys.stderr.write("ps2dff DIAG: %d B of stream yielded %d vertices\n"
                                     % (qwc * 16, got))
            if did == 6:
                # DMAret usually terminates the chain; SA meshes are single-ret
                pass
            continue
        o += 16
    return ms


def _parse_material(b, off, end):
    m = Mat()
    s_off, _ = _find(b, off, end, C_STRUCT)
    if s_off is not None:
        r, g, bb, a = struct.unpack_from("<4B", b, s_off + 4)
        m.color = (r, g, bb, a)
    t_off, t_size = _find(b, off, end, C_TEXTURE)
    if t_off is not None:
        strs = _find_all(b, t_off, t_off + t_size, C_STRING)
        if strs:
            m.texture = b[strs[0][0]:strs[0][0] + strs[0][1]].split(b"\0")[0].decode("ascii", "replace")
        if len(strs) > 1:
            m.mask = b[strs[1][0]:strs[1][0] + strs[1][1]].split(b"\0")[0].decode("ascii", "replace")
    # RpUVAnim: the material-level EXTENSION (0x03) may carry a UVANIMATION (0x135)
    # naming an anim in the DFF's leading dict -> animated (scrolling) texture.
    e_off, e_size = _find(b, off, end, C_EXT)
    if e_off is not None:
        u_off, _u = _find(b, e_off, e_off + e_size, C_UVANIMEXT)
        if u_off is not None:                                   # u_off = 0x135 DATA (past its hdr)
            mask = struct.unpack_from("<I", b, u_off + 12)[0]   # STRUCT hdr(12) then u32 mask
            if mask & 1:                                        # slot-0 anim name @ +16, char[32]
                m.uvanim_name = b[u_off + 16:u_off + 48].split(b"\0")[0].decode("ascii", "replace")
    return m


def _mat_mul(a, b):
    """row-vector 3x3 compose: v*(a.b) == (v*a)*b  (librw Matrix::mult_)."""
    return tuple(sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
                 for r in range(3) for c in range(3))


def _pt_mul(p, m, t):
    return tuple(sum(p[k] * m[k * 3 + c] for k in range(3)) + t[c] for c in range(3))


def _parse_framelist(b, fl_off, fl_size):
    """RW FrameList -> Frame[] with composed local-to-model matrices.

    Stream record (librw FrameStreamData, 56 B): float32 right[3] up[3] at[3],
    float32 pos[3], int32 parent, int32 matrixFlags - then one EXTENSION chunk
    per frame, whose 0x253F2FE node carries the frame NAME (absent = unnamed)."""
    st, st_size = _find(b, fl_off, fl_off + fl_size, C_STRUCT)
    if st is None:
        return []
    num = struct.unpack_from("<I", b, st)[0]
    if num <= 0 or st + 4 + num * 56 > fl_off + fl_size:
        return []
    frames = []
    o = st + 4
    for _ in range(num):
        rot = struct.unpack_from("<9f", b, o)
        pos = struct.unpack_from("<3f", b, o + 36)
        parent = struct.unpack_from("<i", b, o + 48)[0]
        frames.append(Frame(rot=rot, pos=pos, parent=parent))
        o += 56
    # one extension chunk per frame, in frame order
    eo = st + st_size
    for i in range(num):
        if eo + 12 > fl_off + fl_size:
            break
        cid, csize, _v = struct.unpack_from("<III", b, eo)
        if cid != C_EXT:
            break
        nm_off, nm_size = _find(b, eo + 12, eo + 12 + csize, C_FRAMENAME)
        if nm_off is not None:
            frames[i].name = b[nm_off:nm_off + nm_size].split(b"\0")[0].decode("ascii", "replace")
        eo += 12 + csize
    # compose local-to-model down the parent chain (parents always precede
    # children in a RW stream, but guard anyway so a cycle cannot hang us)
    for i, f in enumerate(frames):
        rot, pos = f.rot, f.pos
        p = f.parent
        seen = 0
        while 0 <= p < num and seen < num:
            pf = frames[p]
            rot, pos = _mat_mul(rot, pf.rot), _pt_mul(pos, pf.rot, pf.pos)
            p = pf.parent
            seen += 1
        f.ltm_rot, f.ltm_pos = rot, pos
    return frames


def _parse_atomics(b, cl_off, cl_size):
    """Clump ATOMIC records -> [(frameIndex, geometryIndex, flags)].

    Struct is 4 x int32 (frame, geometry, flags, pad) at RW >= 3.4, which every
    SA disc build is; the geometry index addresses the clump's GeomList."""
    out = []
    for a_off, a_size in _find_all(b, cl_off, cl_off + cl_size, C_ATOMIC):
        s_off, s_size = _find(b, a_off, a_off + a_size, C_STRUCT)
        if s_off is None or s_size < 12:
            continue
        fidx, gidx, flags = struct.unpack_from("<3I", b, s_off)
        out.append((fidx, gidx, flags))
    return out


def _weld_and_triangulate(streams, geo):
    """librw objUninstance semantics: weld duplicate stream verts (first-seen
    order, key pos+uv+colours+skin), then walk each mesh's strip emitting
    triangles, skipping ADC restarts and degenerates. streams: [(MeshStream,
    matIndex)].  For skinned peds the bone weights/indices are part of the weld
    key (librw findVertexSkin masks 0x10000): two verts identical in pos/uv/colour
    but bound to different bones must NOT collapse, and boneIdx/boneW stay parallel
    to geo.verts through the weld exactly as the static attributes do."""
    keymap = {}

    def emit(ms, i):
        bi = ms.bidx[i] if i < len(ms.bidx) else None
        bw = ms.bw[i] if i < len(ms.bw) else None
        key = (ms.pos[i],
               ms.uv[i] if i < len(ms.uv) else None,
               ms.day[i] if i < len(ms.day) else None,
               ms.night[i] if i < len(ms.night) else None,
               bi, bw)
        idx = keymap.get(key)
        if idx is None:
            idx = len(geo.verts)
            keymap[key] = idx
            geo.verts.append(key[0])
            geo.uvs.append(key[1] if key[1] else (0.0, 0.0))
            geo.day.append(key[2] if key[2] else (255, 255, 255, 255))
            geo.night.append(key[3] if key[3] else (255, 255, 255, 255))
            if bi is not None:               # skinned: keep skin parallel to verts
                geo.boneIdx.append(bi)
                geo.boneW.append(bw)
        return idx

    for ms, mat in streams:
        idxs = [emit(ms, i) for i in range(len(ms.pos))]
        for i in range(len(idxs) - 2):
            if ms.adc[i + 2]:
                continue                            # restart marker kills this tri
            a, b_, c = idxs[i], idxs[i + 1], idxs[i + 2]
            if a == b_ or b_ == c or a == c:
                continue                            # degenerate
            if i & 1:
                geo.tris.append((b_, a, c, mat))    # odd tris flip winding
            else:
                geo.tris.append((a, b_, c, mat))


def _read_native_skin(b, off, size):
    """RW native-skin plugin (ID_SKIN 0x116) on a PS2 geometry - librw
    ps2::readNativeSkin.  Inner STRUCT: u32 platform(==4); header[4] =
    {numBones, numUsedBones, numWeights, pad}; u8 usedBones[numUsedBones];
    f32 inverseMatrices[numBones][16]; skip 16B; skin-split data
    {i32 boneLimit, numMeshes, rleSize [, remap tables when numMeshes>0]}.

    The per-vertex weights/indices are NOT in this plugin - on PS2 they ride the
    native geometry VIF stream as the 5th attribute (see _parse_chain).  Returns
    {numBones,numUsed,numWeights,used,invBind} (invBind = numBones x 16 floats,
    mesh-space, byte-identical to the PC twin)."""
    inner_id = struct.unpack_from("<I", b, off)[0]
    assert inner_id == C_STRUCT, "native skin: no inner struct"
    p = off + 12
    platform = struct.unpack_from("<I", b, p)[0]
    p += 4
    assert platform == 4, "native skin platform %d != PS2" % platform
    numBones, numUsed, numWeights, _pad = struct.unpack_from("<4B", b, p)
    p += 4
    old_format = (numUsed == 0)                 # numUsedBones absent in <34003 files
    used = []
    if not old_format:
        used = list(struct.unpack_from("<%dB" % numUsed, b, p))
        p += numUsed
    inv = [struct.unpack_from("<16f", b, p + i * 64) for i in range(numBones)]
    if old_format:                              # fabricate the new-format fields
        numUsed = numBones
        used = list(range(numBones))
        numWeights = numWeights or 4
    return {"numBones": numBones, "numUsed": numUsed, "numWeights": numWeights,
            "used": used, "invBind": inv}


def load_dff(b, pos_scale=POS_SCALE):
    model = Model()
    uvdict = parse_uvanimdict(b)                  # RpUVAnim scroll rates (empty if none)
    cl_off, cl_size = _find(b, 0, len(b), C_CLUMP)
    if cl_off is None:
        raise ValueError("no Clump")
    fl_off, fl_size = _find(b, cl_off, cl_off + cl_size, C_FRAMELIST)
    model.frames = _parse_framelist(b, fl_off, fl_size) if fl_off is not None else []
    frame_of_geo = {}                             # geometry index -> frame index
    for fidx, gidx, _flags in _parse_atomics(b, cl_off, cl_size):
        frame_of_geo.setdefault(gidx, fidx)
    gl_off, gl_size = _find(b, cl_off, cl_off + cl_size, C_GEOMLIST)
    if gl_off is None:
        raise ValueError("no GeomList")
    for g_off, g_size in _find_all(b, gl_off, gl_off + gl_size, C_GEOMETRY):
        g_end = g_off + g_size
        geo = Geo()
        s_off, _ = _find(b, g_off, g_end, C_STRUCT)
        fmt, numTri, numVerts, _m = struct.unpack_from("<IIII", b, s_off)
        geo.declared_verts = numVerts
        if not fmt & FMT_NATIVE:
            raise ValueError("geometry is not PS2-native (unexpected on a PS2 disc)")

        ml_off, ml_size = _find(b, g_off, g_end, C_MATLIST)
        if ml_off is not None:
            for m_off, m_size in _find_all(b, ml_off, ml_off + ml_size, C_MATERIAL):
                mt = _parse_material(b, m_off, m_off + m_size)
                if mt.uvanim_name:
                    mt.uvscroll = uvdict.get(mt.uvanim_name.lower())
                geo.materials.append(mt)

        ext_off, ext_size = _find(b, g_off, g_end, C_EXT)
        ext_end = ext_off + ext_size
        sk_off, sk_size = _find(b, ext_off, ext_end, C_SKIN)
        if sk_off is not None:                  # skinned ped/actor (else world/vehicle)
            geo.skin = _read_native_skin(b, sk_off, sk_size)
        bm_off, _bs = _find(b, ext_off, ext_end, C_BINMESH)
        bmflags, numMeshes, _tot = struct.unpack_from("<III", b, bm_off)
        meshes = [struct.unpack_from("<II", b, bm_off + 12 + i * 8)
                  for i in range(numMeshes)]
        tristrip = bmflags == FMT_TRISTRIP

        nd_off, nd_size = _find(b, ext_off, ext_end, C_NATIVE)
        # the inner Struct header lies about its size (bigger than the parent --
        # an RW stream quirk), so read it directly instead of via _chunks
        inner_id = struct.unpack_from("<I", b, nd_off)[0]
        assert inner_id == C_STRUCT, "native data: no inner struct"
        p_off = nd_off + 12
        assert struct.unpack_from("<I", b, p_off)[0] == 4, "not PS2 native data"
        o = p_off + 4
        streams = []
        for nidx, mat in meshes:
            dataSize, _noFix = struct.unpack_from("<II", b, o)
            o += 8
            ms = _parse_chain(b[o:o + dataSize], tristrip, pos_scale)
            o += dataSize
            streams.append((ms, mat))
        _weld_and_triangulate(streams, geo)
        fidx = frame_of_geo.get(len(model.geometries), -1)
        if 0 <= fidx < len(model.frames):
            fr = model.frames[fidx]
            geo.frame_index = fidx
            geo.frame_name = fr.name
            if not fr.ltm_is_identity:
                geo.frame_ltm = (fr.ltm_rot, fr.ltm_pos)
        model.geometries.append(geo)
    return model


# --- gvcslib bridge -----------------------------------------------------------
# Duck-typed twins of gvcslib.sa_dff.SaMesh/SaModel so the battle exporter
# (gvcslib work/sa_export_pmap.py) can run on THIS decoder via monkey-patch:
# our DMA-tag walk survives layouts its resync-scan skips, and it carries the
# correct day colours (gvcslib's own decoder packs the HIGH bytes = the NIGHT
# set - see docs/gta_sa_psp/research/ps2_dff_format.md). Night colours ride
# along in colors_night for the upcoming .night sidecar export.
#
# Positions stay FRAME-LOCAL (== what the atomic path renders, see the module
# docstring); the frame index/name and its local-to-model matrix ride along on
# every mesh so the exporter can apply the clump rule where it applies
# (sa_export_pmap: bake for a static anim-clump atomic, split for an animated one).

# SA gives the `<model>_dam` atomic its own model-info slot (SetDamagedAtomic ->
# m_pDamagedAtomic) and only instantiates it once the prop is broken, so the
# world export drops it: 69 map models carry one, and today both twins are packed
# on top of each other (same clump origin) = duplicated tris + z-fight. Flip to
# False to get the old both-atomics behaviour back.
DROP_DAMAGED_ATOMICS = True


@dataclass
class SaMeshDuck:
    material_index: int
    positions: list = field(default_factory=list)
    uv: list = field(default_factory=list)
    colors: list = field(default_factory=list)         # RGBA8888 ints, DAY
    colors_night: list = field(default_factory=list)   # RGBA8888 ints, NIGHT
    triangles: list = field(default_factory=list)
    frame_index: int = -1        # clump frame this mesh's atomic hangs off
    frame_name: str = ""         # its node name (== IFP sequence name when animated)
    frame_ltm: tuple = None      # (rot9, pos3) local-to-model, or None when identity


@dataclass
class SaModelDuck:
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # ps2dff.Frame[] (empty on a frameless clump)


def decode_sa(blob, pos_scale=POS_SCALE):
    """gvcslib.sa_dff.decode drop-in: PS2 DFF bytes -> SaModel-shaped object."""
    m = load_dff(bytes(blob), pos_scale)
    out = SaModelDuck(frames=m.frames)
    keep = m.geometries
    if DROP_DAMAGED_ATOMICS:
        live = [g for g in keep if not g.frame_name.lower().endswith("_dam")]
        if live:                      # all-damaged (never seen) -> keep them all
            keep = live
    for geo in keep:
        base = len(out.materials)
        for mt in geo.materials:
            r, g, b_, a = mt.color
            out.materials.append({"texture_name": mt.texture,
                                  "color": (r << 24) | (g << 16) | (b_ << 8) | a,
                                  "uvscroll": mt.uvscroll})   # (du_dt,dv_dt) or None
        by_mat = {}
        for a3, b3, c3, mat in geo.tris:
            by_mat.setdefault(mat, []).append((a3, b3, c3))
        for mat, tris in sorted(by_mat.items()):
            mesh = SaMeshDuck(material_index=base + mat,
                              frame_index=geo.frame_index,
                              frame_name=geo.frame_name,
                              frame_ltm=geo.frame_ltm)
            remap = {}
            for tri in tris:
                loc = []
                for gi in tri:
                    li = remap.get(gi)
                    if li is None:
                        li = len(mesh.positions)
                        remap[gi] = li
                        mesh.positions.append(geo.verts[gi])
                        mesh.uv.append(geo.uvs[gi])
                        d = geo.day[gi]
                        n = geo.night[gi]
                        mesh.colors.append((d[0] << 24) | (d[1] << 16) | (d[2] << 8) | d[3])
                        mesh.colors_night.append((n[0] << 24) | (n[1] << 16) | (n[2] << 8) | n[3])
                    loc.append(li)
                mesh.triangles.append(tuple(loc))
            out.meshes.append(mesh)
    return out


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        m = load_dff(open(path, "rb").read())
        print(f"=== {path}")
        for i, f in enumerate(m.frames):
            print(f"  frame{i}: {f.name!r} parent={f.parent} "
                  f"ltm_pos=({f.ltm_pos[0]:.3f},{f.ltm_pos[1]:.3f},{f.ltm_pos[2]:.3f})"
                  f"{'' if f.ltm_rot == IDENT_ROT else ' [rotated]'}")
        for gi, g in enumerate(m.geometries):
            night_live = sum(1 for c in g.night if c[:3] != (0, 0, 0))
            texs = ",".join(x.texture for x in g.materials[:6])
            print(f"  geo{gi}: verts={len(g.verts)}/{g.declared_verts} "
                  f"tris={len(g.tris)} mats={len(g.materials)} [{texs}] "
                  f"night-lit verts={night_live} "
                  f"frame{g.frame_index} {g.frame_name!r}")
