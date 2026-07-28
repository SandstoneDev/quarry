"""the source game IFP animation package decoder (ANPK / ANP2 / ANP3).

One IFP file = one named anim *block* (e.g. 'ped') holding many named animation
clips; each clip holds one *sequence* per animated bone; each sequence is an
array of keyframes (quaternion rotation + optional translation + frame time).
SA ships the **ANP3** packed/compressed form (s16-quantized quaternions x4096);
ANP2 is the uncompressed packed form and ANPK is the legacy GTA3/VC RW-chunked
form. This module decodes all three: the anim/bone listing + frame counts always,
plus full keyframe values (quaternion + optional translation + time).

Variant seen in the shipping SA files (PED.IFP, anim.img/*.ifp): **ANP3**.

 (confirmed: CAnimManager::LoadAnimFile 0x4df270,
read order from source lines 205-216; keyTypeCode->stride table; quat conjugate;
s16/4096 dequant ).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# keyTypeCode -> (hasRot, hasTrans, isCompressed, on-disk stride). Confirmed A.4.
# 1 = rot-only uncompressed (20B: 4f quat + f time)
# 2 = rot+trans uncompressed (32B: 4f quat + f time + 3f trans)
# 3 = rot-only compressed (10B: 4h quat + h time)
# 4 = rot+trans compressed (16B: 4h quat + h time + 3h trans)
_KEYTYPE = {
    1: (True, False, False, 20),
    2: (True, True, False, 32),
    3: (True, False, True, 10),
    4: (True, True, True, 16),
}

# s16 dequant scale for quaternion components (and packed translations): 1/4096.
_QUAT_SCALE = 1.0 / 4096.0
_TRANS_SCALE = 1.0 / 4096.0   # packed (ANP3 type 4) translations share the 1/4096 scale
_TIME_SCALE = 1.0 / 4096.0    # packed compressed time s16 -> seconds-ish

# legacy keyframe sub-fourccs and their float source-frame strides (B / ANPK form)
_LEGACY_KF = {
    b"KR00": (True, False, False, 20),   # 4f quat + f time
    b"KRT0": (True, True, False, 32),    # 4f quat + 3f trans + f time
    b"KRTS": (True, True, False, 44),    # 4f quat + 3f trans + 3f scale(discarded) + f time
}

# table-driven CRC32 over the UPPERCASED name:
# standard reflected CRC32 (poly 0xEDB88320), init 0xFFFFFFFF, NO final xor, name
# uppercased first. Used for hashKey so names round-trip to the runtime.
_CRC_TABLE = []


def _build_crc_table():
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if (c & 1) else (c >> 1)
        _CRC_TABLE.append(c)


_build_crc_table()


def crc32_upper(name: str) -> int:
    """CKeyGen::GetUppercaseKey: table-driven CRC32 over the uppercased name."""
    crc = 0xFFFFFFFF
    for ch in name.upper():
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ ord(ch)) & 0xFF]
    return crc & 0xFFFFFFFF


def _cstr(buf, off, n) -> str:
    """Latin-1 string from a NUL-padded fixed slot (defensive: clamps to buffer)."""
    return buf[off:off + n].split(b"\x00", 1)[0].decode("latin-1", "replace")


# --------------------------- dataclasses ---------------------------

@dataclass
class Keyframe:
    time: float
    rot: List[float]                       # quaternion [x, y, z, w] (authored sign)
    trans: Optional[List[float]] = None    # [x, y, z] or None for rot-only


@dataclass
class Sequence:
    """One animated-bone track within an animation clip."""
    name: str                              # bone name
    type: int                              # keyTypeCode 1..4
    num_frames: int
    bone_tag: int                          # RpHAnim node id (s32; -1 if none)
    has_rot: bool
    has_trans: bool
    compressed: bool
    hash_key: int                          # CRC32(upper bone name)
    keyframes: List[Keyframe] = field(default_factory=list)


@dataclass
class Animation:
    """One named animation clip (a set of per-bone sequences)."""
    name: str
    num_sequences: int
    hash_key: int                          # CRC32(upper anim name)
    sequences: List[Sequence] = field(default_factory=list)
    total_keyframe_bytes: int = 0          # ANP3 per-anim header; sum(numFrames*stride)
    flags: int = 0                         # ANP3 per-anim flags (bit0 = trans compressed)


@dataclass
class Ifp:
    version: str                           # 'ANPK' | 'ANP2' | 'ANP3'
    block_name: str                        # CAnimBlock.name (first 16 bytes of slot)
    compressed: bool                       # version == 'ANP3'
    animations: List[Animation]


# --------------------------- keyframe decode ---------------------------

def _decode_compressed(buf, off, hastrans) -> Keyframe:
    qx, qy, qz, qw, t = struct.unpack_from("<5h", buf, off)
    rot = [-qx * _QUAT_SCALE, -qy * _QUAT_SCALE, -qz * _QUAT_SCALE, qw * _QUAT_SCALE]
    time = t * _TIME_SCALE
    trans = None
    if hastrans:
        tx, ty, tz = struct.unpack_from("<3h", buf, off + 10)
        trans = [tx * _TRANS_SCALE, ty * _TRANS_SCALE, tz * _TRANS_SCALE]
    return Keyframe(time, rot, trans)


def _decode_uncompressed(buf, off, hastrans) -> Keyframe:
    qx, qy, qz, qw = struct.unpack_from("<4f", buf, off)
    rot = [-qx, -qy, -qz, qw]
    if hastrans:
        # code 2: 4f quat, f time, 3f trans
        time = struct.unpack_from("<f", buf, off + 16)[0]
        tx, ty, tz = struct.unpack_from("<3f", buf, off + 20)
        return Keyframe(time, rot, [tx, ty, tz])
    time = struct.unpack_from("<f", buf, off + 16)[0]
    return Keyframe(time, rot, None)


def _decode_keyframes(buf, off, num_frames, key_type) -> List[Keyframe]:
    """Decode num_frames keyframes of the given keyTypeCode starting at off.

 Quaternion x,y,z are UN-NEGATED here (the loader stores the conjugate; we
 reverse it to recover the authored quaternion). A re-encoder must re-negate.
 """
    has_rot, has_trans, compressed, stride = _KEYTYPE[key_type]
    out: List[Keyframe] = []
    end = len(buf)
    for i in range(num_frames):
        fo = off + i * stride
        if fo + stride > end:
            break  # defensive: truncated blob, keep what decoded
        if compressed:
            out.append(_decode_compressed(buf, fo, has_trans))
        else:
            out.append(_decode_uncompressed(buf, fo, has_trans))
    return out


# --------------------------- packed parser (ANP2 / ANP3) ---------------------------

def _parse_packed(buf, fourcc) -> Ifp:
    compressed = (fourcc == b"ANP3")
    off = 8  # 8-byte file header already probed

    block_name = _cstr(buf, off, 24)[:16]
    off += 24
    num_anims = struct.unpack_from("<I", buf, off)[0]
    off += 4

    anims: List[Animation] = []
    for _ in range(num_anims):
        try:
            off = _parse_packed_anim(buf, off, compressed, anims)
        except (struct.error, IndexError):
            # one corrupt anim header must not kill the whole file: stop cleanly
            break
    return Ifp(fourcc.decode("latin-1"), block_name, compressed, anims)


def _parse_packed_anim(buf, off, compressed, anims) -> int:
    anim_name = _cstr(buf, off, 24)
    off += 24
    num_seq = struct.unpack_from("<I", buf, off)[0]
    off += 4
    total_kf = 0
    flags = 0
    if compressed:  # ANP3 adds totalKeyframeBytes + flags
        total_kf = struct.unpack_from("<I", buf, off)[0]
        off += 4
        flags = struct.unpack_from("<I", buf, off)[0]
        off += 4

    anim = Animation(anim_name, num_seq, crc32_upper(anim_name),
                     total_keyframe_bytes=total_kf, flags=flags)

    for _ in range(num_seq):
        # EXACT loader read order (A.3): name(24), keyType(u32), numFrames(u32), boneTag(s32)
        seq_name = _cstr(buf, off, 24)
        off += 24
        key_type, num_frames = struct.unpack_from("<II", buf, off)
        off += 8
        bone_tag = struct.unpack_from("<i", buf, off)[0]
        off += 4

        if key_type in _KEYTYPE:
            has_rot, has_trans, is_comp, stride = _KEYTYPE[key_type]
            kfs = []
            try:
                kfs = _decode_keyframes(buf, off, num_frames, key_type)
            except (struct.error, IndexError):
                kfs = []  # bad record: keep the listing, drop its frames
            off += num_frames * stride
            anim.sequences.append(Sequence(
                seq_name, key_type, num_frames, bone_tag,
                has_rot, has_trans, is_comp, crc32_upper(seq_name), kfs))
        else:
            # unknown keyType: we cannot know the stride -> record the bone and stop
            # walking this anim's sequences (continuing would mis-align the cursor).
            anim.sequences.append(Sequence(
                seq_name, key_type, num_frames, bone_tag,
                False, False, False, crc32_upper(seq_name), []))
            break

    anims.append(anim)
    return off


# --------------------------- legacy parser (ANPK, RW-chunked) ---------------------------

def _round4(n: int) -> int:
    return (n + 3) & ~3


def _read_chunk(buf, off):
    """Read one RW chunk {fourcc[4], u32 size}; body size is rounded up to 4.

 Returns (fourcc, body_offset, body_size_padded, next_off) or None at EOF.
 """
    if off + 8 > len(buf):
        return None
    fourcc = buf[off:off + 4]
    size = struct.unpack_from("<I", buf, off + 4)[0]
    body = off + 8
    padded = _round4(size)
    return fourcc, body, padded, body + padded


def _parse_legacy(buf) -> Ifp:
    # ANPK { INFO{numAnims, blockName} per-anim: NAME, DGAN{INFO{numSeq}},
    # per-seq: CPAN{ ANIM{boneName,numFrames,...,[tag]}, KR00|KRT0|KRTS } }
    anims: List[Animation] = []
    block_name = ""
    # body of the ANPK file starts right after the 8-byte file header
    inner_off = 8
    end = len(buf)

    # INFO
    ch = _read_chunk(buf, inner_off)
    num_anims = 0
    if ch and ch[0] == b"INFO":
        _, b, _sz, inner_off = ch
        num_anims = struct.unpack_from("<I", buf, b)[0]
        block_name = _cstr(buf, b + 4, 24)[:16]

    for _ in range(num_anims):
        if inner_off >= end:
            break
        try:
            inner_off, anim = _parse_legacy_anim(buf, inner_off, end)
        except (struct.error, IndexError):
            break
        if anim is not None:
            anims.append(anim)
    return Ifp("ANPK", block_name, False, anims)


def _parse_legacy_anim(buf, off, end):
    anim_name = ""
    num_seq = 0

    ch = _read_chunk(buf, off)
    if ch and ch[0] == b"NAME":
        _, b, _sz, off = ch
        anim_name = _cstr(buf, b, _sz)

    ch = _read_chunk(buf, off)
    seq_area_off = off
    if ch and ch[0] == b"DGAN":
        fourcc, dgan_body, dgan_sz, off = ch
        # DGAN body holds an INFO chunk with numSequences
        sub = _read_chunk(buf, dgan_body)
        if sub and sub[0] == b"INFO":
            _, ib, _isz, _ = sub
            num_seq = struct.unpack_from("<I", buf, ib)[0]
        seq_area_off = off

    anim = Animation(anim_name, num_seq, crc32_upper(anim_name))

    cur = seq_area_off
    for _ in range(num_seq):
        ch = _read_chunk(buf, cur)
        if not ch or ch[0] != b"CPAN":
            break
        _, cpan_body, cpan_sz, cur = ch
        seq = _parse_legacy_sequence(buf, cpan_body, cpan_body + cpan_sz)
        if seq is not None:
            anim.sequences.append(seq)
    anim.num_sequences = len(anim.sequences) or num_seq
    return cur, anim


def _parse_legacy_sequence(buf, off, end):
    ch = _read_chunk(buf, off)
    if not ch or ch[0] != b"ANIM":
        return None
    _, anim_body, anim_sz, after_anim = ch
    bone_name = _cstr(buf, anim_body, 24)
    # ANIM body: boneName[24], u32 numFrames, u32 ?, u32 ?, [s32 tag at +0x28 if sz==0x2C]
    num_frames = struct.unpack_from("<I", buf, anim_body + 24)[0]
    bone_tag = -1
    if anim_sz >= 0x2C:
        bone_tag = struct.unpack_from("<i", buf, anim_body + 0x28)[0]

    # one keyframe sub-chunk follows: KR00 / KRT0 / KRTS
    ch = _read_chunk(buf, after_anim)
    key_type = 1
    has_trans = False
    keyframes: List[Keyframe] = []
    if ch and ch[0] in _LEGACY_KF:
        kf_fourcc, kf_body, kf_sz, _ = ch
        has_rot, has_trans, _comp, stride = _LEGACY_KF[kf_fourcc]
        key_type = 2 if has_trans else 1   # legacy is float => map to uncompressed code
        keyframes = _decode_legacy_keyframes(buf, kf_body, num_frames, kf_fourcc)

    return Sequence(
        bone_name, key_type, num_frames, bone_tag,
        True, has_trans, False, crc32_upper(bone_name), keyframes)


def _decode_legacy_keyframes(buf, off, num_frames, kf_fourcc) -> List[Keyframe]:
    _has_rot, has_trans, _comp, stride = _LEGACY_KF[kf_fourcc]
    out: List[Keyframe] = []
    end = len(buf)
    for i in range(num_frames):
        fo = off + i * stride
        if fo + stride > end:
            break
        qx, qy, qz, qw = struct.unpack_from("<4f", buf, fo)
        rot = [-qx, -qy, -qz, qw]
        trans = None
        if kf_fourcc == b"KR00":
            time = struct.unpack_from("<f", buf, fo + 16)[0]
        elif kf_fourcc == b"KRT0":
            tx, ty, tz = struct.unpack_from("<3f", buf, fo + 16)
            trans = [tx, ty, tz]
            time = struct.unpack_from("<f", buf, fo + 28)[0]
        else:  # KRTS: quat, trans, scale(discard), time
            tx, ty, tz = struct.unpack_from("<3f", buf, fo + 16)
            trans = [tx, ty, tz]
            time = struct.unpack_from("<f", buf, fo + 40)[0]
        out.append(Keyframe(time, rot, trans))
    return out


# --------------------------- public API ---------------------------

def parse_ifp(data: bytes) -> Dict:
    """Parse an IFP (ANPK/ANP2/ANP3) into a JSON-shaped dict.

 Returns {version, block_name, compressed,
 anims:[{name, total_keyframe_bytes, flags,
 bones:[{name, type, num_frames, bone_tag,
 has_rot, has_trans, compressed,
 keyframes:[{time, rot:[x,y,z,w], trans:[x,y,z]?}]}]}]}.
 The anim/bone listing + frame counts are always populated; keyframe values are
 decoded too (rot quaternion in authored sign, optional translation, time).
 """
    if len(data) < 8:
        raise ValueError("IFP too small for an 8-byte header")
    fourcc = data[:4]
    if fourcc in (b"ANP2", b"ANP3"):
        ifp = _parse_packed(data, fourcc)
    elif fourcc == b"ANPK":
        ifp = _parse_legacy(data)
    else:
        raise ValueError(f"not an IFP file (bad magic {fourcc!r})")
    return _ifp_to_dict(ifp)


def _kf_dict(kf: Keyframe) -> Dict:
    d = {"time": float(kf.time), "rot": [float(c) for c in kf.rot]}
    if kf.trans is not None:
        d["trans"] = [float(c) for c in kf.trans]
    return d


def _seq_dict(seq: Sequence, with_keyframes: bool) -> Dict:
    d = {
        "name": seq.name,
        "type": int(seq.type),
        "num_frames": int(seq.num_frames),
        "bone_tag": int(seq.bone_tag),
        "has_rot": bool(seq.has_rot),
        "has_trans": bool(seq.has_trans),
        "compressed": bool(seq.compressed),
        "hash_key": int(seq.hash_key),
    }
    if with_keyframes:
        d["keyframes"] = [_kf_dict(k) for k in seq.keyframes]
    return d


def _ifp_to_dict(ifp: Ifp) -> Dict:
    return {
        "version": ifp.version,
        "block_name": ifp.block_name,
        "compressed": bool(ifp.compressed),
        "anims": [
            {
                "name": a.name,
                "hash_key": int(a.hash_key),
                "num_sequences": int(a.num_sequences),
                "total_keyframe_bytes": int(a.total_keyframe_bytes),
                "flags": int(a.flags),
                "bones": [_seq_dict(s, with_keyframes=True) for s in a.sequences],
            }
            for a in ifp.animations
        ],
    }


def to_json(parsed: Dict, include_keyframes: bool = False) -> Dict:
    """Project a parsed IFP into a UI-friendly anim-list JSON view.

 Every leaf is a JSON primitive (str/int/float/bool/None) so the web server can
 hand it straight to the UI. By default keyframe arrays are dropped (the anim
 list only needs name/type/frame-count per bone); pass include_keyframes=True
 to keep the decoded rot/trans/time arrays.
 """
    file_info = {
        "version": parsed["version"],
        "block_name": parsed["block_name"],
        "compressed": bool(parsed["compressed"]),
    }
    anims = []
    for a in parsed["anims"]:
        bones = []
        for b in a["bones"]:
            bone = {
                "name": b["name"],
                "type": int(b["type"]),
                "num_frames": int(b["num_frames"]),
                "bone_tag": int(b["bone_tag"]),
                "has_rot": bool(b["has_rot"]),
                "has_trans": bool(b["has_trans"]),
                "compressed": bool(b["compressed"]),
                "hash_key": int(b["hash_key"]),
            }
            if include_keyframes and "keyframes" in b:
                bone["keyframes"] = b["keyframes"]
            bones.append(bone)
        anims.append({
            "name": a["name"],
            "hash_key": int(a["hash_key"]),
            "num_sequences": int(a["num_sequences"]),
            "total_keyframe_bytes": int(a["total_keyframe_bytes"]),
            "flags": int(a["flags"]),
            "bones": bones,
        })
    return {**file_info, "file": file_info, "anims": anims}
