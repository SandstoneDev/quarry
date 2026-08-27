"""RenderWare binary stream walker - the chunk container under DFF and TXD.

Every RW chunk is a 12-byte header followed by its body:
 u32 type | u32 size | u32 libraryID
`size` is the body length (excluding the 12-byte header). Chunks nest: a parent's
body is itself a sequence of chunks. Unknown chunks are skipped by `size`.

libraryID decodes to an RW version + build (SA retail D3D9 = 0x1803FFFF = 3.6.0.3).


Reference: librw-master/src/base.cpp (libraryIDUnpackVersion)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, Optional

_HDR = struct.Struct("<III")

# --- RW chunk type registry (the ones that matter for SA DFF/TXD/COL) ---
STRUCT = 0x01
STRING = 0x02
EXTENSION = 0x03
CAMERA = 0x05
TEXTURE = 0x06
MATERIAL = 0x07
MATERIAL_LIST = 0x08
ATOMIC_SECTION = 0x09
PLANE_SECTION = 0x0A
WORLD = 0x0B
FRAME_LIST = 0x0E
GEOMETRY = 0x0F
CLUMP = 0x10
LIGHT = 0x12
UNICODE_STRING = 0x13
ATOMIC = 0x14
TEXTURE_NATIVE = 0x15
TEXTURE_DICTIONARY = 0x16
ANIM_DATABASE = 0x17          # rwID_ANIMDATABASE (the platform SDK headers)
IMAGE = 0x18                  # rwID_IMAGE, RwImage non-native (the platform SDK headers)
SKIN_ANIMATION = 0x19         # rwID_SKINANIMATION, legacy skin anim (the platform SDK headers)
GEOMETRY_LIST = 0x1A
# 0x1B is HANIMANIMATION (RtAnimAnimation) per the platform SDK headers - the SDK name.
HANIM_ANIMATION = 0x1B
DMORPH_ANIMATION = 0x1E       # rwID_DMORPHANIMATION (the platform SDK headers) - NOT an alias of 0x1F
RIGHT_TO_RENDER = 0x1F
MT_EFFECT_NATIVE = 0x20       # rwID_MTEFFECTNATIVE (the platform SDK headers)
MT_EFFECT_DICT = 0x21         # rwID_MTEFFECTDICT (the platform SDK headers)
PI_TEXTURE_DICTIONARY = 0x23  # rwID_PITEXDICTIONARY (the platform SDK headers)
TOC = 0x24                    # rwID_TOC (the platform SDK headers)
PRT_STD_GLOBAL_DATA = 0x25    # rwID_PRTSTDGLOBALDATA (the platform SDK headers)
# --- plugin/extension ids (MAKECHUNKID the middleware vendor=1 -> 0x01xx; the platform SDK headers) ---
MORPH_PLG = 0x0105            # rwID_MORPHPLUGIN (the platform SDK headers)
MRM_PLG = 0x0111             # rwID_MRMPLUGIN, multi-res LOD mesh (the platform SDK headers)
COLLIS_PLG = 0x011D          # rwID_COLLISPLUGIN, RW-native collision (the platform SDK headers)
SKIN_PLG = 0x0116            # rwID_SKINPLUGIN (the platform SDK headers)
HANIM_PLG = 0x011E           # rwID_HANIMPLUGIN (the platform SDK headers)
USER_DATA_PLG = 0x011F       # rwID_USERDATAPLUGIN (the platform SDK headers)
MATERIAL_EFFECTS_PLG = 0x0120  # rwID_MATERIALEFFECTSPLUGIN (the platform SDK headers)
NORMAL_MAP_PLG = 0x0133      # rwID_NORMMAPPLUGIN (the platform SDK headers)
ADC_PLG = 0x0134             # rwID_ADCPLUGIN, PS2 tri-strip adjacency (the platform SDK headers)
# --- the middleware vendor=5 -> 0x05xx (registered in the platform SDK headers, not headers) ---
BIN_MESH_PLG = 0x050E
NATIVE_DATA_PLG = 0x0510
# --- the original publisher-private (vendor 0x0253F2, not in RW SDK vendor enum; community-sourced) ---
TWO_D_EFFECT = 0x0253F2F8
NIGHT_VERTEX_COLORS = 0x0253F2F9
FRAME_NAME = 0x0253F2FE

_NAMES = {
    STRUCT: "Struct", STRING: "String", EXTENSION: "Extension", CAMERA: "Camera",
    TEXTURE: "Texture", MATERIAL: "Material", MATERIAL_LIST: "MaterialList",
    ATOMIC_SECTION: "AtomicSection", PLANE_SECTION: "PlaneSection", WORLD: "World",
    FRAME_LIST: "FrameList", GEOMETRY: "Geometry", CLUMP: "Clump", LIGHT: "Light",
    UNICODE_STRING: "UnicodeString", ATOMIC: "Atomic", TEXTURE_NATIVE: "TextureNative",
    TEXTURE_DICTIONARY: "TextureDictionary", ANIM_DATABASE: "AnimDatabase",
    IMAGE: "Image", SKIN_ANIMATION: "SkinAnimation", GEOMETRY_LIST: "GeometryList",
    HANIM_ANIMATION: "HAnimAnimation", DMORPH_ANIMATION: "DMorphAnimation",
    RIGHT_TO_RENDER: "RightToRender", MT_EFFECT_NATIVE: "MTEffectNative",
    MT_EFFECT_DICT: "MTEffectDict", PI_TEXTURE_DICTIONARY: "PITextureDictionary",
    TOC: "TOC", PRT_STD_GLOBAL_DATA: "PrtStdGlobalData",
    MORPH_PLG: "MorphPLG", MRM_PLG: "MRMPLG", COLLIS_PLG: "CollisionPLG",
    SKIN_PLG: "SkinPLG", HANIM_PLG: "HAnimPLG", USER_DATA_PLG: "UserDataPLG",
    MATERIAL_EFFECTS_PLG: "MaterialEffectsPLG", NORMAL_MAP_PLG: "NormalMapPLG",
    ADC_PLG: "ADCPLG", BIN_MESH_PLG: "BinMeshPLG", NATIVE_DATA_PLG: "NativeDataPLG",
    TWO_D_EFFECT: "2dEffect", NIGHT_VERTEX_COLORS: "NightVertexColors",
    FRAME_NAME: "FrameName",
}


def type_name(t: int) -> str:
    return _NAMES.get(t, f"0x{t:08X}")


def unpack_version(lib_id: int) -> tuple[int, int]:
    """Decode a RW libraryID into (version, build). Mirrors librw base.cpp."""
    if lib_id & 0xFFFF0000:
        version = ((lib_id >> 14) & 0x3FF00) + 0x30000 | ((lib_id >> 16) & 0x3F)
        build = lib_id & 0xFFFF
    else:
        version = lib_id << 8
        build = 0
    return version, build


def pack_version(version: int, build: int = 0xFFFF) -> int:
    """Encode an RW (version, build) into a libraryID. Inverse of unpack_version.

 Mirrors librw base.cpp `libraryIDPackVersion`. For the modern (>= 0x30000)
 layout the version's high nibbles go to bits 16-31 and bits 0-5, with the
 build in the low 16 bits:

 pack_version(0x36003) -> 0x1803FFFF (SA retail D3D9, 3.6.0.3)

 Pre-0x30000 ids stored the version as `lib_id << 8`; we emit that compact form
 only when the value is too small for the packed encoding (kept for symmetry).
 """
    if version <= 0x31000:
        # legacy compact id: version was recovered as (lib_id << 8)
        return (version >> 8) & 0xFFFFFFFF
    bumped = version - 0x30000
    lib = ((bumped & 0x3FF00) << 14) | ((version & 0x3F) << 16) | (build & 0xFFFF)
    return lib & 0xFFFFFFFF


def version_string(lib_id: int) -> str:
    v, _ = unpack_version(lib_id)
    return f"{(v >> 16) & 0xF}.{(v >> 12) & 0xF}.{(v >> 8) & 0xF}.{v & 0xF}"


@dataclass
class ChunkHeader:
    type: int
    size: int
    lib_id: int
    offset: int  # where this 12-byte header starts

    @property
    def body_offset(self) -> int:
        return self.offset + 12

    @property
    def end(self) -> int:
        return self.offset + 12 + self.size

    @property
    def type_name(self) -> str:
        return type_name(self.type)

    @property
    def version(self) -> str:
        return version_string(self.lib_id)


def read_header(buf, off: int) -> ChunkHeader:
    t, size, lib = _HDR.unpack_from(buf, off)
    return ChunkHeader(t, size, lib, off)


def iter_chunks(buf, start: int, end: int) -> Iterator[ChunkHeader]:
    """Yield each chunk header in [start, end). Stops at padding (type 0) or overflow."""
    off = start
    while off + 12 <= end:
        h = read_header(buf, off)
        if h.type == 0 and h.size == 0:
            break  # sector padding
        yield h
        nxt = h.body_offset + h.size
        if nxt <= off:  # malformed: avoid infinite loop
            break
        off = nxt


def find_chunk(buf, type_: int, start: int, end: int) -> Optional[ChunkHeader]:
    for h in iter_chunks(buf, start, end):
        if h.type == type_:
            return h
    return None
