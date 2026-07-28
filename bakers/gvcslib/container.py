"""Shared the original publisher Leeds relocatable 'chunk' container.

Used identically (verified) by *.LVZ, PSP/GAME.DTZ, GTA3PSP.IMG, and *.XTX, plus the
0x20-byte streaming descriptors embedded in a relocated LVZ.

Layout of the 0x20-byte header (at the START of the *inflated* payload for zlib-wrapped
files; at file offset 0 for uncompressed ones like XTX):
    +0x00 u32 magic        'DLRW'=0x57524C44 | 'GTAG'=0x47544147 | 'tex\\0'=0x00746578
    +0x04 u32 version       (LVZ=0, DTZ=1)
    +0x08 u32 total_size    == len(payload) exactly
    +0x0c u32 payload_size
    +0x10 u32 reloc_off     (== payload_size; u32 fixup-site list follows the payload)
    +0x14 u32 reloc_count
    +0x18 u32 import_off
    +0x1c u16 import_count1 ; +0x1e u16 import_count2

Relocation (engine FUN_0022d780): for each u32 site offset E in the reloc list,
*(payload+E) += base.  The consumer root is payload+0x20.

NOTE on round-trip: for zlib-wrapped containers the byte-exact gate is at the *payload*
level (decode/encode of the inflated payload), not the compressed file - zlib output is
not guaranteed identical. Re-deflate with .deflate() produces an engine-loadable file.
"""
import struct
import zlib

from ._io import u16, u32

MAGIC_DLRW = 0x57524C44
MAGIC_GTAG = 0x47544147
MAGIC_TEX = 0x00746578
MAGICS = {MAGIC_DLRW: 'DLRW', MAGIC_GTAG: 'GTAG', MAGIC_TEX: 'tex'}

HDR_SIZE = 0x20


class Container:
    def __init__(self, payload):
        self.payload = bytes(payload)
        (self.magic, self.version, self.total_size, self.payload_size,
         self.reloc_off, self.reloc_count) = struct.unpack_from('<6I', self.payload, 0)
        self.import_off = u32(self.payload, 0x18)
        self.import_count1 = u16(self.payload, 0x1c)
        self.import_count2 = u16(self.payload, 0x1e)
        self.root = HDR_SIZE  # payload offset of the relocated object root

    @classmethod
    def load(cls, data):
        data = bytes(data)
        if data[:2] == b'\x78\xda' or data[:2] == b'\x78\x9c' or data[:2] == b'\x78\x01':
            payload = zlib.decompress(data)
        else:
            payload = data
        return cls(payload)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rb') as f:
            return cls.load(f.read())

    @property
    def magic_str(self):
        return MAGICS.get(self.magic, hex(self.magic))

    def reloc_sites(self):
        return [u32(self.payload, self.reloc_off + i * 4) for i in range(self.reloc_count)]

    def check(self):
        assert self.total_size == len(self.payload), (self.total_size, len(self.payload))
        return True

    def deflate(self):
        """Re-compress the payload into an engine-loadable file (NOT byte-equal to original)."""
        return zlib.compress(self.payload, 9)
