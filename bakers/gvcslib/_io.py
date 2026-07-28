"""Little-endian byte reader/writer helpers for the console title (PSP) codecs."""
import struct


def u8(d, o):  return d[o]
def u16(d, o): return struct.unpack_from('<H', d, o)[0]
def u32(d, o): return struct.unpack_from('<I', d, o)[0]
def i16(d, o): return struct.unpack_from('<h', d, o)[0]
def f32(d, o): return struct.unpack_from('<f', d, o)[0]


class R:
    """Cursor reader over a bytes-like object."""
    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def u8(self):  v = self.d[self.p]; self.p += 1; return v
    def u16(self): v = struct.unpack_from('<H', self.d, self.p)[0]; self.p += 2; return v
    def u32(self): v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v
    def i16(self): v = struct.unpack_from('<h', self.d, self.p)[0]; self.p += 2; return v
    def f32(self): v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4; return v
    def take(self, n): v = self.d[self.p:self.p + n]; self.p += n; return v
    def at(self, off): return R(self.d, off)


class W:
    """Append-only byte writer."""
    def __init__(self):
        self.b = bytearray()

    def u8(self, v):  self.b.append(v & 0xff)
    def u16(self, v): self.b += struct.pack('<H', v & 0xffff)
    def u32(self, v): self.b += struct.pack('<I', v & 0xffffffff)
    def i16(self, v): self.b += struct.pack('<h', v)
    def f32(self, v): self.b += struct.pack('<f', v)
    def raw(self, bs): self.b += bs
    def pad_to(self, n, fill=0): self.b += bytes((fill,)) * (n - len(self.b)) if n > len(self.b) else b''
    def getvalue(self): return bytes(self.b)
    def __len__(self): return len(self.b)
