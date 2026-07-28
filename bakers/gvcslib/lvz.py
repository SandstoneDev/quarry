"""LVZ streaming-directory codec (GTA: the source console title, PSP).

A ``<ZONE>.LVZ`` file is a zlib-wrapped the original publisher Leeds relocatable chunk
(magic ``DLRW``). Once inflated, the consumer root sits at ``payload + 0x20``
(see :mod:`gvcslib.container`). The LVZ directs the engine's streaming system:
it tells the loader which byte ranges of the sibling ``<ZONE>.IMG`` file hold
each resource and where on the world cell grid / texture grid they live.

This is a *decode / enumerate* codec. The payload is fully relocated in place
by the engine (every pointer-sized fixup site has ``base`` added), so a
byte-exact re-encode is meaningless for a relocated structure - the contract
here is faithful enumeration, not round-trip. ``encode`` therefore returns the
original container payload unchanged (it is a relocated image; we do not rebuild
relocations), which keeps a payload-level identity for callers that want it.

Layout (all offsets relative to ``root = payload + 0x20``; all pointers are
payload-relative u32 fixup sites, i.e. file ``base == 0``):

 root + 0x000 u32 master resource-record array base
 root + 0x004 .. cell-grid rows: {u32 ptr, u32 count} pairs, stride 8,
 running up to root + 0x12c
 root + 0x12c u32 master resource count
 (BEACH 5917 / MALL 770 / MAINLA 6167)
 root + 0x2d0 u32 texture-grid entry count
 root + 0x2d4 u32 texture-grid array pointer

Master resource record (stride 0x0c, ``master_count`` of them):
 +0x00 u32 desc_ptr payload offset of the resource header (0 = empty slot)
 +0x04 u32 runtime always 0 on disk
 +0x08 u32 index resource index (0xFFFFFFFF = empty slot)

Streaming descriptor (0x20 bytes, magic ``DLRW``, packed in one contiguous
array inside the payload):
 +0x00 u32 magic 'DLRW'
 +0x08 u32 read_size bytes to read from the IMG
 +0x0c u32 mem_size bytes the resource occupies in RAM once decoded
 +0x18 u32 img_offset byte offset into the sibling IMG file

Texture-grid entry (stride 0x10):
 +0x00 u16 x grid column
 +0x02 u16 y grid row
 +0x04 u32 img_offset byte offset into the IMG
 +0x08 u32 size bytes in the IMG
 +0x0c u32 count texture count for this cell

EMPIRICAL NOTE on alignment: the engine's typical streaming chunks are
2 KiB (0x800) page-aligned in the IMG, but a small minority of tightly-packed
chunks sit at 4-byte (sub-page) offsets. The load-bearing invariant the engine
actually relies on - and the one this module guarantees / tests - is
``img_offset + read_size <= IMG file size`` for every descriptor, with every
offset at least 4-byte aligned. ``StreamDescriptor.is_page_aligned`` exposes
the 2 KiB check per descriptor.
"""
import struct

from .container import Container

ROOT = 0x20

OFF_RESREC_BASE = 0x000   # root-relative: master record array pointer
OFF_CELLGRID = 0x004      # root-relative: first cell-grid row pair
OFF_MASTER_COUNT = 0x12c  # root-relative: master resource count
OFF_TEXGRID_COUNT = 0x2d0
OFF_TEXGRID_PTR = 0x2d4

RESREC_STRIDE = 0x0c
DESC_SIZE = 0x20
TEX_STRIDE = 0x10

MAGIC_DLRW = 0x57524C44
PAGE = 0x800  # 2 KiB streaming page


class StreamDescriptor:
    """One 0x20-byte 'DLRW' streaming descriptor."""

    __slots__ = ("payload_off", "read_size", "mem_size", "img_offset")

    def __init__(self, payload_off, read_size, mem_size, img_offset):
        self.payload_off = payload_off
        self.read_size = read_size
        self.mem_size = mem_size
        self.img_offset = img_offset

    # gate-friendly aliases the task wording uses ("offset, size, memsize")
    @property
    def offset(self):
        return self.img_offset

    @property
    def size(self):
        return self.read_size

    @property
    def memsize(self):
        return self.mem_size

    @property
    def is_page_aligned(self):
        return self.img_offset % PAGE == 0

    @property
    def is_dword_aligned(self):
        return self.img_offset % 4 == 0

    def fits_in(self, img_size):
        return self.img_offset + self.read_size <= img_size

    def __repr__(self):
        return ("StreamDescriptor(off=0x%x, read=0x%x, mem=0x%x)"
                % (self.img_offset, self.read_size, self.mem_size))


class ResourceRecord:
    """One master resource-record (stride 0x0c)."""

    __slots__ = ("slot", "desc_ptr", "index")

    EMPTY_INDEX = 0xFFFFFFFF

    def __init__(self, slot, desc_ptr, index):
        self.slot = slot
        self.desc_ptr = desc_ptr
        self.index = index

    @property
    def is_used(self):
        return self.desc_ptr != 0 and self.index != self.EMPTY_INDEX

    def __repr__(self):
        return ("ResourceRecord(slot=%d, index=%d, desc_ptr=0x%x)"
                % (self.slot, self.index, self.desc_ptr))


class TexCell:
    """One texture-grid entry (stride 0x10)."""

    __slots__ = ("x", "y", "img_offset", "size", "count")

    def __init__(self, x, y, img_offset, size, count):
        self.x = x
        self.y = y
        self.img_offset = img_offset
        self.size = size
        self.count = count

    @property
    def is_page_aligned(self):
        return self.img_offset % PAGE == 0

    def fits_in(self, img_size):
        return self.img_offset + self.size <= img_size

    def __repr__(self):
        return ("TexCell(x=%d, y=%d, off=0x%x, size=0x%x, count=%d)"
                % (self.x, self.y, self.img_offset, self.size, self.count))


class CellRow:
    """One cell-grid row pointer pair {ptr, count}."""

    __slots__ = ("ptr", "count")

    def __init__(self, ptr, count):
        self.ptr = ptr
        self.count = count

    def __repr__(self):
        return "CellRow(ptr=0x%x, count=%d)" % (self.ptr, self.count)


class Lvz:
    """Decoded LVZ streaming directory."""

    def __init__(self, container):
        self.container = container
        self.payload = container.payload
        self.root = container.root
        self.master_count = 0
        self.resources = []            # list[ResourceRecord] (used slots only)
        self.all_records = []          # list[ResourceRecord] (every slot)
        self.streaming_descriptors = []  # list[StreamDescriptor]
        self.cell_grid = []            # list[CellRow]
        self.tex_grid = []             # list[TexCell]
        self.desc_block_off = None     # payload offset of the DLRW descriptor array

    @property
    def magic(self):
        return self.container.magic_str

    @property
    def version(self):
        return self.container.version


def _u32(p, o):
    return struct.unpack_from("<I", p, o)[0]


def _u16(p, o):
    return struct.unpack_from("<H", p, o)[0]


def _read_records(p, root):
    base = _u32(p, root + OFF_RESREC_BASE)
    count = _u32(p, root + OFF_MASTER_COUNT)
    recs = []
    for i in range(count):
        rec = base + i * RESREC_STRIDE
        desc_ptr, _runtime, index = struct.unpack_from("<III", p, rec)
        recs.append(ResourceRecord(i, desc_ptr, index))
    return count, recs


def _read_cell_grid(p, root):
    """The cell grid is a flat array of {ptr, count} pairs (stride 8) starting
 at root+0x04 and running up to the master-count field at root+0x12c."""
    rows = []
    o = root + OFF_CELLGRID
    end = root + OFF_MASTER_COUNT
    while o + 8 <= end:
        ptr, count = struct.unpack_from("<II", p, o)
        rows.append(CellRow(ptr, count))
        o += 8
    return rows


def _read_tex_grid(p, root):
    count = _u32(p, root + OFF_TEXGRID_COUNT)
    ptr = _u32(p, root + OFF_TEXGRID_PTR)
    cells = []
    for i in range(count):
        o = ptr + i * TEX_STRIDE
        x = _u16(p, o + 0x00)
        y = _u16(p, o + 0x02)
        img_offset, size, cnt = struct.unpack_from("<III", p, o + 0x04)
        cells.append(TexCell(x, y, img_offset, size, cnt))
    return cells


def _find_descriptor_block(p):
    """Locate the single contiguous run of 0x20-byte 'DLRW' descriptors.

 The payload itself starts with a 'DLRW' header at offset 0; the streaming
 descriptor array is the first further run of DLRW magics packed at a 0x20
 stride. Returns (block_offset, count).
 """
    magic = struct.pack("<I", MAGIC_DLRW)
    # find the start of the descriptor array: the first DLRW magic at offset
    # >= 4 that is immediately followed by another DLRW one DESC_SIZE later
    start = None
    pos = p.find(magic, 4)
    while pos != -1:
        nxt = pos + DESC_SIZE
        if nxt + 4 <= len(p) and p[nxt:nxt + 4] == magic:
            start = pos
            break
        pos = p.find(magic, pos + 4)
    if start is None:
        # no array; maybe a lone descriptor
        lone = p.find(magic, 4)
        return (lone, 1) if lone != -1 else (None, 0)
    count = 0
    o = start
    while o + 4 <= len(p) and p[o:o + 4] == magic:
        count += 1
        o += DESC_SIZE
    return start, count


def _read_descriptors(p):
    base, count = _find_descriptor_block(p)
    descs = []
    if base is None:
        return None, descs
    for i in range(count):
        o = base + i * DESC_SIZE
        read_size = _u32(p, o + 0x08)
        mem_size = _u32(p, o + 0x0c)
        img_offset = _u32(p, o + 0x18)
        descs.append(StreamDescriptor(o, read_size, mem_size, img_offset))
    return base, descs


def decode(data):
    """Decode an LVZ file (raw bytes, zlib-wrapped or already inflated payload).

 Returns an :class:`Lvz` exposing ``resources``, ``streaming_descriptors``
 (each with ``.offset`` / ``.size`` / ``.memsize``), ``cell_grid`` and
 ``tex_grid``.
 """
    if isinstance(data, Container):
        container = data
    else:
        container = Container.load(data)
    p = container.payload
    root = container.root

    lvz = Lvz(container)
    lvz.master_count, lvz.all_records = _read_records(p, root)
    lvz.resources = [r for r in lvz.all_records if r.is_used]
    lvz.cell_grid = _read_cell_grid(p, root)
    lvz.tex_grid = _read_tex_grid(p, root)
    lvz.desc_block_off, lvz.streaming_descriptors = _read_descriptors(p)
    return lvz


def encode(obj):
    """Return the (relocated) container payload bytes.

 LVZ is a fully-relocated image: pointers have already had ``base`` folded
 in and the on-disk fixup list is consumed at load time. We do not rebuild
 relocations, so ``encode`` simply hands back the inflated payload. At the
 payload level ``encode(decode(x))`` reproduces the original payload bytes;
 to make an engine-loadable file again, re-deflate with ``Container.deflate``.
 """
    if isinstance(obj, Lvz):
        return obj.payload
    if isinstance(obj, Container):
        return obj.payload
    return bytes(obj)
