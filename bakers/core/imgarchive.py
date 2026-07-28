"""the source game IMG archive reader (VER2).

VER2 layout (the only format SA 1.0 ships):
 "VER2" (4) + u32 count + count * 32-byte directory records, then sector-aligned payload.
Directory record (32 B):
 u32 offsetSectors | u16 streamingSize | u16 sizeInArchive | char name[24]
Everything is quantized to 2048-byte sectors. Reads are lazy: the directory is
parsed once, payloads are pulled by seek (the 937 MB gta3.img is never fully read).


"""
from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from typing import List, Optional

SECTOR = 2048
_REC = struct.Struct("<IHH")  # offsetSectors, streamingSize, sizeInArchive


@dataclass
class ImgEntry:
    name: str
    offset_sectors: int
    streaming_size: int
    size_in_archive: int

    @property
    def ext(self) -> str:
        return self.name.rsplit(".", 1)[1].lower() if "." in self.name else ""

    @property
    def size_sectors(self) -> int:
        # sizeInArchive overrides streamingSize when non-zero (retail ships it 0)
        return self.size_in_archive or self.streaming_size

    @property
    def offset_bytes(self) -> int:
        return self.offset_sectors * SECTOR

    @property
    def size_bytes(self) -> int:
        return self.size_sectors * SECTOR


class ImgArchive:
    def __init__(self, path: str, version: str, entries: List[ImgEntry]):
        self.path = path
        self.version = version
        self.entries = entries
        self._fh = open(path, "rb")
        self._io_lock = threading.Lock()   # seek+read must be atomic: archives are shared across server threads
        self._by_name = {e.name.lower(): e for e in entries}
        # staged edits (applied on save_as)
        self._replaced: dict = {}     # idx -> new bytes
        self._deleted: set = set()    # idx
        self._added: List[tuple] = []  # (name, bytes)

    @property
    def count(self) -> int:
        return len(self.entries)

    @classmethod
    def open(cls, path: str) -> "ImgArchive":
        with open(path, "rb") as f:
            head = f.read(8)
            magic = head[:4]
            if magic != b"VER2":
                raise ValueError(
                    f"Unsupported IMG magic {magic!r} "
                    "(VER1 external .DIR not handled here)"
                )
            count = struct.unpack_from("<I", head, 4)[0]
            dir_bytes = f.read(count * 32)
        return cls(path, "VER2", cls._parse_dir(dir_bytes, count))

    @staticmethod
    def _parse_dir(buf: bytes, count: int) -> List[ImgEntry]:
        entries: List[ImgEntry] = []
        for i in range(count):
            off = i * 32
            offset_sectors, streaming, in_archive = _REC.unpack_from(buf, off)
            # top byte of the offset is reused by the engine for the image id; mask it off
            offset_sectors &= 0x00FFFFFF
            name = buf[off + 8: off + 32].split(b"\x00", 1)[0].decode("latin-1")
            entries.append(ImgEntry(name, offset_sectors, streaming, in_archive))
        return entries

    def extract(self, entry: ImgEntry) -> bytes:
        # one lock per archive: without it two threads interleave seek/read on
        # the shared handle and each gets the other's (truncated/foreign) bytes.
        with self._io_lock:
            self._fh.seek(entry.offset_bytes)
            return self._fh.read(entry.size_bytes)

    def find(self, name: str) -> Optional[ImgEntry]:
        return self._by_name.get(name.lower())

    def by_ext(self, ext: str) -> List[ImgEntry]:
        ext = ext.lower().lstrip(".")
        return [e for e in self.entries if e.ext == ext]

    # ---- editing (Inc.2): stage changes, then save_as a new VER2 ----
    def _idx(self, key) -> int:
        if isinstance(key, int):
            if not 0 <= key < len(self.entries):
                raise IndexError(key)
            return key
        e = self.find(key)
        if e is None:
            raise KeyError(key)
        return self.entries.index(e)

    def replace(self, key, data: bytes) -> None:
        self._replaced[self._idx(key)] = bytes(data)

    def delete(self, key) -> None:
        self._deleted.add(self._idx(key))

    def add(self, name: str, data: bytes) -> None:
        self._added.append((name, bytes(data)))

    @property
    def is_dirty(self) -> bool:
        return bool(self._replaced or self._deleted or self._added)

    def save_as(self, out_path: str) -> None:
        """Write a new VER2 archive with staged edits. Unchanged payloads are
 stream-copied from the source by seek (the 937 MB source is never fully read)."""
        if os.path.abspath(out_path) == os.path.abspath(self.path):
            raise ValueError("save_as target must differ from the source path")
        items = []
        for i, e in enumerate(self.entries):
            if i in self._deleted:
                continue
            if i in self._replaced:
                d = self._replaced[i]
                items.append((e.name, len(d), ("bytes", d)))
            else:
                items.append((e.name, e.size_bytes, ("copy", self.path, e.offset_bytes, e.size_bytes)))
        for name, d in self._added:
            items.append((name, len(d), ("bytes", d)))
        _write_img(out_path, items)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ImgArchive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- VER2 writer ----------------------------------------------------------
def _write_img(out_path: str, items: List[tuple]) -> None:
    """items: list of (name, size_bytes, source) where source is
 ('bytes', data) or ('copy', src_path, offset_bytes, nbytes)."""
    count = len(items)
    header_bytes = 8 + count * 32
    header_sectors = (header_bytes + SECTOR - 1) // SECTOR
    recs = []
    cur = header_sectors
    for name, size, _src in items:
        ssec = (size + SECTOR - 1) // SECTOR
        recs.append((cur, ssec, name))
        cur += ssec
    with open(out_path, "wb") as out:
        out.write(b"VER2" + struct.pack("<I", count))
        for off, ssec, name in recs:
            nm = name.encode("latin-1")[:23]
            out.write(struct.pack("<IHH", off, ssec, 0) + nm + b"\x00" * (24 - len(nm)))
        out.write(b"\x00" * (header_sectors * SECTOR - header_bytes))
        for (name, size, src), (off, ssec, _n) in zip(items, recs):
            if src[0] == "bytes":
                out.write(src[1])
                pad = ssec * SECTOR - len(src[1])
            else:
                _, sp, so, nb = src
                with open(sp, "rb") as fin:
                    fin.seek(so)
                    remaining = nb
                    while remaining > 0:
                        chunk = fin.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                pad = ssec * SECTOR - nb
            if pad > 0:
                out.write(b"\x00" * pad)


def build_img(out_path: str, entries: List[tuple]) -> None:
    """Build a VER2 .img from [(name, data_bytes), ...]."""
    _write_img(out_path, [(name, len(data), ("bytes", data)) for name, data in entries])
