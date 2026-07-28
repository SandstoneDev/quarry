"""VFS path-hash codec (GTA: the source console title, PSP).

The game's virtual file system (the disc / RUNDATA archive index) addresses
every packed resource by a 32-bit hash of its canonical path rather than by a
string. The hash is Bob Jenkins' classic ``lookup2`` (the 1996/97 ``hash()``
function, predecessor of ``hashlittle``), seeded with the IEEE CRC-32
polynomial ``0x04C11DB7``.

The VFS index table is a flat array of ``{u32 hash, u32 lbn, u32 size}``
records kept sorted ascending by ``hash`` so the loader can binary-search a
path's hash to find its logical block number (LBN) and byte size on the UMD.

Path normalization (applied before hashing):
 * backslash ``\\`` -> forward slash ``/``
 * lowercase -> UPPERCASE
 * trailing spaces trimmed

Verified anchor::

 path_hash(normalize('DISC0:/PSP_GAME/USRDIR/RUNDATA/PSP/GAME.DTZ'),
 0x04C11DB7) == 0x3a344db9

lookup2 ``mix`` shift schedule {13, 8, 13, 12, 16, 5, 3, 10, 15}.
"""
import struct

U32 = 0xFFFFFFFF

DEFAULT_SEED = 0x04C11DB7
RECORD_FMT = "<III"      # hash, lbn, size
RECORD_SIZE = 12


def normalize(path):
    """Canonicalize a VFS path before hashing.

 Backslashes become forward slashes, the whole string is upper-cased, and
 trailing spaces are trimmed. Returns a ``str``.
 """
    s = path.replace("\\", "/")
    s = s.upper()
    s = s.rstrip(" ")
    return s


def _mix(a, b, c):
    """Bob Jenkins lookup2 ``mix`` macro (32-bit wrapping arithmetic)."""
    a = (a - b - c) & U32; a ^= c >> 13
    b = (b - c - a) & U32; b ^= (a << 8) & U32
    c = (c - a - b) & U32; c ^= b >> 13
    a = (a - b - c) & U32; a ^= c >> 12
    b = (b - c - a) & U32; b ^= (a << 16) & U32
    c = (c - a - b) & U32; c ^= b >> 5
    a = (a - b - c) & U32; a ^= c >> 3
    b = (b - c - a) & U32; b ^= (a << 10) & U32
    c = (c - a - b) & U32; c ^= b >> 15
    return a & U32, b & U32, c & U32


def path_hash(path, seed=DEFAULT_SEED):
    """Bob Jenkins lookup2 hash of ``path`` (already-normalized ``str`` or
 raw ``bytes``), seeded with ``seed`` (the engine uses ``0x04C11DB7``).

 Returns a 32-bit unsigned int. Callers that want the canonical VFS hash
 should pass ``normalize(path)``.
 """
    if isinstance(path, str):
        data = path.encode("latin-1")
    else:
        data = bytes(path)

    length = len(data)
    a = b = 0x9E3779B9
    c = seed & U32

    # main loop: consume 12-byte little-endian blocks
    i = 0
    n = length
    while n >= 12:
        a = (a + struct.unpack_from("<I", data, i)[0]) & U32
        b = (b + struct.unpack_from("<I", data, i + 4)[0]) & U32
        c = (c + struct.unpack_from("<I", data, i + 8)[0]) & U32
        a, b, c = _mix(a, b, c)
        i += 12
        n -= 12

    # tail: fold remaining 0..11 bytes; c picks up the length (lookup2 quirk:
    # the length is added to c *before* the tail switch).
    c = (c + length) & U32
    # remaining bytes from i.. i+n-1
    if n >= 11: c = (c + (data[i + 10] << 24)) & U32
    if n >= 10: c = (c + (data[i + 9] << 16)) & U32
    if n >= 9:  c = (c + (data[i + 8] << 8)) & U32
    # byte data[i+8] would go into c's low byte but that slot is reserved
    if n >= 8:  b = (b + (data[i + 7] << 24)) & U32
    if n >= 7:  b = (b + (data[i + 6] << 16)) & U32
    if n >= 6:  b = (b + (data[i + 5] << 8)) & U32
    if n >= 5:  b = (b + data[i + 4]) & U32
    if n >= 4:  a = (a + (data[i + 3] << 24)) & U32
    if n >= 3:  a = (a + (data[i + 2] << 16)) & U32
    if n >= 2:  a = (a + (data[i + 1] << 8)) & U32
    if n >= 1:  a = (a + data[i]) & U32

    a, b, c = _mix(a, b, c)
    return c & U32


def vfs_hash(path, seed=DEFAULT_SEED):
    """Convenience: normalize then hash."""
    return path_hash(normalize(path), seed)


class VfsRecord:
    """One VFS index record: {hash, lbn, size}."""

    __slots__ = ("hash", "lbn", "size", "path")

    def __init__(self, hash_, lbn, size, path=None):
        self.hash = hash_ & U32
        self.lbn = lbn & U32
        self.size = size & U32
        self.path = path        # optional source path (not serialized)

    def to_bytes(self):
        return struct.pack(RECORD_FMT, self.hash, self.lbn, self.size)

    def __repr__(self):
        return ("VfsRecord(hash=0x%08x, lbn=%d, size=%d)"
                % (self.hash, self.lbn, self.size))


class VfsTable:
    """A VFS path-hash index table.

 Records are kept sorted ascending by ``hash`` at all times so the table can
 be binary-searched (and serialized) exactly as the engine expects.
 """

    def __init__(self, seed=DEFAULT_SEED):
        self.seed = seed & U32
        self._records = []      # list[VfsRecord], maintained hash-sorted

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self):
        return list(self._records)

    def hash_of(self, path):
        """Canonical VFS hash for ``path`` under this table's seed."""
        return path_hash(normalize(path), self.seed)

    def add(self, path, lbn, size):
        """Hash ``path`` and insert a record, keeping the table hash-sorted.

 Returns the created :class:`VfsRecord`.
 """
        h = self.hash_of(path)
        rec = VfsRecord(h, lbn, size, path=path)
        # insertion sort by hash (stable for equal hashes -> insertion order)
        lo, hi = 0, len(self._records)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._records[mid].hash <= h:
                lo = mid + 1
            else:
                hi = mid
        self._records.insert(lo, rec)
        return rec

    def _find_index(self, h):
        """Index of the first record whose hash == ``h`` (binary search), or -1."""
        lo, hi = 0, len(self._records)
        while lo < hi:
            mid = (lo + hi) // 2
            mh = self._records[mid].hash
            if mh < h:
                lo = mid + 1
            elif mh > h:
                hi = mid
            else:
                # walk back to the first record with this hash
                while mid > 0 and self._records[mid - 1].hash == h:
                    mid -= 1
                return mid
        return -1

    def lookup(self, path):
        """Look a path up by hash. Returns the matching :class:`VfsRecord` or
 ``None``. If multiple records collide on the same hash, the first
 inserted is returned."""
        h = self.hash_of(path)
        idx = self._find_index(h)
        return self._records[idx] if idx != -1 else None

    def to_bytes(self):
        """Serialize the sorted record array (no header) -> ``bytes``."""
        out = bytearray()
        for r in self._records:
            out += r.to_bytes()
        return bytes(out)

    @classmethod
    def from_bytes(cls, data, seed=DEFAULT_SEED):
        """Parse a flat ``{hash,lbn,size}`` record array back into a table.

 The records are assumed already hash-sorted (they are re-sorted to be
 safe). Source paths are unknown, so ``lookup`` by path will only work
 for entries re-``add``-ed with their path.
 """
        t = cls(seed=seed)
        n = len(data) // RECORD_SIZE
        recs = []
        for i in range(n):
            h, lbn, size = struct.unpack_from(RECORD_FMT, data, i * RECORD_SIZE)
            recs.append(VfsRecord(h, lbn, size))
        recs.sort(key=lambda r: r.hash)
        t._records = recs
        return t

    def is_sorted(self):
        return all(
            self._records[i].hash <= self._records[i + 1].hash
            for i in range(len(self._records) - 1)
        )
