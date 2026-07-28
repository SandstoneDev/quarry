"""the source game RenderWare IMG v2 archive reader (read-only)."""
import struct

SECTOR = 2048

class SaImg:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(8)
            if head[:4] != b"VER2":
                raise ValueError("not a VER2 IMG: %r" % head[:4])
            self.count = struct.unpack_from("<I", head, 4)[0]
            tbl = f.read(self.count * 32)
        self._ent = {}      # lower-name -> (off_sectors, stream_sectors)
        self._names = []
        for i in range(self.count):
            off, strm, _sz = struct.unpack_from("<IHH", tbl, i * 32)
            name = tbl[i * 32 + 8:i * 32 + 32].split(b"\x00", 1)[0].decode("latin1")
            self._ent[name.lower()] = (off, strm)
            self._names.append(name)

    def names(self):
        return list(self._names)

    def extract(self, name):
        e = self._ent.get(name.lower())
        if e is None:
            raise KeyError(name)
        off, strm = e
        with open(self.path, "rb") as f:
            f.seek(off * SECTOR)
            return f.read(strm * SECTOR)
