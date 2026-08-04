"""the source game placement (.IPL) parser - text `inst` + binary `bnry` (read-only).

Two physical carriers of the same world placement data:

* **Text IPL** - disc ``DATA/MAPS/**/*.IPL``. Human-readable; the ``inst``
 section holds 11 CSV fields per instance. Carries LODs and the
 non-streamed instances.
* **Binary IPL** - ``*_stream*.ipl`` entries inside ``GTA3.IMG``, magic
 ``bnry``. Holds the bulk (streamed building instances). 40-byte INST
 records. PS2 layout is identical to the PC ``bnry`` format.

Both decode into the same :class:`Inst`. The binary form does **not** store
the model *name* (resolve via :mod:`gvcslib.sa_ide`); ``name`` is ``''`` there.

Field order (both forms)::

 model_id, name, interior, posX, posY, posZ, rotX, rotY, rotZ, rotW, lod

``lod`` is an index into the *same IPL's* instance list (the high-LOD proxy),
or -1 for none. ``rot`` is the stored unit quaternion (engine conjugates it
at load; we keep the on-disk value verbatim).

In **binary** IPLs the third on-disk field is **not** ``interior`` but
``flags_area = (flags<<8)|areaCode`` (see :class:`Inst`): ``interior`` then holds
just the low-byte area code, ``flags_area`` the raw word, and :attr:`Inst.flags`
the high bits (0x100 underwater, 0x400 tunnel, ...). Text IPLs keep a genuine
``interior`` column and ``flags_area is None``.
"""
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Binary `bnry` header:,,. INST record
# = 7 float32 (pos xyz + quat xyzw) then 3 int32: model_id, flagsArea, lod.
# The middle int32 is NOT an interior id - it is flagsArea = (flags<<8)|areaCode.
# SA loader CFileLoader::CreateEntityFromInstance (the original loader) reads it
# as: low byte -> entity bM_areaCode; high bits -> flags (0x100 underwater,
# 0x200 invert-rotation, 0x400 tunnel, 0x800 tunnel-transition, 0x1000 unknown).
# Proven against gta3.img (observed 0x0/0x12/0x100/0x200/0x400, never 0..18).
_BNRY_NUM_INST = 0x04
_BNRY_OFF_INST = 0x1c
_INST_REC = struct.Struct("<7f3i")
INST_SIZE = _INST_REC.size  # 40


@dataclass
class Inst:
    """One world placement (text or binary IPL).

 ``interior`` is the genuine interior id for **text** IPLs. For **binary**
 (`bnry`) IPLs the on-disk middle int32 is ``flags_area = (flags<<8)|areaCode``
 (NOT an interior); there ``interior`` holds only the low-byte area code and the
 raw word is kept in ``flags_area`` (``None`` for text). Use :attr:`area` and
 :attr:`flags` to read either form uniformly.
 """
    __slots__ = ("model_id", "name", "interior", "pos", "rot", "lod", "flags_area")
    model_id: int
    name: str
    interior: int
    pos: Tuple[float, float, float]
    rot: Tuple[float, float, float, float]
    lod: int
    flags_area: Optional[int]  # raw bnry middle word; None for text IPLs

    @property
    def area(self) -> int:
        """Area/interior selector (engine ``bM_areaCode``); low byte for binary."""
        return (self.flags_area & 0xFF) if self.flags_area is not None else self.interior

    @property
    def flags(self) -> int:
        """Binary-IPL entity flag bits (0x100 underwater, 0x400 tunnel, ...); 0 for text."""
        return (self.flags_area & ~0xFF) if self.flags_area is not None else 0


def parse_text_ipl(path) -> List[Inst]:
    """Parse the ``inst`` section of one text .IPL into a list of :class:`Inst`."""
    out = []
    section = None
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if section is None:
                if low in ("inst", "cull", "zone", "pick", "path", "occl",
                           "mult", "grge", "enex", "cars", "jump", "tcyc",
                           "auzo", "2dfx"):
                    section = low
                continue
            if low == "end":
                section = None
                continue
            if section != "inst":
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue
            try:
                mid = int(parts[0])
                interior = int(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                rx, ry, rz, rw = (float(parts[6]), float(parts[7]),
                                  float(parts[8]), float(parts[9]))
                lod = int(parts[10])
            except ValueError:
                continue
            out.append(Inst(mid, parts[1], interior, (x, y, z),
                            (rx, ry, rz, rw), lod, None))
    return out


def parse_binary_ipl(blob) -> List[Inst]:
    """Parse a binary ``bnry`` IPL blob (from GTA3.IMG) into :class:`Inst` list."""
    if blob[:4] != b"bnry":
        raise ValueError("not a bnry IPL: %r" % blob[:4])
    num = struct.unpack_from("<I", blob, _BNRY_NUM_INST)[0]
    off = struct.unpack_from("<I", blob, _BNRY_OFF_INST)[0]
    out = []
    for i in range(num):
        o = off + i * INST_SIZE
        if o + INST_SIZE > len(blob):
            break
        x, y, z, rx, ry, rz, rw, mid, flags_area, lod = _INST_REC.unpack_from(blob, o)
        # middle int32 = flagsArea, not interior; interior gets the low-byte area code
        out.append(Inst(mid, "", flags_area & 0xFF, (x, y, z),
                        (rx, ry, rz, rw), lod, flags_area))
    return out


def load_all(data_dir, img=None) -> List[Inst]:
    """All placements: text ``inst`` from ``DATA/MAPS/**`` + binary from ``img``.

 ``data_dir`` = the ``DATA`` folder. ``img`` = an open
 :class:`gvcslib.sa_img.SaImg` (its ``*_stream*.ipl`` entries are parsed);
 pass ``None`` to skip the binary set.
 """
    out = []
    maps = os.path.join(data_dir, "MAPS")
    for root, _d, files in os.walk(maps):
        for fn in files:
            if fn.lower().endswith(".ipl"):
                out.extend(parse_text_ipl(os.path.join(root, fn)))
    if img is not None:
        for n in img.names():
            if n.lower().endswith(".ipl"):
                out.extend(parse_binary_ipl(img.extract(n)))
    return out
