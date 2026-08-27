#!/usr/bin/env python3
"""sa_col - decode the source game collision (.col) libraries from gta3.img.

Port-side codec (gvcslib stays READ-ONLY). 1:1 with the reverse:
CFileLoader::LoadCollisionModel{,Ver2,Ver3,Ver4} (the reference sources FileLoader.cpp)
and the ColHelpers struct layout (ColHelpers.h). COL spec also on
https://gtamods.com/wiki/Collision_File .

A .col *library* (e.g. `LAs_0.col`, stored as an IMG entry) is a back-to-back
sequence of ColModels, each prefixed with a 32-byte FileHeader:

 char fourcc[4] "COLL"|"COL2"|"COL3"|"COL4"
 u32 size bytes AFTER the 8-byte FileInfo (so entry = size + 8)
 char modelName[22]
 u16 modelId

Version bodies (offsets in V2+ are measured from the fourcc, +4 trick below):

 V1 (COLL): sequential - bounds, then count-prefixed sphere/line/box/vert/face
 V2/3/4 : header with counts+flags+section offsets, then one packed blob.
 vertices are int16/128 (CompressedVector); faces are u16 idx + 2x u8.

Usage:
 python sa_col.py # self-test: index+decode every .col in gta3.img
 python sa_col.py <modelName> # decode one model, dump stats
"""
import os
import struct
import sys
import math

# SA_GTA3_IMG env override: the Quarry converter points this at the archive
# extracted from the USER'S disc (PS2 COL containers are byte-identical in
# format); the default keeps the historical dev loop alive.
IMG = os.environ.get("SA_GTA3_IMG",
                     "")
SECTOR = 2048


# ---------------------------------------------------------------- IMG VER2 ----
class ImgArchive:
    """Read-only the source game IMG (VER2) archive directory + entry fetch."""

    def __init__(self, path):
        self.path = path
        self.entries = {}  # lower-name -> (offset_bytes, size_bytes)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"VER2":
                raise ValueError(f"{path}: not a VER2 IMG ({magic!r})")
            n = struct.unpack("<I", f.read(4))[0]
            dirblob = f.read(n * 32)
        for i in range(n):
            off, szs, sz = struct.unpack_from("<IHH", dirblob, i * 32)
            name = dirblob[i * 32 + 8 : i * 32 + 32].split(b"\x00")[0].decode("latin1")
            size = (sz if sz else szs) * SECTOR
            self.entries[name.lower()] = (off * SECTOR, size)

    def names(self, ext=None):
        if ext is None:
            return list(self.entries)
        ext = ext.lower()
        return [n for n in self.entries if n.endswith(ext)]

    def read(self, name):
        off, size = self.entries[name.lower()]
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(size)


# ------------------------------------------------------------- COL structs ----
class ColModel:
    __slots__ = ("name", "model_id", "version", "bmin", "bmax",
                 "center", "radius", "spheres", "boxes",
                 "verts", "faces")

    def __init__(self):
        self.name = ""
        self.model_id = 0
        self.version = 0
        self.bmin = (0.0, 0.0, 0.0)
        self.bmax = (0.0, 0.0, 0.0)
        self.center = (0.0, 0.0, 0.0)
        self.radius = 0.0
        self.spheres = []   # (cx,cy,cz, r, material)
        self.boxes = []     # (minx,miny,minz, maxx,maxy,maxz, material)
        self.verts = []     # (x,y,z) floats, model-local
        self.faces = []     # (a,b,c, material)

    @property
    def n_tris(self):
        return len(self.faces)

    def __repr__(self):
        return (f"<ColModel {self.name!r} v{self.version} "
                f"sph={len(self.spheres)} box={len(self.boxes)} "
                f"v={len(self.verts)} f={len(self.faces)} r={self.radius:.1f}>")


_FOURCC = {b"COLL": 1, b"COL2": 2, b"COL3": 3, b"COL4": 4}
_HDR_SIZE = {2: 76, 3: 88, 4: 92}  # version Header size (after the 32-byte FileHeader)


def _decode_v1(cm, buf, base):
    """COLL: sequential layout. `base` = offset of the version Header (=32)."""
    p = base
    # TBounds(40): sphere{radius,center}(16) + box{min,max}(24)
    cm.radius = struct.unpack_from("<f", buf, p)[0]
    cm.center = struct.unpack_from("<3f", buf, p + 4)
    cm.bmin = struct.unpack_from("<3f", buf, p + 16)
    cm.bmax = struct.unpack_from("<3f", buf, p + 28)
    p += 40
    (nsph,) = struct.unpack_from("<I", buf, p); p += 4
    for _ in range(nsph):
        r, cx, cy, cz = struct.unpack_from("<f3f", buf, p)
        mat = buf[p + 16]
        cm.spheres.append((cx, cy, cz, r, mat)); p += 20
    (nlines,) = struct.unpack_from("<I", buf, p); p += 4
    p += nlines * 24  # lines unused
    (nbox,) = struct.unpack_from("<I", buf, p); p += 4
    for _ in range(nbox):
        mnx, mny, mnz, mxx, mxy, mxz = struct.unpack_from("<6f", buf, p)
        mat = buf[p + 24]
        cm.boxes.append((mnx, mny, mnz, mxx, mxy, mxz, mat)); p += 28
    (nvert,) = struct.unpack_from("<I", buf, p); p += 4
    for _ in range(nvert):
        cm.verts.append(struct.unpack_from("<3f", buf, p)); p += 12
    (nface,) = struct.unpack_from("<I", buf, p); p += 4
    for _ in range(nface):
        a, b, c = struct.unpack_from("<3I", buf, p)
        mat = buf[p + 12]
        cm.faces.append((a, b, c, mat)); p += 16


def _decode_v234(cm, buf, ver):
    """COL2/3/4: offset-based. Offsets are relative to fourcc; element absolute
 offset within the entry = fileOffset + 4 (the 4 fourcc bytes are folded into
 the header size in the reverse, so we add them back here)."""
    p = 32  # version Header starts right after the 32-byte FileHeader
    # bounds TBounds(40): box CBoundingBox(min,max)=24 then CSphere(center,radius)=16
    cm.bmin = struct.unpack_from("<3f", buf, p)
    cm.bmax = struct.unpack_from("<3f", buf, p + 12)
    cm.center = struct.unpack_from("<3f", buf, p + 24)
    cm.radius = struct.unpack_from("<f", buf, p + 36)[0]
    nsph, nbox, nface = struct.unpack_from("<3H", buf, p + 40)
    nlines = buf[p + 46]  # u8 (+1 pad byte to align the u32 below)
    flags = struct.unpack_from("<I", buf, p + 48)[0]
    off_sph, off_box, off_lines, off_verts, off_faces, off_planes = \
        struct.unpack_from("<6I", buf, p + 52)
    # (V3/V4 shadow fields follow; unused for our purposes)

    def at(off):
        return off + 4 if off else 0

    # spheres - CSphere(center vec3, radius)=16 + surface(4)
    o = at(off_sph)
    for i in range(nsph):
        q = o + i * 20
        cx, cy, cz, r = struct.unpack_from("<4f", buf, q)
        cm.spheres.append((cx, cy, cz, r, buf[q + 16]))
    # boxes - CBox(min,max)=24 + surface(4)
    o = at(off_box)
    for i in range(nbox):
        q = o + i * 28
        mnx, mny, mnz, mxx, mxy, mxz = struct.unpack_from("<6f", buf, q)
        cm.boxes.append((mnx, mny, mnz, mxx, mxy, mxz, buf[q + 24]))
    # faces - a,b,c u16, material u8, light u8
    o = at(off_faces)
    maxidx = -1
    for i in range(nface):
        q = o + i * 8
        a, b, c = struct.unpack_from("<3H", buf, q)
        cm.faces.append((a, b, c, buf[q + 6]))
        maxidx = max(maxidx, a, b, c)
    # vertices - CompressedVector: 3x int16 / 128.0 (count not stored -> derive)
    o = at(off_verts)
    nvert = maxidx + 1 if maxidx >= 0 else 0
    for i in range(nvert):
        x, y, z = struct.unpack_from("<3h", buf, o + i * 6)
        cm.verts.append((x / 128.0, y / 128.0, z / 128.0))


def parse_library(blob):
    """Decode a .col library blob -> list[ColModel] (in file order)."""
    out = []
    pos = 0
    n = len(blob)
    while pos + 8 <= n:
        fourcc = blob[pos:pos + 4]
        ver = _FOURCC.get(fourcc)
        if ver is None:
            break  # padding / end of meaningful data
        size = struct.unpack_from("<I", blob, pos + 4)[0]
        total = size + 8
        if pos + total > n:
            break
        name = blob[pos + 8:pos + 30].split(b"\x00")[0].decode("latin1")
        model_id = struct.unpack_from("<H", blob, pos + 30)[0]
        entry = blob[pos:pos + total]
        cm = ColModel()
        cm.name, cm.model_id, cm.version = name, model_id, ver
        if ver == 1:
            _decode_v1(cm, entry, 32)
        else:
            _decode_v234(cm, entry, ver)
        out.append(cm)
        pos += total
    return out


def build_index(img):
    """name(lower) -> ColModel, scanning every .col library in the IMG.
 Later libraries win on name collision (rare)."""
    idx = {}
    libs = img.names(".col")
    for lib in libs:
        for cm in parse_library(img.read(lib)):
            idx[cm.name.lower()] = cm
    return idx, libs


# --------------------------------------------------------------- self-test ----
def _selftest():
    img = ImgArchive(IMG)
    libs = img.names(".col")
    print(f"IMG {IMG}\n  .col libraries: {len(libs)}")
    vh = {1: 0, 2: 0, 3: 0, 4: 0}
    tot_models = tot_tris = tot_box = tot_sph = 0
    bad = 0
    for lib in libs:
        try:
            models = parse_library(img.read(lib))
        except Exception as e:
            print(f"  !! {lib}: {e}")
            bad += 1
            continue
        for cm in models:
            vh[cm.version] += 1
            tot_models += 1
            tot_tris += cm.n_tris
            tot_box += len(cm.boxes)
            tot_sph += len(cm.spheres)
            # sanity: every face index in range
            nv = len(cm.verts)
            for (a, b, c, _m) in cm.faces:
                if a >= nv or b >= nv or c >= nv:
                    raise AssertionError(f"{cm.name}: face idx OOB ({a},{b},{c} >= {nv})")
    print(f"  models={tot_models}  version hist={vh}")
    print(f"  total: tris={tot_tris}  boxes={tot_box}  spheres={tot_sph}  bad libs={bad}")
    # dump a few examples
    sample = parse_library(img.read(libs[0]))
    print(f"\n  sample from {libs[0]}:")
    for cm in sample[:5]:
        print("   ", cm)


def _dump_one(name):
    img = ImgArchive(IMG)
    idx, libs = build_index(img)
    cm = idx.get(name.lower())
    if not cm:
        print(f"model {name!r} not found in {len(idx)} col models")
        # suggest near matches
        near = [k for k in idx if name.lower() in k][:10]
        if near:
            print("  near:", near)
        return
    print(cm)
    print("  bbox", cm.bmin, cm.bmax)
    if cm.verts:
        xs = [v[0] for v in cm.verts]; ys = [v[1] for v in cm.verts]; zs = [v[2] for v in cm.verts]
        print(f"  vert extent x[{min(xs):.2f},{max(xs):.2f}] "
              f"y[{min(ys):.2f},{max(ys):.2f}] z[{min(zs):.2f},{max(zs):.2f}]")
    for f in cm.faces[:4]:
        print("  face", f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _dump_one(sys.argv[1])
    else:
        _selftest()
