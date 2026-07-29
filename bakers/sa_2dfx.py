#!/usr/bin/env python3
"""the source game 2dfx LIGHT (type-0) reader: pulls the corona/point-light effects baked into a
model's DFF (RW plugin chunk 0x0253F2F8 in a GEOMETRY EXTENSION). Read-only; reuses
gvcslib.sa_dff for the RW chunk tree.

Returns a list of dicts per type-0 entry:
 pos(x,y,z local), color(r,g,b,a), farClip, ptRange, coronaSize, shadowSize,
 showMode, flags (= flags1 | flags2<<8; bit5 AtDay, bit6 AtNight, ...).
"""
import os
import struct, sys
sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_dff

TWODFX = 0x0253F2F8


def _walk(chunk, out):
    if chunk.type == TWODFX:
        out.append(chunk)
    for c in chunk.children:
        _walk(c, out)


def parse_lights(blob):
    blob = bytes(blob)
    try:
        root = sa_dff.parse_chunks(blob)
    except Exception:
        return []
    chunks = []
    _walk(root, chunks)
    lights = []
    for ch in chunks:
        o = ch.data_off
        if o + 4 > len(blob):
            continue
        count = struct.unpack_from("<I", blob, o)[0]; o += 4
        for _ in range(count):
            if o + 20 > len(blob):
                break
            px, py, pz, etype, dsize = struct.unpack_from("<3fII", blob, o)
            pay = o + 20
            if (etype & 0xFF) == 0 and dsize >= 0x4C and pay + 0x4C <= len(blob):
                r, g, b, a = struct.unpack_from("<4B", blob, pay + 0x00)
                farClip, ptRange, cSize, shSize = struct.unpack_from("<4f", blob, pay + 0x04)
                showMode = blob[pay + 0x14]
                flags1   = blob[pay + 0x18]
                flags2   = blob[pay + 0x4A] if (dsize >= 0x50 and pay + 0x4B <= len(blob)) else 0
                lights.append(dict(pos=(px, py, pz), color=(r, g, b, a),
                                   farClip=farClip, ptRange=ptRange,
                                   coronaSize=cSize, shadowSize=shSize,
                                   showMode=showMode, flags=flags1 | (flags2 << 8)))
            o = pay + dsize
    return lights


if __name__ == "__main__":
    from gvcslib import sa_img
    img = sa_img.SaImg("")
    name = sys.argv[1] if len(sys.argv) > 1 else "streetlamp1.dff"
    L = parse_lights(img.extract(name))
    print(name, "->", len(L), "type-0 lights")
    for l in L:
        print("  ", l)
