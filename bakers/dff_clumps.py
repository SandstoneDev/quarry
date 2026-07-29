#!/usr/bin/env python3
"""dff_clumps.py - split a the source game clothes DFF into its concatenated RenderWare CLUMPs.

Each player.img clothes model (torso/head/hands/legs/feet/vest/...) is THREE complete skinned
CLUMPs in one file, named by the atomic's frame node-name: "Normal" / "Fat" / "Ripped"
(identical topology + UVs; only vertex positions, and Fat's belly bone-weights, differ). SA
CPU-blends them by the fat/muscle stats. Our hero_bake historically parsed the FIRST clump
(= "Ripped") -> a slightly-muscular base body; use split_clumps()["normal"] for the true base,
and the "fat"/"ripped" clumps as morph targets (see the fat/muscle sliders).

Non-morph DFFs (a single clump) return one entry keyed by that clump's frame name.
"""
import os
import struct

CLUMP = 0x10
FRAMELIST = 0x0E
ATOMIC = 0x14
EXTENSION = 0x03
FRAME_NODENAME = 0x253F2FE


def _children(blob, o, end):
    """Walk sibling RW chunks in [o, end). Each -> (type, size, bodyOff) where the 12-byte
 header (type,size,libId) precedes bodyOff."""
    out = []
    while o + 12 <= end:
        typ, sz, _lib = struct.unpack_from("<III", blob, o)
        body = o + 12
        out.append((typ, sz, body))
        o = body + sz
    return out


def _frame_names(blob, fl_body, fl_end):
    """FRAMELIST -> {frameIndex: nodeName}. The frame array STRUCT is first; a per-frame
 EXTENSION follows, and the name lives in a FRAME node-name (0x253F2FE) chunk inside it."""
    exts = [k for k in _children(blob, fl_body, fl_end) if k[0] == EXTENSION]
    names = {}
    for i, (_typ, sz, body) in enumerate(exts):
        for (t2, s2, b2) in _children(blob, body, body + sz):
            if t2 == FRAME_NODENAME:
                names[i] = blob[b2:b2 + s2].split(b"\x00", 1)[0].decode("latin1", "replace")
    return names


def split_clumps(blob):
    """{clumpFrameName.lower(): clump_sub_blob_bytes} for a (multi-)CLUMP DFF. The sub-blob is a
 self-contained clump (its own 12-byte CLUMP header + body), so parse_geometry() reads it directly."""
    blob = bytes(blob)
    out = {}
    for (typ, sz, body) in _children(blob, 0, len(blob)):
        if typ != CLUMP:
            continue
        start = body - 12                       # include the CLUMP header
        sub = blob[start:body + sz]
        kids = _children(blob, body, body + sz)
        fl = [k for k in kids if k[0] == FRAMELIST]
        atoms = [k for k in kids if k[0] == ATOMIC]
        names = _frame_names(blob, fl[0][2], fl[0][2] + fl[0][1]) if fl else {}
        nm = None
        if atoms:                               # atomic -> its frame index -> that frame's name
            fidx = struct.unpack_from("<I", blob, atoms[0][2] + 12)[0]
            nm = names.get(fidx)
        key = (nm if nm else ("clump%d" % len(out))).lower()
        out[key] = sub
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
    from gvcslib import sa_img
    PLAYER_IMG = ""
    im = sa_img.SaImg(PLAYER_IMG)
    for d in (sys.argv[1:] or ["torso.dff", "head.dff", "hands.dff", "legs.dff", "feet.dff"]):
        try:
            cl = split_clumps(im.extract(d))
            print("%-14s -> %s" % (d, {k: len(v) for k, v in cl.items()}))
        except Exception as e:
            print("%-14s ERR %s" % (d, e))
