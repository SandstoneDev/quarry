#!/usr/bin/env python3
"""sa_skin - decode the RenderWare SKIN (0x116) + HANIM (0x11E) plugin chunks and
the FRAMELIST skeleton from a the source game PC ped DFF. Port-side codec (gvcslib stays
READ-ONLY); reuses gvcslib.sa_dff.parse_chunks for the chunk tree.

Byte layout (byte-validated):
 SKIN (0x116) in GEOMETRY->EXTENSION:
 u8 numBones, u8 numUsedBones, u8 maxWeights, u8 pad
 u8[numUsedBones] usedBoneArray
 u8[nVert][4] boneIndices (HAnim node indices)
 f32[nVert][4] boneWeights (sum 1.0)
 f32[nBones][16] inverseBind (row-major; trans in row 3)
 12 bytes trailer (zero)
 HANIM (0x11E) in each FRAMELIST frame EXTENSION:
 u32 version(0x100), i32 nodeId, u32 numNodes [, u32 flags, u32 keyFrameSize,
 node[numNodes] {i32 nodeId, i32 nodeIndex, u32 flags}] (root frame has the table)
 FRAMELIST STRUCT: u32 numFrames, then per frame 56B {f32 rot[9], f32 pos[3], i32 parent, u32 flags}

Returns a dict: { frames:[{rot,pos,parent,nodeId}], nodes:[(id,idx,flags)], geoms:[skin...] }.
Run directly to validate against a ped (default fam1) from GTA3.IMG.
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_img
from gvcslib.sa_dff import (parse_chunks, STRUCT, EXTENSION, FRAMELIST,
                            GEOMETRYLIST, GEOMETRY, CLUMP)

SKIN  = 0x116
HANIM = 0x11E
GTA3  = ""


def decode_skeleton(blob):
    """FRAMELIST 56-byte frame records + HANIM per-frame node table. Split out of
 decode() because this half is platform-NEUTRAL - the frame records and the
 HANIM hierarchy are byte-for-byte identical on the PC and PS2 twins (verified:
 node table + nodeId->parent map match), so the PS2-native codec (tools/ps2skin)
 reuses this exact parse for the skeleton and only supplies the skin weights +
 inverse-bind matrices from the native path.
 Returns {frames:[{rot,pos,parent,nodeId}], nodes:[(id,idx,flags)]}."""
    blob = bytes(blob)
    root = parse_chunks(blob)
    clump = root if root.type == CLUMP else root.find(CLUMP)
    if not clump:
        raise ValueError("no CLUMP")
    fl = clump.find(FRAMELIST)
    fst = fl.find(STRUCT)
    o = fst.data_off
    numFrames = struct.unpack_from("<I", blob, o)[0]; o += 4
    frames = []
    for _i in range(numFrames):
        rot = struct.unpack_from("<9f", blob, o); o += 36
        pos = struct.unpack_from("<3f", blob, o); o += 12
        parent, flags = struct.unpack_from("<iI", blob, o); o += 8
        frames.append({"rot": rot, "pos": pos, "parent": parent, "nodeId": -1})

    nodetable = None
    fexts = [c for c in fl.children if c.type == EXTENSION]
    for i, ext in enumerate(fexts):
        for c in ext.children:
            if c.type != HANIM:
                continue
            p = c.data_off
            ver, nodeId, numNodes = struct.unpack_from("<IiI", blob, p); p += 12
            if i < numFrames:
                frames[i]["nodeId"] = nodeId
            if numNodes > 0:
                flg, kfs = struct.unpack_from("<II", blob, p); p += 8
                nodes = []
                for _n in range(numNodes):
                    nid, nidx, nflg = struct.unpack_from("<iiI", blob, p); p += 12
                    nodes.append((nid, nidx, nflg))
                nodetable = nodes
    return {"frames": frames, "nodes": nodetable}


def decode(blob):
    blob = bytes(blob)
    root = parse_chunks(blob)
    clump = root if root.type == CLUMP else root.find(CLUMP)
    if not clump:
        raise ValueError("no CLUMP")

    # ---- FRAMELIST: 56-byte frame records + HANIM per-frame extensions ----
    skel = decode_skeleton(blob)
    frames = skel["frames"]
    nodetable = skel["nodes"]

    # ---- SKIN per geometry ----
    gl = clump.find(GEOMETRYLIST)
    geoms = []
    for geo in gl.find_all(GEOMETRY):
        gst = geo.find(STRUCT)
        gflags, gntri, gnvert, gnmorph = struct.unpack_from("<4I", blob, gst.data_off)
        ext = geo.find(EXTENSION)
        if not ext:
            continue
        for c in ext.children:
            if c.type != SKIN:
                continue
            p = c.data_off
            numBones, numUsed, maxW, _pad = struct.unpack_from("<4B", blob, p); p += 4
            used = list(struct.unpack_from("<%dB" % numUsed, blob, p)); p += numUsed
            bidx = [struct.unpack_from("<4B", blob, p + v*4) for v in range(gnvert)]
            p += gnvert * 4
            bw = [struct.unpack_from("<4f", blob, p + v*16) for v in range(gnvert)]
            p += gnvert * 16
            inv = [struct.unpack_from("<16f", blob, p + b*64) for b in range(numBones)]
            p += numBones * 64
            budget = 4 + numUsed + gnvert*4 + gnvert*16 + numBones*64 + 12
            geoms.append({"nvert": gnvert, "numBones": numBones, "numUsed": numUsed,
                          "maxW": maxW, "used": used, "boneIdx": bidx, "boneW": bw,
                          "invBind": inv, "skin_size": c.size, "budget": budget})
    return {"frames": frames, "nodes": nodetable, "geoms": geoms}


if __name__ == "__main__":
    ped = sys.argv[1] if len(sys.argv) > 1 else "fam1"
    im = sa_img.SaImg(GTA3)
    sk = decode(im.extract(ped + ".dff"))
    print("=== %s.dff skeleton ===" % ped)
    print("frames:", len(sk["frames"]), " HANIM nodes:", len(sk["nodes"]) if sk["nodes"] else 0)
    for i, f in enumerate(sk["frames"][:6]):
        print("  frame %d: parent=%d nodeId=%d pos=(%.3f,%.3f,%.3f)"
              % (i, f["parent"], f["nodeId"], f["pos"][0], f["pos"][1], f["pos"][2]))
    if sk["nodes"]:
        print("  nodes[0..5]:", sk["nodes"][:6])
    for gi, g in enumerate(sk["geoms"]):
        print("=== geom %d skin ===" % gi)
        print("  nvert=%d numBones=%d numUsed=%d maxW=%d  size=%d budget=%d %s"
              % (g["nvert"], g["numBones"], g["numUsed"], g["maxW"], g["skin_size"],
                 g["budget"], "OK" if g["skin_size"] == g["budget"] else "!! MISMATCH"))
        print("  used:", g["used"])
        for v in (0, 1, 2):
            print("  vert%d idx=%s w=%s" % (v, g["boneIdx"][v],
                  tuple(round(x,3) for x in g["boneW"][v])))
