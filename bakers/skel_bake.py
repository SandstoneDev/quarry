#!/usr/bin/env python3
"""skel_bake - bake a ped rig (skeleton + skin weights) + selected locomotion clips
into skel.bin for the PSP skeletal-animation runtime. Mesh + textures stay in
char.bin (same DFF -> same vertex order); skel.bin adds the rig & anim only.

Sources (both validated byte-exact):
 tools/sa_skin.decode(dff) -> {frames:[{rot(3x3),pos,parent,nodeId}], nodes:[(id,idx,flg)],
 geoms:[{nvert,numBones,boneIdx[v][4],boneW[v][4],invBind[b][16]}]}
 tools/sa_ifp.decode(ifp) -> {anims:[{name,seqs:[{name,boneTag,keyType,numFrames,kf bytes}]}]}

Bone ordering in skel.bin == the SKIN bone order (0..numBones-1), which is also the
invBind order and what vertex boneIdx[] already references. Each bone stores its
HANIM nodeId, parent (remapped into skin-bone order), bind quat+pos, and invBind.
Clip tracks resolve boneTag(=nodeId) -> skin-bone index; keyframes kept compressed
(s16: quat/4096, trans/1024, time/60s).

skel.bin layout (little-endian):
 'SKL1' | u16 numBones | u16 numClips | u32 numVerts
 bones[numBones]: s16 parent | s16 nodeId | f32 bindQuat[4] | f32 bindPos[3] | f32 invBind[16]
 skin[numVerts]: u8 boneIdx[4] | f32 boneW[4]
 clips[numClips]: char name[24] | f32 duration | u16 numTracks | u16 pad
 track: s16 boneIdx | u8 hasTrans | u8 pad | u16 numKeys
 keys[numKeys]: s16 q[4] | s16 time | (s16 t[3] if hasTrans)
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_skin
import sa_ifp

OUT = ""
DEPLOY = ""
CLIPS = ["IDLE_stance", "WALK_civi", "run_civi", "sprint_civi"]   # idle/walk/run/sprint


def mat3_to_quat(m):
    """row-major 3x3 (m[0..8]) -> quat (x,y,z,w)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
    n = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
    return (x/n, y/n, z/n, w/n)


def crc32_upper(name):
    """CKeyGen::GetUppercaseKey 0x54f410 - table CRC32 over uppercased name."""
    crc = 0xFFFFFFFF
    for ch in name.encode("ascii", "replace"):
        c = ch - 32 if 97 <= ch <= 122 else ch   # toupper
        crc = ((crc >> 8) ^ _CRCTAB[(crc ^ c) & 0xFF]) & 0xFFFFFFFF
    return crc

def _mk_crctab():
    tab = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (0xEDB88320 ^ (c >> 1)) if (c & 1) else (c >> 1)
        tab.append(c & 0xFFFFFFFF)
    return tab
_CRCTAB = _mk_crctab()


def main():
    ped = sys.argv[1] if len(sys.argv) > 1 else "fam1"
    im = sa_skin.sa_img.SaImg(sa_skin.GTA3)
    sk = sa_skin.decode(im.extract(ped + ".dff"))
    pkg = sa_ifp.decode(open(sa_ifp._find_ped_ifp(), "rb").read())

    frames, nodes, geoms = sk["frames"], sk["nodes"], sk["geoms"]
    geo = geoms[0]
    numBones = geo["numBones"]
    invBind = geo["invBind"]

    # SKIN bone order b (0..numBones-1) -> HANIM nodeId via the HANIM node table.
    # nodes is the hierarchy table in node order; entry = (nodeId, nodeIndex, flags).
    if not nodes or len(nodes) < numBones:
        raise SystemExit("HANIM node table missing/short (%s)" % (len(nodes) if nodes else None))
    bone_nodeId = [nodes[b][0] for b in range(numBones)]
    nodeId_to_bone = {nid: b for b, nid in enumerate(bone_nodeId)}

    # frame (skeleton) lookup by nodeId, to grab bind pose for each skin bone.
    frame_by_node = {f["nodeId"]: f for f in frames if f["nodeId"] >= 0}
    # frame index -> nodeId, to remap parent frame index -> parent nodeId -> bone.
    fidx_node = {i: f["nodeId"] for i, f in enumerate(frames)}

    bones = []
    for b in range(numBones):
        nid = bone_nodeId[b]
        f = frame_by_node.get(nid)
        if f is None:
            # bone has no matching frame (rare); identity bind
            bones.append({"parent": -1, "nodeId": nid,
                          "q": (0, 0, 0, 1), "p": (0, 0, 0), "inv": invBind[b]})
            continue
        pq = mat3_to_quat(f["rot"])
        # parent: frame.parent is a frame index -> its nodeId -> skin bone index
        par = -1
        pf = f["parent"]
        if pf is not None and pf >= 0 and pf in fidx_node:
            par = nodeId_to_bone.get(fidx_node[pf], -1)
        bones.append({"parent": par, "nodeId": nid, "q": pq, "p": f["pos"], "inv": invBind[b]})

    # ---- resolve clips ----
    by_name = {a["name"].lower(): a for a in pkg["anims"]}
    out_clips = []
    for cname in CLIPS:
        a = by_name.get(cname.lower())
        if not a:
            print("  !! clip %s not found" % cname); continue
        tracks = []
        maxtime = 0.0
        for s in a["seqs"]:
            # resolve track bone: prefer boneTag(nodeId), else bone-name CRC vs frame names? (no names baked)
            bi = nodeId_to_bone.get(s["boneTag"], -1)
            if bi < 0:
                continue   # track for a bone not in this skin - skip
            hasTrans = 1 if s["keyType"] in (2, 4) else 0
            stride = {1: 20, 2: 32, 3: 10, 4: 16}[s["keyType"]]
            comp = s["keyType"] in (3, 4)
            keys = []
            kf = s["kf"]
            for fi in range(s["numFrames"]):
                base = fi * stride
                if comp:
                    q = struct.unpack_from("<4h", kf, base)
                    t16 = struct.unpack_from("<h", kf, base + 8)[0]
                    tr = struct.unpack_from("<3h", kf, base + 10) if hasTrans else (0, 0, 0)
                else:  # uncompressed float -> requantize to s16 (same scales)
                    qf = struct.unpack_from("<4f", kf, base)
                    q = tuple(int(round(c * 4096.0)) for c in qf)
                    tsec = struct.unpack_from("<f", kf, base + 16)[0]
                    t16 = int(round(tsec * 60.0))
                    if hasTrans:
                        trf = struct.unpack_from("<3f", kf, base + 20)
                        tr = tuple(int(round(c * 1024.0)) for c in trf)
                    else:
                        tr = (0, 0, 0)
                keys.append((q, t16, tr))
                maxtime = max(maxtime, t16 / 60.0)
            tracks.append({"bone": bi, "hasTrans": hasTrans, "keys": keys})
        out_clips.append({"name": cname, "dur": maxtime, "tracks": tracks})

    # ---- emit ----
    buf = bytearray()
    buf += b"SKL1"
    buf += struct.pack("<HHI", numBones, len(out_clips), geo["nvert"])
    for b in bones:
        buf += struct.pack("<hh", b["parent"], b["nodeId"])
        buf += struct.pack("<4f", *b["q"])
        buf += struct.pack("<3f", *b["p"])
        buf += struct.pack("<16f", *b["inv"])
    for v in range(geo["nvert"]):
        bi = geo["boneIdx"][v]
        buf += struct.pack("<4B", *(min(x, numBones-1) for x in bi))
        buf += struct.pack("<4f", *geo["boneW"][v])
    for c in out_clips:
        nm = c["name"].encode("ascii")[:23]; nm += b"\x00" * (24 - len(nm))
        buf += nm
        buf += struct.pack("<fHH", c["dur"], len(c["tracks"]), 0)
        for t in c["tracks"]:
            buf += struct.pack("<hBBH", t["bone"], t["hasTrans"], 0, len(t["keys"]))
            for (q, tm, tr) in t["keys"]:
                buf += struct.pack("<4hh", q[0], q[1], q[2], q[3], tm)
                if t["hasTrans"]:
                    buf += struct.pack("<3h", tr[0], tr[1], tr[2])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "wb").write(buf)
    try:
        open(DEPLOY, "wb").write(buf)
        deployed = "deployed"
    except OSError:
        deployed = "(deploy skip)"
    print("=== skel.bin: %s  bones=%d verts=%d clips=%d  %d bytes  %s ==="
          % (ped, numBones, geo["nvert"], len(out_clips), len(buf), deployed))
    for c in out_clips:
        res = len(c["tracks"])
        tk = sum(len(t["keys"]) for t in c["tracks"])
        print("  %-14s dur=%.2fs tracks=%d keys=%d" % (c["name"], c["dur"], res, tk))
    # structural validation
    bad = sum(1 for v in range(geo["nvert"]) for x in geo["boneIdx"][v] if x >= numBones)
    print("  vert boneIdx out-of-range: %d %s" % (bad, "OK" if bad == 0 else "!!"))
    rootless = sum(1 for b in bones if b["parent"] < 0)
    print("  bones with no parent (roots): %d" % rootless)


if __name__ == "__main__":
    main()
