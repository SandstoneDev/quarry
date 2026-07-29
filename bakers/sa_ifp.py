#!/usr/bin/env python3
"""sa_ifp - parse a the source game ANP3 packed IFP animation block (PED.IFP and friends).
Port-side codec (gvcslib READ-ONLY). Byte spec: docs 18_data_formats/ifp_format.md
(decoded from CAnimManager::LoadAnimFile 0x4df270).

ANP3 layout (little-endian) - VERIFIED against CAnimManager::LoadAnimFile in the original (static analysis):
 "ANP3" | u32 blockSize | char blockName[24] | u32 numAnimations
 per animation: char animName[24] | u32 numSeq | u32 totalKeyframeBytes | u32 flags
 per sequence: char seqName[24] | u32 keyTypeCode | u32 numFrames | s32 boneTag
 then numFrames keyframes contiguous. Stride is keyed by CODE
 (the ANP3 flag only governs cursor accounting, not stride):
 code 1 -> 0x14 (20) uncompressed rot-only : f32 qx,qy,qz,qw, time
 code 2 -> 0x20 (32) uncompressed rot+trans : f32 qx,qy,qz,qw, time, tx,ty,tz
 code 3 -> 0x0A (10) compressed rot-only : s16 qx,qy,qz,qw, time
 code 4 -> 0x10 (16) compressed rot+trans : s16 qx,qy,qz,qw, time, tx,ty,tz
 dequant (compressed): quat = s16/4096, trans = s16/1024, time = s16/60 (sec).
 quat x,y,z are stored NEGATED (conjugate) on disk.

Returns: [ {name, flags, seqs:[ {name, boneTag, keyType, stride, numFrames, kf(bytes)} ]} ].
Run directly to validate against ANIM/PED.IFP (cursor == totalKeyframeBytes per anim).
"""
import os
import struct
import sys

# SA_ROOT env override: Quarry points this at the user's extracted PS2 disc so the
# base clips come from the staged ANIM/PED.IFP (byte-exact ANP3, identical on PS2 &
# PC). Defaults keep the PC dev loop. (col_bake.py uses the same idiom.)
ROOT = os.environ.get("SA_ROOT", "")


def _find_ped_ifp():
    for p in (ROOT + "/anim/ped.ifp", ROOT + "/ANIM/PED.IFP", ROOT + "/anim/PED.IFP"):
        if os.path.exists(p):
            return p
    # case-insensitive search under anim/
    for d in ("anim", "ANIM"):
        dd = os.path.join(ROOT, d)
        if os.path.isdir(dd):
            for f in os.listdir(dd):
                if f.lower() == "ped.ifp":
                    return os.path.join(dd, f)
    return None


def _cstr(b):
    z = b.find(b"\x00")
    return b[:z if z >= 0 else len(b)].decode("ascii", "replace")


def decode(blob):
    blob = bytes(blob)
    if blob[:4] != b"ANP3":
        raise ValueError("not ANP3 (got %r) - only the SA shipping packed form is handled" % blob[:4])
    o = 4
    blockSize = struct.unpack_from("<I", blob, o)[0]; o += 4
    blockName = _cstr(blob[o:o+24]); o += 24
    numAnim = struct.unpack_from("<I", blob, o)[0]; o += 4

    anims = []
    for _a in range(numAnim):
        animName = _cstr(blob[o:o+24]); o += 24
        numSeq, totalKf, flags = struct.unpack_from("<III", blob, o); o += 12
        seqs = []
        kfsum = 0
        STRIDE = {1: 20, 2: 32, 3: 10, 4: 16}
        for _s in range(numSeq):
            seqName = _cstr(blob[o:o+24]); o += 24
            keyType, numFrames = struct.unpack_from("<II", blob, o); o += 8
            boneTag = struct.unpack_from("<i", blob, o)[0]; o += 4
            stride = STRIDE.get(keyType, 10)
            fb = numFrames * stride
            kf = blob[o:o+fb]; o += fb
            kfsum += fb
            seqs.append({"name": seqName, "boneTag": boneTag, "keyType": keyType,
                         "stride": stride, "numFrames": numFrames, "kf": kf})
        anims.append({"name": animName, "flags": flags, "numSeq": numSeq,
                      "totalKf": totalKf, "kfsum": kfsum, "seqs": seqs})
    return {"block": blockName, "anims": anims, "consumed": o, "size": len(blob)}


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else _find_ped_ifp()
    if not p:
        sys.exit("ped.ifp not found under " + ROOT + "/anim")
    pkg = decode(open(p, "rb").read())
    print("block=%r  anims=%d  consumed=%d/%d %s"
          % (pkg["block"], len(pkg["anims"]), pkg["consumed"], pkg["size"],
             "OK" if pkg["consumed"] == pkg["size"] else "!! TRAILING"))
    bad = [a for a in pkg["anims"] if a["kfsum"] != a["totalKf"]]
    print("cursor check: %d/%d anims have kfsum==totalKeyframeBytes %s"
          % (len(pkg["anims"]) - len(bad), len(pkg["anims"]),
             "OK" if not bad else ("!! %d MISMATCH" % len(bad))))
    want = ["walk_civi", "run_civi", "sprint_civi", "idle_stance", "walk", "run", "sprint", "idle"]
    for a in pkg["anims"]:
        if any(w in a["name"].lower() for w in want):
            s0 = a["seqs"][0] if a["seqs"] else None
            print("  %-20s seq=%2d flags=%d  seq0=%s frames=%d type=%d tag=%d"
                  % (a["name"], a["numSeq"], a["flags"],
                     s0["name"] if s0 else "-", s0["numFrames"] if s0 else 0,
                     s0["keyType"] if s0 else -1, s0["boneTag"] if s0 else -1))
    print("first 8 anim names:", [a["name"] for a in pkg["anims"][:8]])
