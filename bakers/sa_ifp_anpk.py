#!/usr/bin/env python3
"""sa_ifp_anpk - decode a the source game ANPK IFP anim block (the cutscene IFPs in cuts.img,
e.g. intro1a.ifp). ANP3 (sa_ifp.py) is the shipping packed ped form; ANPK is the older
RW chunk-tree form used by cutscenes, with UNCOMPRESSED float keyframes and bones bound
by NAME (not a numeric tag).

Chunk tree (each chunk = fourcc[4] + u32 size + body; bodies 4-byte aligned):
 ANPK { INFO(u32 count, char name[24])
 per actor: NAME(str) + DGAN { INFO(u32 numObjects)
 per bone: CPAN { ANIM(char bone[24], f32 ?, u32 numFrames, 12 pad)
 + KRxx keyframes } } }
Keyframes:
 KRT0 = 32 B/frame: f32 quat[4] (x,y,z,w) + f32 trans[3] + f32 time
 KR00 = 20 B/frame: f32 quat[4] + f32 time

Returns {block, anims:[ {name, seqs:[ {bone, keyType, numFrames, kf(bytes)} ]} ]}.
"""
import struct, sys

def _u32(b, o): return struct.unpack_from("<I", b, o)[0]
def _a4(n): return (n + 3) & ~3
def _cstr(b):
    z = b.find(b"\x00"); return b[:z if z >= 0 else len(b)].decode("latin1", "replace")

def decode(blob):
    blob = bytes(blob)
    if blob[:4] != b"ANPK":
        raise ValueError("not ANPK (got %r)" % blob[:4])
    o = 8                                            # skip ANPK fourcc+size
    if blob[o:o+4] != b"INFO":
        raise ValueError("ANPK: expected INFO")
    isz = _u32(blob, o+4); count = _u32(blob, o+8); block = _cstr(blob[o+12:o+36])
    o += 8 + isz
    anims = []
    for _a in range(count):
        if blob[o:o+4] != b"NAME":
            raise ValueError("ANPK: expected NAME @%d got %r" % (o, blob[o:o+4]))
        nsz = _u32(blob, o+4); aname = _cstr(blob[o+8:o+8+nsz]); o += 8 + _a4(nsz)
        if blob[o:o+4] != b"DGAN":
            raise ValueError("ANPK: expected DGAN @%d got %r" % (o, blob[o:o+4]))
        dsz = _u32(blob, o+4); dend = o + 8 + dsz; do = o + 8
        if blob[do:do+4] != b"INFO":
            raise ValueError("DGAN: expected INFO")
        iisz = _u32(blob, do+4); nobj = _u32(blob, do+8); do += 8 + iisz
        seqs = []
        for _b in range(nobj):
            if blob[do:do+4] != b"CPAN":
                raise ValueError("expected CPAN @%d got %r" % (do, blob[do:do+4]))
            csz = _u32(blob, do+4); cend = do + 8 + csz; co = do + 8
            if blob[co:co+4] != b"ANIM":
                raise ValueError("expected ANIM @%d got %r" % (co, blob[co:co+4]))
            asz = _u32(blob, co+4)
            bone = _cstr(blob[co+8:co+8+24])
            numFrames = _u32(blob, co+8+28)
            co += 8 + _a4(asz)
            kfc = blob[co:co+4]; ksz = _u32(blob, co+4); kf = blob[co+8:co+8+ksz]
            seqs.append({"bone": bone, "keyType": _cstr(kfc), "numFrames": numFrames, "kf": kf})
            do = cend
        anims.append({"name": aname, "seqs": seqs})
        o = dend
    return {"block": block, "anims": anims}

if __name__ == "__main__":
    sys.path.insert(0, "")
    from sa_img import SaImg
    img = SaImg("")
    name = sys.argv[1] if len(sys.argv) > 1 else "intro1a.ifp"
    pkg = decode(img.extract(name).rstrip(b"\x00"))
    print("block=%r anims=%d" % (pkg["block"], len(pkg["anims"])))
    for a in pkg["anims"]:
        b0 = a["seqs"][0] if a["seqs"] else None
        tot = sum(len(s["kf"]) for s in a["seqs"])
        print("  %-14s bones=%2d  bone0=%-16s type=%s frames=%d  kfbytes=%d"
              % (a["name"], len(a["seqs"]), b0["bone"] if b0 else "-",
                 b0["keyType"] if b0 else "-", b0["numFrames"] if b0 else 0, tot))
