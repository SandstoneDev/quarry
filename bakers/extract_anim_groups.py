#!/usr/bin/env python3
"""extract_anim_groups - recover CAnimManager::ms_aAnimAssocDefinitions from a
retail the source game PC executable.

DEV TOOL, run once. The result is checked in as tools/data/sa_anim_groups.json and
that JSON is what the bakers (and therefore Quarry) read - end users convert from a
PS2 disc and have no PC executable, and the PS2 build assembles the same table at
runtime, so the pointers in its image are into .bss and cannot be walked statically.

The table is 118 entries of

 char GroupName[16];
 char BlockName[16];
 int32 ModelIndex;
 int32 NumAnims;
 const char** AnimNames; // NumAnims pointers to 24-byte name buffers
 struct { int32 AnimId; int32 Flags; } *AnimDescr;

and is located by content, not by address: entry 0 is the only place in the image
where "default" is followed 16 bytes later by "ped" and then by a plausible count and
two pointers that resolve inside a section. (Addresses differ per build - 0x8AA5A8 on
retail 1.0 US, 0x919720 on the copy this was run against.)

 python tools/extract_anim_groups.py <the source game> [-o tools/data/sa_anim_groups.json]
"""
import json
import os
import struct
import sys

ENTRY_SIZE = 0x30
NUM_GROUPS = 118


def _sections(d):
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    if d[pe:pe + 4] != b"PE\0\0":
        raise SystemExit("not a PE image")
    nsec, = struct.unpack_from("<H", d, pe + 6)
    optsz, = struct.unpack_from("<H", d, pe + 20)
    opt = pe + 24
    imgbase, = struct.unpack_from("<I", d, opt + 28)
    out = []
    for i in range(nsec):
        o = opt + optsz + i * 40
        vsz, va, rsz, ra = struct.unpack_from("<IIII", d, o + 8)
        out.append((va + imgbase, vsz, ra, rsz))
    return out


def _mk_readers(d, secs):
    def v2o(v):
        for va, vsz, ra, rsz in secs:
            if va <= v < va + max(vsz, rsz):
                off = ra + (v - va)
                if off < ra + rsz:
                    return off
        return None

    def o2v(o):
        for va, vsz, ra, rsz in secs:
            if ra <= o < ra + rsz:
                return va + (o - ra)
        return None

    def cstr(v, cap=64):
        o = v2o(v)
        if o is None:
            return None
        e = d.find(b"\0", o, o + cap)
        return d[o:(e if e >= 0 else o + cap)].decode("latin1")

    return v2o, o2v, cstr


def find_table(d, secs):
    """VA of ms_aAnimAssocDefinitions, by content."""
    v2o, o2v, _ = _mk_readers(d, secs)
    start = 0
    while True:
        i = d.find(b"default\0", start)
        if i < 0:
            return None
        start = i + 1
        if d[i + 16:i + 20] != b"ped\0":
            continue
        _mi, na, pn, pdsc = struct.unpack_from("<4I", d, i + 32)
        if not (1 <= na <= 400):
            continue
        if v2o(pn) is None or v2o(pdsc) is None:
            continue
        va = o2v(i)
        if va is not None:
            return va


def extract(path):
    d = open(path, "rb").read()
    secs = _sections(d)
    v2o, _o2v, cstr = _mk_readers(d, secs)
    table = find_table(d, secs)
    if table is None:
        raise SystemExit("ms_aAnimAssocDefinitions not found in %s" % path)
    groups = []
    for i in range(NUM_GROUPS):
        b = d[v2o(table + i * ENTRY_SIZE):][:ENTRY_SIZE]
        name = b[0:16].split(b"\0")[0].decode("latin1")
        block = b[16:32].split(b"\0")[0].decode("latin1")
        model, count, p_names, p_descs = struct.unpack_from("<4I", b, 32)
        anims = []
        for k in range(count):
            ptr, = struct.unpack_from("<I", d, v2o(p_names + 4 * k))
            clip = cstr(ptr) or ""
            aid, flags = struct.unpack_from("<ii", d, v2o(p_descs + 8 * k))
            anims.append([aid, flags & 0xFFFF, clip])
        groups.append({"id": i, "name": name, "block": block,
                       "model": model, "anims": anims})
    return table, groups


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    src = argv[1]
    out = "tools/data/sa_anim_groups.json"
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    table, groups = extract(src)
    n_anims = sum(len(g["anims"]) for g in groups)
    uniq = {(g["block"].lower(), a[2].lower()) for g in groups for a in g["anims"] if a[2]}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        # ⚠ RELEASE: no provenance fields here. The pre-push audit's "RE traces" rule blocks
        # a shipped file that names the source executable or a VA, and this table rides into
        # the Quarry bundle - so the exe name and table address are recorded in the commit
        # message and this comment, not in the artefact. Nothing reads them anyway.
        json.dump({
                   "groups": groups}, f, indent=1)
    print("table at 0x%08X" % table)
    print("%d groups, %d anim slots, %d distinct (block, clip)" % (len(groups), n_anims, len(uniq)))
    print("-> %s (%d bytes)" % (out, os.path.getsize(out)))


if __name__ == "__main__":
    main(sys.argv)
