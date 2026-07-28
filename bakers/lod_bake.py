#!/usr/bin/env python3
"""Bake lod.bin - the real SA m_pLod link, one int32 per .pmap instance.

The .pmap drops the IPL `lod` field: sa_ipl.load_all() flattens all IPLs into one
list, but `lod` is a LOCAL index into the same IPL file's instance list, so once
flattened it can't be resolved. We recover it here WITHOUT re-baking the .pmap:

  1. read the shipped .pmap instance array (final, cell-sorted order) and key each
     instance by (pos float32, quat s16) -> its global instance index.
  2. re-read every IPL PER FILE; within each file resolve inst.lod -> the LOD
     instance, and record key(inst) -> key(lod_inst).
  3. for each .pmap instance, look up its LOD key, then that key's .pmap index.

The key is the exact on-disk bytes psp_scene writes (pos = verbatim float32, quat =
round(c*32767) clamped s16), so the match is byte-exact. Instances whose LOD lies
outside the baked region get -1 (no proxy -> the detail simply vanishes far out,
as on the real SA map edge).

lod.bin layout (little-endian, aligned 1:1 to PmapInstance[]):
    u32 magic 'PLOD'(0x444F4C50), u32 version(1), u32 count, int32 lod_idx[count]
"""
import os, struct, sys

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import sa_ipl, sa_ide
from gvcslib.sa_img import SaImg

PMAP = ""
OUT  = ""
# IPL placement source. The .pmap placements are float32-identical to one of these
# roots; the tool reports the match rate so we can confirm/switch.
# SA_ROOT env override first (the Quarry converter's user-disc extract): the
# region baker picks the root with the best link match, and instances exported
# FROM that root always match it best - the legacy roots stay as dev fallback.
ROOTS = ([os.environ["SA_ROOT"]] if os.environ.get("SA_ROOT") else []) + [
    "",          # PC
    "",  # PS2
]

S16 = 32767
def _qpack(rot):
    out = []
    for v in rot:
        q = int(round(v * S16))
        out.append(-32768 if q < -32768 else (S16 if q > S16 else q))
    return struct.pack('<4h', *out)

def ipl_key(inst, lodset):
    # pos f32 + quat s16 + is_lod byte. The is_lod byte is REQUIRED: in SA the
    # low-poly LOD model is usually placed at the SAME pos+rot as the detail it
    # replaces, so pos+quat alone collides a detail with its own LOD. The baker
    # marks the LOD with interior=1 by the very same DFF-name test, so the byte
    # matches the .pmap side exactly.
    is_lod = 1 if inst.model_id in lodset else 0
    return struct.pack('<3f', *inst.pos) + _qpack(inst.rot) + bytes((is_lod,))

def load_pmap_keys():
    f = open(PMAP, 'rb')
    v = struct.unpack_from('<20I', f.read(80), 0)
    ic, ioff = v[9], v[10]
    f.seek(ioff); idata = f.read(ic * 36)
    f.close()
    key_to_idx = {}; keys = []; dups = 0
    for k in range(ic):
        o = k * 36
        interior = struct.unpack_from('<i', idata, o + 28)[0]
        key = idata[o + 4:o + 24] + bytes((1 if interior else 0,))   # pos+quat+is_lod
        keys.append(key)
        if key in key_to_idx: dups += 1
        else: key_to_idx[key] = k
    return ic, keys, key_to_idx, dups

def load_ipl_links(root):
    """key(inst) -> key(lod_inst) for every IPL instance with a valid lod, plus the
    SET of all instance keys (to measure how well this root matches the .pmap)."""
    maps = None
    for cand in ("data/maps", "DATA/MAPS"):
        p = os.path.join(root, cand)
        if os.path.isdir(p): maps = p; break
    img = None
    for cand in ("models/gta3.img", "MODELS/GTA3.IMG"):
        p = os.path.join(root, cand)
        if os.path.isfile(p): img = SaImg(p); break
    # LOD model-ids (DFF name "lod*") - same test the baker uses for interior=1.
    ide = sa_ide.parse_maps(os.path.join(root, "DATA"))
    lodset = set()
    for mid, d in ide.items():
        dff = getattr(d, "dff", "") or ""
        if dff.lower().startswith("lod"): lodset.add(mid)
    lodkey = {}; allkeys = set(); nlinks = 0
    def add(insts):
        nonlocal nlinks
        for j, inst in enumerate(insts):
            allkeys.add(ipl_key(inst, lodset))
            lod = inst.lod
            if lod is not None and 0 <= lod < len(insts):
                lodkey[ipl_key(inst, lodset)] = ipl_key(insts[lod], lodset); nlinks += 1
    if maps:
        for r, _d, files in os.walk(maps):
            for fn in files:
                if fn.lower().endswith(".ipl"):
                    try: add(sa_ipl.parse_text_ipl(os.path.join(r, fn)))
                    except Exception: pass
    if img:
        for n in img.names():
            if n.lower().endswith(".ipl"):
                try: add(sa_ipl.parse_binary_ipl(img.extract(n)))
                except Exception: pass
    return lodkey, allkeys, nlinks

def main():
    ic, keys, key_to_idx, dups = load_pmap_keys()
    print("pmap instances: %d  (key collisions: %d)" % (ic, dups))

    best = None
    for root in ROOTS:
        if not os.path.isdir(root):
            print("  skip (missing):", root); continue
        lodkey, allkeys, nlinks = load_ipl_links(root)
        matched = sum(1 for k in keys if k in allkeys)
        rate = 100.0 * matched / ic if ic else 0.0
        print("  root %-55s  IPL links %d  pmap-match %d/%d (%.1f%%)"
              % (os.path.basename(root), nlinks, matched, ic, rate))
        if best is None or matched > best[0]:
            best = (matched, root, lodkey)
    if best is None:
        print("ERROR: no IPL root available"); return 1
    matched, root, lodkey = best
    print("using root:", root)

    lod_idx = [-1] * ic
    resolved = out_region = 0
    for k in range(ic):
        lk = lodkey.get(keys[k])
        if lk is None: continue
        gi = key_to_idx.get(lk, -1)
        if gi >= 0 and gi != k: lod_idx[k] = gi; resolved += 1   # drop self-links
        elif gi < 0: out_region += 1
    print("resolved LOD links: %d  (lod outside region: %d)" % (resolved, out_region))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'wb') as o:
        o.write(struct.pack('<3I', 0x444F4C50, 1, ic))
        o.write(struct.pack('<%di' % ic, *lod_idx))
    print("wrote %s : %d bytes" % (OUT, 12 + 4 * ic))
    return 0

if __name__ == "__main__":
    sys.exit(main())
