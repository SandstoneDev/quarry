#!/usr/bin/env python3
"""PS2 DFF geometry unpack PROTOTYPE (librw ps2.cpp as the reference).

Parses one PS2 DFF: Geometry + BinMeshPLG + NativeDataPLG -> per-mesh VIF
batches -> welded vertex list (librw objUninstance order) -> validates against
the PC version of the same model (vertex count + prelit bytes + night chunk).

VIF chain layout (librw MatPipeline::instance, numBrokenAttribs==0):
 16B header: DMAret | size-1, 0, VIF_FLUSH, msk_flush
 per batch:
 per attrib: [STMASK+mask | NOP+STMOD] STCYCL UNPACK(tag) payload(QWC-padded)
 ITOP | nverts ; MSCALF/MSCNT ; then NOP,NOP (mid) or FLUSH,msk (last)
 tristrip batches OVERLAP by 2 verts (collectData: datap advance nverts-2).

UNPACK tag: (attribCode & 0xFF004000) | 0x8000 | nverts<<16 | vuAddr
 codes: 0x68 V3_32(xyz f32) 0x6D V4_16 0x6E V4_8(rgba) 0x6A V3_8(normal)
 0x64 V2_32(uv f32) 0x65 V2_16 | 0x4000 = unsigned
"""
import struct
import sys

CHUNK_STRUCT, CHUNK_EXT = 0x01, 0x03
CHUNK_GEOMLIST, CHUNK_GEOM = 0x1A, 0x0F
CHUNK_BINMESH, CHUNK_NATIVE = 0x50E, 0x510
CHUNK_NIGHT = 0x0253F2F9

FMT_TRISTRIP = 0x01
FMT_PRELIT = 0x08
FMT_NATIVE = 0x01000000


def chunks(b, off, end):
    while off + 12 <= end:
        cid, size, _ = struct.unpack_from("<III", b, off)
        yield cid, off + 12, size
        off += 12 + size


def find(b, off, end, cid):
    for c, o, s in chunks(b, off, end):
        if c == cid:
            return o, s
    return None, 0


def descend(b, path):
    off, end = 0, len(b)
    for want in path:
        o, s = find(b, off, end, want)
        assert o is not None, f"chunk {want:#x} not found"
        off, end = o, o + s
    return off, end


def unpack_info(cmd):
    """VIF UNPACK cmd byte -> (ncomp, bytes_per_comp) or None.
 cmd = 011m vnvl: vn[3:2]+1 components, vl[1:0]: 0=32bit 1=16bit 2=8bit."""
    if (cmd & 0x60) != 0x60:
        return None
    vn = (cmd >> 2) & 3
    vl = cmd & 3
    if vl == 3:
        return None                            # V4-5 packed, not used here
    return vn + 1, 4 >> vl


def parse_ps2(path):
    b = open(path, "rb").read()
    # Geometry struct: flags
    g_off, g_end = descend(b, [0x10, CHUNK_GEOMLIST, CHUNK_GEOM])
    s_off, s_size = find(b, g_off, g_end, CHUNK_STRUCT)
    fmt, numTri, numVerts, numMorphs = struct.unpack_from("<IIII", b, s_off)
    print(f"  geom fmt={fmt:#x} native={bool(fmt & FMT_NATIVE)} "
          f"declared verts={numVerts} tris={numTri}")

    ext_off, ext_size = find(b, g_off, g_end, CHUNK_EXT)
    ext_end = ext_off + ext_size
    bm_off, bm_size = find(b, ext_off, ext_end, CHUNK_BINMESH)
    bmflags, numMeshes, totalIdx = struct.unpack_from("<III", b, bm_off)
    print(f"  binmesh flags={bmflags} meshes={numMeshes} totalIndices={totalIdx}")
    meshes = []
    o = bm_off + 12
    for _ in range(numMeshes):
        nidx, mat = struct.unpack_from("<II", b, o)
        meshes.append((nidx, mat))
        o += 8
    tristrip = bmflags == FMT_TRISTRIP

    nd_off, nd_size = find(b, ext_off, ext_end, CHUNK_NATIVE)
    nd_end = nd_off + nd_size
    p_off, p_size = find(b, nd_off, nd_end, CHUNK_STRUCT)
    assert p_off is not None, "native data has no struct wrapper"
    platform = struct.unpack_from("<I", b, p_off)[0]
    assert platform == 4, f"platform {platform} != PS2"
    o = p_off + 4

    verts, uvs, cols, norms = [], [], [], []
    mesh_ranges = []
    for mi, (nidx, mat) in enumerate(meshes):
        dataSize, noFix = struct.unpack_from("<II", b, o)
        o += 8
        raw = b[o:o + dataSize]
        o += dataSize
        v0 = len(verts)
        parse_chain(raw, nidx, tristrip, verts, uvs, cols, norms)
        mesh_ranges.append((v0, len(verts), mat))
        print(f"  mesh {mi}: dataSize={dataSize} noFix={noFix} "
              f"indices={nidx} -> stream verts={len(verts) - v0}")

    night = None
    n_off, n_size = find(b, ext_off, ext_end, CHUNK_NIGHT)
    if n_off is not None:
        night = b[n_off:n_off + n_size]
    return dict(verts=verts, uvs=uvs, cols=cols, norms=norms,
                meshes=mesh_ranges, night=night, numVerts=numVerts,
                tristrip=tristrip, fmt=fmt)


def parse_chain(raw, expect_idx, tristrip, verts, uvs, cols, norms):
    """Walk the VIF chain, collecting per-batch attribute payloads with the
 tristrip 2-vert overlap dropped (librw collectData semantics)."""
    # DMA-chain walk (SA world = broken-out sections referenced by DMAref).
    # Per batch: DMAref pairs (payload at addr, format+slot in the UNPACK tag)
    # then a DMAcnt inline block whose ITOP carries the batch's TRUE nverts
    # (ref counts are padded up to whole quadwords).
    o = 0
    first = True
    pend = []                                  # (slot, ncomp, csz, payload)
    got = 0
    while o + 16 <= len(raw):
        w0, addr, v0, v1 = struct.unpack_from("<4I", raw, o)
        did = (w0 >> 28) & 7
        qwc = w0 & 0xFFFF
        if did == 3:                           # DMAref -> one attribute batch
            info = unpack_info((v1 >> 24) & 0x7F)
            if info:
                slot = v1 & 0xFF
                pend.append((slot, info[0], info[1],
                             raw[0:0]  # placeholder, replaced below
                             if False else (addr, qwc)))
            o += 16
            continue
        if did in (1, 6):                      # DMAcnt / DMAret inline block
            inline = raw[o + 16: o + 16 + qwc * 16]
            nverts = None
            for k in range(0, len(inline) - 3, 4):
                t = struct.unpack_from("<I", inline, k)[0]
                if ((t >> 24) & 0x7F) == 0x04: # VIF_ITOP
                    nverts = t & 0xFFFF
                    break
            if nverts is not None and pend:
                skip = 0 if first else (2 if tristrip else 0)
                for slot, ncomp, csz, (a, q) in pend:
                    payload = raw[a * 16: a * 16 + q * 16]
                    asz = ncomp * csz
                    for i in range(skip, nverts):
                        el = payload[i * asz:(i + 1) * asz]
                        if slot == 0:          # XYZ(+flags) s16 or f32
                            if csz == 2:
                                vals = struct.unpack("<%dh" % ncomp, el)
                            else:
                                vals = struct.unpack("<%df" % ncomp, el)
                            verts.append(vals[:3] + (vals[3] if ncomp > 3 else 0,))
                        elif slot == 1:        # UV
                            if csz == 2:
                                u, v = struct.unpack("<2h", el[:4])
                                uvs.append((u, v))
                            else:
                                uvs.append(struct.unpack("<2f", el[:8]))
                        elif slot == 2:        # colours
                            if csz == 2:
                                cols.append(struct.unpack("<%dh" % ncomp, el))
                            else:
                                cols.append(tuple(el))
                        elif slot == 3:
                            norms.append(struct.unpack("<%db" % ncomp, el[:ncomp]))
                got += nverts - (0 if first else (2 if tristrip else 0))
                first = False
                pend = []
            o += 16 + qwc * 16
            if did == 6 and got >= expect_idx:
                break
            continue
        o += 16                                # unknown tag: step a quadword
    return got


def parse_pc(path):
    b = open(path, "rb").read()
    g_off, g_end = descend(b, [0x10, CHUNK_GEOMLIST, CHUNK_GEOM])
    s_off, s_size = find(b, g_off, g_end, CHUNK_STRUCT)
    fmt, numTri, numVerts, numMorphs = struct.unpack_from("<IIII", b, s_off)
    o = s_off + 16
    prelit = None
    if fmt & FMT_PRELIT:
        prelit = b[o:o + numVerts * 4]
        o += numVerts * 4
    numTexSets = (fmt >> 16) & 0xFF
    if numTexSets == 0 and fmt & 0x04:
        numTexSets = 1
    uvs = b[o:o + numVerts * 8 * numTexSets]
    o += numVerts * 8 * numTexSets
    o += numTri * 8                            # face indices
    o += 16 + 8                                # sphere + hasVerts/hasNormals
    verts = [struct.unpack_from("<3f", b, o + i * 12) for i in range(numVerts)]
    ext_off, ext_size = find(b, g_off, g_end, CHUNK_EXT)
    night = None
    n_off, n_size = find(b, ext_off, ext_off + ext_size, CHUNK_NIGHT)
    if n_off is not None:
        night = b[n_off:n_off + n_size]
    return dict(fmt=fmt, numVerts=numVerts, numTri=numTri,
                prelit=prelit, verts=verts, night=night)


def main():
    ps2 = parse_ps2(sys.argv[1])
    print(f"PS2 stream verts total={len(ps2['verts'])} uvs={len(ps2['uvs'])} "
          f"cols={len(ps2['cols'])} norms={len(ps2['norms'])}")

    # librw-style weld: first-seen order; key WITHOUT the xyz w flag (ADC
    # differs on batch seams for otherwise-identical vertices)
    welded = {}
    order = []
    wcols = []
    for i, v in enumerate(ps2["verts"]):
        key = (v[:3],
               ps2["uvs"][i] if i < len(ps2["uvs"]) else None,
               ps2["cols"][i] if i < len(ps2["cols"]) else None)
        if key not in welded:
            welded[key] = len(order)
            order.append(key)
            wcols.append(key[2])
    print(f"PS2 welded verts={len(order)}")

    pc = parse_pc(sys.argv[2])
    print(f"PC  numVerts={pc['numVerts']} tris={pc['numTri']} fmt={pc['fmt']:#x}")

    if ps2["night"] and pc["night"]:
        print(f"night chunks: ps2={len(ps2['night'])} pc={len(pc['night'])} "
              f"payload identical={ps2['night'][4:] == pc['night'][4:]}")

    # raw ranges -> deduce scales
    xs = [v[0] for v in ps2["verts"]]
    ws = [v[3] for v in ps2["verts"]]
    us = [u[0] for u in ps2["uvs"]]
    c0 = [c[0] for c in ps2["cols"]]
    c3 = [c[3] for c in ps2["cols"]]
    pxs = [v[0] for v in pc["verts"]]
    print(f"PS2 raw X range {min(xs)}..{max(xs)}  W set {sorted(set(ws))[:6]}")
    print(f"PC  X range {min(pxs):.3f}..{max(pxs):.3f}  "
          f"ratio {min(xs)/min(pxs):.2f} / {max(xs)/max(pxs):.2f}")
    print(f"PS2 raw U range {min(us)}..{max(us)}  col0 {min(c0)}..{max(c0)} "
          f"col3 {min(c3)}..{max(c3)}")
    print(f"first 3 PS2 verts: {ps2['verts'][:3]}")
    print(f"first 3 PC  verts: {pc['verts'][:3]}")
    print(f"first 3 PS2 cols: {ps2['cols'][:3]}")
    print(f"PC prelit first 3: {[tuple(pc['prelit'][i*4:i*4+4]) for i in range(3)] if pc['prelit'] else None}")

    # position-keyed match: for sample PS2 verts find the PC vertex at the same
    # spot, compare PS2 16-bit colour vs PC prelit (day) and PC night chunk.
    pcmap = {}
    for i, v in enumerate(pc["verts"]):
        pcmap.setdefault((round(v[0] * 128), round(v[1] * 128), round(v[2] * 128)), []).append(i)
    night = pc["night"][4:]
    hits = 0
    print("--- pos-matched colour probes ---")
    for i in range(0, len(order), max(1, len(order) // 8)):
        pos, uvk, col = order[i]
        cand = pcmap.get(pos)
        if not cand:
            continue
        j = cand[0]
        day = tuple(pc["prelit"][j * 4:j * 4 + 4])
        nit = tuple(night[j * 4:j * 4 + 4])
        raw = [c & 0xFFFF for c in col]
        hi = tuple(r >> 8 for r in raw)
        lo = tuple(r & 0xFF for r in raw)
        print(f" ps2[{i}] raw={raw} hi={hi} lo={lo} | pc[{j}] day={day} night={nit}")
        hits += 1
        if hits >= 8:
            break


if __name__ == "__main__":
    main()
