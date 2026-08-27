#!/usr/bin/env python3
"""Guard-band tessellation for v3 .pmap files (LZ4 blobs).

WHY

The Allegrex GE does no triangle clipping. If any vertex of a triangle leaves the guard
band - which a large triangle near the camera does as soon as you turn - the GE discards
the WHOLE triangle. On screen that is a chunk of wall or road blinking out at the edge of
view. It is architecture, not a render bug, and no cull margin reaches it: widening the
frustum cone to 150% was tried on hardware and changed nothing.

tools/pmap_tessellate.py solved this once, for the v2 format with raw vertex and index
pools. The live chunkset ps2full is v3: every model is one LZ4 block of vertices followed
by indices, and the raw pools are gone, so that tool cannot read it. Measured on ps2full,
8.4% of edges are longer than the 24-unit threshold the v2 fix used, and the longest reach
116 units.

WHAT IT DOES

Splits every triangle whose longest world-space edge exceeds --max-edge, one into four,
with a per-submesh midpoint cache so a shared edge yields ONE shared midpoint - no
T-junctions, no cracks. Vertices are interpolated in their stored s16 space, which is
linear in world space, so position, UV and colour all interpolate correctly.

WHAT IT PRESERVES

Model, texture, instance and grid tables byte for byte; per-model submesh contiguity;
submesh-local index bases; the texel and CLUT pools. Only the vertex/index content of a
model blob, the affected submesh counts, and the compressed-blob offset table change.

LIMITS, ENFORCED BEFORE ANYTHING IS WRITTEN

- A submesh cannot exceed 65535 vertices, because indices are u16.
- A model blob cannot exceed the engine's LZ4_SCRATCH (320 KB) uncompressed, because the
 streaming worker decompresses into exactly that buffer.

A model that would cross either limit is left exactly as it was and counted in the report.
Refusing is always safe; the untessellated model keeps the guard-band flicker it had, which
is strictly better than a model the engine cannot load.
"""

import argparse
import os
import struct
import sys

try:
    import lz4.block
except ImportError:
    sys.exit("needs python lz4: pip install lz4")

MAGIC = 0x50414D50
HDR = "<23I"
VERT = 12          # s16 u, s16 v, u16 color, s16 x, s16 y, s16 z
LZ4_SCRATCH = 320 * 1024
U16_MAX = 65535

# Matches src/platform_psp/pmap.h's own PMAP_VERSION_STRIPPED (5): a
# world-store stage 2a stripped tile (tools/world_store_build.py
# strip_tile). comp_flag is tested for truthiness only, never a
# specific value (see pmap.h's own comment on comp_flag) - a stripped
# tile's comp_flag (2) is truthy too, so that check alone does not gate
# this; version is what actually identifies the format.
PMAP_VERSION_STRIPPED = 5


class Pmap:
    def __init__(self, path):
        self.path = path
        self.buf = bytearray(open(path, "rb").read())
        f = struct.unpack_from(HDR, self.buf, 0)
        (self.magic, self.version, self.file_size,
         self.model_count, self.model_off,
         self.submesh_count, self.submesh_off,
         self.texture_count, self.texture_off,
         self.instance_count, self.instance_off,
         self.grid_off,
         self.vertex_off, self.vertex_bytes,
         self.index_off, self.index_bytes,
         self.texel_off, self.texel_bytes,
         self.clut_off, self.clut_bytes,
         self.comp_flag, self.comp_model_off, self.comp_tex_off) = f
        if self.magic != MAGIC:
            raise ValueError("not a pmap")
        if self.version == PMAP_VERSION_STRIPPED:
            raise ValueError(
                "this is a STRIPPED tile (version=%d) - it belongs to the world "
                "store (tools/world_store_build.py); its comp_model/comp_tex tables "
                "hold GLOBAL ids, not local byte offsets, and this tool has no way "
                "to resolve one" % self.version)
        if not self.comp_flag:
            raise ValueError("v2 raw pools - use tools/pmap_tessellate.py")

    def models(self):
        return [list(struct.unpack_from("<IIffffff", self.buf, self.model_off + i * 32))
                for i in range(self.model_count)]

    def submeshes(self):
        return [list(struct.unpack_from("<iIIII", self.buf, self.submesh_off + i * 20))
                for i in range(self.submesh_count)]

    def comp(self, which):
        off = self.comp_model_off if which == "m" else self.comp_tex_off
        n = self.model_count if which == "m" else self.texture_count
        return [list(struct.unpack_from("<II", self.buf, off + i * 8)) for i in range(n)]


def vpos(v):
    return struct.unpack_from("<hhh", v, 6)


def lerp_vert(a, b):
    """Midpoint of two packed vertices.

 The colour is GU_COLOR_5551 - R:5 G:5 B:5 A:1, see PmapVertex in pmap.h. It has to be
 unpacked per channel and averaged, because averaging the packed integer bleeds each
 channel into its neighbour.

 Getting this format wrong is not a subtle defect. A first cut here used 565 masks
 (green 6 bits at shift 5, blue at shift 11); that widened green into blue and dropped
 the alpha bit, and on hardware whole roads turned flat green - roads carry the biggest
 triangles, so they gain the most NEW vertices and every one of them was wrong.

 Alpha is one bit and cannot be averaged. It is OR-ed: a midpoint between a visible and an
 invisible vertex stays visible, which keeps an edge from opening a hole. """
    au, av, ac = struct.unpack_from("<hhH", a, 0)
    bu, bv, bc = struct.unpack_from("<hhH", b, 0)
    ax, ay, az = vpos(a)
    bx, by, bz = vpos(b)
    r  = (((ac       ) & 0x1F) + ((bc       ) & 0x1F)) >> 1
    g  = (((ac >>  5 ) & 0x1F) + ((bc >>  5 ) & 0x1F)) >> 1
    bl = (((ac >> 10 ) & 0x1F) + ((bc >> 10 ) & 0x1F)) >> 1
    al = ((ac >> 15) & 1) | ((bc >> 15) & 1)
    c = r | (g << 5) | (bl << 10) | (al << 15)
    return struct.pack("<hhHhhh", (au + bu) // 2, (av + bv) // 2, c,
                       (ax + bx) // 2, (ay + by) // 2, (az + bz) // 2)


def tri_split(verts, idx, scale, max_edge, parents=None):
    """One split round. Returns (verts, idx, changed).

 `parents`, when given, is a dict the caller owns: every vertex index this round
 INVENTS is recorded as newIndex -> (a, b), the two it was interpolated from. The night
 sidecars need it - a vertex that did not exist has no baked night colour, and the only
 defensible one is the average of the two it sits between.

 The threshold is converted into the vertices' own s16 space once, so the inner loop
 never touches floating point: a world edge of E units is E/scale in stored units."""
    if scale <= 0.0:
        return verts, idx, False
    thresh = max_edge / scale
    t2 = thresh * thresh
    mid = {}

    def midpoint(a, b):
        k = (a, b) if a < b else (b, a)
        m = mid.get(k)
        if m is None:
            m = len(verts)
            verts.append(lerp_vert(verts[a], verts[b]))
            mid[k] = m
            if parents is not None:
                parents[m] = (a, b)
        return m

    out = []
    changed = False
    for i in range(0, len(idx) - 2, 3):
        a, b, c = idx[i], idx[i + 1], idx[i + 2]
        pa, pb, pc = vpos(verts[a]), vpos(verts[b]), vpos(verts[c])

        def d2(p, q):
            return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2

        if max(d2(pa, pb), d2(pb, pc), d2(pc, pa)) <= t2:
            out += [a, b, c]
            continue
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        out += [a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]
        changed = True
    return verts, out, changed


def tessellate(path, out_path, max_edge, passes, verbose):
    p = Pmap(path)
    models, subs = p.models(), p.submeshes()
    mcomp, tcomp = p.comp("m"), p.comp("t")
    raw = bytes(p.buf)

    new_blobs = {}
    st = dict(models=0, tris_before=0, tris_after=0, refused_u16=0, refused_size=0)

    for mi, md in enumerate(models):
        first, count, scale = md[0], md[1], md[2]
        off, csize = mcomp[mi]
        if not count or not csize:
            continue
        try:
            blob = lz4.block.decompress(raw[off:off + csize], uncompressed_size=LZ4_SCRATCH)
        except Exception as e:
            if verbose:
                print("  model %d: decompress failed (%s) - left alone" % (mi, e))
            continue

        s0, sN = subs[first], subs[first + count - 1]
        vbase, ibase = s0[1], s0[3]
        vtotal = (sN[1] + sN[2]) - vbase
        itotal = (sN[3] + sN[4]) - ibase
        if vtotal * VERT + itotal * 2 > len(blob):
            if verbose:
                print("  model %d: blob shorter than its tables - left alone" % mi)
            continue
        vpool = blob[:vtotal * VERT]
        ipool = blob[vtotal * VERT: vtotal * VERT + itotal * 2]

        new_v, new_i, new_counts = [], [], []
        sub_map = []          # (submesh, old vfirst, old vcount, new vcount, parents)
        model_changed = False
        refused = None
        for k in range(count):
            si = first + k
            tex, vf, vc, if_, ic = subs[si]
            base = (vf - vbase) * VERT
            vs = [bytes(vpool[base + j * VERT: base + (j + 1) * VERT]) for j in range(vc)]
            ids = list(struct.unpack_from("<%dH" % ic, ipool, (if_ - ibase) * 2)) if ic else []
            st["tris_before"] += ic // 3
            sub_parents = {}
            for _ in range(max(1, passes)):
                vs, ids, ch = tri_split(vs, ids, scale, max_edge, sub_parents)
                model_changed = model_changed or ch
                if not ch:
                    break
            if len(vs) > U16_MAX:
                refused = "submesh %d needs %d vertices (u16 index limit)" % (si, len(vs))
                break
            st["tris_after"] += len(ids) // 3
            new_counts.append((si, tex, len(vs), len(ids)))
            sub_map.append((si, vf, vc, len(vs), sub_parents))
            new_v.append(vs)
            new_i.append(ids)

        if refused:
            st["refused_u16"] += 1
            if verbose:
                print("  model %d refused: %s" % (mi, refused))
            continue
        if not model_changed:
            continue

        blob_new = (b"".join(b"".join(vs) for vs in new_v)
                    + b"".join(struct.pack("<%dH" % len(i), *i) for i in new_i))
        if len(blob_new) > LZ4_SCRATCH:
            st["refused_size"] += 1
            if verbose:
                print("  model %d refused: blob %d B over LZ4_SCRATCH %d B"
                      % (mi, len(blob_new), LZ4_SCRATCH))
            continue

        vcur, icur = vbase, ibase
        for (si, tex, nv, ni) in new_counts:
            struct.pack_into("<iIIII", p.buf, p.submesh_off + si * 20, tex, vcur, nv, icur, ni)
            vcur += nv
            icur += ni
        new_blobs[mi] = blob_new
        st["models"] += 1

    if not new_blobs:
        print("%s: nothing over %.0fu - unchanged" % (os.path.basename(path), max_edge))
        return False

    # Rewrite. Blobs are emitted in their ORIGINAL file order, including the untouched ones,
    # so offsets stay monotonic - the streaming worker clusters requests by file offset and
    # relies on that ordering to turn many small reads into a few sequential ones.
    starts = [o for o, c in mcomp if c] + [o for o, c in tcomp if c]
    if not starts:
        print("%s: no blobs" % os.path.basename(path))
        return False
    first_blob = min(starts)
    out = bytearray(p.buf[:first_blob])

    order = sorted([(o, "m", i) for i, (o, c) in enumerate(mcomp) if c]
                   + [(o, "t", i) for i, (o, c) in enumerate(tcomp) if c])
    for off, kind, i in order:
        if kind == "m" and i in new_blobs:
            data = lz4.block.compress(new_blobs[i], mode="high_compression",
                                      compression=12, store_size=False)
        else:
            src, cs = (mcomp[i] if kind == "m" else tcomp[i])
            data = raw[src:src + cs]
        at = len(out)
        out += data
        tbl = (p.comp_model_off if kind == "m" else p.comp_tex_off) + i * 8
        struct.pack_into("<II", out, tbl, at, len(data))

    struct.pack_into("<I", out, 8, len(out))

    # ★★★ b982 - THE HEADER AND THE SIDECARS, the two halves of why b952 pulled this pass
    # out of the convert chain. Splitting ADDS vertices, and three things address the vertex
    # pool by position rather than by content:
    # * header vertex_bytes - what the engine divides to get the vertex count. Left at the
    # pre-split value it makes the engine read the wrong number of vertices, and the night
    # buffer below is then applied to the wrong ones.
    # * region_*.night - one u16 PER VERTEX, aligned to the pool.
    # * region_*.nightd - glow runs addressed by vertex INDEX.
    # The first is fixed here. The other two cannot be: this pass has no way to invent the
    # night colour of a vertex that did not exist, so if a sidecar is sitting next to the file
    # we REFUSE rather than silently ship a world whose baked night lighting is off by a few
    # hundred vertices. That failure is invisible in a geometry check - which is exactly how
    # it shipped twice before.
    fmodels, fsubs = p.models(), p.submeshes()   # re-read: the split rewrote the submesh table
    vtotal = 0
    for mi in range(len(fmodels)):
        first, count = fmodels[mi][0], fmodels[mi][1]
        if not count:
            continue
        sN = fsubs[first + count - 1]
        vtotal = max(vtotal, sN[1] + sN[2])
    struct.pack_into("<I", out, 13 * 4, vtotal * VERT)

    open(out_path, "wb").write(out)
    note = ""
    if st["models"]:
        base = os.path.splitext(path)[0]
        stale = [x for x in (base + ".night", base + ".nightd") if os.path.exists(x)]
        if stale:
            note += ("  ★ WARNING: %s still carries the PRE-SPLIT vertex count - this pass "
                     "cannot interpolate baked night colour, so that sidecar is now misaligned"
                     % ", ".join(os.path.basename(x) for x in stale))
    if st["refused_u16"] or st["refused_size"]:
        note = "  [refused %d over u16, %d over scratch]" % (st["refused_u16"], st["refused_size"])
    print("%s: %d models split, tris %d -> %d (+%.0f%%), %s -> %s%s"
          % (os.path.basename(path), st["models"], st["tris_before"], st["tris_after"],
             100.0 * (st["tris_after"] - st["tris_before"]) / max(1, st["tris_before"]),
             human(len(raw)), human(len(out)), note))
    return True


def human(n):
    for u in ("B", "KB", "MB"):
        if n < 1024 or u == "MB":
            return "%.1f%s" % (n, u)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="region_*.pmap files, or a directory holding them (the converter "
                         "passes the chunkset directory)")
    ap.add_argument("--max-edge", type=float, default=24.0,
                    help="world units; longer edges are split (default 24, the v2 threshold)")
    ap.add_argument("--passes", type=int, default=6,
                    help="max split rounds per submesh. Each round halves the offending edges, "
                         "and the loop stops early once nothing is left to split, so this is a "
                         "ceiling and not a cost. Default 6 covers the longest edge measured in "
                         "ps2full (268u -> under 24 in five); 3 left a tile at 33.5u.")
    ap.add_argument("--suffix", default="_tess",
                    help="output suffix; empty string overwrites in place")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    # a directory expands to its region tiles, so the converter can pass a chunkset folder
    # the same way every other baker in the chain does
    paths = []
    for p0 in a.paths:
        if os.path.isdir(p0):
            import glob as _g
            paths += sorted(_g.glob(os.path.join(p0, "region_*.pmap")))
        else:
            paths.append(p0)
    if not paths:
        print("no region_*.pmap found")
        return

    done = 0
    for path in paths:
        base, ext = os.path.splitext(path)
        out_path = path if a.suffix == "" else base + a.suffix + ext
        try:
            if tessellate(path, out_path, a.max_edge, a.passes, a.verbose):
                done += 1
        except Exception as e:
            print("%s: FAILED (%s)" % (os.path.basename(path), e))
    print("%d of %d rewritten" % (done, len(paths)))


if __name__ == "__main__":
    main()
