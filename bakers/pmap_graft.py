#!/usr/bin/env python3
"""pmap_graft.py - transplant instances a DONOR chunkset has but a TARGET lacks.

Why: the production set (chunks_v2_world_uvfix) carries 44157 instances, but
every surviving whole-map source .pmap holds only 43806 - 351 instances (the
Grove bridge LOD sections + prop clusters) were added by an undocumented
post-step whose tool is lost. Any fresh re-bake therefore ships a poorer world
(the b39x "bridge vanished" reports). This tool closes that hole for ANY future
set: diff each region tile against the donor by instance position+quat, and
graft the missing instances - together with their models and textures when the
target doesn't already carry identical ones (content-hash match) - into the
target's RAW v2 tiles.

Run BEFORE col/lod/road/dyn sidecar bakes and BEFORE pmap_lz4 (the sidecars must
see the grafted instances; the tiles must still be v2).

Donor tiles may be v2 or v3 (v3 per-model/per-texture LZ4 blobs are inflated on
demand - same layout pmap.c streams).

Usage: python pmap_graft.py <donor_dir> <target_dir>
"""
import glob
import hashlib
import os
import struct
import sys

import lz4.block

GVCS = os.environ.get("GVCS_ROOT", "")
sys.path.insert(0, GVCS)
from gvcslib import psp_scene

VERT = 12          # PmapVertex bytes
IDX = 2            # u16 index

PMAP_MAGIC = 0x50414D50   # 'PMAP' little-endian
# Matches src/platform_psp/pmap.h's own PMAP_VERSION_STRIPPED (5): a world-store
# stage 2a stripped tile (tools/world_store_build.py strip_tile). Its comp_model/
# comp_tex tables hold GLOBAL ids, not local byte offsets - a donor in this format
# read by the code below (which assumes offsets) silently sliced the wrong bytes at
# a handful of real positions and fed the result to LZ4, which in most cases raised
# but in 3 of 53917 simulated cases decompressed successfully to the WRONG, SHORT
# length with no error at all (see model_pools/texture_pools's own length
# assertion, which is what actually catches that class of bug - this constant only
# closes the one door that was open by name).
PMAP_VERSION_STRIPPED = 5


class DonorTile:
    """Minimal v2/v3 reader: raw tables + on-demand pool slices (v3 inflates)."""

    def __init__(self, path):
        self.path = path
        d = self.data = open(path, "rb").read()
        (self.magic, self.version, _fsz,
         self.model_count, self.model_off,
         self.submesh_count, self.submesh_off,
         self.texture_count, self.texture_off,
         self.instance_count, self.instance_off,
         self.grid_off,
         self.vertex_off, self.vertex_bytes,
         self.index_off, self.index_bytes,
         self.texel_off, self.texel_bytes,
         self.clut_off, self.clut_bytes) = struct.unpack_from("<20I", d, 0)
        # No check here at all previously - found in review. This tool grafts
        # geometry from a donor into a target's RAW pools and runs BEFORE the rest
        # of the bake chain; a wrong-format donor must be refused loudly, not read
        # as though its tables meant what they mean for a real v2/v3 tile. Magic
        # first (this module never checked it at all before), then an EXACT
        # whitelist - not the open-ended `self.version >= 3` this used to branch
        # on, which a stripped tile (5) or a v4 UVR tile also satisfies - matching
        # this file's OWN docstring: "Donor tiles may be v2 or v3."
        if self.magic != PMAP_MAGIC:
            raise ValueError("%s: not a .pmap (bad magic %08x)" % (path, self.magic))
        if self.version == PMAP_VERSION_STRIPPED:
            raise ValueError(
                "%s: this is a STRIPPED tile (version=%d) - pmap_graft.py cannot read "
                "one as a donor; its comp_model/comp_tex tables hold GLOBAL ids into a "
                "companion world.idx/world.dat, not local byte offsets, and it belongs "
                "to the world store (tools/world_store_build.py)" % (path, self.version))
        if self.version not in (2, 3):
            raise ValueError(
                "%s: donor tiles must be v2 or v3 (got version=%d) - this tool's own "
                "docstring says so" % (path, self.version))

        self.comp_models = self.comp_tex = None
        if self.version == 3:
            flag, cmo, cto = struct.unpack_from("<3I", d, 80)
            if flag:
                self.comp_models = [struct.unpack_from("<2I", d, cmo + 8 * i)
                                    for i in range(self.model_count)]
                self.comp_tex = [struct.unpack_from("<2I", d, cto + 8 * i)
                                 for i in range(self.texture_count)]

    def submesh(self, i):
        return struct.unpack_from("<i4I", self.data, self.submesh_off + 20 * i)

    def model(self, i):
        o = self.model_off + 32 * i
        fs, sc = struct.unpack_from("<2I", self.data, o)
        scale, cx, cy, cz, br, dd = struct.unpack_from("<6f", self.data, o + 8)
        return fs, sc, scale, (cx, cy, cz), br, dd

    def model_span(self, i):
        fs, sc, *_ = self.model(i)
        v0 = v1 = i0 = i1 = None
        for s in range(fs, fs + sc):
            _t, vf, vc, if_, ic = self.submesh(s)
            v0 = vf if v0 is None else min(v0, vf)
            v1 = vf + vc if v1 is None else max(v1, vf + vc)
            i0 = if_ if i0 is None else min(i0, if_)
            i1 = if_ + ic if i1 is None else max(i1, if_ + ic)
        return v0, v1, i0, i1

    def model_pools(self, i):
        """(vertex_bytes, index_bytes) for model i, decompressed if v3."""
        v0, v1, i0, i1 = self.model_span(i)
        vb, ib = (v1 - v0) * VERT, (i1 - i0) * IDX
        if self.comp_models:
            off, csize = self.comp_models[i]
            need = vb + ib
            try:
                raw = lz4.block.decompress(self.data[off:off + csize], uncompressed_size=need)
            except Exception as exc:
                raise ValueError("%s: model %d blob does not decompress (%s: %s)"
                                 % (self.path, i, type(exc).__name__, exc)) from exc
            # lz4.block.decompress does NOT pad or error when the compressed stream's
            # own content is SHORTER than uncompressed_size - it silently returns
            # however many bytes the stream actually held (confirmed: requesting MORE
            # than a stream's real length succeeds and returns the real, shorter
            # length; only requesting LESS raises). Reading the wrong bytes at the
            # wrong offset - exactly what a version mismatch (or any other table
            # corruption) produces - is therefore invisible to the try/except above
            # in general: this length check is the one that actually generalises,
            # catching a short/wrong decompression regardless of WHY it happened, not
            # just the one cause (a stripped donor) that prompted adding it.
            if len(raw) != need:
                raise ValueError(
                    "%s: model %d blob decompressed to %d bytes, its own tables "
                    "promise %d - refusing to graft truncated or wrong data"
                    % (self.path, i, len(raw), need))
            return raw[:vb], raw[vb:vb + ib]
        v = self.data[self.vertex_off + v0 * VERT: self.vertex_off + v1 * VERT]
        x = self.data[self.index_off + i0 * IDX: self.index_off + i1 * IDX]
        return v, x

    def texture(self, i):
        o = self.texture_off + 32 * i
        w, h, fmt = struct.unpack_from("<2HI", self.data, o)
        tf, tb, bw, cf, ce, nl = struct.unpack_from("<6I", self.data, o + 8)
        return w, h, fmt, tf, tb, bw, cf, ce, nl

    def texture_pools(self, i):
        w, h, fmt, tf, tb, bw, cf, ce, nl = self.texture(i)
        cb = ce * 4
        if self.comp_tex:
            off, csize = self.comp_tex[i]
            need = tb + cb
            try:
                raw = lz4.block.decompress(self.data[off:off + csize], uncompressed_size=need)
            except Exception as exc:
                raise ValueError("%s: texture %d blob does not decompress (%s: %s)"
                                 % (self.path, i, type(exc).__name__, exc)) from exc
            # See model_pools's own comment: a short/wrong decompression is not an
            # exception, it is a silently shorter return value, and this is the check
            # that actually catches it - this is literally the shape of the bug
            # found live: region_10_9.pmap texture 56 expected 2880 bytes and got 43,
            # with no exception anywhere in the original code.
            if len(raw) != need:
                raise ValueError(
                    "%s: texture %d blob decompressed to %d bytes, its own tables "
                    "promise %d - refusing to graft truncated or wrong data"
                    % (self.path, i, len(raw), need))
            return raw[:tb], raw[tb:tb + cb]
        t = self.data[self.texel_off + tf: self.texel_off + tf + tb]
        c = self.data[self.clut_off + cf: self.clut_off + cf + cb] if ce else b""
        return t, c

    def instances(self):
        out = []
        for i in range(self.instance_count):
            o = self.instance_off + 36 * i
            mi, px, py, pz = struct.unpack_from("<Ifff", self.data, o)
            qx, qy, qz, qw = struct.unpack_from("<4h", self.data, o + 16)
            scale, interior, cell = struct.unpack_from("<fii", self.data, o + 24)
            out.append((mi, (px, py, pz), (qx, qy, qz, qw), scale, interior, cell))
        return out


def inst_key(pos, quat):
    return (round(pos[0], 1), round(pos[1], 1), round(pos[2], 1),
            quat[0], quat[1], quat[2], quat[3])


def model_key(scale, center, br, vbytes, ibytes):
    h = hashlib.md5(vbytes)
    h.update(ibytes)
    return (round(scale, 6), round(center[0], 2), round(center[1], 2),
            round(center[2], 2), round(br, 2), h.hexdigest())


def tex_key(tbytes, cbytes):
    return hashlib.md5(tbytes + b"|" + cbytes).hexdigest()


def graft_tile(donor_path, target_path):
    dt = DonorTile(donor_path)
    sc = psp_scene.read_scene(open(target_path, "rb").read())

    have = {inst_key(i.pos, tuple(int(round(q * 32767)) for q in i.quat))
            for i in sc.instances}
    missing = [(mi, pos, quat, iscale, interior, cell)
               for (mi, pos, quat, iscale, interior, cell) in dt.instances()
               if inst_key(pos, quat) not in have]
    if not missing:
        return 0, 0, 0

    tgt_models = {}
    for li, m in enumerate(sc.models):
        vb = b"".join(s.vertex_bytes for s in m.submeshes)
        ib = b"".join(s.index_bytes for s in m.submeshes)
        tgt_models[model_key(m.scale, m.center, m.bound_radius, vb, ib)] = li
    tgt_tex = {}
    for li, t in enumerate(sc.textures):
        tgt_tex[tex_key(t.texel_bytes, t.clut_bytes)] = li

    added_models = added_tex = 0
    donor_model_map = {}
    for (mi, pos, quat, iscale, interior, cell) in missing:
        if mi in donor_model_map:
            continue
        fs, scnt, mscale, mcenter, br, dd = dt.model(mi)
        vb, ib = dt.model_pools(mi)
        mk = model_key(mscale, mcenter, br, vb, ib)
        if mk in tgt_models:
            donor_model_map[mi] = tgt_models[mk]
            continue
        # transplant the model: submeshes carve their slices out of the pools
        v0, _v1, i0, _i1 = dt.model_span(mi)
        subs = []
        for s in range(fs, fs + scnt):
            tex, vf, vc, if_, ic = dt.submesh(s)
            lt = -1
            if tex >= 0:
                tb, cb = dt.texture_pools(tex)
                tk = tex_key(tb, cb)
                if tk not in tgt_tex:
                    w, h, fmt, _tf, _tbb, bw, _cf, ce, nl = dt.texture(tex)
                    sc.textures.append(psp_scene.Texture(
                        width=w, height=h, format=fmt, texel_bytes=tb,
                        buffer_width=bw, clut_bytes=cb, clut_entries=ce,
                        num_levels=nl))
                    tgt_tex[tk] = len(sc.textures) - 1
                    added_tex += 1
                lt = tgt_tex[tk]
            subs.append(psp_scene.Submesh(
                texture=lt,
                vertex_bytes=vb[(vf - v0) * VERT:(vf - v0 + vc) * VERT],
                index_bytes=ib[(if_ - i0) * IDX:(if_ - i0 + ic) * IDX]))
        sc.models.append(psp_scene.Model(
            submeshes=subs, scale=mscale, center=mcenter,
            bound_radius=br, draw_dist=dd))
        li = len(sc.models) - 1
        tgt_models[mk] = li
        donor_model_map[mi] = li
        added_models += 1

    for (mi, pos, quat, iscale, interior, cell) in missing:
        sc.instances.append(psp_scene.Instance(
            model=donor_model_map[mi], pos=pos,
            quat=tuple(q / 32767.0 for q in quat),
            scale=iscale, interior=interior, cell=cell))

    out = psp_scene.write_scene(sc.models, sc.textures, sc.instances, sc.grid)
    open(target_path, "wb").write(out)
    return len(missing), added_models, added_tex


def main():
    if len(sys.argv) < 3:
        print("usage: pmap_graft.py <donor_dir> <target_dir>"); return 1
    donor, target = sys.argv[1], sys.argv[2]
    tot_i = tot_m = tot_t = files = 0
    for tp in sorted(glob.glob(os.path.join(target, "region_*_*.pmap"))):
        dp = os.path.join(donor, os.path.basename(tp))
        if not os.path.exists(dp):
            continue
        ni, nm, nt = graft_tile(dp, tp)
        if ni:
            print("%-22s +%d inst (+%d models, +%d tex)"
                  % (os.path.basename(tp), ni, nm, nt))
            tot_i += ni; tot_m += nm; tot_t += nt; files += 1
    print("GRAFT TOTAL: +%d instances (+%d models, +%d textures) across %d tiles"
          % (tot_i, tot_m, tot_t, files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
