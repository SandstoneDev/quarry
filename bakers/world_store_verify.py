#!/usr/bin/env python3
"""world_store_verify.py - independent checker for the world-store container
tools/world_store_build.py writes (
-design.md section 4, stage 2a): world.idx + world.dat + 184 stripped region_*.pmap tiles.

THE ONE RULE THIS MODULE FOLLOWS: verify the container independently of the writer. This
file imports NOTHING from world_store_build.py - no struct, no constant, no helper. Every
layout fact below (the WIDX_* structs, the PMAP_* structs, the .night/.nightd sidecar
formats) is re-derived straight from the format AS DOCUMENTED: world_store_build.py's own
module docstring for world.idx/world.dat, and - for the base .pmap tables the stripped
tiles still carry (header, model, submesh, texture descriptor, instance) - directly from
src/platform_psp/pmap.h/pmap.c, the engine-side C source that is the actual ground truth,
cross-checked live against a real region_0_0.pmap (see the per-field comments below for
what was confirmed how). Reusing world_store_build's parsing code would make this checker
share the writer's idea of the format - unable to disagree with it even when it is wrong.
This project has already paid for that mistake twice (a test whose oracle was blind to the
field under test, and a verifier gate shared with the pass it graded); this module exists
so a bug in world_store_build.py's understanding of ANY field has an independent reader
standing a chance of noticing it.

FORMAT 2 (b658dfde..0254f82): global ids narrowed u32 -> u16 (REFS_ENTRY), a build_stamp
field inserted into world.idx's header, every non-.pmap sidecar (.col/.night/.nightd/
.grass/.lod/.sway/.dyn/.spin/.mflags/.road/.tobj/.anim/regions.bin) now copied through
unchanged, and world.idx's own `version` bumped to 2. Re-derived from world_store_build.
py's docstring exactly as before - NOT its code - and this round is the actual test of
whether that docstring is now SUFFICIENT: the first pass through this format left two
things unreported by the docstring, both raised as findings rather than patched over by
reading the implementation, and both are now fixed at the source, independently confirmed
by a from-scratch re-read of the rewritten docstring (not by trusting that the fix
happened):

 (1) the build stamp's exact byte formula - "name and bytes, concatenated" did not say
 order, separator, or chaining. The docstring's FORMAT section now gives literal
 pseudocode (see _compute_build_stamp's own comment); it matches, term for term, what
 this module had already worked out empirically against the real store's stamp before
 the docstring said so - the one place this round's from-scratch re-read had
 something to cross-check against, and it checked out exactly.

 (2) world.idx's own `version` field now reads 2, refused BY NAME when it is 1 (see the
 version check below) rather than folded into a generic "unsupported version"
 message - version 1 is a format this module used to be able to read and no longer
 can, which is worth explaining; a version it has never seen is not.

 Everything else in the rewritten FORMAT section - byte offsets and widths for the
 header, both global tables, the tile directory, names, refs, and the stripped-tile
 field changes keyed by header index AND byte offset - required NO inference on this
 pass: every struct below came directly off that table. See the WIDX_HDR/PMAP_HDR
 comments for the one wording point that is not fully precise (names_off's padding
 description) and why it does not block a correct reader even so.

Two scopes, matching what needs the tiles/sidecars and what does not:

 verify_store(idx_bytes, dat_bytes) -> (ok, [problems])
 CONTAINER ONLY: every blob lies wholly inside world.dat and decompresses to exactly
 its recorded dsize, no id references a zero-length blob, blob offsets are
 non-decreasing with gaps under 16 bytes (pure alignment padding, not holes - what
 lets a future reader coalesce adjacent blobs into one read), every blob starts
 16-byte aligned, refs_off is 4-byte aligned and every per-tile ref array is at least
 2-byte aligned (what u16 entries actually need), neither global table exceeds the
 65535-entry ceiling a u16 id can address, and the tile directory's name/ref spans and
 every per-tile global id are in range. Never raises - an unparseable idx_bytes comes
 back as (False, ["one problem string"]), same contract as pmap_lattice_verify.
 verify_bytes.

 verify_dir(path, ref=None) -> int
 THE FULL JOB over a ps2global/-shaped directory: everything verify_store does, PLUS
 every stripped tile actually parses as version PMAP_VERSION_STRIPPED (5) with its
 comp_model/comp_tex global ids resolving (in range, decompressing, AND agreeing with
 what world.idx's own per-tile refs say they should be - the two artifacts are
 written in the same run and must never disagree), its build stamp (index_off,
 repurposed - read ONLY when version is exactly 5, never otherwise) agreeing with
 world.idx's own, submesh ranges contiguous within each model with every index
 falling inside its OWN submesh (confirmed empirically against real region_0_0.pmap
 geometry that indices are LOCAL to their owning submesh, not the model or a global
 pool - see _check_model_geometry's docstring), every instance's model field in
 range, the .night/.nightd sidecars (if present) internally consistent with the
 tile's own vertex count, and - when `ref` (the original, unstripped world) is given
 - the fidelity claim the whole store exists to make: every model and texture blob
 resolved through the store is byte-identical to the same blob in the original tile,
 the resident prefix around those blobs (model/submesh/texture-descriptor/instance
 tables, grid) is itself byte-identical to the reference exactly as strip_tile()'s own
 docstring promises, EVERY sidecar file is byte-identical to the reference's (or named
 as missing/extra), and the build stamp recomputed from the reference's own tile set
 matches world.idx's - proving the stamp is not just internally consistent but
 actually correct. Prints per-tile problems and a summary; returns a shell exit code
 (0 = clean).

Usage:
 python tools/world_store_verify.py <ps2global_dir>
 python tools/world_store_verify.py <ps2global_dir> --ref <original_unstripped_dir>
"""
import os
import struct
import sys
import zlib

import lz4.block

# ---------------------------------------------------------------------------
# world.idx / world.dat container format - re-derived from world_store_build.py's
# own module docstring (the FORMAT section: a byte-offset/width table for the
# header, both global tables, the tile directory, names and refs, and the
# stripped-tile field changes keyed by header index AND byte offset), NOT from
# its code - and, this round, the docstring is now BYTE-EXACT enough that no
# field below required inference: every offset, width, and field order came
# straight off the table, cross-checked against the real world.idx's own
# offset arithmetic (model_table_off + 10264*12 == tex_table_off exactly) as a
# second, independent confirmation, not as the source of the layout itself.
#
# VERSION 2 (this module's own history): format 2's byte layout (44-byte header,
# u16 refs) was already on disk BEFORE world.idx's own `version` field caught up
# to say so - it stayed 1 across the header-growing, ref-narrowing change for
# one full round, which this module flagged as a finding (a format-1 reader had
# no signal at all that it was misreading a 44-byte header as 40). Now fixed at
# the source: WIDX_VERSION=2, and a version-1 file is refused BY NAME below, not
# folded into the generic "unsupported version" message an unknown version 3
# would get - version 1 is a format this module used to speak and now cannot,
# which is worth explaining; an unseen future version is not.
# ---------------------------------------------------------------------------
WIDX_MAGIC = b"WIDX"
WIDX_VERSION = 2
WIDX_HDR = struct.Struct("<4s10I")      # magic,version,build_stamp,M,T,N,model_table_off,
                                         # tex_table_off,tile_dir_off,names_off,refs_off
GLOBAL_ENTRY = struct.Struct("<3I")     # off,csize,dsize - UNCHANGED (still u32 x3); confirmed
                                         # against the real world.idx's own table_off arithmetic
                                         # (model_table_off + 10264*12 == tex_table_off exactly)
TILE_DIR_ENTRY = struct.Struct("<6I")   # name_off,name_len,model_count,model_refs_off,
                                         # tex_count,tex_refs_off - UNCHANGED (still u32 x6)
DAT_ALIGN = 16                          # world.dat blob alignment (GE DMA requirement)

# Format 2: per-tile ref arrays are u16 global ids, not u32 - "sized to what it
# actually needs to hold... rather than defaulted to u32" per the docstring.
# Confirmed against the real world.idx: (file_size - refs_off) == (14129 model refs +
# 39788 tex refs) * 2 bytes exactly.
REFS_ENTRY = struct.Struct("<H")
REFS_ENTRY_ALIGN = 2                    # what a uint16_t* cast on MIPS needs
# refs_off ITSELF is still 4-byte aligned (the docstring says padding there is
# unchanged); an INDIVIDUAL tile's model_refs_off/tex_refs_off is only guaranteed
# REFS_ENTRY_ALIGN (2) now that entries are u16 - a tile with an odd total ref
# count shifts the next tile's start by an odd number of 2-byte units.
#
# The one place the rewritten docstring's wording is not fully precise: it says
# "names_off is padded to the next 4-byte boundary from wherever the tile dir
# ends" - but names_off is a single scalar offset, not a padded region, and the
# tile dir's own end is already a multiple of 4 (44 + M*12 + T*12 + N*24, every
# term already a multiple of 4), so there is nothing to pad there at all. What
# actually needs padding, and does, is the END of the (variable-length, since
# names are not fixed width) names blob, so that refs_off itself lands aligned --
# the docstring's own stated PURPOSE ("so refs_off starts aligned") is what
# actually pins the padding's location, just not the sentence's literal subject.
# This does not block a correct READER, which this module is: nothing here
# reconstructs the padding-insertion algorithm, only checks that refs_off is
# 4-byte aligned (below) and that every declared (name_off, name_len) span stays
# inside [names_off, refs_off) - both independent of exactly how the writer
# arrived at that alignment.
REFS_OFF_ALIGN = 4
# "build_store REFUSES to build... if either global table would ever exceed
# 65535 entries" - independently re-enforced here as a container-format invariant
# (an id has to fit in one REFS_ENTRY), not merely trusted of the writer's own
# say-so: a corrupted or hand-edited world.idx claiming more unique entries than a
# u16 id space can address is a broken container regardless of how it was made.
MAX_UNIQUE_ENTRIES = 65535

# ---------------------------------------------------------------------------
# Base.pmap v3-family tables the stripped tiles still carry, re-derived directly
# from src/platform_psp/pmap.h (the engine's own C struct definitions - the actual
# ground truth, not any Python tool's idea of it) and cross-checked live against a
# real region_0_0.pmap while this module was written (see the field comments).
# ---------------------------------------------------------------------------
PMAP_MAGIC = 0x50414D50                 # 'PMAP' little-endian (pmap.h PMAP_MAGIC)
PMAP_VERSION_STRIPPED = 5               # pmap.h PMAP_VERSION_STRIPPED; cross-checked against
                                         # the #define itself in test_stripped_version_
                                         #_constant_matches_pmap_h (pmap_lattice_verify.py's
                                         # own test, not this module's - both sides of the
                                         # C/Python boundary are pinned there already).

# 23 u32 header, pmap.h PmapHeader: magic,version,file_size, model_count,model_off,
# submesh_count,submesh_off, texture_count,texture_off, instance_count,instance_off,
# grid_off, vertex_off,vertex_bytes, index_off,index_bytes, texel_off,texel_bytes,
# clut_off,clut_bytes, comp_flag, comp_model_off,comp_tex_off. v4 adds a 24th field
# (uvrange_off) this module never reads - nothing it checks needs UV range.
PMAP_HDR = struct.Struct("<23I")
(_H_MAGIC, _H_VERSION, _H_FILESIZE, _H_MODEL_COUNT, _H_MODEL_OFF, _H_SUBMESH_COUNT,
 _H_SUBMESH_OFF, _H_TEX_COUNT, _H_TEX_OFF, _H_INST_COUNT, _H_INST_OFF, _H_GRID_OFF,
 _H_VERTEX_OFF, _H_VERTEX_BYTES, _H_INDEX_OFF, _H_INDEX_BYTES, _H_TEXEL_OFF,
 _H_TEXEL_BYTES, _H_CLUT_OFF, _H_CLUT_BYTES, _H_COMP_FLAG, _H_COMP_MODEL_OFF,
 _H_COMP_TEX_OFF) = range(23)

# PmapModel, 32 bytes: first_submesh,submesh_count(u32 x2), scale,center_x,center_y,
# center_z,bound_radius,draw_dist (f32 x6).
PMAP_MODEL = struct.Struct("<2I6f")
# PmapSubmesh, 20 bytes: texture(i32, -1=untextured), vertex_first,vertex_count,
# index_first,index_count (u32 x4).
PMAP_SUBMESH = struct.Struct("<i4I")
# PmapTexture, 32 bytes: width,height(u16x2), format,texel_first,texel_bytes,
# buffer_width,clut_first,clut_entries,num_levels (u32 x7).
PMAP_TEX = struct.Struct("<HHIIIIIII")
# PmapComp, 8 bytes: off (a stripped tile's global id, in this module's context)
# and csize.
PMAP_COMP = struct.Struct("<2I")
# PmapInstance, 36 bytes: model(u32), pos_x,pos_y,pos_z(f32x3), qx,qy,qz,qw(s16x4,
# unit quaternion fixed 1.15), scale(f32), interior,cell(i32x2).
PMAP_INSTANCE = struct.Struct("<I3f4hfii")

VERT_SZ = 12                            # PmapVertex: s16 u,v; u16 colour; s16 x,y,z

#.night sidecar (pmap.c pmap_load_night): raw u16[vertex_bytes/12] array, one
# GU_COLOR_5551 value per vertex, aligned 1:1 to the tile's OWN vertex pool.
#.nightd sidecar (pmap.c pmap_load_nightd / pmap.h PmapNightRun): magic 'NDL2'
# (checked here as the raw 4 bytes b"NDL2", equal to the LE u32 0x324C444E pmap.c
# compares against) + u32 run_count + run_count x {u32 vidx; u16 n; u16 col} (8
# bytes each) - run `n` consecutive vertices starting at pool index `vidx` glow at
# night; every run's addressed vertices must exist in the tile's own vertex pool
# (confirmed against pmap.c: nv = vertex_bytes/12 is the SAME whole-tile count
# both sidecars are checked against on load).
NIGHTD_MAGIC = b"NDL2"
NIGHTD_HDR = struct.Struct("<I")        # run_count (the 4-byte magic precedes it)
PMAP_NIGHT_RUN = struct.Struct("<IHH")  # vidx,n,col


# ---------------------------------------------------------------------------
# world.idx parsing - never raises past its own boundary; verify_store turns a
# ValueError into (False, [str(exc)]), same contract as pmap_lattice_verify._parse.
# ---------------------------------------------------------------------------
def _parse_widx(idx_bytes):
    if len(idx_bytes) < WIDX_HDR.size:
        raise ValueError("world.idx: file is shorter than the header (%d < %d bytes)"
                          % (len(idx_bytes), WIDX_HDR.size))
    magic, version, build_stamp, mc, tc, nc, mt_off, tt_off, td_off, names_off, refs_off = \
        WIDX_HDR.unpack_from(idx_bytes, 0)
    if magic != WIDX_MAGIC:
        raise ValueError("world.idx: bad magic %r (expected %r)" % (magic, WIDX_MAGIC))
    # Refuse an unknown version BY NAME, in both directions - this whole round
    # exists because a version field went stale through a breaking change, so a
    # generic "wrong version" message is exactly the failure mode being guarded
    # against. Version 1 is a format this module USED to be able to read and now
    # cannot (a known, understood prior shape - 40-byte header, u32 refs, no
    # build_stamp); an unseen future version (3, or anything else) gets the
    # honest generic message instead, since nothing here knows what changed.
    if version == 1:
        raise ValueError(
            "world.idx: version 1 (the original 40-byte header, u32 refs, no "
            "build_stamp field) cannot be read by this module - version 2 "
            "inserted a build_stamp field into the header and narrowed every ref "
            "array from u32 to u16; a version-1 file needs rebuilding, not a "
            "compatible reader")
    if version != WIDX_VERSION:
        raise ValueError("world.idx: unsupported version %d (expected %d)"
                          % (version, WIDX_VERSION))

    def _read_global_table(off, count, label):
        end = off + count * GLOBAL_ENTRY.size
        if off < 0 or end > len(idx_bytes):
            raise ValueError(
                "world.idx: %s table (%d entries at offset %d, ending at %d) runs beyond "
                "the end of the file (%d bytes)" % (label, count, off, end, len(idx_bytes)))
        return [GLOBAL_ENTRY.unpack_from(idx_bytes, off + i * GLOBAL_ENTRY.size)
                for i in range(count)]

    model_table = _read_global_table(mt_off, mc, "model")
    tex_table = _read_global_table(tt_off, tc, "texture")

    td_end = td_off + nc * TILE_DIR_ENTRY.size
    if td_off < 0 or td_end > len(idx_bytes):
        raise ValueError(
            "world.idx: tile directory (%d entries at offset %d, ending at %d) runs beyond "
            "the end of the file (%d bytes)" % (nc, td_off, td_end, len(idx_bytes)))
    tile_dir = [TILE_DIR_ENTRY.unpack_from(idx_bytes, td_off + i * TILE_DIR_ENTRY.size)
                for i in range(nc)]

    return {
        "version": version, "build_stamp": build_stamp,
        "model_table": model_table, "tex_table": tex_table,
        "tile_count": nc, "tile_dir": tile_dir, "mt_off": mt_off, "tt_off": tt_off,
        "td_off": td_off, "names_off": names_off, "refs_off": refs_off,
    }


def _decode_tile_name(idx_bytes, name_off, name_len, problems, label):
    end = name_off + name_len
    if name_off < 0 or end > len(idx_bytes):
        problems.append("%s: name span [%d,%d) is beyond the end of world.idx (%d bytes)"
                         % (label, name_off, end, len(idx_bytes)))
        return None
    raw = idx_bytes[name_off:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append("%s: name bytes are not valid UTF-8 (%r)" % (label, raw))
        return None


def verify_store(idx_bytes, dat_bytes):
    """(ok, [problems]) - container-only checks over world.idx + world.dat. See the
 module docstring for the exact list. Never raises."""
    try:
        info = _parse_widx(idx_bytes)
    except ValueError as exc:
        return False, [str(exc)]

    problems = []
    dat_size = len(dat_bytes)
    model_table, tex_table = info["model_table"], info["tex_table"]
    mc, tc = len(model_table), len(tex_table)

    # ---- ceiling: a per-tile ref is one REFS_ENTRY (u16) wide, so a global table
    # past MAX_UNIQUE_ENTRIES holds ids no ref array could ever address, regardless
    # of how the file came to say so (build_store's own refusal to WRITE such a
    # file is a different, writer-side guarantee this check does not rely on). ----
    for kind, count in (("model", mc), ("texture", tc)):
        if count > MAX_UNIQUE_ENTRIES:
            problems.append(
                "%s table has %d entries, past the %d-entry ceiling a %d-bit global id "
                "(REFS_ENTRY) can address" % (kind, count, MAX_UNIQUE_ENTRIES,
                                              REFS_ENTRY.size * 8))

    # ---- refs_off itself must be 4-byte aligned (the docstring: "refs_off itself
    # is still 4-byte aligned (padding is unchanged)") - INDIVIDUAL tile ref-array
    # offsets are checked separately below, to the weaker 2-byte guarantee u16
    # entries actually need. ----
    if info["refs_off"] % REFS_OFF_ALIGN != 0:
        problems.append("refs_off %d is not %d-byte aligned"
                        % (info["refs_off"], REFS_OFF_ALIGN))

    # ---- per-entry: bounds, alignment, decompression, zero-length-blob rule ----
    for kind, table in (("model", model_table), ("texture", tex_table)):
        for gid, (off, csize, dsize) in enumerate(table):
            # Alignment is a property of the offset value alone - checked
            # unconditionally, BEFORE the bounds check below, so a misaligned
            # offset that also happens to run off the end of world.dat (e.g. the
            # very last blob in the file) still gets its alignment problem
            # reported, not silently swallowed by the bounds check's own `continue`.
            if off % DAT_ALIGN != 0:
                problems.append("%s blob %d: offset %d is not %d-byte aligned"
                                 % (kind, gid, off, DAT_ALIGN))
            end = off + csize
            if off < 0 or csize < 0 or end > dat_size:
                problems.append(
                    "%s blob %d: [%d,%d) is beyond the end of world.dat (%d bytes)"
                    % (kind, gid, off, end, dat_size))
                continue
            if csize == 0:
                if dsize != 0:
                    problems.append(
                        "%s blob %d: csize=0 (zero-length blob) but dsize=%d - zero "
                        "compressed bytes cannot decompress to nonzero content"
                        % (kind, gid, dsize))
                continue
            try:
                got = lz4.block.decompress(dat_bytes[off:end], uncompressed_size=dsize)
            except Exception as exc:                                    # noqa: BLE001
                problems.append("%s blob %d: does not decompress (%s)" % (kind, gid, exc))
                continue
            if len(got) != dsize:
                problems.append(
                    "%s blob %d: decompresses to %d bytes, world.idx's table promises %d"
                    % (kind, gid, len(got), dsize))

    # ---- ordering: world.dat lays every model blob first (in model-table order),
    # then every texture blob (in texture-table order), contiguously - see the
    # module docstring's BLOB ORDERING cross-reference in world_store_build.py's own
    # docstring. Concatenating the two tables in that order reproduces the exact
    # physical layout order; consecutive offsets must never decrease, and any gap
    # between one blob's end and the next one's start must be pure 16-byte alignment
    # padding (0..15 bytes), not an unexplained hole - a future reader coalescing
    # adjacent blobs into one read depends on there being nothing else in between. ----
    combined = list(model_table) + list(tex_table)
    for k in range(1, len(combined)):
        o0, c0, _d0 = combined[k - 1]
        o1, _c1, _d1 = combined[k]
        if o1 < o0:
            problems.append(
                "blob table: offset is not non-decreasing at entry %d (%d, then %d)"
                % (k, o0, o1))
            continue
        gap = o1 - (o0 + c0)
        if gap < 0 or gap >= DAT_ALIGN:
            problems.append(
                "blob table: a %d-byte hole sits between entry %d (ending at %d) and "
                "entry %d (starting at %d) - beyond what 16-byte alignment padding "
                "explains" % (gap, k - 1, o0 + c0, k, o1))

    # ---- tile directory: name/ref spans in bounds, every referenced id in range
    # and not a zero-length blob. ----
    for i, (name_off, name_len, mcount, mrefs_off, tcount, trefs_off) in \
            enumerate(info["tile_dir"]):
        label = "tile %d" % i
        name = _decode_tile_name(idx_bytes, name_off, name_len, problems, label)
        if name is not None:
            label = name

        if mrefs_off % REFS_ENTRY_ALIGN != 0:
            problems.append("%s: model_refs_off %d is not %d-byte aligned"
                            % (label, mrefs_off, REFS_ENTRY_ALIGN))
        mrefs_end = mrefs_off + mcount * REFS_ENTRY.size
        if mrefs_off < 0 or mrefs_end > len(idx_bytes):
            problems.append(
                "%s: model ref span [%d,%d) is beyond the end of world.idx (%d bytes)"
                % (label, mrefs_off, mrefs_end, len(idx_bytes)))
        else:
            mrefs = struct.unpack_from("<%dH" % mcount, idx_bytes, mrefs_off)
            for k, gid in enumerate(mrefs):
                if gid >= mc:
                    problems.append(
                        "%s: model ref %d -> global id %d is out of range (%d unique "
                        "models)" % (label, k, gid, mc))
                elif model_table[gid][1] == 0:
                    problems.append(
                        "%s: model ref %d -> global id %d references a zero-length blob"
                        % (label, k, gid))

        if trefs_off % REFS_ENTRY_ALIGN != 0:
            problems.append("%s: tex_refs_off %d is not %d-byte aligned"
                            % (label, trefs_off, REFS_ENTRY_ALIGN))
        trefs_end = trefs_off + tcount * REFS_ENTRY.size
        if trefs_off < 0 or trefs_end > len(idx_bytes):
            problems.append(
                "%s: texture ref span [%d,%d) is beyond the end of world.idx (%d bytes)"
                % (label, trefs_off, trefs_end, len(idx_bytes)))
        else:
            trefs = struct.unpack_from("<%dH" % tcount, idx_bytes, trefs_off)
            for k, gid in enumerate(trefs):
                if gid >= tc:
                    problems.append(
                        "%s: texture ref %d -> global id %d is out of range (%d unique "
                        "textures)" % (label, k, gid, tc))
                elif tex_table[gid][1] == 0:
                    problems.append(
                        "%s: texture ref %d -> global id %d references a zero-length blob"
                        % (label, k, gid))

    return (not problems), problems


# ---------------------------------------------------------------------------
# Stripped.pmap tile parsing - the base v3-family tables, independent of
# pmap_lattice_verify.py and world_store_build.py alike (see the module docstring).
# ---------------------------------------------------------------------------
def _parse_pmap_tables(buf):
    """Header + model/submesh/comp_model/comp_tex/instance tables for ANY v3-family
 .pmap (stripped or not - the caller decides what to make of `version` and of
 comp_model/comp_tex's meaning). Raises ValueError, never lets a struct.error or
 an IndexError escape, for anything not well-formed."""
    if len(buf) < PMAP_HDR.size:
        raise ValueError("file is shorter than the header (%d < %d bytes)"
                          % (len(buf), PMAP_HDR.size))
    h = PMAP_HDR.unpack_from(buf, 0)
    if h[_H_MAGIC] != PMAP_MAGIC:
        raise ValueError("not a .pmap (bad magic %r)" % (h[_H_MAGIC],))

    mc, moff = h[_H_MODEL_COUNT], h[_H_MODEL_OFF]
    sc, soff = h[_H_SUBMESH_COUNT], h[_H_SUBMESH_OFF]
    tc, toff = h[_H_TEX_COUNT], h[_H_TEX_OFF]
    ic, ioff = h[_H_INST_COUNT], h[_H_INST_OFF]
    cmoff, ctoff = h[_H_COMP_MODEL_OFF], h[_H_COMP_TEX_OFF]

    try:
        models = [PMAP_MODEL.unpack_from(buf, moff + i * PMAP_MODEL.size) for i in range(mc)]
        subs = [PMAP_SUBMESH.unpack_from(buf, soff + i * PMAP_SUBMESH.size) for i in range(sc)]
        tex_descs = [PMAP_TEX.unpack_from(buf, toff + i * PMAP_TEX.size) for i in range(tc)]
        comp_model = [PMAP_COMP.unpack_from(buf, cmoff + i * PMAP_COMP.size) for i in range(mc)]
        comp_tex = [PMAP_COMP.unpack_from(buf, ctoff + i * PMAP_COMP.size) for i in range(tc)]
        instances = [PMAP_INSTANCE.unpack_from(buf, ioff + i * PMAP_INSTANCE.size)
                     for i in range(ic)]
    except struct.error as exc:
        raise ValueError("a table runs past the end of the file (%s)" % exc) from exc

    return {"h": h, "models": models, "subs": subs, "tex_descs": tex_descs,
            "comp_model": comp_model, "comp_tex": comp_tex, "instances": instances}


def _model_span(models, subs, i):
    """(vfirst, vcount, ifirst, icount) for model i - the model's own first and
 last submesh mark its overall vertex/index span (pmap.c's own streaming loader
 computes this identically: pmap_load_finish's blob-fetch path reads
 `vstart = s0->vertex_first; ...; vbytes = (sN->vertex_first+sN->vertex_count-vstart)
 * 12`, cross-checked live against pmap.c). Raises IndexError/struct.error-derived
 ValueError via the caller's try/except, same as _parse_pmap_tables."""
    first, count = models[i][0], models[i][1]
    if count == 0:
        return 0, 0, 0, 0
    s0 = subs[first]
    sN = subs[first + count - 1]
    vfirst = s0[1]
    vcount = sN[1] + sN[2] - vfirst
    ifirst = s0[3]
    icount = sN[3] + sN[4] - ifirst
    return vfirst, vcount, ifirst, icount


def _check_model_geometry(tile_label, i, models, subs, raw, problems):
    """Submesh contiguity within model i, and every index falling inside its OWN
 submesh's local vertex range - confirmed empirically (not assumed from the
 header comment alone) against a real region_0_0.pmap model: decompressed one
 real multi-submesh model, and for each of its submeshes the stored u16 index
 values are LOCAL to that submesh (0..vertex_count-1), needing the submesh's own
 (vertex_first - model's vfirst) added back to find the actual local vertex in
 the model's decompressed blob - NOT the model's overall vcount, and NOT a
 submesh-table-relative global offset. Guards exactly the shape this project's
 own history flags: a check that only reads a model's first and last submesh
 misses a corrupted MIDDLE one (`_model_span` above has that same blind spot by
 construction; this function is what actually opens every submesh in between)."""
    first, count = models[i][0], models[i][1]
    if count == 0:
        return
    vfirst0, vcount_total, ifirst0, icount_total = _model_span(models, subs, i)
    if icount_total == 0:
        return
    # index bytes start right after this model's own vcount_total*VERT_SZ vertex bytes
    idx_byte_off = vcount_total * VERT_SZ
    idx_vals = struct.unpack_from("<%dH" % icount_total, raw, idx_byte_off)

    expect_v, expect_i = vfirst0, ifirst0
    for si in range(first, first + count):
        tex, vfirst, vcount, ifirst, icount = subs[si]
        if vfirst != expect_v or ifirst != expect_i:
            problems.append(
                "%s model %d submesh %d: not contiguous with the previous submesh - "
                "expected vertex_first=%d index_first=%d, got %d/%d"
                % (tile_label, i, si, expect_v, expect_i, vfirst, ifirst))
        local_v_lo = vfirst - vfirst0
        local_i_lo = ifirst - ifirst0
        sub_idx = idx_vals[local_i_lo:local_i_lo + icount]
        for k, v in enumerate(sub_idx):
            if not (0 <= v < vcount):
                problems.append(
                    "%s model %d submesh %d: index %d (value %d) falls outside its own "
                    "submesh's vertex range [0,%d) - escapes into a neighbour"
                    % (tile_label, i, si, k, v, vcount))
                break   # one report per submesh is enough to point at the problem
        expect_v = vfirst + vcount
        expect_i = ifirst + icount


def _get_stripped_model_bytes(idx_info, dat_bytes, gid, need_dsize, problems, label):
    """Resolve global model id `gid` through the store to decompressed bytes, or
 None + an appended problem. `need_dsize` is what the CALLING tile's own tables
 say this model should decompress to - cross-checked against world.idx's own
 dsize for the same id (a mismatch here means the tile and the store disagree
 about what this blob even is)."""
    model_table = idx_info["model_table"]
    if gid >= len(model_table):
        problems.append("%s: global model id %d is out of range (%d unique models)"
                         % (label, gid, len(model_table)))
        return None
    off, csize, dsize = model_table[gid]
    if dsize != need_dsize:
        problems.append(
            "%s: global model %d's own dsize (%d) does not match what this tile's "
            "submesh table says the model should decompress to (%d)"
            % (label, gid, dsize, need_dsize))
    if csize == 0:
        return b""
    if off + csize > len(dat_bytes):
        problems.append("%s: global model %d's blob [%d,%d) is beyond the end of "
                         "world.dat (%d bytes)" % (label, gid, off, off + csize, len(dat_bytes)))
        return None
    try:
        return lz4.block.decompress(dat_bytes[off:off + csize], uncompressed_size=dsize)
    except Exception as exc:                                            # noqa: BLE001
        problems.append("%s: global model %d's blob does not decompress (%s)"
                         % (label, gid, exc))
        return None


def _verify_stripped_tile(name, buf, idx_tile, idx_info, dat_bytes, problems, counts):
    """Everything verify_dir checks about ONE stripped tile that does not need a
 reference: version, comp_model/comp_tex ids (in range, agree with world.idx's
 own per-tile refs, decompress, and their csize matches the global table's),
 submesh contiguity + index bounds, and every instance's model field in range.
 idx_tile: (name_off,name_len,model_count,model_refs_off,tex_count,trefs_off) - world.idx's OWN directory entry for this tile (already known, by name, to
 belong to it). Returns the parsed tables dict - even when version is wrong
 (the caller still finds the header/vertex_bytes useful for the sidecar checks,
 it just skips everything that assumes comp_model/comp_tex hold global ids) - or None only when the tile could not be parsed AT ALL (bad magic, or a table
 running past the end of the file)."""
    try:
        t = _parse_pmap_tables(buf)
    except ValueError as exc:
        problems.append("%s: %s" % (name, exc))
        return None

    h = t["h"]
    if h[_H_VERSION] != PMAP_VERSION_STRIPPED:
        problems.append(
            "%s: expected a STRIPPED tile (version=%d), got version=%d - this tile was "
            "never run through world_store_build.py's strip_tile(), or a stale copy "
            "survived a rebuild" % (name, PMAP_VERSION_STRIPPED, h[_H_VERSION]))
        return t   # tables still parsed; caller may still find this useful, but skip
                   # everything below that assumes comp_model/comp_tex hold global ids

    # ---- build stamp: index_off is repurposed to hold it, and ONLY means that
    # when version is exactly PMAP_VERSION_STRIPPED (checked immediately above --
    # this module never reads _H_INDEX_OFF as a stamp on any other version, matching
    # how every engine reader gates every other version-dependent field). A
    # mismatch here is the "partial redeploy" scenario the docstring names as the
    # likeliest real way this format breaks: this tile is stale (or world.idx/
    # world.dat are), and without this check a global id would silently resolve
    # into the WRONG build's blob instead of refusing. ----
    counts["build_stamp_checked"] += 1
    tile_stamp = h[_H_INDEX_OFF]
    if tile_stamp != idx_info["build_stamp"]:
        problems.append(
            "%s: build stamp 0x%08x does not match world.idx's 0x%08x - this tile is "
            "from a different build than the store (a stale tile, or a world.idx/"
            "world.dat rebuilt without it)" % (name, tile_stamp, idx_info["build_stamp"]))

    _name_off, _name_len, exp_mc, exp_mrefs_off, exp_tc, exp_trefs_off = idx_tile
    mc, tc = len(t["comp_model"]), len(t["comp_tex"])
    if mc != exp_mc or tc != exp_tc:
        problems.append(
            "%s: model/texture counts (%d/%d) do not match world.idx's own tile "
            "directory entry (%d/%d)" % (name, mc, tc, exp_mc, exp_tc))
    else:
        exp_mrefs = struct.unpack_from("<%dH" % exp_mc, idx_info_bytes_of(idx_info), exp_mrefs_off) \
            if exp_mc else ()
        exp_trefs = struct.unpack_from("<%dH" % exp_tc, idx_info_bytes_of(idx_info), exp_trefs_off) \
            if exp_tc else ()
        model_table, tex_table = idx_info["model_table"], idx_info["tex_table"]

        for i, (gid, csize) in enumerate(t["comp_model"]):
            counts["models"] += 1
            if i < len(exp_mrefs) and gid != exp_mrefs[i]:
                problems.append(
                    "%s: model %d's own global id (%d) disagrees with world.idx's "
                    "model_refs[%d] (%d) for this tile" % (name, i, gid, i, exp_mrefs[i]))
            if gid >= len(model_table):
                problems.append("%s: model %d's global id %d is out of range (%d unique "
                                "models)" % (name, i, gid, len(model_table)))
            elif model_table[gid][1] != csize:
                problems.append(
                    "%s: model %d's own csize (%d) does not match global model %d's csize "
                    "in world.idx (%d)" % (name, i, csize, gid, model_table[gid][1]))

        for i, (gid, csize) in enumerate(t["comp_tex"]):
            counts["textures"] += 1
            if i < len(exp_trefs) and gid != exp_trefs[i]:
                problems.append(
                    "%s: texture %d's own global id (%d) disagrees with world.idx's "
                    "tex_refs[%d] (%d) for this tile" % (name, i, gid, i, exp_trefs[i]))
            if gid >= len(tex_table):
                problems.append("%s: texture %d's global id %d is out of range (%d unique "
                                "textures)" % (name, i, gid, len(tex_table)))
            elif tex_table[gid][1] != csize:
                problems.append(
                    "%s: texture %d's own csize (%d) does not match global texture %d's "
                    "csize in world.idx (%d)" % (name, i, csize, gid, tex_table[gid][1]))

    # ---- submesh contiguity + index bounds, per model, decompressed through the
    # store (this is the ONE thing here that actually needs world.dat: the
    # stripped tile itself carries no blob region any more). ----
    models, subs = t["models"], t["subs"]
    for i in range(len(models)):
        try:
            _vf, vcount, _if, icount = _model_span(models, subs, i)
        except (IndexError, struct.error) as exc:
            problems.append("%s model %d: submesh range runs past the submesh table (%s)"
                             % (name, i, exc))
            continue
        if vcount == 0:
            continue
        gid, _csize = t["comp_model"][i]
        need = vcount * VERT_SZ + icount * 2
        raw = _get_stripped_model_bytes(idx_info, dat_bytes, gid, need, problems,
                                        "%s model %d" % (name, i))
        if raw is None:
            continue
        if len(raw) != need:
            problems.append("%s model %d: resolved blob is %d bytes, tables promise %d"
                             % (name, i, len(raw), need))
            continue
        _check_model_geometry(name, i, models, subs, raw, problems)

    # ---- every instance references a model that exists ----
    mc_real = len(models)
    for ii, inst in enumerate(t["instances"]):
        counts["instances"] += 1
        model_idx = inst[0]
        if model_idx >= mc_real:
            problems.append(
                "%s instance %d: references model %d, but this tile only has %d models"
                % (name, ii, model_idx, mc_real))

    return t


def idx_info_bytes_of(idx_info):
    """world.idx's raw bytes, stashed on the info dict by verify_dir so the
 per-tile helper above can re-read a ref array without re-parsing the whole
 file. Kept as a tiny named accessor (not a bare dict key) so a reader sees at
 a glance this is intentional plumbing, not an accident."""
    return idx_info["_raw"]


# ---------------------------------------------------------------------------
#.night /.nightd sidecars
# ---------------------------------------------------------------------------
def _verify_night_sidecar(name, dir_path, vertex_bytes, problems, counts):
    path = os.path.join(dir_path, name[:-5] + ".night")   # name ends ".pmap"
    if not os.path.exists(path):
        return
    counts["night_checked"] += 1
    if vertex_bytes % VERT_SZ != 0:
        problems.append("%s: header vertex_bytes (%d) is not a multiple of %d - cannot "
                        "even compute this tile's vertex count" % (name, vertex_bytes, VERT_SZ))
        return
    nv = vertex_bytes // VERT_SZ
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) != nv * 2:
        problems.append(
            "%s: .night sidecar is %d bytes, expected exactly %d (2 bytes x %d vertices "
            "in this tile's own vertex pool)" % (name, len(buf), nv * 2, nv))


def _verify_nightd_sidecar(name, dir_path, vertex_bytes, problems, counts):
    path = os.path.join(dir_path, name[:-5] + ".nightd")
    if not os.path.exists(path):
        return
    counts["nightd_checked"] += 1
    if vertex_bytes % VERT_SZ != 0:
        problems.append("%s: header vertex_bytes (%d) is not a multiple of %d - cannot "
                        "even compute this tile's vertex count" % (name, vertex_bytes, VERT_SZ))
        return
    nv = vertex_bytes // VERT_SZ
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 4 + NIGHTD_HDR.size or buf[:4] != NIGHTD_MAGIC:
        problems.append("%s: .nightd sidecar has a bad magic/header (expected b'NDL2' + "
                        "a run count)" % (name,))
        return
    (run_count,) = NIGHTD_HDR.unpack_from(buf, 4)
    need = 4 + NIGHTD_HDR.size + run_count * PMAP_NIGHT_RUN.size
    if len(buf) != need:
        problems.append(
            "%s: .nightd sidecar is %d bytes, expected exactly %d for %d run(s)"
            % (name, len(buf), need, run_count))
        return
    off = 4 + NIGHTD_HDR.size
    for ri in range(run_count):
        vidx, n, _col = PMAP_NIGHT_RUN.unpack_from(buf, off + ri * PMAP_NIGHT_RUN.size)
        counts["nightd_runs_checked"] += 1
        if vidx + n > nv:
            problems.append(
                "%s: .nightd run %d addresses vertices [%d,%d), but this tile's vertex "
                "pool only has %d vertices" % (name, ri, vidx, vidx + n, nv))


# ---------------------------------------------------------------------------
# Build stamp, recomputed from the true source (--ref only): a STRONGER check
# than the internal tile-vs-world.idx agreement above, which only proves the
# store is self-consistent, not that it actually reflects the source it claims
# to. The exact byte formula below is NOT fully specified by world_store_build.
# py's own docstring - see the comment on _compute_build_stamp for what the
# docstring says, what it leaves out, and how the gap was closed.
# ---------------------------------------------------------------------------
def _compute_build_stamp(tiles):
    """tiles: {name: raw source .pmap bytes}. Returns the u32 CRC32 build stamp
 for this exact tile set, by the SAME rule world.idx's own build_stamp field
 and every stripped tile's index_off are supposed to agree on.

 HISTORY (kept because it is the actual evidence this formula is right, not
 just an assertion of it): the first version of world_store_build.py's own
 docstring said only "a u32 CRC32 ... over every source tile's name and
 bytes, concatenated in SORTED-name order" - it did not say byte order,
 separator, or chaining, and this module could not reproduce the real
 world.idx's own stamp from that sentence alone. That gap was reported (not
 silently patched over by reading strip_tile()'s implementation) and closed
 empirically instead, the same way this module settled index locality
 earlier: by computing candidate formulas over the REAL ps2full/ tiles and
 checking which one reproduces the real world.idx's own build_stamp
 (0x8ba38c79, measured live) - treating the writer's OWN produced artifact
 as ground truth, not its source. Plain name+bytes concatenation (with or
 without a separating newline), bytes+name, all-names-then-all-bytes, and a
 crc-of-per-tile-crcs were all tried and did NOT reproduce it; the formula
 below did, exactly.

 The docstring was THEN rewritten to give this exact formula as literal
 pseudocode (world_store_build.py's own FORMAT section, BUILD STAMP
 paragraph) - re-read from scratch on a later pass, independently of this
 function's own text below, and it matches term for term: same order (name,
 then a single 0x00, then bytes), same chaining (one running CRC across
 every tile, not per-tile digests combined some other way), same sort key.
 That agreement is the actual confirmation the docstring gap is closed, not
 just the rewrite existing."""
    crc = 0
    for name in sorted(tiles):
        crc = zlib.crc32(name.encode("utf-8"), crc)
        crc = zlib.crc32(b"\x00", crc)
        crc = zlib.crc32(tiles[name], crc)
    return crc & 0xFFFFFFFF


def _verify_build_stamp_against_source(ref_dir, expected_names, expected_stamp, problems):
    """Recomputes the build stamp from region_*.pmap in `ref_dir` and compares it
 to world.idx's own build_stamp field. Unlike the per-tile internal check in
 _verify_stripped_tile (which only proves the store agrees with ITSELF), this
 proves the stamp is not just consistent but actually CORRECT relative to the
 true source - the fidelity claim's own root, one level up from any individual
 blob. Does nothing (returns 0) if `ref_dir` does not hold EXACTLY the tile set
 world.idx's own tile directory names (`expected_names`) - a partial --ref
 directory is a normal, supported case elsewhere in this module (see
 test_verify_dir_ref_mode_handles_a_missing_reference_tile_gracefully), and a
 stamp recomputed over the wrong tile set would not mean anything; reporting a
 mismatch there would be a manufactured problem, not a real one. A count match
 alone is not enough either - this checks the actual NAME SET agrees, not just
 how many files happen to be present."""
    names = sorted(f for f in os.listdir(ref_dir)
                   if f.startswith("region_") and f.endswith(".pmap"))
    if set(names) != set(expected_names):
        return 0
    tiles = {}
    for n in names:
        with open(os.path.join(ref_dir, n), "rb") as fh:
            tiles[n] = fh.read()
    got = _compute_build_stamp(tiles)
    if got != expected_stamp:
        problems.append(
            "build stamp: recomputed 0x%08x from %d source tile(s) in %s, but world.idx "
            "says 0x%08x - the store does not match this source (built from a different "
            "or since-edited copy)" % (got, len(names), ref_dir, expected_stamp))
    return len(names)


# ---------------------------------------------------------------------------
# Sidecar files: every non-.pmap, non-container file next to the tiles (.col,
#.night,.nightd,.grass,.lod,.sway,.dyn,.spin,.mflags,.road,.tobj,
#.anim, regions.bin, and anything else that shows up later) must be copied
# through byte-for-byte unchanged, per the docstring's SIDECAR FILES section.
# Discovered from the directory listing itself, not a hardcoded extension list,
# so this does not need updating - or the docstring's own enumeration
# repeated here - if another sidecar type is ever added.
# ---------------------------------------------------------------------------
def _list_sidecar_files(dir_path):
    out = set()
    for f in os.listdir(dir_path):
        full = os.path.join(dir_path, f)
        if f in ("world.idx", "world.dat"):
            continue
        if f.startswith("region_") and f.endswith(".pmap"):
            continue
        if os.path.isdir(full):
            continue
        out.add(f)
    return out


def _verify_sidecars_against_ref(path, ref, problems, counts):
    store_side = _list_sidecar_files(path)
    ref_side = _list_sidecar_files(ref)
    for missing in sorted(ref_side - store_side):
        problems.append(
            "%s: sidecar present in the reference (%s) but missing from the store"
            % (missing, ref))
    for extra in sorted(store_side - ref_side):
        problems.append(
            "%s: sidecar present in the store but not in the reference (%s)"
            % (extra, ref))
    for name in sorted(store_side & ref_side):
        with open(os.path.join(path, name), "rb") as fh:
            a = fh.read()
        with open(os.path.join(ref, name), "rb") as fh:
            b = fh.read()
        counts["sidecars_checked"] += 1
        if a != b:
            problems.append(
                "%s: sidecar differs from the reference byte-for-byte (%d bytes -> %d "
                "bytes)" % (name, len(b), len(a)))


# ---------------------------------------------------------------------------
# --ref fidelity: the claim nothing else proves.
# ---------------------------------------------------------------------------
def _verify_resident_prefix_byte_identity(name, stripped_buf, ref_buf, s_h, r_h, problems):
    """world_store_build.py's own strip_tile() docstring promises the resident
 prefix (header through comp_tex) is "byte-identical to the source EXCEPT"
 five header fields (version, file_size, vertex_off, comp_flag, and - format 2
 - index_off, repurposed to the build stamp) and comp_model[i]/comp_tex[i]'s
 own off/gid field. Nothing else in this module ever re-opens that promise: the
 per-model/per-texture blob comparison below only ever looks at comp_model/
 comp_tex's TARGETS (the blobs they point to), never at the MODEL table's
 scale/centre, a SUBMESH's texture id, or the INSTANCE table/grid themselves - a regression that quietly rewrote any of those (e.g. a future refactor of
 strip_tile() that also touched the model table) would sail through every
 other check in this module, --ref included, with nothing here to catch it.
 This is that check: a direct byte-range compare of everything between the two
 files that is supposed to never change at all."""
    for i in range(len(s_h)):
        if i in (_H_VERSION, _H_FILESIZE, _H_VERTEX_OFF, _H_COMP_FLAG, _H_INDEX_OFF):
            continue   # the five fields strip_tile is documented to change
        if s_h[i] != r_h[i]:
            problems.append(
                "%s: header field %d changed from the reference (%r -> %r) - strip_tile() "
                "has no legitimate reason to touch this field" % (name, i, r_h[i], s_h[i]))

    comp_model_off = s_h[_H_COMP_MODEL_OFF]
    if comp_model_off != r_h[_H_COMP_MODEL_OFF]:
        return   # already reported above as a header-field mismatch; the range below
                 # would not even mean the same thing between the two files
    prefix_end = min(len(stripped_buf), comp_model_off)
    a = stripped_buf[PMAP_HDR.size:prefix_end]
    b = ref_buf[PMAP_HDR.size:prefix_end]
    if a != b:
        # find roughly where, for a useful message, without assuming word alignment
        first_diff = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        problems.append(
            "%s: the resident prefix (model/submesh/texture-descriptor/instance/grid "
            "tables, byte range [%d,%d)) differs from the reference starting at relative "
            "offset %d - strip_tile() must not touch any of this"
            % (name, PMAP_HDR.size, comp_model_off, first_diff))


def _verify_ref_fidelity(name, stripped_buf, stripped_t, ref_buf, idx_info, dat_bytes,
                         problems, counts):
    try:
        r = _parse_pmap_tables(ref_buf)
    except ValueError as exc:
        problems.append("%s: reference tile: %s" % (name, exc))
        return

    _verify_resident_prefix_byte_identity(name, stripped_buf, ref_buf,
                                          stripped_t["h"], r["h"], problems)

    s_models, s_subs = stripped_t["models"], stripped_t["subs"]
    r_models, r_subs = r["models"], r["subs"]
    if len(s_models) != len(r_models) or len(s_subs) != len(r_subs):
        problems.append(
            "%s: model/submesh counts differ from the reference (%d/%d -> %d/%d) - "
            "cannot compare blobs index-for-index" % (name, len(r_models), len(r_subs),
                                                       len(s_models), len(s_subs)))
        return

    model_table = idx_info["model_table"]
    for i, (gid, _csize) in enumerate(stripped_t["comp_model"]):
        try:
            _vf, vcount, _if, icount = _model_span(s_models, s_subs, i)
            _rvf, rvcount, _rif, ricount = _model_span(r_models, r_subs, i)
        except (IndexError, struct.error) as exc:
            problems.append("%s model %d: submesh range runs past the submesh table (%s)"
                             % (name, i, exc))
            continue
        if vcount != rvcount or icount != ricount:
            problems.append(
                "%s model %d: vertex/index count differs from the reference (%d/%d -> "
                "%d/%d)" % (name, i, rvcount, ricount, vcount, icount))
            continue
        if vcount == 0:
            continue
        counts["ref_models"] += 1
        need = vcount * VERT_SZ + icount * 2

        r_off, r_csize = r["comp_model"][i]
        try:
            ref_bytes = (lz4.block.decompress(ref_buf[r_off:r_off + r_csize],
                                              uncompressed_size=need) if r_csize else b"")
        except Exception as exc:                                        # noqa: BLE001
            problems.append("%s model %d: reference blob does not decompress (%s)"
                            % (name, i, exc))
            continue

        if gid >= len(model_table):
            problems.append("%s model %d: global id %d is out of range (%d unique models)"
                            % (name, i, gid, len(model_table)))
            continue
        off, csize, dsize = model_table[gid]
        try:
            store_bytes = (lz4.block.decompress(dat_bytes[off:off + csize],
                                                uncompressed_size=dsize) if csize else b"")
        except Exception as exc:                                        # noqa: BLE001
            problems.append("%s model %d: store blob (global id %d) does not decompress "
                            "(%s)" % (name, i, gid, exc))
            continue

        if store_bytes != ref_bytes:
            problems.append(
                "%s model %d: the blob resolved through the store (global id %d) is NOT "
                "byte-identical to the original tile's own blob" % (name, i, gid))

    tex_table = idx_info["tex_table"]
    s_tex_descs, r_tex_descs = stripped_t["tex_descs"], r["tex_descs"]
    if len(s_tex_descs) != len(r_tex_descs):
        problems.append(
            "%s: texture count differs from the reference (%d -> %d)"
            % (name, len(r_tex_descs), len(s_tex_descs)))
        return
    for i, (gid, _csize) in enumerate(stripped_t["comp_tex"]):
        counts["ref_textures"] += 1
        tbytes, centries = s_tex_descs[i][4], s_tex_descs[i][7]
        need = tbytes + centries * 4

        r_off, r_csize = r["comp_tex"][i]
        try:
            ref_bytes = (lz4.block.decompress(ref_buf[r_off:r_off + r_csize],
                                              uncompressed_size=need) if r_csize else b"")
        except Exception as exc:                                        # noqa: BLE001
            problems.append("%s texture %d: reference blob does not decompress (%s)"
                            % (name, i, exc))
            continue

        if gid >= len(tex_table):
            problems.append("%s texture %d: global id %d is out of range (%d unique "
                            "textures)" % (name, i, gid, len(tex_table)))
            continue
        off, csize, dsize = tex_table[gid]
        try:
            store_bytes = (lz4.block.decompress(dat_bytes[off:off + csize],
                                                uncompressed_size=dsize) if csize else b"")
        except Exception as exc:                                        # noqa: BLE001
            problems.append("%s texture %d: store blob (global id %d) does not decompress "
                            "(%s)" % (name, i, gid, exc))
            continue

        if store_bytes != ref_bytes:
            problems.append(
                "%s texture %d: the blob resolved through the store (global id %d) is NOT "
                "byte-identical to the original tile's own blob" % (name, i, gid))


# ---------------------------------------------------------------------------
# verify_dir: the full job.
# ---------------------------------------------------------------------------
def verify_dir(path, ref=None):
    """Walk a ps2global/-shaped directory: world.idx, world.dat, and every
 region_*.pmap (stripped) tile in it, plus .night/.nightd sidecars where
 present. With `ref` given, also checks byte-identity against the original
 (unstripped) tiles in that directory. Prints per-tile problems and a summary
 of what was actually checked; returns a shell exit code (0 = clean)."""
    idx_path = os.path.join(path, "world.idx")
    dat_path = os.path.join(path, "world.dat")
    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        print("no world.idx/world.dat in %s" % path)
        return 1
    with open(idx_path, "rb") as fh:
        idx_bytes = fh.read()
    with open(dat_path, "rb") as fh:
        dat_bytes = fh.read()

    problems = []
    ok, store_problems = verify_store(idx_bytes, dat_bytes)
    problems.extend(store_problems)

    try:
        idx_info = _parse_widx(idx_bytes)
    except ValueError as exc:
        problems.append(str(exc))
        print("world.idx: %s" % exc)
        print("checked 0 tiles, 0 blobs, 0 instances - world.idx itself could not be read")
        return 1
    idx_info["_raw"] = idx_bytes

    names_on_disk = sorted(f for f in os.listdir(path)
                           if f.startswith("region_") and f.endswith(".pmap"))
    names_in_idx = [_decode_tile_name(idx_bytes, e[0], e[1], problems, "tile %d" % i)
                    for i, e in enumerate(idx_info["tile_dir"])]
    names_in_idx_set = set(n for n in names_in_idx if n is not None)
    names_on_disk_set = set(names_on_disk)
    structural_problems = []
    for missing in sorted(names_in_idx_set - names_on_disk_set):
        structural_problems.append(
            "%s: listed in world.idx's tile directory but missing from %s" % (missing, path))
    for extra in sorted(names_on_disk_set - names_in_idx_set):
        structural_problems.append(
            "%s: a region_*.pmap file on disk with no entry in world.idx's tile directory"
            % (extra,))
    problems.extend(structural_problems)

    counts = {"tiles": 0, "models": 0, "textures": 0, "instances": 0,
             "ref_models": 0, "ref_textures": 0, "ref_tiles": 0,
             "build_stamp_checked": 0, "night_checked": 0, "nightd_checked": 0,
             "nightd_runs_checked": 0, "sidecars_checked": 0}

    for i, name in enumerate(names_in_idx):
        if name is None or name not in names_on_disk_set:
            continue
        counts["tiles"] += 1
        tile_path = os.path.join(path, name)
        with open(tile_path, "rb") as fh:
            buf = fh.read()

        tile_problems = []
        t = _verify_stripped_tile(name, buf, idx_info["tile_dir"][i], idx_info, dat_bytes,
                                  tile_problems, counts)

        if t is not None:
            vertex_bytes = t["h"][_H_VERTEX_BYTES]
            _verify_night_sidecar(name, path, vertex_bytes, tile_problems, counts)
            _verify_nightd_sidecar(name, path, vertex_bytes, tile_problems, counts)

            if ref is not None and t["h"][_H_VERSION] == PMAP_VERSION_STRIPPED:
                ref_path = os.path.join(ref, name)
                if os.path.exists(ref_path):
                    counts["ref_tiles"] += 1
                    with open(ref_path, "rb") as fh:
                        ref_buf = fh.read()
                    _verify_ref_fidelity(name, buf, t, ref_buf, idx_info, dat_bytes,
                                        tile_problems, counts)

        if tile_problems:
            print("%s:" % name)
            for p in tile_problems[:8]:
                print("   ", p)
            if len(tile_problems) > 8:
                print("    ... and %d more" % (len(tile_problems) - 8))
        problems.extend(tile_problems)

    global_problems = []
    if ref is not None:
        # Two whole-store checks, each meaningless per-tile: every sidecar file
        # (keyed by name, not per-region-tile) byte-identical to the reference,
        # and the build stamp recomputed from the FULL reference tile set (not
        # just the internal tile-vs-world.idx agreement already checked above).
        _verify_sidecars_against_ref(path, ref, global_problems, counts)
        counts["build_stamp_source_tiles"] = _verify_build_stamp_against_source(
            ref, names_in_idx_set, idx_info["build_stamp"], global_problems)
    problems.extend(global_problems)

    if global_problems:
        print("whole-store (--ref):")
        for p in global_problems[:8]:
            print("   ", p)
        if len(global_problems) > 8:
            print("    ... and %d more" % (len(global_problems) - 8))

    if store_problems:
        print("world.idx/world.dat (container-level):")
        for p in store_problems[:8]:
            print("   ", p)
        if len(store_problems) > 8:
            print("    ... and %d more" % (len(store_problems) - 8))

    if structural_problems:
        print("directory structure (world.idx vs the files actually on disk):")
        for p in structural_problems[:8]:
            print("   ", p)
        if len(structural_problems) > 8:
            print("    ... and %d more" % (len(structural_problems) - 8))

    print("checked %d tiles, %d unique model blob(s) + %d unique texture blob(s) in the "
         "store, %d model ref(s), %d texture ref(s), %d instance(s), %d build stamp(s), "
         "%d .night + %d .nightd (%d run(s))%s"
         % (counts["tiles"], len(idx_info["model_table"]), len(idx_info["tex_table"]),
            counts["models"], counts["textures"], counts["instances"],
            counts["build_stamp_checked"], counts["night_checked"], counts["nightd_checked"],
            counts["nightd_runs_checked"],
            (", %d tile(s) fidelity-checked against %d model ref(s) + %d texture ref(s), "
             "%d sidecar(s), build stamp recomputed from %d source tile(s) in --ref"
             % (counts["ref_tiles"], counts["ref_models"], counts["ref_textures"],
                counts["sidecars_checked"], counts.get("build_stamp_source_tiles", 0)))
            if ref is not None else ""))
    print("%d problem(s)" % len(problems))
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        ref = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1 or args[0].startswith("-"):
        if args:
            print("unrecognised argument(s): %s\n" % " ".join(args))
        print(__doc__)
        return 2
    return verify_dir(args[0], ref=ref)


if __name__ == "__main__":
    raise SystemExit(main())
