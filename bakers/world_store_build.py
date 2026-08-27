#!/usr/bin/env python3
"""world_store_build.py - stage 2a of
 (section 4): collapse
184 self-contained region_*.pmap tiles' DUPLICATE blob bytes into one global store - world.dat (unique compressed blobs, back to back) + world.idx (two global tables plus,
per tile, the array of global ids that replaces its former comp_model/comp_tex tables) - and write the STRIPPED tiles themselves (resident prefix only, blob region removed)
so the saving is realised on disk, not only computed.

WHY: the same building appears in many tiles, and today each tile carries its own full
copy of that building's geometry and textures - an offline census measured 4.33 copies
of the average texture across tiles. This pass builds that half of the fix. Nothing in
the engine changes in this task; it is built and verified entirely on PC.

THE DESIGN THIS IMPLEMENTS (spec section 4, stage 2a): tiles and their streaming stay
EXACTLY as they are - region slots, the 3x3 window, generations, the admission budget,
the worker, collision. Only the source and ownership of blobs changes: world.dat appears
alongside the tiles as one archive, opened once, addressed by a global blob id; in each
tile's OWN tables, a global id sits where a local blob offset used to
 This is
deliberately NOT the fully-global index section 3.1 of the same spec describes - one
world-wide model table, one instance list, one grid, tiles removed entirely - that is a
later, separate, optional stage (2d); section 3.1 says so explicitly. What stage 2a keeps
resident per tile and what it moves into the shared store are both measured on the real
184-tile world, not guessed:

 world.idx 0.44 MiB global blob tables + per-tile id arrays (see FORMAT below)
 world.dat 108.68 MiB unique compressed blobs, 16-byte aligned, Hilbert-curve order
 stripped
 tiles 5.67 MiB 184 region_*.pmap, resident prefix only, blobs removed

(An earlier draft of this figure said 5.26 MiB - the sum of the tiles' own comp_model_off
values, i.e. the resident prefix NOT counting the comp_model/comp_tex tables themselves.
That is a real quantity but not the one either place using it meant to report: 5.67 MiB is
the measured size of the stripped tiles actually sitting on disk, comp_model/comp_tex
included - there is nothing else in a stripped tile, since its blob region is gone.)

Why the per-tile tables (MODEL, SUBMESH, texture descriptors, instance, grid) stay WITH
the tiles instead of moving into world.idx: 5.67 MiB of tables inside one index means
either holding all of it resident against a ~24 MiB memory budget, or seeking inside the
index once per tile. Both are worse than 184 small local prefixes the engine already
knows how to read. The win comes from one permanently-open world.dat, not from merging
metadata: measured on this hardware, 94% of a tile's load time is file OPENS, at ~34ms
each (see this project's own tile-load-is-file-opens research) - replacing many small
per-tile blob reads with one already-open archive is the entire saving; collapsing 184
tables into one index does nothing for that number and only costs resident memory to get
there.

THE DEDUP KEY is blob-only: byte equality of the compressed bytes, for BOTH models and
textures, exactly as the plan specifies. A model's compressed blob holds ONLY the raw
int16 vertex/index pool - confirmed directly: tools/test_pmap_lattice.py's
make_tile(scale=X) with the same default vertices produces byte-IDENTICAL compressed
blobs regardless of X (see test_scale_alone_does_not_change_the_compressed_blob_bytes,
this pass's own test file). Scale and centre, which turn those ints into a world-space
position, live in the separate PmapModel record - and because stage 2a keeps that
record exactly where it already lives, in the tile's own .pmap (see above), two models
with the SAME vertex pool and DIFFERENT scale or centre are genuinely the same geometry
placed differently: they SHOULD share one blob, and each tile's own model record still
dequantises it its own way. That is precisely what a global store is for. (An earlier
version of this module widened the key to (scale, cx, cy, cz, blob) to satisfy a test
that assumed two such models must NOT collapse - that test encoded a misunderstanding
of this module's own design, not a real constraint, and has been replaced; see
test_same_vertex_pool_different_placement_shares_one_blob.) Textures were always
blob-only (no placement concept applies to a texture at all), so this brings models onto
the identical rule textures already used - both dedup through the same _dedup_blobs().

A DECOMPRESSED-CONTENT key (two blobs whose compressed bytes differ because they were
compressed with different settings, but which decompress to the same bytes) is strictly
a superset of what the compressed-byte key finds. It is MEASURED here - _content_dedup_extra() decompresses every already-unique blob and groups by content - and reported in stats (*_unique_by_content, *_content_collisions,
*_bytes_saved_extra_by_content), but never applied to the actual store: switching keys
changes what "the same blob" means for a system with no independent way to notice a bad
merge, and that is a decision for a human with the measured numbers in hand, not
something this pass decides on its own.

SHARED NIGHT AGREEMENT. The dedup key above is deliberately blob-only, and that is
still correct for the blob's OWN bytes - but a model's vertex COLOUR is not entirely
inside that blob at runtime. src/platform_psp/pmap.c's relight_model (called from
pmap_use_model whenever the day/night balance changes) MUTATES a loaded model's vertex
bytes IN PLACE: it rewrites the 2-byte colour field of every vertex, reading w->night
and w->nightd - both PER-TILE sidecars, indexed by the model's vertex offset into
THAT TILE's own vertex pool. A shared blob has no single `w` to relight from. Two
tiles that disagree about a shared model's night colours would fight over the SAME
physical bytes: whichever tile's relight ran most recently wins, and the other tile's
geometry is now wearing the wrong tile's colours until its own relight runs again (if
it ever does). That is live-bytes corruption, not a misclassification - and it is a
hazard for the (not yet built) engine-side world-store reader specifically, the same
class of incidental-safety mistake already paid for twice with comp_flag and
IoBench.c: today it cannot happen (PMAP_VERSION_STRIPPED is not in any load whitelist
yet), but the writer is the one place this can be closed for good before there is a
reader to get it wrong.

Measured against the real 184-tile world rather than designed around a maybe: 10264
unique model blobs, 689 of them shared (referenced by more than one model-slot,
counting BOTH across tiles and, in principle, twice within one tile - see
_resolve_night_share_keys' own docstring for why tile identity is not special-cased),
max 69 distinct tiles sharing a single blob. Of those 689: 679 agree (0 of the 679
had NO tile asserting any night colour at all for that model - every shared blob in
today's world has at least one tile with real .night/.nightd data; this count is
measured every build, not assumed, see night_absent_groups in build_store()'s stats
and the printed report's own line). 10 genuinely DIFFER (8 across two different
tiles, 2 within a single tile referencing its own blob twice at different night-
authored spans - both shapes are the same hazard and both are excluded the same
way). Excluding those 10 from sharing costs 20127 bytes (19.7 KiB) of extra,
un-deduplicated blob copies - 0.0177% of a ~114.68 MiB combined store. Reproduce
this measurement with tools/pmap_lattice_verify.py's own _model_spans and the
_parse_night/_parse_nightd/_night_signature functions below against a real ps2full/
directory; if a future rebuild moves these numbers by much more than a handful, that
is a signal the world's content changed in a way worth understanding, not a reason to
silently accept a different number.

So: do not solve shared relighting (no attempt is made to give a shared blob multiple
simultaneous colour states, or to have the engine copy-on-write a shared buffer before
relighting it - both are real designs, neither is free, and neither is needed here).
Exclude the hard case instead, and let the measurement justify it - the same shape as
the lattice pass's too_small gate, where refusing the awkward models cost nothing
measurable and removed a whole class of problem rather than solving it in general. A
model blob may be shared only if every tile referencing it agrees BYTE-FOR-BYTE on the
.night bytes covering that model's vertex span, AND on any .nightd runs that address
that span (clipped to it, the same clip relight_model itself applies - see
_night_signature). A tile with no usable .night file for a given model does not block
sharing: missing night data is read as "this tile asserts nothing", compatible with
any other tile's assertion, concrete or absent - see _resolve_night_share_keys and
its night_absent_groups counter, which exists specifically so that "missing means no
constraint" is a number every build re-confirms rather than a rule taken on faith.
Where tiles disagree, each distinct night-colour assertion gets its OWN physical blob
copy in world.dat (never a new field, never a per-model flag - just a second global
id for the SAME bytes), so relight_model mutating one tile's copy can never reach
another tile's.

★ Do not "optimise" this exclusion away (e.g. by widening the dedup key back to
blob-only and accepting the eventual colour-fighting as a rendering detail to fix
later) without re-running the measurement above against whatever the world has become
by then - 19.7 KiB was cheap for THIS world; a different one is not guaranteed to be.

Also checked, and settled rather than deferred: `isLod`, one bit per model stamped
from a tile's own instances, had the same SHAPE of risk (a census that only ever
looked at one tile at a time deciding "last write wins" for a value that is actually
GLOBAL per shared blob). Measured directly, with the instance stride verified against
real data first (PmapInstance is 36 bytes, `interior` at byte offset 28): zero
conflicts across all 689 shared blobs - no shared blob is an LOD proxy in one tile
and detail geometry in another. Nothing to do here; recorded so the next person does
not have to re-derive the worry from scratch.

STRIPPED TILES: writing world.idx/world.dat alone only COMPUTES the saving - the source
.pmap files still carry every blob, so nothing has actually gotten smaller on disk until
they are rewritten too. main() also writes a stripped copy of every tile into the same
output directory: same filename, same resident prefix (header through the comp_tex
table) byte-for-byte, EXCEPT comp_model[i]/comp_tex[i] are rewritten from
(local_offset, csize) to (global_id, csize) and the blob region is removed entirely
(vertex_off becomes the new, smaller file_size - the blob region is empty, "starting"
at EOF). version becomes PMAP_VERSION_STRIPPED (5, matching src/platform_psp/pmap.h's
own constant of the same name) - NOT comp_flag, which an earlier version of this module
picked and which review found does not work: pmap.c's pmap_load_finish and
pmap_lattice_verify.py both test comp_flag for truthiness only, never a specific value,
so any nonzero comp_flag passes identically. Version is what those two enforce to an
exact whitelist, so a stripped tile is refused by them today - with no engine change - simply by using a version number that whitelist does not contain yet.

★ THE GENERAL LESSON, learned the hard way in the same review that found the comp_flag
mistake: a version is only a gate where something actually checks it, and this codebase
has roughly a dozen independent .pmap header parsers. Defining PMAP_VERSION_STRIPPED does
not gate anything by itself - it took a second pass to find that
src/platform_psp/IoBench.c opens a .pmap itself (does not go through pmap_load_finish at
all) and gated on an open-ended `version < PMAP_VERSION_LZ4`, which a stripped tile (5) is
not less than, so it fell through and fed a global id to LZ4 as though it were a byte
offset - fails cleanly there, but silently, as wrong benchmark numbers rather than a
refusal. Fixed alongside this comment (now an exact whitelist, matching pmap.c's own
style) together with tools/stream_io_model.py (checked `comp_flag == 1`, silently read a
stripped tile as "no compressed blobs" instead of refusing it by name). Both are believed
fixed now; neither this module nor pmap.h claims that is true of every .pmap reader that
exists - only of the ones actually found and checked. Anything encountered later needs
the same treatment: an exact version whitelist, not comp_flag, and not an open-ended
comparison either. comp_flag is still set to STRIPPED_COMP_FLAG (2) as a secondary,
human-readable marker, but the version field is the actual gate. See strip_tile() for
the exact byte-level contract, and strip_all() for applying it to every tile using the
ids a just-built (or just-reopened) world.idx already recorded for each one.

A stripped tile also carries a BUILD STAMP (see BUILD STAMP below) so a stale or partial
copy of world.idx/world.dat/a stripped tile can be detected instead of silently resolving
a global id into the wrong blob, and main() copies every SIDECAR file (.col, .night,
.nightd, .grass, .lod, .sway, .dyn, .spin, .mflags, .road, .tobj, .anim, regions.bin - anything that is not itself a region_*.pmap) through byte-for-byte unchanged: they are
keyed by tile name, and stripping touches neither vertex counts nor vertex order, so they
stay valid exactly as they are. Without them, an output directory holding only world.idx/
world.dat/stripped .pmap files is not a usable alternative world - switching to it would
silently lose collision, baked night lighting, grass, LOD proxy links and render flags
with no error at all. See the FORMAT section's SIDECAR FILES paragraph for the measured
sizes.

BLOB ORDERING: unique blobs are appended to world.dat in FIRST-SEEN order while walking
tiles in a spatial traversal (models before textures, matching the source .pmap's own
per-tile convention - see tools/pmap_lz4.py). This is the spec's own requirement R1

img_archive_v2_and_streaming_scheduler.md) - SA itself reads several files in one call
only because they sit contiguously in read order, and the same physical reasoning
applies here. The traversal is a single named knob, TILE_ORDER (also the `order=`
parameter on build_store/main), defaulting to a Hilbert space-filling curve rather than
row-major: row-major (sort by y, then x) is correct along one row but has a seam at
every row boundary - the last tile of row y and the first tile of row y+1 sit
back-to-back in the file while potentially `width` tiles apart in reality, and the tile
directly above the last tile of a row is `width` tiles AWAY in file order. On a FULLY
OCCUPIED grid a Hilbert curve has no such seam at all: any two consecutive positions are
always grid-adjacent, everywhere - proven as a structural property, not measured, in
test_hilbert_curve_is_a_bijection_with_adjacent_steps (a dense synthetic 16x16 grid).

★ On the REAL map that property does not hold, and the earlier wording here claiming
"always... everywhere in the grid" overstated it: the world is only 71.9% occupied (184
of a 256-cell bounding square), so consecutive Hilbert positions are grid-adjacent on
169 of 183 steps - 92.3%, measured directly on the real tile set, not 100%. Sparse
occupancy is what breaks it: the curve still visits every cell of the notional square in
adjacency order, but 28.1% of those cells are empty, and skipping an empty one can hand
the next OCCUPIED tile a non-adjacent predecessor.

★ Separately, and more importantly for what Part E should expect: neither ordering
actually solves the problem a read-ahead window cares about. The reviewer rebuilt the
world under both orderings and measured the byte span each tile's 3x3 window needs
inside world.dat. Hilbert beats row-major by about 5% overall but LOSES on 30% of
individual tiles, and the median window span is 90.4 MiB out of a 108.7 MiB world.dat - a median 12.7x read amplification versus the window's own uncompressed working set.
Global dedup anchors a shared blob wherever it was FIRST seen while walking tiles in
whatever order was chosen, and duplication in this world is not spatially confined (see
sa-lod-sharing-myth and the census this module's own WHY paragraph cites) - a building's
one shared copy can just as easily be anchored by a tile on the far side of the map as by
a neighbour. No tile ordering fixes that; ordering only ever controls how tiles that
happen to need DISTINCT, never-shared blobs are laid out relative to each other, and most
of a real tile's window is shared blobs anchored elsewhere. Hilbert is kept as the
default anyway - it is free and slightly better on the aggregate number above - but
this comment must not promise a locality guarantee the format does not deliver; a future
read-ahead built assuming a 3x3 window is a small, mostly-sequential read will be wrong
by an order of magnitude for a typical tile. This is what a future streaming read-ahead
depends on knowing: a sequential read inside an already-open window costs nothing on the
memory card, a new offset costs a full seek (see this project's own
sa-img-elevator-and-coalescing research), so keeping spatially-adjacent tiles adjacent in
world.dat is still worth doing where it is free - just not sufficient on its own. Pass
order="row_major" for the (worse, but simple) baseline.

FORMAT

world.dat: unique blobs back to back, each starting on a 16-byte boundary (the GE's DMA
alignment requirement). No 2048-byte sectors - this is a PC-side global store, not an
ISO layer.

world.idx - VERSION 2 (all integers little-endian; version 1, the original 40-byte
header with u32 refs, cannot be read by this reader at all, see read_index()'s own
version==1 message):
 offset bytes field type
 [ 0] 4 magic 'WIDX' 4s
 [ 4] 4 version (=2) u32
 [ 8] 4 build_stamp u32 - see BUILD STAMP below for the EXACT formula
 [12] 4 model_unique_count u32 (M)
 [16] 4 tex_unique_count u32 (T)
 [20] 4 tile_count u32 (N)
 [24] 4 model_table_off u32
 [28] 4 tex_table_off u32
 [32] 4 tile_dir_off u32
 [36] 4 names_off u32
 [40] 4 refs_off u32
 - header is exactly 44 bytes [0:44); model_table_off is always 44 (== header
 size), since the model table is the first thing that follows it, but every offset
 is still stored explicitly rather than assumed, and a reader should use the stored
 value, not recompute it.

 model table, at model_table_off, M entries, 12 bytes each, no padding between
 entries: (off:u32, csize:u32, dsize:u32) into world.dat, in that field order - off first, then csize, then dsize. Entry i starts at model_table_off + i*12.

 tex table, at tex_table_off, T entries, same 12-byte (off, csize, dsize) shape as
 the model table. Entry i starts at tex_table_off + i*12.

 tile dir, at tile_dir_off, N entries, 24 bytes each, no padding between entries:
 (name_off:u32, name_len:u32, model_count:u32, model_refs_off:u32, tex_count:u32,
 tex_refs_off:u32), in that field order. Entry i starts at tile_dir_off + i*24. One
 entry per tile, in the SAME spatial order the blobs themselves were emitted in
 (see BLOB ORDERING above).

 names, at names_off: tile name bytes UTF-8 encoded, back to back with NO separator
 and NO terminator between them - a name's own (name_off, name_len) from its tile
 dir entry is the only thing that bounds it; do not scan for a NUL. names_off is
 padded to the next 4-byte boundary from wherever the tile dir ends, purely so
 refs_off (below) starts aligned; the padding bytes themselves (0..3 zero bytes) are
 never referenced by any (name_off, name_len) pair and must not be interpreted as
 a name.

 refs, at refs_off (guaranteed 4-byte aligned; see _pack_index()'s own comment for
 why): for each tile, in tile-dir order, model_count u16 global model ids (in the
 tile's OWN original per-model order, i.e. entry i here is model i's global id in
 THAT tile's own MODEL table) immediately followed by tex_count u16 global texture
 ids (same rule, texture order) - model refs before tex refs, no padding between
 the two arrays or between one tile's refs and the next tile's. u16, not u32: the
 only genuinely new per-reference surface this format adds, so it is sized to what
 it actually needs (real world: <=10264 models, <=9198 textures, both comfortably
 under the 65535 ceiling) rather than defaulted to u32 - halves refs_blob.
 build_store() REFUSES to build (raises ValueError naming which table and by how
 much) if either global table would ever exceed 65535 entries, rather than silently
 wrapping a ref that no longer fits. An individual tile's own model_refs_off/
 tex_refs_off is only GUARANTEED 2-byte aligned (not 4-byte) once entries are u16 - correct and sufficient for a uint16_t* cast on MIPS, not a regression from u32's
 4-byte guarantee.

 A global id is an index into the GLOBAL table (model table or tex table above),
 not a byte offset - resolve it via model_table[gid] / tex_table[gid] to get the
 (off, csize, dsize) that actually locates the blob in world.dat.

 Everything else about a tile (its MODEL table's scale/centre/bound_radius/
 draw_dist, its SUBMESH table, instance table, grid) is untouched by this pass and
 stays exactly where it already lives - in the tile's own .pmap, and, after
 stripping, in the stripped copy of it too.

stripped .pmap (written alongside world.idx/world.dat, one per source tile, same
filename): the same v3 header shape (23 u32 fields, 92 bytes, see pmap.h's own
PmapHeader) and every table up to and including comp_tex, byte-identical to the source
EXCEPT exactly these fields (by header index, 0-based, matching pmap.h's own field
order - byte offset = index*4):
 index 1 (byte 4) version = PMAP_VERSION_STRIPPED (5) - the actual gate; not
 in any load path's version whitelist yet, refused
 index 2 (byte 8) file_size = the new, smaller size - the blob region is gone
 index 12 (byte 48) vertex_off = same new size as file_size - the blob region is
 empty, "starting" (and ending) at EOF
 index 14 (byte 56) index_off = the BUILD STAMP (see below), REPURPOSED - only
 for version==PMAP_VERSION_STRIPPED; index_off is
 always 0 and unread on every v3-family tile
 regardless (the raw index pool it used to point
 at does not exist past v2, and index_bytes, the
 field right after it, is unread at every version,
 not just v3+ - see pmap.h's own loud comment at
 the struct declaration), so this reuses dead
 space instead of growing the header
 index 20 (byte 80) comp_flag = STRIPPED_COMP_FLAG (2) - secondary marker only,
 tested for truthiness everywhere, not a real gate
 comp_model[i], comp_tex[i] = (global_id, csize) - the FIRST u32 of each
 8-byte PmapComp entry (formerly a local byte
 offset) becomes a global id; the SECOND u32
 (csize) is left as-is (still meaningful: 0 iff
 no blob, the same convention as before stripping)

BUILD STAMP - the EXACT formula, not a description of one (an earlier version of this
docstring said "CRC32 over name and bytes, concatenated", which does not say the byte
order, the separator, or whether the CRC is chained per tile or restarted - found in
review: another reader could not reproduce the value from that sentence alone and had
to derived it by testing candidate formulas against a real, already-built
world.idx's known stamp, which only works when a known-good value already exists to
check against). The reference implementation is _compute_build_stamp(); this is
EXACTLY what it computes, in order:

 crc = 0
 for name in sorted(tiles.keys()): # Python string sort, ascending;
 # NOT insertion/dict order
 crc = zlib.crc32(name.encode("utf-8"), crc) # the tile's name, UTF-8, no
 # trailing NUL of its own
 crc = zlib.crc32(b"\x00", crc) # ONE separator byte, 0x00
 crc = zlib.crc32(tiles[name], crc) # the tile's raw .pmap bytes,
 # in full, unmodified
 build_stamp = crc & 0xFFFFFFFF

The CRC is CHAINED, not per-tile: each zlib.crc32() call passes the PREVIOUS call's
result as its own second argument (zlib.crc32's own running-CRC parameter), so the
whole loop is one continuous 32-bit CRC over the concatenation of every
(name + 0x00-separator + bytes) triple, in sorted order - not N independent per-tile
CRCs combined some other way (XORed, summed, etc.). The 0x00 separator exists so that,
e.g., tile "a" with bytes "bc" cannot hash the same as tile "ab" with bytes "c" - without it two different (name-set, content) pairs could collide. Sorting by name
(never insertion order) is what makes the stamp reproducible regardless of what order
a caller's dict happened to be built in - see
test_build_store_result_is_independent_of_input_dict_order, and, for the stamp
specifically, test_build_stamp_is_content_addressed_and_order_independent.

The SAME resulting value is written into world.idx's own header (byte offset 8, see
the field table above) and into every stripped tile's index_off field (byte offset 56,
see the stripped-.pmap field table above). World data is not in git and gets copied by
hand between a PC, an emulator directory and a memory stick - a partial redeploy
(world.dat rebuilt, one stripped tile left stale, or the reverse) is not hypothetical,
it is the likeliest way this format actually breaks, and the failure is silently WRONG
geometry (a global id resolves, just into the wrong build's blob) rather than a crash.
verify_build_stamp() checks every stripped tile's stamp against a world.idx's and names
exactly which tile disagrees, rather than leaving that to be discovered as a rendering
bug with no error anywhere in the chain.

SIDECAR FILES: every tile also has non-.pmap sidecars sitting next to it in the source
directory - .col (collision, NOT 1:1 with tiles, some extra), .night/.nightd (baked
night-time vertex colours and their vertex-indexed run table), .grass, .lod, .sway, .dyn,
.spin, .mflags, .road, .tobj, .anim, and one directory-level regions.bin. On the real
184-tile world these total 55.27 MiB against 317.90 MiB of .pmap - collision alone
(36.93 MiB) is bigger than the entire stripped-tile total. main() copies every one of
them into the output directory byte-for-byte unchanged: they are keyed by tile name, and
stripping touches neither vertex counts nor vertex order (see the DEDUP KEY section - placement and geometry both stay exactly where a sidecar's own indexing expects them),
so they need no transformation at all to remain valid. Skipping them would make the
output directory look like a complete world while silently missing collision, lighting,
grass, LOD links and render flags the moment anything tried to use it as one.

Every global entry carries both csize (compressed) and dsize (uncompressed) - the
brief's own framing: "that is the second u16 of an IMG v2 entry, which the original publisher
reserved and never used." dsize is not stored anywhere in the source .pmap either; it
is recomputed here exactly the way the original v2->v3 compressor (pmap_lz4.py)
derived it in the first place: a model's span from its own submeshes (vcount*VERT_SZ +
icount*2), a texture's from its own PmapTexture descriptor (tbytes + clut_entries*4).

Usage:
 python tools/world_store_build.py <in_dir> --out <out_dir> [--force]

 Refuses (exit 2, nothing written) if <out_dir> is the SAME real directory as
 <in_dir> (os.path.realpath comparison, so a trailing separator or a redundant
 '.' component does not slip past it - see main()'s own SAFETY comment) - always refused, --force does not override this one, because there is no safe
 version of overwriting the only thing you are still reading from. Also refuses
 if <out_dir> already contains region_*.pmap files this run did not just create
 (looks like an existing world, quite possibly someone's rollback copy) UNLESS
 --force is given. A fresh, empty, or absent <out_dir> needs no flag.
"""
import os
import shutil
import struct
import sys
import zlib

import lz4.block

# Reused, not reinvented: HDR/MODEL/SUBMESH/COMP/VERT_SZ are the exact v3 layout
# constants tools/pmap_lattice_verify.py cross-checked against a real region_0_0.pmap.
# _parse is that module's own safe header+table reader - it already turns "bad
# magic", "file shorter than the header", "wrong version", and "a table runs past the
# end of the file" into a plain ValueError instead of letting a corrupt tile crash a
# 184-tile batch; this module wraps that ValueError with the tile's own name (_parse
# has no tile name of its own to report) and adds the one table _parse deliberately
# never reads (comp_tex / the texture descriptor table - a v3-only concept with
# nothing for _parse's model/submesh-focused contract to check).
from pmap_lattice_verify import (HDR, MODEL, SUBMESH, COMP, VERT_SZ, _parse,
                                 _model_spans, PMAP_VERSION_STRIPPED)

IDX_MAGIC = b"WIDX"
IDX_VERSION = 2
# version 1 (the original shape of this format) was a 40-byte header (magic + 9 u32:
# version,M,T,N,model_table_off,tex_table_off,tile_dir_off,names_off,refs_off - no
# build_stamp) with refs_blob entries at u32. version 2 (current) inserts build_stamp
# as a NEW u32 field right after version (44-byte header, magic + 10 u32) and narrows
# every refs_blob entry from u32 to u16 (see REF_ID_MAX_COUNT's own comment) --
# BOTH are breaking changes to the byte layout, not merely additive ones: a version-1
# reader given a version-2 file would misparse a 44-byte header as 40 (reading
# model_unique_count from what is actually the low half of build_stamp) and read every
# ref array at HALF its real element count, at double the real per-element width.
# Bumped for exactly the reason PMAP_VERSION_STRIPPED exists: a version is only a gate
# where something actually checks it, and this container's own version field is that
# something for world.idx - see read_index's own check below, which names version 1
# specifically now rather than only saying "unsupported".

# THE REAL GATE IS THE VERSION FIELD, NOT THIS ONE. An earlier version of this module
# picked comp_flag=2 believing it would stop a v3-or-earlier reader from mistaking a
# stripped tile for a self-contained one - wrong, found in review: every load path
# (src/platform_psp/pmap.c's pmap_load_finish, and this project's own
# pmap_lattice_verify.py before this change) tests comp_flag for TRUTHINESS
# (`h->comp_flag`), never for a specific value, so 2 is exactly as accepted as 1. What
# every load path DOES enforce to an exact set is the version field - see
# PMAP_VERSION_STRIPPED (imported above, matching src/platform_psp/pmap.h's own
# constant of the same name) and strip_tile's own docstring. comp_flag=2 is kept
# anyway as a secondary, human-readable marker - harmless, and it still means
# something to a reader - but it is not what makes a stripped tile get refused.
STRIPPED_COMP_FLAG = 2

IDX_HDR = struct.Struct("<4s10I")       # magic,version,build_stamp,M,T,N,model_table_off,
                                         # tex_table_off,tile_dir_off,names_off,refs_off
GLOBAL_ENTRY = struct.Struct("<3I")     # off, csize, dsize
TILE_DIR_ENTRY = struct.Struct("<6I")   # name_off,name_len,model_count,model_refs_off,
                                         # tex_count,tex_refs_off

# The per-tile ref arrays (world.idx's refs_blob) hold u16 global ids - the only
# genuinely new per-reference surface this format adds, so it is sized to what it
# actually needs (real world: 10264 models, 9198 textures) rather than defaulted to
# u32; halves refs_blob. build_store refuses to build (see its own check) rather
# than silently wrap a ref that no longer fits once a global table exceeds this.
REF_ID_MAX_COUNT = 65535

# PmapTexture, 32 bytes: w,h,format,texel_first,texel_bytes,bufw,clut_first,
# clut_entries,num_levels - cross-checked against tools/pmap_lz4.py's own TEX_STRIDE=32
# unpack (`"<HHIIIIIII"`). Only tbytes (index 4) and clut_entries (index 7) are read
# here: they are what a texture blob decompresses to (tbytes + clut_entries*4), exactly
# how pmap_lz4.py's own compress sliced texel/clut pools when it built this same blob.
TEX_DESC = struct.Struct("<HHIIIIIII")

#.night sidecar (pmap.c pmap_load_night): raw u16[vertex_bytes/12], one GU_COLOR_5551
# value per vertex, aligned 1:1 to the tile's OWN vertex pool - no header, no magic; a
# wrong-sized file is treated as absent (pmap_load_night's own "size-mismatch -> stays
# NULL" fallback - see _parse_night below, which mirrors it exactly).
#.nightd sidecar (pmap.c pmap_load_nightd / pmap.h PmapNightRun): magic 'NDL2' (the LE
# u32 0x324C444E pmap.c compares against, here checked as the raw 4 bytes b"NDL2") + u32
# run_count + run_count x {u32 vidx; u16 n; u16 col} (8 bytes each) - run `n` consecutive
# vertices starting at pool index `vidx` glow at night. Same source (pmap.c) as
# world_store_verify.py's own independent NIGHTD_MAGIC/NIGHTD_HDR/PMAP_NIGHT_RUN --
# kept as two separate readings deliberately, same reasoning as the IDX_VERSION/
# WIDX_VERSION pin (test_writer_and_verifier_version_constants_agree).
NIGHTD_MAGIC = b"NDL2"
NIGHTD_HDR = struct.Struct("<I")        # run_count (the 4-byte magic precedes it)
PMAP_NIGHT_RUN = struct.Struct("<IHH")  # vidx, n, col - 8 bytes


def _pad_to_16(buf):
    while len(buf) % 16:
        buf.append(0)


# ---------------------------------------------------------------------------
# Spatial traversal order
# ---------------------------------------------------------------------------
def _tile_xy(name):
    """Parse 'region_<x>_<y>.pmap' -> (x, y) ints. Raises ValueError naming the tile
 for anything else - a silent fallback to some arbitrary order here would scramble
 locality without ever saying so, and locality is this function's entire purpose."""
    stem = name[:-5] if name.endswith(".pmap") else name
    parts = stem.split("_")
    if len(parts) == 3 and parts[0] == "region":
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            pass
    raise ValueError("tile name %r does not match region_<x>_<y>.pmap" % (name,))


def _hilbert_d(order, x, y):
    """Distance along a Hilbert curve for (x, y) in [0, order) x [0, order), `order`
 a power of two. Classic rot-and-flip formulation (Wikipedia, "Hilbert curve",
 xy2d). See test_hilbert_curve_is_a_bijection_with_adjacent_steps for the two
 properties this is actually relied on for: every cell gets a distinct distance,
 and consecutive distances are always grid-adjacent cells."""
    d = 0
    s = order // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


ORDER_ROW_MAJOR = "row_major"
ORDER_HILBERT = "hilbert"

# The single knob controlling world.dat's blob order. Change this constant (or pass
# order= into build_store/main) to switch the whole store's layout - see the module
# docstring's BLOB ORDERING section for why Hilbert is the default.
TILE_ORDER = ORDER_HILBERT


def _spatial_order(names, order=TILE_ORDER):
    """Sort tile names so spatially-adjacent tiles land close together in world.dat.
 See the module docstring for why Hilbert (the default) beats row-major (the
 explicit, simpler alternative - pass order="row_major") for this specific
 purpose: no seam at row boundaries."""
    coords = {n: _tile_xy(n) for n in names}
    if not coords:
        return []
    if order == ORDER_ROW_MAJOR:
        return sorted(coords, key=lambda n: (coords[n][1], coords[n][0], n))
    if order == ORDER_HILBERT:
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        min_x, min_y = min(xs), min(ys)
        span = max(max(xs) - min_x, max(ys) - min_y) + 1
        side = 1
        while side < span:
            side *= 2

        def key(n):
            x, y = coords[n]
            return (_hilbert_d(side, x - min_x, y - min_y), x, y, n)
        return sorted(coords, key=key)
    raise ValueError("unknown tile order %r" % (order,))


# ---------------------------------------------------------------------------
# Per-tile parsing
# ---------------------------------------------------------------------------
def _read_tile(name, buf):
    """(h, models, subs, comp_model, tex_descs, comp_tex) for one tile. Raises
 ValueError, always prefixed with `name`, for anything that is not a well-formed
 v3-or-later LZ4 .pmap - with 184 real tiles in one batch, an error that does not
 say WHICH file is broken is barely better than a bare traceback."""
    try:
        h, models, subs, comp_model = _parse(buf)
    except ValueError as exc:
        raise ValueError("%s: %s" % (name, exc)) from exc

    tex_count, tex_off, comp_tex_off = h[7], h[8], h[22]
    try:
        tex_descs = [TEX_DESC.unpack_from(buf, tex_off + i * TEX_DESC.size)
                     for i in range(tex_count)]
        comp_tex = [COMP.unpack_from(buf, comp_tex_off + i * COMP.size)
                    for i in range(tex_count)]
    except struct.error as exc:
        raise ValueError("%s: texture table runs past the end of the file (%s)"
                         % (name, exc)) from exc
    return h, models, subs, comp_model, tex_descs, comp_tex


def _parse_night(buf, total_nv):
    """Raw .night bytes for one tile -> the same bytes back, or None if `buf` is None
 or the wrong size for this tile's OWN vertex pool. Mirrors pmap_load_night's own
 "size-mismatch -> stays NULL, tile renders day-only" fallback (pmap.c) exactly: a
 malformed .night file is not a signal of anything at build time either - it is
 the SAME as no .night file at all, so a model's night signature (_night_signature
 below) must treat both identically rather than treating malformed-but-present as
 somehow more constraining than absent."""
    if buf is None or len(buf) != total_nv * 2:
        return None
    return buf


def _parse_nightd(buf, total_nv):
    """Raw .nightd bytes for one tile -> [(vidx, n, col), ...], or [] if `buf` is
 None or the file is malformed, or has a run count pmap_load_nightd itself would
 refuse (0 or > total_nv - see its own "absurd" comment). Same "not loaded ->
 behaves exactly like absent" mirroring as _parse_night above."""
    if buf is None:
        return []
    if len(buf) < 4 + NIGHTD_HDR.size or buf[:4] != NIGHTD_MAGIC:
        return []
    (run_count,) = NIGHTD_HDR.unpack_from(buf, 4)
    if run_count == 0 or run_count > total_nv:
        return []
    need = 4 + NIGHTD_HDR.size + run_count * PMAP_NIGHT_RUN.size
    if len(buf) != need:
        return []
    off = 4 + NIGHTD_HDR.size
    return [PMAP_NIGHT_RUN.unpack_from(buf, off + i * PMAP_NIGHT_RUN.size)
           for i in range(run_count)]


def _night_signature(night_bytes, nightd_runs, vfirst, vcount):
    """What ONE tile asserts about ONE model's night colours - a hashable value two
 tiles' assertions can be compared for exact agreement. See the module docstring's
 SHARED NIGHT AGREEMENT section for the policy this implements and the measurement
 that justifies it.

 Returns None ("this tile asserts nothing about this model's night colours - not
 a constraint on sharing, compatible with anything, including another tile's
 concrete assertion") when night_bytes is None (no usable .night; see
 _parse_night) AND no .nightd run overlaps [vfirst, vfirst+vcount).

 Otherwise returns (night_slice, clipped_runs): night_slice is the raw 2*vcount
 byte slice of night_bytes covering this model (or None if there was no usable
 .night but at least one .nightd run overlapped), and clipped_runs is a tuple of
 (lo, hi, col) - each overlapping .nightd run CLIPPED to the model's span and
 made relative to vfirst (0 <= lo < hi <= vcount), sorted by lo. This is the exact
 clip relight_model itself applies at runtime (pmap.c: `a = rs[ri].vidx > base ?
 rs[ri].vidx - base : 0; b = rs[ri].vidx + rs[ri].n - base; if (b > nv) b = nv;`),
 reproduced here so two tiles that only PARTIALLY overlap a shared run at the
 model's boundary are compared on what they'd actually apply, not on the raw
 (possibly wider) run bytes.

 Two tiles whose night_slice AND clipped_runs are both exactly equal are agreeing
 on EVERY vertex colour this model could ever be relit to; anything else - one
 concrete value differing from another, full stop - is a real disagreement,
 however small."""
    night_slice = night_bytes[vfirst * 2:(vfirst + vcount) * 2] if night_bytes is not None else None
    clipped = []
    for vidx, n, col in nightd_runs:
        lo = max(vidx, vfirst) - vfirst
        hi = min(vidx + n, vfirst + vcount) - vfirst
        if hi > lo:
            clipped.append((lo, hi, col))
    clipped.sort()
    if night_slice is None and not clipped:
        return None
    return (night_slice, tuple(clipped))


def _resolve_night_share_keys(names, model_entries):
    """model_entries: {name: [(cbytes, dsize, sig), ...]} - sig from
 _night_signature, already computed per (tile, model). Returns (resolved, stats):
 resolved is the same shape with sig replaced by share_key, an opaque value
 _dedup_blobs folds into its lookup key alongside cbytes (see that function's own
 docstring); stats is the measurement this whole mechanism exists to report - see
 the module docstring's SHARED NIGHT AGREEMENT section for the real numbers
 measured against the actual 184-tile world. Nothing in this module asserts stats
 stays near those numbers on a future rebuild - a legitimately different world
 (new tiles, re-authored night data) is allowed to move them, and a hard-coded
 check against today's values would be wrong the day that happens for a good
 reason. main()'s own printed report is where those numbers surface every build,
 for a human to notice and re-measure by hand if they move by more than a
 handful - see the module docstring's own instruction to do exactly that.

 Algorithm: group all (tile, model-index) occurrences project-wide by their
 compressed bytes - this is exactly _dedup_blobs's own grouping, done here one
 step early, over ALL models regardless of tile (a group of size 1 - this cbytes
 used exactly once anywhere - can never disagree with itself and is left alone).
 Within a cbytes group of size > 1, collect the DISTINCT concrete (non-None) sigs,
 in first-seen order:
 - 0 distinct concrete sigs (every occurrence unconstrained) or exactly 1 (every
 concrete occurrence agrees, unconstrained ones assert nothing) -> the whole
 group shares one blob, share_key=None for everyone.
 - 2+ distinct concrete sigs -> genuine disagreement: partition into one
 share_key per distinct sig (0..len-1), every concrete occurrence routed to
 its own sig's group, every unconstrained occurrence folded into group 0
 (share_key=None - the same value the 0-or-1-sig case above uses, so a reader
 does not need to know a split happened to understand what a None key means).

 This never merges two DIFFERENT concrete sigs into one blob, and it never blocks
 an unconstrained occurrence from sharing - the two guarantees the docstring
 promises. It also protects a same-tile self-conflict (one tile using the same
 compressed bytes twice, at two spans with genuinely different night colours) the
 exact same way, even though that shape never actually occurs in the measured
 184-tile world (see night_same_tile_disagree_groups in the returned stats, and
 the module docstring) - the grouping is by cbytes across ALL occurrences, tile
 identity plays no special role in it."""
    by_cbytes = {}
    for name in names:
        for idx, (cbytes, dsize, sig) in enumerate(model_entries[name]):
            by_cbytes.setdefault(cbytes, []).append((name, idx, sig))

    resolved = {name: [None] * len(model_entries[name]) for name in names}

    shared_groups = 0            # a cbytes value used more than once, anywhere
    max_sharers = 0              # most DISTINCT tiles any single shared blob touches
    night_absent_groups = 0      # shared, and not one occurrence had usable night data
    night_agree_groups = 0       # shared, and every occurrence that HAD data agreed
    night_disagree_groups = 0    # shared, and at least two occurrences disagreed
    night_same_tile_disagree_groups = 0   #...of which, disagreement within ONE tile
    night_disagree_extra_bytes = 0        # extra physical copies x len(cbytes)

    for cbytes, occ in by_cbytes.items():
        if len(occ) <= 1:
            (name, idx, sig), = occ
            resolved[name][idx] = (cbytes, model_entries[name][idx][1], None)
            continue

        shared_groups += 1
        distinct_tiles = {name for name, idx, sig in occ}
        max_sharers = max(max_sharers, len(distinct_tiles))

        concrete_order = []
        seen = set()
        for name, idx, sig in occ:
            if sig is not None and sig not in seen:
                seen.add(sig)
                concrete_order.append(sig)

        if len(concrete_order) > 1:
            night_disagree_groups += 1
            if len(distinct_tiles) == 1:
                night_same_tile_disagree_groups += 1
            night_disagree_extra_bytes += (len(concrete_order) - 1) * len(cbytes)
            sig_to_group = {s: gi for gi, s in enumerate(concrete_order)}
            for name, idx, sig in occ:
                group = sig_to_group[sig] if sig is not None else 0
                resolved[name][idx] = (cbytes, model_entries[name][idx][1],
                                       None if group == 0 else group)
        else:
            if concrete_order:
                night_agree_groups += 1
            else:
                night_absent_groups += 1
            for name, idx, sig in occ:
                resolved[name][idx] = (cbytes, model_entries[name][idx][1], None)

    stats = dict(
        night_shared_model_blob_groups=shared_groups,
        night_max_sharers=max_sharers,
        night_absent_groups=night_absent_groups,
        night_agree_groups=night_agree_groups,
        night_disagree_groups=night_disagree_groups,
        night_same_tile_disagree_groups=night_same_tile_disagree_groups,
        night_disagree_extra_bytes=night_disagree_extra_bytes,
    )
    return resolved, stats


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------
def _dedup_blobs(names, per_tile_entries, kind):
    """per_tile_entries: {name: [(cbytes, dsize, share_key), ...]}, in each tile's
 own per-model (or per-texture, per `kind`) order. Deduplicates by (cbytes,
 share_key) equality - see the module docstring's DEDUP KEY section for why
 byte equality alone is correct for both models and textures: each tile keeps its
 own MODEL table (scale, centre) untouched, so a model's blob is nothing but its
 raw vertex/index pool, with no placement baked in for two equal blobs to
 legitimately disagree about.

 share_key is an extra, arbitrary hashable value folded into the lookup key
 alongside cbytes - for textures it is always None (every texture entry
 effectively keys on cbytes alone, unchanged from before this parameter existed:
 textures have no per-vertex night data, so nothing ever needs to force two
 byte-identical texture blobs apart). For models, a non-None share_key is what
 keeps two byte-identical blobs from merging when their tiles disagree about
 night colours - see the module docstring's SHARED NIGHT AGREEMENT section and
 _resolve_night_share_keys(), the only place that ever computes one. share_key
 itself is NEVER stored in global_table or written to world.dat; it only
 influences which entries end up sharing a gid.

 Returns (global_table [(cbytes, dsize), ...] in first-seen order,
 refs {name: [gid, ...]}). Raises ValueError, naming the tile/kind/index, if the
 same (cbytes, share_key) ever show up with a different decompressed size - that
 would mean two genuinely different things hashed to the same key, and merging
 them would silently hand a future reader the wrong dsize."""
    key_to_gid = {}
    global_table = []
    refs = {}
    for name in names:
        tile_refs = []
        for i, (cbytes, dsize, share_key) in enumerate(per_tile_entries[name]):
            key = (cbytes, share_key)
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(global_table)
                key_to_gid[key] = gid
                global_table.append((cbytes, dsize))
            elif global_table[gid][1] != dsize:
                raise ValueError(
                    "%s %s %d: identical compressed bytes but a different "
                    "decompressed size (%d vs %d) - refusing to merge"
                    % (name, kind, i, dsize, global_table[gid][1]))
            tile_refs.append(gid)
        refs[name] = tile_refs
    return global_table, refs


def _compute_build_stamp(tiles):
    """u32 identity for this exact {name: bytes} set - see the module docstring's
 BUILD STAMP section for why: the same value goes into world.idx's own header and
 into every stripped tile's (repurposed) index_off field, so a stale or partial
 redeploy - one file rebuilt, another left behind, copied by hand between a PC, an
 emulator directory and a memory stick - is DETECTABLE (a mismatch) instead of
 silently resolving a global id into the wrong build's blob.

 A CRC32 over every tile's name and bytes, concatenated in SORTED-name order - never insertion/dict order, matching this format's own established
 order-independence guarantee (see test_build_store_result_is_independent_of_input_
 dict_order): two callers building from the same {name: bytes} content always get
 the same stamp regardless of what order they happened to construct the dict in.
 Folding the NAME into the hash, not just the bytes, means a same-content tile
 renamed (or the reverse: two differently-named tile sets that coincidentally share
 byte content) is still correctly treated as a different build."""
    crc = 0
    for name in sorted(tiles):
        crc = zlib.crc32(name.encode("utf-8"), crc)
        crc = zlib.crc32(b"\x00", crc)          # separator: "ab"+"c" must not hash the
                                                  # same as "a"+"bc"
        crc = zlib.crc32(tiles[name], crc)
    return crc & 0xFFFFFFFF


def build_store(tiles, night=None, nightd=None, order=TILE_ORDER):
    """tiles: {name: raw .pmap bytes}. night/nightd: optional {name: raw sidecar
 bytes} for tiles that HAVE that sidecar - omit a name (or pass night=None /
 nightd=None entirely, the default) to mean "no usable sidecar for that tile",
 which is a real, common, and SAFE case (see the module docstring's SHARED NIGHT
 AGREEMENT section), not an error; every existing caller that predates this
 parameter gets the exact same output as before, since "no night data anywhere"
 only ever WIDENS what may share, never narrows it. Returns (idx_bytes,
 dat_bytes, stats). Raises ValueError (naming the offending tile) for any
 malformed .pmap input rather than letting a struct.error/IndexError escape
 uncaught; a malformed or wrong-sized night/nightd sidecar is NOT an error here
 - see _parse_night/_parse_nightd, it degrades to "no usable data for this
 tile", mirroring pmap_load_night/pmap_load_nightd's own silent fallback."""
    names = _spatial_order(tiles.keys(), order=order)
    night = night or {}
    nightd = nightd or {}

    parsed = {name: _read_tile(name, tiles[name]) for name in names}

    model_entries = {}
    for name in names:
        h, models, subs, comp_model, _tex_descs, _comp_tex = parsed[name]
        buf = tiles[name]
        total_nv = h[13] // VERT_SZ
        night_bytes = _parse_night(night.get(name), total_nv)
        nightd_runs = _parse_nightd(nightd.get(name), total_nv)
        entries = []
        for i in range(len(models)):
            vfirst, vcount, icount = _model_spans(models, subs, i)
            dsize = vcount * VERT_SZ + icount * 2
            off, csize = comp_model[i]
            cbytes = buf[off:off + csize] if csize else b""
            sig = _night_signature(night_bytes, nightd_runs, vfirst, vcount)
            entries.append((cbytes, dsize, sig))
        model_entries[name] = entries
    model_entries, night_stats = _resolve_night_share_keys(names, model_entries)
    model_global, model_refs = _dedup_blobs(names, model_entries, "model")

    tex_entries = {}
    for name in names:
        h, models, subs, comp_model, tex_descs, comp_tex = parsed[name]
        buf = tiles[name]
        entries = []
        for ti in range(len(tex_descs)):
            tbytes, centries = tex_descs[ti][4], tex_descs[ti][7]
            dsize = tbytes + centries * 4
            off, csize = comp_tex[ti]
            cbytes = buf[off:off + csize] if csize else b""
            entries.append((cbytes, dsize, None))   # textures: share_key is always
                                                       # None, no per-vertex night data
        tex_entries[name] = entries
    tex_global, tex_refs = _dedup_blobs(names, tex_entries, "texture")

    # Per-tile refs are u16 (see REF_ID_MAX_COUNT's own comment) - fail loudly, not by
    # silently wrapping a ref, the moment either global table would no longer fit one.
    # This is a real ceiling to hit eventually (the world grows, dedup gets better and
    # merges fewer things, or both), and a wrapped id is silently-wrong geometry with
    # no error anywhere - exactly the failure class this whole review round is about.
    if len(model_global) > REF_ID_MAX_COUNT:
        raise ValueError(
            "%d unique models exceeds the u16 ref ceiling (%d) - the per-tile ref "
            "width must widen back to u32 (or the global table split) before this "
            "many can be built" % (len(model_global), REF_ID_MAX_COUNT))
    if len(tex_global) > REF_ID_MAX_COUNT:
        raise ValueError(
            "%d unique textures exceeds the u16 ref ceiling (%d) - the per-tile ref "
            "width must widen back to u32 (or the global table split) before this "
            "many can be built" % (len(tex_global), REF_ID_MAX_COUNT))

    # ---- world.dat: unique blobs back to back, 16-byte aligned, models then
    # textures, each in first-seen (== spatial traversal) order. ----
    dat = bytearray()
    model_table = []
    for cbytes, dsize in model_global:
        _pad_to_16(dat)
        off = len(dat)
        dat += cbytes
        model_table.append((off, len(cbytes), dsize))
    tex_table = []
    for cbytes, dsize in tex_global:
        _pad_to_16(dat)
        off = len(dat)
        dat += cbytes
        tex_table.append((off, len(cbytes), dsize))

    build_stamp = _compute_build_stamp(tiles)
    idx = _pack_index(names, model_table, tex_table, model_refs, tex_refs, build_stamp)
    stats = _build_stats(names, model_global, tex_global, model_refs, tex_refs)
    stats["build_stamp"] = build_stamp
    stats.update(night_stats)
    return bytes(idx), bytes(dat), stats


def _pack_index(names, model_table, tex_table, model_refs, tex_refs, build_stamp):
    model_table_off = IDX_HDR.size
    tex_table_off = model_table_off + len(model_table) * GLOBAL_ENTRY.size
    tile_dir_off = tex_table_off + len(tex_table) * GLOBAL_ENTRY.size

    names_blob = bytearray()
    name_spans = {}
    for name in names:
        nb = name.encode("utf-8")
        name_spans[name] = (len(names_blob), len(nb))
        names_blob += nb
    # Pad to a 4-byte boundary before refs_blob starts. refs_off itself must be at
    # LEAST 4-byte aligned even though the ref arrays it points into are u16s (2-byte
    # natural alignment): the base has to be a fixed, simple invariant a reader can
    # rely on regardless of what is inside, and 4-byte costs nothing extra to keep
    # (padding is at most 3 bytes either way) - a reader casting idx_bytes + refs_off
    # to a uint16_t* still gets a correctly-aligned pointer, same as it would need for
    # a uint32_t* if this were ever widened back. What is NOT guaranteed past this base
    # is that every INDIVIDUAL tile's model_refs_off/tex_refs_off also lands on a
    # 4-byte boundary: with u16 entries, a tile with an ODD model or texture count
    # advances the cursor by an odd number of u16s (an even byte count, but not
    # necessarily a multiple of 4), so only 2-byte alignment is guaranteed per tile --
    # correct and sufficient for a uint16_t* cast on MIPS, see
    # test_refs_region_is_2_byte_aligned_regardless_of_ref_widths. Nothing here
    # (Python's struct module, this process's x86 CPU) enforces or even notices EITHER
    # requirement, which is why both are arranged for explicitly: names_blob is a raw
    # concatenation of variable-length tile-name strings, so its length is NOT
    # generally a multiple of 4 (e.g. three "region_0_N.pmap" names, 15 bytes each, sum
    # to 45) - confirmed to actually misalign refs_off before this padding was added,
    # on a small fixture, even though the real 184-tile world happened to land on a
    # multiple of 4 by coincidence and would have shipped looking clean.
    while len(names_blob) % 4:
        names_blob.append(0)

    refs_blob = bytearray()
    ref_spans = {}
    for name in names:
        mrefs, trefs = model_refs[name], tex_refs[name]
        mr_off = len(refs_blob)
        refs_blob += struct.pack("<%dH" % len(mrefs), *mrefs)
        tr_off = len(refs_blob)
        refs_blob += struct.pack("<%dH" % len(trefs), *trefs)
        ref_spans[name] = (mr_off, tr_off)

    names_off = tile_dir_off + len(names) * TILE_DIR_ENTRY.size
    refs_off = names_off + len(names_blob)
    total = refs_off + len(refs_blob)

    out = bytearray(total)
    IDX_HDR.pack_into(out, 0, IDX_MAGIC, IDX_VERSION, build_stamp,
                      len(model_table), len(tex_table), len(names),
                      model_table_off, tex_table_off, tile_dir_off,
                      names_off, refs_off)
    for i, entry in enumerate(model_table):
        GLOBAL_ENTRY.pack_into(out, model_table_off + i * GLOBAL_ENTRY.size, *entry)
    for i, entry in enumerate(tex_table):
        GLOBAL_ENTRY.pack_into(out, tex_table_off + i * GLOBAL_ENTRY.size, *entry)
    for i, name in enumerate(names):
        noff, nlen = name_spans[name]
        mr_off, tr_off = ref_spans[name]
        TILE_DIR_ENTRY.pack_into(out, tile_dir_off + i * TILE_DIR_ENTRY.size,
                                 names_off + noff, nlen,
                                 len(model_refs[name]), refs_off + mr_off,
                                 len(tex_refs[name]), refs_off + tr_off)
    out[names_off:names_off + len(names_blob)] = names_blob
    out[refs_off:refs_off + len(refs_blob)] = refs_blob
    return out


def read_index(idx_bytes):
    """Parses world.idx into a plain dict: version, build_stamp (see the module
 docstring's BUILD STAMP section and verify_build_stamp()), model_table/tex_table
 (lists of (off,csize,dsize)), tile_order (emission order), tiles ({name:
 {"model_refs": [...], "tex_refs": [...]}} - global ids, read back as the u16s
 they are stored as)."""
    magic, version, build_stamp, mc, tc, nc, mt_off, tt_off, td_off, names_off, refs_off = \
        IDX_HDR.unpack_from(idx_bytes, 0)
    if magic != IDX_MAGIC:
        raise ValueError("not a world.idx (bad magic %r)" % (magic,))
    if version == 1:
        raise ValueError(
            "this world.idx is version 1 (pre-build-stamp, u32 refs) - version 2 "
            "inserted the build_stamp field right after version (header grew from "
            "40 to 44 bytes) and narrowed every ref array from u32 to u16; a version-1 "
            "file cannot be read by this version-2 reader without misparsing both the "
            "header and every ref array. Rebuild it with the current "
            "tools/world_store_build.py rather than trying to read it as-is.")
    if version != IDX_VERSION:
        raise ValueError("unsupported world.idx version %d (expected %d)"
                         % (version, IDX_VERSION))

    model_table = [GLOBAL_ENTRY.unpack_from(idx_bytes, mt_off + i * GLOBAL_ENTRY.size)
                  for i in range(mc)]
    tex_table = [GLOBAL_ENTRY.unpack_from(idx_bytes, tt_off + i * GLOBAL_ENTRY.size)
                for i in range(tc)]

    tile_order = []
    tiles = {}
    for i in range(nc):
        name_off, name_len, model_count, model_refs_off, tex_count, tex_refs_off = \
            TILE_DIR_ENTRY.unpack_from(idx_bytes, td_off + i * TILE_DIR_ENTRY.size)
        name = idx_bytes[name_off:name_off + name_len].decode("utf-8")
        mrefs = list(struct.unpack_from("<%dH" % model_count, idx_bytes, model_refs_off))
        trefs = list(struct.unpack_from("<%dH" % tex_count, idx_bytes, tex_refs_off))
        tile_order.append(name)
        tiles[name] = {"model_refs": mrefs, "tex_refs": trefs}

    return {"version": version, "build_stamp": build_stamp,
            "model_table": model_table, "tex_table": tex_table,
            "tile_order": tile_order, "tiles": tiles}


def get_model_blob(dat_bytes, index, global_id):
    off, csize, dsize = index["model_table"][global_id]
    if not csize:
        return b""
    return lz4.block.decompress(dat_bytes[off:off + csize], uncompressed_size=dsize)


def get_tex_blob(dat_bytes, index, global_id):
    off, csize, dsize = index["tex_table"][global_id]
    if not csize:
        return b""
    return lz4.block.decompress(dat_bytes[off:off + csize], uncompressed_size=dsize)


# ---------------------------------------------------------------------------
# Stripped tiles: the actual on-disk saving, not just the computed one.
# ---------------------------------------------------------------------------
# v3 header field 14 is index_off: always 0 and never read on any v3-family tile
# (v3+ replaced the raw index pool it used to point at with per-model LZ4 blobs; see
# pmap.h's own comment on it). A stripped tile has even less use for it - there is no
# pool of any kind left - so this reuses that dead space for the build stamp instead
# of growing the header. Named here so strip_tile and read_stripped_stamp cannot
# drift about which field it is.
STRIPPED_STAMP_FIELD = 14


def strip_tile(buf, model_ids, tex_ids, build_stamp):
    """Returns a stripped tile: the same resident prefix (header through the comp_tex
 table) as `buf`, byte-identical except for exactly the fields the module
 docstring's STRIPPED TILES section names, with the blob region removed entirely.
 Critically, version becomes PMAP_VERSION_STRIPPED: that is what gets this file
 refused by every load path this project has actually checked (see the module
 docstring's ★ GENERAL LESSON paragraph for the two that were NOT refusing it until
 this was found) with no change to any of them - comp_flag is set too, but only as
 a secondary marker; see STRIPPED_COMP_FLAG's own comment.

 model_ids/tex_ids: this tile's own model_refs/tex_refs - global ids, in the SAME
 order as its own model/texture tables - exactly what
 read_index(idx)["tiles"][name] already carries once build_store(tiles) has run
 (see strip_all()).

 build_stamp: the SAME u32 build_stamp read_index(idx)["build_stamp"] carries for
 the world.idx this tile is meant to be paired with - written into the tile's own
 (repurposed) index_off field so a later mismatch is detectable; see the module
 docstring's BUILD STAMP section and verify_build_stamp(). Required, not defaulted:
 a caller that forgot to pass the real stamp is exactly the bug this field exists to
 catch, so there is no silent fallback value to accidentally ship.

 Raises ValueError if the id counts do not match this tile's own tables - a
 mismatch here would silently attach the wrong tile's global ids to this one's
 models, corrupting the geometry a future reader would resolve."""
    h, models, subs, comp_model, tex_descs, comp_tex = _read_tile("<strip>", buf)
    if len(model_ids) != len(comp_model):
        raise ValueError("model id count (%d) does not match this tile's own model "
                         "table (%d)" % (len(model_ids), len(comp_model)))
    if len(tex_ids) != len(comp_tex):
        raise ValueError("texture id count (%d) does not match this tile's own "
                         "texture table (%d)" % (len(tex_ids), len(comp_tex)))

    prefix_end = h[12]   # vertex_off in the SOURCE tile: end of the resident prefix
    out = bytearray(buf[:prefix_end])

    cmoff, ctoff = h[21], h[22]
    for i, gid in enumerate(model_ids):
        _, csize = comp_model[i]
        COMP.pack_into(out, cmoff + i * COMP.size, gid, csize)
    for i, gid in enumerate(tex_ids):
        _, csize = comp_tex[i]
        COMP.pack_into(out, ctoff + i * COMP.size, gid, csize)

    new_size = len(out)
    hdr_fields = list(h)
    hdr_fields[1] = PMAP_VERSION_STRIPPED  # version - the field every load path
                                            # actually enforces to an exact set; see the
                                            # STRIPPED_COMP_FLAG comment above for why
                                            # comp_flag alone cannot do this job.
    hdr_fields[2] = new_size            # file_size
    hdr_fields[STRIPPED_STAMP_FIELD] = build_stamp  # index_off, repurposed - see its own comment
    hdr_fields[12] = new_size           # vertex_off - empty blob region, starts at EOF
    hdr_fields[20] = STRIPPED_COMP_FLAG  # comp_flag (secondary marker, see above)
    HDR.pack_into(out, 0, *hdr_fields)
    return bytes(out)


def read_stripped_stamp(buf):
    """The build stamp embedded in a stripped tile's header (the repurposed index_off
 field - see STRIPPED_STAMP_FIELD). Raises ValueError, naming the actual version
 found, if `buf` is not a stripped tile at all - reading this field on anything
 else would return a meaningless number (0, on every ordinary v3-family tile) rather
 than a real build stamp."""
    h = HDR.unpack_from(buf, 0)
    if h[1] != PMAP_VERSION_STRIPPED:
        raise ValueError("not a stripped tile (version=%d, expected %d)"
                         % (h[1], PMAP_VERSION_STRIPPED))
    return h[STRIPPED_STAMP_FIELD]


def verify_build_stamp(idx_bytes, stripped_tiles):
    """stripped_tiles: {name: bytes}. Returns (ok, [problems]): every stripped tile's
 own build stamp must equal the world.idx's - a mismatch means this tile and this
 world.idx/world.dat were built from DIFFERENT inputs (a stale or partial redeploy:
 world.dat rebuilt and one tile left behind, or the reverse), and global ids read
 through one cannot be trusted to mean what the other thinks they mean, silently,
 with no other symptom until something renders wrong. Reports every offending tile
 by name rather than stopping at the first one, matching this module's existing
 per-tile error style."""
    info = read_index(idx_bytes)
    want = info["build_stamp"]
    problems = []
    for name, buf in stripped_tiles.items():
        try:
            got = read_stripped_stamp(buf)
        except ValueError as exc:
            problems.append("%s: %s" % (name, exc))
            continue
        if got != want:
            problems.append(
                "%s: build stamp %08x does not match world.idx's %08x - this tile "
                "and this world.idx/world.dat were built from DIFFERENT inputs (a "
                "stale or partial redeploy); its global ids cannot be trusted to "
                "resolve correctly through this world.dat" % (name, got, want))
    return (not problems), problems


def strip_all(tiles, idx_bytes):
    """{name: stripped_bytes} for every tile in `tiles`, using the global ids AND the
 build stamp world.idx (as packed by build_store -> _pack_index, in `idx_bytes`)
 recorded for each one. Reads them back out with read_index() rather than accepting
 a model_refs/tex_refs dict directly - this is exactly what a caller re-opening a
 previously-built world.idx from disk would also do, and it is the only path
 exercised here (see main())."""
    info = read_index(idx_bytes)
    build_stamp = info["build_stamp"]
    out = {}
    for name in info["tile_order"]:
        refs = info["tiles"][name]
        out[name] = strip_tile(tiles[name], refs["model_refs"], refs["tex_refs"], build_stamp)
    return out


# ---------------------------------------------------------------------------
# Stats, including the decompressed-content measurement (never applied, only reported)
# ---------------------------------------------------------------------------
def _content_dedup_extra(global_table):
    """global_table: [(cbytes, dsize), ...], already unique by the ACTUAL key this
 pass uses. Groups these further by DECOMPRESSED content - what a
 decompressed-content key would additionally find. Returns (unique_by_content,
 extra_bytes_saved, collision_groups): extra_bytes_saved is ON TOP of what the
 actual key already saved - for every content-group with more than one distinct
 compressed encoding, all but the cheapest (smallest csize) member would become
 redundant under a content key."""
    by_content = {}
    for cbytes, dsize in global_table:
        raw = b"" if not cbytes else lz4.block.decompress(cbytes, uncompressed_size=dsize)
        by_content.setdefault(raw, []).append(len(cbytes))
    extra_saved = 0
    collisions = 0
    for sizes in by_content.values():
        if len(sizes) > 1:
            collisions += 1
            extra_saved += sum(sizes) - min(sizes)
    return len(by_content), extra_saved, collisions


def _build_stats(names, model_global, tex_global, model_refs, tex_refs):
    model_entries = sum(len(model_refs[n]) for n in names)
    tex_entries = sum(len(tex_refs[n]) for n in names)

    model_csize = [len(c) for c, _ in model_global]
    tex_csize = [len(c) for c, _ in tex_global]
    model_instance_bytes = sum(model_csize[gid] for n in names for gid in model_refs[n])
    tex_instance_bytes = sum(tex_csize[gid] for n in names for gid in tex_refs[n])
    model_unique_bytes = sum(model_csize)
    tex_unique_bytes = sum(tex_csize)
    model_saved = model_instance_bytes - model_unique_bytes
    tex_saved = tex_instance_bytes - tex_unique_bytes

    model_unique_by_content, model_extra_saved, model_collisions = \
        _content_dedup_extra(model_global)
    tex_unique_by_content, tex_extra_saved, tex_collisions = \
        _content_dedup_extra(tex_global)

    largest_model = max((dsize for _, dsize in model_global), default=0)
    largest_tex = max((dsize for _, dsize in tex_global), default=0)
    largest_model_c = max(model_csize, default=0)
    largest_tex_c = max(tex_csize, default=0)

    return {
        "model_entries": model_entries,
        "model_unique": len(model_global),
        "tex_entries": tex_entries,
        "tex_unique": len(tex_global),
        "bytes_saved": model_saved + tex_saved,
        "model_bytes_saved": model_saved,
        "tex_bytes_saved": tex_saved,
        "model_instance_bytes": model_instance_bytes,
        "model_unique_bytes": model_unique_bytes,
        "tex_instance_bytes": tex_instance_bytes,
        "tex_unique_bytes": tex_unique_bytes,
        "model_unique_by_content": model_unique_by_content,
        "tex_unique_by_content": tex_unique_by_content,
        "model_content_collisions": model_collisions,
        "tex_content_collisions": tex_collisions,
        "bytes_saved_extra_by_content": model_extra_saved + tex_extra_saved,
        "model_bytes_saved_extra_by_content": model_extra_saved,
        "tex_bytes_saved_extra_by_content": tex_extra_saved,
        "largest_model_dsize": largest_model,
        "largest_tex_dsize": largest_tex,
        "largest_model_csize": largest_model_c,
        "largest_tex_csize": largest_tex_c,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = None
    if "--out" in args:
        i = args.index("--out")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        out_dir = args[i + 1]
        del args[i:i + 2]
    force = "--force" in args
    if force:
        args.remove("--force")
    if len(args) != 1 or args[0].startswith("-") or not out_dir:
        if args or not out_dir:
            print("usage: world_store_build.py <in_dir> --out <out_dir> [--force]\n")
        print(__doc__)
        return 2

    in_dir = args[0]

    # SAFETY: in_dir is very often a live world with a same-shaped sibling that IS
    # its only rollback (e.g. ps2full vs ps2full_pre_lattice) - nothing on disk
    # marks either one as special, so a typo in --out is not hypothetical. Two
    # distinct real paths do not raise shutil.SameFileError (nothing is being copied
    # ONTO itself to catch); os.makedirs(out_dir, exist_ok=True) below is a silent
    # no-op on an existing directory; and the stripped-tile write loop further down
    # overwrites every region_*.pmap whose NAME came from in_dir's own listing, by
    # filename, into out_dir. Put together: `world_store_build.py ps2full --out
    # ps2full_pre_lattice` destroys the only rollback and exits 0 with a
    # normal-looking report. Refuse both ways this can happen instead of trusting
    # the operator to type correctly.
    in_real = os.path.realpath(in_dir)
    out_real = os.path.realpath(out_dir)
    if in_real == out_real:
        print("refusing: --out %r is the same directory as %r (in_dir) - this would "
             "overwrite the only source this run reads from while it is still "
             "reading it" % (out_dir, in_dir))
        return 2
    if not force and os.path.isdir(out_dir):
        existing = sorted(f for f in os.listdir(out_dir)
                          if f.startswith("region_") and f.endswith(".pmap"))
        if existing:
            print("refusing: --out %r already contains %d region_*.pmap file(s) (e.g. "
                 "%r) that this run did not create - looks like an existing world "
                 "(possibly someone's rollback copy), not a fresh output directory. "
                 "Pass --force if overwriting it is really what you want."
                 % (out_dir, len(existing), existing[0]))
            return 2

    names = sorted(f for f in os.listdir(in_dir) if f.startswith("region_") and f.endswith(".pmap"))
    if not names:
        print("no region_*.pmap in %s" % in_dir)
        return 1
    tiles = {}
    for name in names:
        with open(os.path.join(in_dir, name), "rb") as fh:
            tiles[name] = fh.read()

    #.night/.nightd feed the SHARED NIGHT AGREEMENT check inside build_store (see
    # its own docstring and the module docstring's section of the same name) - read
    # whichever sidecar exists per tile; a tile missing one (or both) simply is not
    # in this dict, which build_store reads as "no usable data for that tile", the
    # same as any other absent sidecar.
    night = {}
    nightd = {}
    for name in names:
        stem = name[:-5]                      # name always ends ".pmap"
        night_path = os.path.join(in_dir, stem + ".night")
        if os.path.isfile(night_path):
            with open(night_path, "rb") as fh:
                night[name] = fh.read()
        nightd_path = os.path.join(in_dir, stem + ".nightd")
        if os.path.isfile(nightd_path):
            with open(nightd_path, "rb") as fh:
                nightd[name] = fh.read()

    idx, dat, stats = build_store(tiles, night=night, nightd=nightd)

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "world.idx")
    dat_path = os.path.join(out_dir, "world.dat")
    with open(idx_path, "wb") as fh:
        fh.write(idx)
    with open(dat_path, "wb") as fh:
        fh.write(dat)

    # The idx/dat pair only COMPUTES the saving - the source tiles still carry every
    # blob until they are rewritten too. Strip and write every tile under the same
    # name, into the same output directory, so it holds a complete, self-sufficient
    # world on its own.
    stripped = strip_all(tiles, idx)
    stripped_total = 0
    for name in names:
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(stripped[name])
        stripped_total += len(stripped[name])

    # Self-check before declaring success: every stripped tile this run just wrote
    # must carry the SAME build stamp as the world.idx this same run just wrote. This
    # can only fail here if build_store/strip_all disagreed with themselves within
    # one process - a real bug, not the redeploy scenario the stamp mainly exists to
    # catch (see the module docstring's BUILD STAMP section) - but the whole point of
    # a self-check is to not assume that just because nothing SHOULD go wrong.
    stamp_ok, stamp_problems = verify_build_stamp(idx, stripped)
    if not stamp_ok:
        print("!!! BUILD STAMP SELF-CHECK FAILED (this should never happen - a bug in "
             "build_store()/strip_all(), not a stale redeploy, since every file here "
             "was just written by this same run):")
        for p in stamp_problems[:8]:
            print("    ", p)
        return 1

    # world.idx/world.dat/stripped tiles alone are NOT a usable alternative world --
    # every sidecar (.col,.night,.nightd,.grass,.lod,.sway,.dyn,.spin,.mflags,
    #.road,.tobj,.anim, regions.bin,...) still lives only in in_dir. They are keyed
    # by tile name and unaffected by stripping (vertex counts and order never change),
    # so a plain byte-for-byte copy is the whole job - see the module docstring's
    # SIDECAR FILES paragraph.
    sidecar_total = 0
    sidecar_count = 0
    for fname in sorted(os.listdir(in_dir)):
        src_path = os.path.join(in_dir, fname)
        if fname.endswith(".pmap") or not os.path.isfile(src_path):
            continue
        shutil.copy2(src_path, os.path.join(out_dir, fname))
        sidecar_total += os.path.getsize(src_path)
        sidecar_count += 1

    combined_pmap_src = sum(len(b) for b in tiles.values())
    store_total = len(idx) + len(dat) + stripped_total          # world.idx+dat+stripped.pmap only
    combined_dir_src = combined_pmap_src + sidecar_total         # the WHOLE source directory
    true_total = store_total + sidecar_total                     # the WHOLE output directory

    print("tiles: %d" % len(names))
    print("models   : %6d entries -> %6d unique (compressed-byte key) | "
         "content key -> %6d unique (%d collision group(s), %d extra byte(s) saveable)"
         % (stats["model_entries"], stats["model_unique"],
            stats["model_unique_by_content"], stats["model_content_collisions"],
            stats["model_bytes_saved_extra_by_content"]))
    print("textures : %6d entries -> %6d unique (compressed-byte key) | "
         "content key -> %6d unique (%d collision group(s), %d extra byte(s) saveable)"
         % (stats["tex_entries"], stats["tex_unique"],
            stats["tex_unique_by_content"], stats["tex_content_collisions"],
            stats["tex_bytes_saved_extra_by_content"]))
    print("bytes saved (duplicate blob copies avoided): %d bytes (%.2f MiB)"
         % (stats["bytes_saved"], stats["bytes_saved"] / 1048576.0))
    print("shared model blobs (by >1 model-slot): %6d | max sharers (distinct tiles): %d"
         % (stats["night_shared_model_blob_groups"], stats["night_max_sharers"]))
    print("  night colours agree across sharers : %6d  (of which no tile asserted "
         "any: %d)" % (stats["night_agree_groups"] + stats["night_absent_groups"],
                       stats["night_absent_groups"]))
    print("                              DIFFER  : %6d  (%d within a single tile) -> "
         "%d extra byte(s) NOT shared (%.2f KiB) so each side keeps its own colours"
         % (stats["night_disagree_groups"], stats["night_same_tile_disagree_groups"],
            stats["night_disagree_extra_bytes"], stats["night_disagree_extra_bytes"] / 1024.0))
    print("largest single blob: model dsize=%d csize=%d | texture dsize=%d csize=%d"
         % (stats["largest_model_dsize"], stats["largest_model_csize"],
            stats["largest_tex_dsize"], stats["largest_tex_csize"]))
    print("build stamp: %08x (written into world.idx and every stripped tile)"
         % stats["build_stamp"])
    print("wrote %s (%d bytes)" % (idx_path, len(idx)))
    print("wrote %s (%d bytes)" % (dat_path, len(dat)))
    print("wrote %d stripped tiles: %d bytes (%.2f MiB)"
         % (len(names), stripped_total, stripped_total / 1048576.0))
    print("copied %d sidecar file(s) unchanged: %d bytes (%.2f MiB)"
         % (sidecar_count, sidecar_total, sidecar_total / 1048576.0))
    print("store only (world.idx + world.dat + stripped .pmap) vs combined .pmap "
         "source: %d bytes (%.2f MiB) vs %d bytes (%.2f MiB) - %.1f%% of the .pmap "
         "portion ALONE, not the whole directory; see the total below for that"
         % (store_total, store_total / 1048576.0, combined_pmap_src,
            combined_pmap_src / 1048576.0,
            100.0 * (combined_pmap_src - store_total) / combined_pmap_src
            if combined_pmap_src else 0.0))
    print("TRUE ON-DISK TOTAL, WHOLE DIRECTORY (store + sidecars): "
         "%d bytes (%.2f MiB)  vs  %d bytes (%.2f MiB) combined ORIGINAL directory "
         "(.pmap + every sidecar) - %.2f MiB saved (%.1f%%)"
         % (true_total, true_total / 1048576.0, combined_dir_src, combined_dir_src / 1048576.0,
            (combined_dir_src - true_total) / 1048576.0,
            100.0 * (combined_dir_src - true_total) / combined_dir_src if combined_dir_src else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
