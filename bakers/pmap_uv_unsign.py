#!/usr/bin/env python3
"""pmap_uv_unsign - move every submesh's UVs into the GE's UNSIGNED u16 window.

WHY (striped-textures root cause, research/striped_textures_rootcause_and_fix.md):
the GE decodes 16-bit vertex texcoords as UNSIGNED u1.15 (Sony GE-UM 6.1/6.5,
GE-CR p13) - there is no signed path. With the global sceGuTexScale(8,8) the
sampling window is [0,16) tiles, wrapping mod 16. Our bakers packed SIGNED
s16 = round(uv*4096) centred around 0, so any triangle whose UV range crosses 0
interpolates the long way through the window: ~(16-span) reversed repeats.
Small span -> dense "compressed repeated" stripes; big span -> "stretched"
mirrored patch. This tool shifts each submesh by an integer tile count
(GU_REPEAT-invariant, appearance-neutral) so every raw UV lands in [0, 65536)
as an unsigned 16-bit value. Nothing but the UV words changes.

Usage:
  python pmap_uv_unsign.py <file-or-dir> [more ...] [--backup DIR] [--dry]

v2 files are patched directly; v3 (LZ4) are decompressed, patched, recompressed
via pmap_lz4_decompress.py / pmap_lz4.py (same folder). A submesh whose UV span
exceeds the 16-tile window (corrupt source) falls back to per-vertex fract-wrap
(identical under GU_REPEAT, matches geom.py _sanitize_uv policy) and is logged.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
UV_ONE = 4096                     # raw units per tile (baker s16 = uv*4096)
WINDOW = 16 * UV_ONE              # 65536 = the GE u16 window at TexScale 8
SPAN_OK = WINDOW - UV_ONE // 4    # 15.75 tiles: max span accepted in ANY domain
                                  # (real data maxes at ~12.6; keeps the tool
                                  # idempotent - output spans stay <= SPAN_OK)

VDT = np.dtype([('u', '<i2'), ('v', '<i2'), ('c', '<u2'),
                ('x', '<i2'), ('y', '<i2'), ('z', '<i2')])


def _read_header(data):
    fields = struct.unpack_from('<20I', data, 0)
    keys = ('magic', 'version', 'file_size', 'model_count', 'model_off',
            'submesh_count', 'submesh_off', 'texture_count', 'texture_off',
            'instance_count', 'instance_off', 'grid_off',
            'vertex_off', 'vertex_bytes', 'index_off', 'index_bytes',
            'texel_off', 'texel_bytes', 'clut_off', 'clut_bytes')
    return dict(zip(keys, fields))


def unsign_v2(data):
    """Patch a v2 pmap bytearray in place. Returns stats dict."""
    h = _read_header(data)
    assert h['magic'] == 0x50414D50 and h['version'] == 2
    subs = [struct.unpack_from('<i4I', data, h['submesh_off'] + 20 * i)
            for i in range(h['submesh_count'])]

    # submesh vertex slices must not overlap (each shifts independently)
    ranges = sorted((s[1], s[1] + s[2]) for s in subs if s[2])
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert a1 <= b0, "overlapping vertex slices %r %r" % ((a0, a1), (b0, b1))

    st = dict(subs_shifted=0, neg_before=0, wrapped=0)
    vo = h['vertex_off']
    for tex, vfirst, vcount, ifirst, icount in subs:
        if vcount == 0:
            continue
        off = vo + vfirst * 12
        v = np.frombuffer(data, dtype=VDT, count=vcount, offset=off).copy()
        u_in = v['u'].astype(np.int32)          # signed-baker view
        w_in = v['v'].astype(np.int32)
        st['neg_before'] += int(((u_in < 0) | (w_in < 0)).sum())
        tri = None
        if icount >= 3:
            idx = np.frombuffer(data, dtype='<u2', count=icount,
                                offset=h['index_off'] + ifirst * 2)
            if idx.max(initial=0) < vcount:
                tri = idx.reshape(-1, 3)
        out = []
        for comp in (u_in, w_in):
            cu = comp & 0xFFFF                  # the GE's unsigned view
            # Domain pick: the AUTHOR's domain is the one where triangles are
            # locally small; in the wrong domain a seam-straddling triangle
            # spans ~(16 - real) tiles. Same rule makes reruns no-ops (the
            # patched file is coherent in unsigned, garbage in signed).
            if tri is not None:
                span_tri_s = int((comp[tri].max(1) - comp[tri].min(1)).max())
                span_tri_u = int((cu[tri].max(1) - cu[tri].min(1)).max())
            else:                               # no indices: whole-set spans
                span_tri_s = int(comp.max() - comp.min())
                span_tri_u = int(cu.max() - cu.min())
            base = cu if span_tri_u <= span_tri_s else comp
            span = int(base.max() - base.min())
            res = None
            if span <= SPAN_OK:
                # integer-tile min-floor shift into [0, span] (REPEAT-invariant)
                res = base - int(np.floor(base.min() / float(UV_ONE))) * UV_ONE
                assert int(res.max() - res.min()) == span        # shift-exact
                if tri is not None:
                    # non-dominance guard: if the s16 READING of the result
                    # still has smaller triangle spans somewhere, the source
                    # mixes both conventions inside one submesh (junk content;
                    # seen on tiny props) - no single shift can win. Fall
                    # through to fract-wrap so the tool stays idempotent.
                    res_s = np.where(res > 32767, res - 65536, res)
                    if int((res_s[tri].max(1) - res_s[tri].min(1)).max()) < \
                       int((res[tri].max(1) - res[tri].min(1)).max()):
                        res = None
            if res is None:
                # incoherent (corrupt/mixed source): per-vertex fract-wrap,
                # identical under GU_REPEAT (geom._sanitize policy)
                res = np.mod(base, UV_ONE)
                st['wrapped'] += 1
            assert res.min() >= 0 and res.max() < WINDOW, (res.min(), res.max())
            out.append(res)

        packed = np.empty(vcount * 2, np.uint16)
        packed[0::2] = out[0].astype(np.uint16)
        packed[1::2] = out[1].astype(np.uint16)
        raw = np.frombuffer(data, dtype=np.uint8, count=vcount * 12, offset=off)
        raw = raw.reshape(-1, 12)
        new_uv = packed.view(np.uint8).reshape(-1, 4)
        if not np.array_equal(raw[:, 0:4], new_uv):
            raw[:, 0:4] = new_uv
            st['subs_shifted'] += 1
    return st


def verify_v2(data):
    """Post-check: the GE's unsigned view must now be the BEST view - for
    every submesh component, the max per-triangle span in the unsigned domain
    is <= the max span in the signed reading, and the whole submesh sits
    inside one [0, SPAN_OK] stretch (no wrap crossing during interpolation;
    wrap applies per-pixel AFTER interpolation). Returns violation count."""
    h = _read_header(data)
    subs = [struct.unpack_from('<i4I', data, h['submesh_off'] + 20 * i)
            for i in range(h['submesh_count'])]
    bad = 0
    for tex, vfirst, vcount, ifirst, icount in subs:
        if vcount == 0:
            continue
        v = np.frombuffer(data, dtype=VDT, count=vcount,
                          offset=h['vertex_off'] + vfirst * 12)
        s_u = v['u'].astype(np.int32); s_w = v['v'].astype(np.int32)
        tri = None
        if icount >= 3:
            idx = np.frombuffer(data, dtype='<u2', count=icount,
                                offset=h['index_off'] + ifirst * 2)
            if idx.max(initial=0) < vcount:
                tri = idx.reshape(-1, 3)
        for comp in (s_u, s_w):
            cu = comp & 0xFFFF
            if int(cu.max() - cu.min()) > SPAN_OK:
                bad += 1
                continue
            if tri is not None:
                if int((cu[tri].max(1) - cu[tri].min(1)).max()) > \
                   int((comp[tri].max(1) - comp[tri].min(1)).max()):
                    bad += 1
    return bad


def process(path, backup_dir=None, dry=False):
    blob = open(path, 'rb').read()
    ver = struct.unpack_from('<I', blob, 4)[0]
    name = os.path.basename(path)
    if ver == 2:
        data = bytearray(blob)
        st = unsign_v2(data)
        out_bytes = bytes(data)
    elif ver == 3:
        tmp = tempfile.mkdtemp(prefix='uvunsign_')
        v2p = os.path.join(tmp, 'in.v2.pmap')
        v3p = os.path.join(tmp, 'out.v3.pmap')
        subprocess.check_call([sys.executable,
                               os.path.join(TOOLS, 'pmap_lz4_decompress.py'),
                               path, v2p], stdout=subprocess.DEVNULL)
        data = bytearray(open(v2p, 'rb').read())
        st = unsign_v2(data)
        if st['subs_shifted'] == 0:
            out_bytes = blob                 # untouched: keep original v3 bytes
        else:
            open(v2p, 'wb').write(bytes(data))
            subprocess.check_call([sys.executable,
                                   os.path.join(TOOLS, 'pmap_lz4.py'),
                                   v2p, v3p], stdout=subprocess.DEVNULL)
            out_bytes = open(v3p, 'rb').read()
        for f in (v2p, v3p):
            try: os.remove(f)
            except OSError: pass
        try: os.rmdir(tmp)
        except OSError: pass
    else:
        print(f"  {name}: version {ver} unsupported - skip")
        return None
    bad = verify_v2(data)
    tag = "DRY " if dry else ""
    print(f"  {tag}{name}: subs_shifted={st['subs_shifted']} "
          f"neg_before={st['neg_before']} wrapped={st['wrapped']} "
          f"verify_bad={bad}")
    assert bad == 0, f"{name}: verify failed"
    if dry or st['subs_shifted'] == 0:
        return st
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        bp = os.path.join(backup_dir, name)
        if not os.path.exists(bp):
            open(bp, 'wb').write(blob)
    open(path, 'wb').write(out_bytes)
    return st


def main():
    argv = sys.argv[1:]
    backup = None
    dry = "--dry" in argv
    if "--backup" in argv:
        k = argv.index("--backup")
        backup = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    argv = [a for a in argv if a != "--dry"]
    files = []
    for a in argv:
        if os.path.isdir(a):
            files += sorted(
                os.path.join(a, f) for f in os.listdir(a)
                if f.lower().endswith('.pmap'))
        else:
            files.append(a)
    total = dict(files=0, subs=0, wrapped=0)
    for p in files:
        st = process(p, backup, dry)
        if st and st['subs_shifted']:
            total['files'] += 1
            total['subs'] += st['subs_shifted']
            total['wrapped'] += st['wrapped']
    print(f"DONE: {total['files']}/{len(files)} files changed, "
          f"{total['subs']} submeshes shifted, {total['wrapped']} fract-wrapped")


if __name__ == "__main__":
    main()
