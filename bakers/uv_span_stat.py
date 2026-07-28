#!/usr/bin/env python3
"""uv_span_stat - report the per-triangle UV span distribution of baked region tiles.

This is the measurement that identified the stretched-texture regression, and the
one that verifies the fix. The GE decodes a 16-bit texcoord as UNSIGNED u1.15, so
with the engine's sceGuTexScale(8,8) the sampling window is [0,16) tiles. A
triangle whose UV extent approaches that window repeats the texture many times
across itself at 16-bit precision, which reads as a stretched surface.

The known-good PC-derived world caps every triangle at 4.00 tiles (geom.py
UV_EDGE_MAX). A PS2 bake that has not been through ps2_uv_tess reaches ~14. So:

  python uv_span_stat.py <dir-or-pmap> [more ...] [--sample N] [--label NAME]

prints tris / median / p99 / max and the share of triangles above 2, 4, 8 and 15
tiles. Point it at two world sets to compare them, e.g. a reference set and a
fresh bake. v3 (lz4) tiles are decompressed in a temp dir; nothing is modified.
"""
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
UV_ONE = 4096
VFMT = "<HHHhhh"          # u,v raw 16-bit; colour u16; x,y,z s16
HDR_KEYS = ('magic', 'version', 'file_size', 'model_count', 'model_off',
            'submesh_count', 'submesh_off', 'texture_count', 'texture_off',
            'instance_count', 'instance_off', 'grid_off',
            'vertex_off', 'vertex_bytes', 'index_off', 'index_bytes',
            'texel_off', 'texel_bytes', 'clut_off', 'clut_bytes')


def load_v2(path):
    """Return the uncompressed pmap bytes (decompressing a v3 tile if needed)."""
    blob = open(path, 'rb').read()
    ver = struct.unpack_from('<I', blob, 4)[0]
    if ver == 2:
        return blob
    if ver == 3:
        tmp = tempfile.mkdtemp(prefix='uvspan_')
        v2p = os.path.join(tmp, 'in.v2.pmap')
        subprocess.check_call([sys.executable,
                               os.path.join(TOOLS, 'pmap_lz4_decompress.py'),
                               path, v2p], stdout=subprocess.DEVNULL)
        data = open(v2p, 'rb').read()
        os.remove(v2p)
        os.rmdir(tmp)
        return data
    raise SystemExit("unsupported pmap version %d in %s" % (ver, path))


def tri_spans(path):
    """Every triangle's max(U extent, V extent) in tiles, as the GE sees it."""
    data = load_v2(path)
    h = dict(zip(HDR_KEYS, struct.unpack_from('<20I', data, 0)))
    out = []
    for i in range(h['submesh_count']):
        _tex, vfirst, vcount, ifirst, icount = struct.unpack_from(
            '<i4I', data, h['submesh_off'] + 20 * i)
        if vcount == 0 or icount < 3:
            continue
        verts = list(struct.iter_unpack(
            VFMT, data[h['vertex_off'] + vfirst * 12:
                       h['vertex_off'] + (vfirst + vcount) * 12]))
        idx = struct.unpack_from('<%dH' % icount, data, h['index_off'] + ifirst * 2)
        if max(idx) >= vcount:
            continue
        for t in range(icount // 3):
            a, b, c = idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]
            us = (verts[a][0], verts[b][0], verts[c][0])
            vs = (verts[a][1], verts[b][1], verts[c][1])
            out.append(max(max(us) - min(us), max(vs) - min(vs)) / float(UV_ONE))
    return out


def pct(spans, above):
    return 100.0 * sum(1 for s in spans if s > above) / len(spans)


def report(label, spans):
    if not spans:
        print("%-22s no triangles" % label)
        return
    s = sorted(spans)
    n = len(s)
    median = s[n // 2]
    p99 = s[min(n - 1, int(n * 0.99))]
    print("%-22s tris=%-9d median=%.3f p99=%.2f max=%.2f | "
          ">2t=%.3f%% >4t=%.3f%% >8t=%.4f%% >15t=%.4f%%"
          % (label, n, median, p99, s[-1],
             pct(s, 2), pct(s, 4), pct(s, 8), pct(s, 15)))


def collect(target, sample):
    if os.path.isdir(target):
        files = sorted(f for f in os.listdir(target) if f.endswith('.pmap'))
        if sample and len(files) > sample:
            files = files[::max(1, len(files) // sample)][:sample]
        paths = [os.path.join(target, f) for f in files]
    else:
        paths = [target]
    spans = []
    for p in paths:
        spans += tri_spans(p)
    return spans, len(paths)


def main():
    argv = sys.argv[1:]
    sample = 8
    label = None
    if "--sample" in argv:
        k = argv.index("--sample")
        sample = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    if "--label" in argv:
        k = argv.index("--label")
        label = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    if not argv:
        raise SystemExit(__doc__)
    for target in argv:
        spans, nfiles = collect(target, sample)
        name = label or os.path.basename(os.path.normpath(target))
        report("%s (%d)" % (name, nfiles), spans)


if __name__ == "__main__":
    main()
