#!/usr/bin/env python3
"""col_audit.py - collision coverage audit: chunks_small LS .col vs the whole-map backup.

User reports collision holes across the map on the LS chunk set. This audit answers
BAKE vs RUNTIME: sample a grid of points over the LS bbox and raycast ground in BOTH
.col sets (offline, exact same WCOL query the PSP runs). A point where the proven
whole-map backup finds ground but the LS set does NOT = a bake regression (data hole).
If the sets agree, the runtime (COL window) is the suspect instead.

Usage: python col_audit.py
"""
import struct, glob, os, math

SMALL = ""
BACKUP = ""
BX0, BX1, BY0, BY1 = 471.0, 3171.0, -2745.0, -495.0   # LS set extent
STEP = 30.0
CEIL = 120.0          # look for ground at/below this z

MODEL_STRIDE, FACE_STRIDE, INST_STRIDE = 36, 8, 72


class ColWorld:
    __slots__ = ("buf","n_models","n_insts","gx0","gy0","gcell","gcx","gcy",
                 "off_m","off_v","off_f","off_i","off_co","off_ci")
    def __init__(self, buf):
        H = struct.unpack_from("<17I", buf, 0)
        F = struct.unpack_from("<17f", buf, 0)
        if H[0] != 0x4C4F4357:
            raise ValueError("bad WCOL magic")
        self.buf = buf
        self.n_models, self.n_insts = H[2], H[3]
        self.gx0, self.gy0, self.gcell = F[6], F[7], F[8]
        self.gcx, self.gcy = H[9], H[10]
        self.off_m, self.off_v, self.off_f, self.off_i = H[11], H[12], H[13], H[14]
        self.off_co, self.off_ci = H[15], H[16]

    def groundz(self, px, py, ceil=CEIL):
        gx = int((px - self.gx0) / self.gcell)
        gy = int((py - self.gy0) / self.gcell)
        if gx < 0 or gy < 0 or gx >= self.gcx or gy >= self.gcy:
            return None
        cell = gy * self.gcx + gx
        cs, ce = struct.unpack_from("<2I", self.buf, self.off_co + cell * 4)
        best = None
        b = self.buf
        for k in range(cs, ce):
            (j,) = struct.unpack_from("<I", b, self.off_ci + k * 4)
            ip = self.off_i + j * INST_STRIDE
            mi = struct.unpack_from("<I", b, ip)[0]
            m = struct.unpack_from("<9f", b, ip + 4)
            pos = struct.unpack_from("<3f", b, ip + 40)
            wc = struct.unpack_from("<3f", b, ip + 52)
            wr = struct.unpack_from("<f", b, ip + 64)[0]
            dx, dy = wc[0] - px, wc[1] - py
            if dx * dx + dy * dy > (wr + 0.5) ** 2:
                continue
            mp = self.off_m + mi * MODEL_STRIDE
            vf, _vc, ff, fc = struct.unpack_from("<4I", b, mp)
            vscale = struct.unpack_from("<f", b, mp + 16)[0]
            for t in range(fc):
                fp = self.off_f + (ff + t) * FACE_STRIDE
                i0, i1, i2 = struct.unpack_from("<3H", b, fp)
                P = []
                for idx in (i0, i1, i2):
                    vx, vy, vz = struct.unpack_from("<3h", b, self.off_v + (vf + idx) * 6)
                    lx, ly, lz = vx * vscale, vy * vscale, vz * vscale
                    P.append((m[0]*lx + m[1]*ly + m[2]*lz + pos[0],
                              m[3]*lx + m[4]*ly + m[5]*lz + pos[1],
                              m[6]*lx + m[7]*ly + m[8]*lz + pos[2]))
                A, B, C = P
                d = (B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1])
                if -1e-6 < d < 1e-6:
                    continue
                l1 = ((B[1]-C[1])*(px-C[0]) + (C[0]-B[0])*(py-C[1])) / d
                l2 = ((C[1]-A[1])*(px-C[0]) + (A[0]-C[0])*(py-C[1])) / d
                l3 = 1.0 - l1 - l2
                if l1 < -0.01 or l2 < -0.01 or l3 < -0.01:
                    continue
                z = l1*A[2] + l2*B[2] + l3*C[2]
                if z <= ceil and (best is None or z > best):
                    best = z
        return best


def load_set(d):
    worlds = []
    for f in sorted(glob.glob(os.path.join(d, "region_*.col"))):
        try:
            worlds.append((os.path.basename(f), ColWorld(open(f, "rb").read())))
        except Exception as e:
            print("  %s: PARSE FAIL %s" % (os.path.basename(f), e))
    return worlds


def query(worlds, px, py):
    best = None
    for _n, w in worlds:
        z = w.groundz(px, py)
        if z is not None and (best is None or z > best):
            best = z
    return best


def main():
    small = load_set(SMALL)
    backup = load_set(BACKUP)
    print("small set: %d col tiles | backup: %d col tiles" % (len(small), len(backup)))
    nx = int((BX1 - BX0) / STEP) + 1
    ny = int((BY1 - BY0) / STEP) + 1
    total = both = only_backup = only_small = neither = 0
    regress = []
    for iy in range(ny):
        py = BY0 + iy * STEP
        for ix in range(nx):
            px = BX0 + ix * STEP
            zs = query(small, px, py)
            zb = query(backup, px, py)
            total += 1
            if zs is not None and zb is not None: both += 1
            elif zb is not None:
                only_backup += 1
                if len(regress) < 40: regress.append((px, py, zb))
            elif zs is not None: only_small += 1
            else: neither += 1
    print("points=%d  both=%d  BACKUP-only(=LS-set hole)=%d  small-only=%d  neither=%d" %
          (total, both, only_backup, only_small, neither))
    print("LS-set coverage: %.1f%% of backup's" % (100.0 * both / max(1, both + only_backup)))
    if regress:
        print("sample regression points (backup has ground, LS set does NOT):")
        for px, py, zb in regress[:20]:
            print("  (%.0f, %.0f) backup z=%.1f" % (px, py, zb))


if __name__ == "__main__":
    main()
