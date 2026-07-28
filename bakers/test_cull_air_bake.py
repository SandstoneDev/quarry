# Standalone test for cull_air_bake (run: python tools/test_cull_air_bake.py).
import os, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BAKE = os.path.join(HERE, "cull_air_bake.py")

# synthetic cull.ipl: one axis-aligned air zone (flag 0x4000), one non-air zone
# (0x40 LOAD_COLLISION), one 14-field mirror zone. Only the air zone must survive.
FIXTURE = """\
# comment
cull
0, 0, 10,  0, 5,  -1,  40, 0,  20,  16384, 0
100, 100, 3,  0, 8,  0,  4, 0,  6,  64, 0
500, 500, 2,  0, 3, -2,  3, 0,  8,  16384, 0, 1.0, 0.0, 0.0
end
"""
# fields: [0]cx [1]cy [2]cz [3]v1x [4]v1y [5]minZ [6]v2x [7]v2y [8]maxZ [9]flags [10]flags2
# air zone: v1=(0,5) v2=(40,0): corner=(0-0-40,0-5-0)=(-40,-5); edges=(0,10),(80,0)
#   corners x in [-40,40], y in [-5,5] -> AABB (-40,-5,-1, 40,5,20)

def main():
    d = tempfile.mkdtemp()
    ipl = os.path.join(d, "cull.ipl"); out = os.path.join(d, "cull_air.bin")
    open(ipl, "w").write(FIXTURE)
    r = subprocess.run([sys.executable, BAKE, ipl, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    b = open(out, "rb").read()
    magic, n = struct.unpack_from("<II", b, 0)
    assert magic == 0x414C5543, hex(magic)          # 'CULA'
    assert n == 1, n                                  # only the 0x4000 zone
    mnx,mny,mnz,mxx,mxy,mxz = struct.unpack_from("<6f", b, 8)
    assert (mnx,mny,mnz,mxx,mxy,mxz) == (-40.0,-5.0,-1.0, 40.0,5.0,20.0), (mnx,mny,mnz,mxx,mxy,mxz)
    assert len(b) == 8 + n*24, len(b)
    print("OK cull_air_bake:", n, "zone(s)")

if __name__ == "__main__":
    main()
