#!/usr/bin/env python3
"""Bake the SA vehicle env-map sprite (xvehicleenv128 from generic/vehicle.txd)
into data/effects/carenv.bin for the b614 vehicle env/spec additive pass
(skygfx vehiclePipe env1). Downscaled 128 -> 64 (the gradient is soft; 16KB
RGBA fits texture-friendly sizes).

carenv.bin: 'CENV' + u16 w,h + w*h*4 RGBA8888.
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_txd                        # PS2 TXD decoder (PC sa_txd_d3d9 chokes on PS2)

# INPUT: SA_ROOT points at the extracted PS2 disc (Quarry sets it); PC dev tree is the fallback.
SA_ROOT  = os.environ.get("SA_ROOT", "")
VEH_TXD  = SA_ROOT + "/models/generic/vehicle.txd"
PART_TXD = SA_ROOT + "/models/particle.txd"
# OUTPUT: argv[1] = the data dir (Quarry passes <OutDir>) -> <dir>/effects/{carenv,headlight}.bin;
# no argv keeps the dev-loop flat assets_build path.
OUT = ""


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    carenv_path = os.path.join(outdir, "effects", "carenv.bin") if outdir else OUT
    os.makedirs(os.path.dirname(carenv_path), exist_ok=True)
    texs = sa_txd.decode(open(VEH_TXD, "rb").read())
    w, h, rgba = texs["xvehicleenv128"]
    # 2x2 box downscale to 64x64
    dw, dh = w // 2, h // 2
    out = bytearray(dw * dh * 4)
    for y in range(dh):
        for x in range(dw):
            for c in range(4):
                s = (rgba[(2*y*w + 2*x)*4 + c] + rgba[(2*y*w + 2*x+1)*4 + c]
                   + rgba[((2*y+1)*w + 2*x)*4 + c] + rgba[((2*y+1)*w + 2*x+1)*4 + c])
                out[(y*dw + x)*4 + c] = s >> 2
    # b619: darken the BOTTOM band. The runtime's V axis is world-up
    # (v = N.up*0.5+0.5), so down-facing normals (undersides, arches) sample
    # the bottom of this sprite - fade it to black and the chassis stops
    for y in range(dh):
        v = y / (dh - 1.0)
        if v <= 0.55:
            continue
        k = max(0.0, 1.0 - (v - 0.55) / 0.30)   # 1 at v=0.55 -> 0 at v>=0.85
        for x in range(dw):
            p = (y*dw + x) * 4
            out[p]   = int(out[p]   * k)
            out[p+1] = int(out[p+1] * k)
            out[p+2] = int(out[p+2] * k)
    buf = b"CENV" + struct.pack("<HH", dw, dh) + bytes(out)
    open(carenv_path, "wb").write(buf)
    print("wrote %s (%dx%d, %d bytes)" % (carenv_path, dw, dh, len(buf)))

    # Same CENV container, data/effects/headlight.bin. White RGB + alpha from
    # luminance so the additive ground quad tints warm and fades at the edges.
    ptex = sa_txd.decode(open(PART_TXD, "rb").read())
    hw, hh, hrgba = ptex["headlight"]     # the SA twin-spot projection sprite
    hout = bytearray(hw * hh * 4)
    for i in range(hw * hh):
        r, g, b = hrgba[i*4], hrgba[i*4+1], hrgba[i*4+2]
        lum = (r*77 + g*150 + b*29) >> 8
        hout[i*4] = hout[i*4+1] = hout[i*4+2] = 255
        hout[i*4+3] = lum
    hbuf = b"CENV" + struct.pack("<HH", hw, hh) + bytes(hout)
    hpath = carenv_path.replace("carenv.bin", "headlight.bin")
    open(hpath, "wb").write(hbuf)
    print("wrote %s (%dx%d, %d bytes)" % (hpath, hw, hh, len(hbuf)))


if __name__ == "__main__":
    main()
