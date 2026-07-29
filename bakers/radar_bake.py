#!/usr/bin/env python3
"""radar_bake.py - bake the the source game radar map into a single stitched atlas (radar.bin)
for the PSP port's minimap.

SA radar (CRadar): the world map is a 12x12 grid of 128x128 tiles, each covering
500 world units, the grid spanning -3000..3000 on both axes. Tile (x,y) texture is
the TXD "radarNN" with NN = y*12 + x, in gta3.img. North = +Y = up; the tile grid's
row y=0 is the northern (+Y) edge.

We stitch all 144 tiles into one 1536x1536 atlas (tile i at cell (i%12, i//12), so
atlas u=0 is world X=-3000 / east-growing, atlas v=0 is world Y=+3000 / south-growing)
then box-downsample to 512x512 (PSP max texture size) RGBA8888. The runtime draws a
disc-fan whose UVs come from the inverse radar transform -> a clean rotating circular
minimap from one textured draw.

geo-ref: atlasU = (wx + 3000) / 6000 ; atlasV = (3000 - wy) / 6000 (both 0..1)

Also packs the real HUD radar sprites from models/hud.txd: radar_centre (player
marker) + radar_north (the N marker), 16x16 RGBA each.

radar.bin (RDR5):
 'RDR5'
 u32 atlasW, atlasH (512, 512)
 u32 spriteW, spriteH (16, 16)
 u32 nBlips (len(BLIP_ORDER))
 u32 ringW, ringH (64, 64) - radarRingPlane flight overlay
 atlasW*atlasH RGBA8888 (map atlas)
 spriteW*spriteH RGBA8888 (radar_centre)
 spriteW*spriteH RGBA8888 (radar_north)
 ringW*ringH RGBA8888 (radarRingPlane, green-recolored ground + white horizon)
 nBlips * spriteW*spriteH RGBA8888 (all radar_* blip sprites, in BLIP_ORDER)
BLIP_ORDER == the runtime BLIP_* enum in Radar.h (index = enum value). Keep in sync.
"""
import os
import struct
import sys

sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
from gvcslib import sa_img, sa_txd    # PS2-native TXD codec (Quarry: user's PS2 disc)

import numpy as np

# INPUT: SA_ROOT env (Quarry -> user's extracted PS2 disc, for hud.txd) + SA_GTA3_IMG
# (the 144 radar tiles + blip sprites live in gta3.img); PC install = dev fallback.
SA_ROOT  = os.environ.get("SA_ROOT", "")
GTA3_IMG = os.environ.get("SA_GTA3_IMG", SA_ROOT + "/MODELS/GTA3.IMG")
# OUTPUT: argv[1] dir (Quarry passes <data>/hud), else the dev assets_build tree.
OUT_DIR  = sys.argv[1] if len(sys.argv) > 1 else ""

NX, NY   = 12, 12
TILE     = 128            # native tile resolution
ATLAS    = 512           # final atlas (PSP max texture); 1536 stitched -> /3 box

# All hud.txd radar_* blip sprites (16x16), in a FIXED order == the BLIP_* enum in
# Radar.h. The runtime indexes blips by this position, so NEVER reorder - only append.
BLIP_ORDER = [
    "centre", "north", "waypoint", "cj", "flag", "qmark", "cash", "light", "fire",
    "enemyattack", "crash1", "race", "impound",
    # services (static map icons)
    "hostpital", "police", "savegame", "ammugun", "spray", "gym", "barbers", "tattoo",
    "modgarage", "tshirt", "school", "burgershot", "chicken", "pizza", "diner",
    "girlfriend", "datefood", "datedrink", "datedisco",
    # missions / characters
    "bigsmoke", "ryder", "sweet", "ogloc", "maddog", "thetruth", "cesarviapando",
    "catalinapink", "toreno", "torenoranch", "woozie", "zero", "emmetgun", "mcstrap",
    "locosyndicate",
    # gangs / properties / places
    "gangb", "gangg", "gangn", "gangp", "gangy", "triads", "triadscasino",
    "mafiacasino", "propertyg", "propertyr", "airyard", "boatyard", "runway",
    "bulldozer", "truck",
]


def box_downsample(a, out_w, out_h):
    """a: (H,W,4) uint8 -> (out_h,out_w,4) uint8 by integer-factor block average."""
    h, w = a.shape[:2]
    fy, fx = h // out_h, w // out_w
    a = a[:out_h*fy, :out_w*fx].astype(np.uint32)
    a = a.reshape(out_h, fy, out_w, fx, 4).sum(axis=(1, 3)) // (fx * fy)
    return a.astype(np.uint8)


def main():
    im = sa_img.SaImg(GTA3_IMG)
    have = {n.lower() for n in im.names()}

    stitched = np.zeros((NY*TILE, NX*TILE, 4), dtype=np.uint8)
    ok = 0
    for nn in range(NX*NY):
        name = "radar%02d.txd" % nn            # IMG: radar00..radar143 (min 2-digit pad)
        if name.lower() not in have:
            continue
        d = sa_txd.decode(im.extract(name))
        if not d:
            continue
        w, h, rgba = next(iter(d.values()))
        tile = np.frombuffer(rgba, np.uint8).reshape(h, w, 4)
        if (h, w) != (TILE, TILE):
            tile = box_downsample(tile, TILE, TILE) if (h > TILE) else tile
        tx, ty = nn % NX, nn // NX
        stitched[ty*TILE:(ty+1)*TILE, tx*TILE:(tx+1)*TILE] = tile[:TILE, :TILE]
        ok += 1

    atlas = box_downsample(stitched, ATLAS, ATLAS)
    atlas[:, :, 3] = 255                        # force opaque

    # real HUD radar sprites (player + north marker) from models/hud.txd
    SPR = 16
    hud = sa_txd.decode(open(SA_ROOT + "/models/hud.txd", "rb").read())
    def sprite(name):
        w, h, rgba = hud[name]
        a = np.frombuffer(rgba, np.uint8).reshape(h, w, 4)
        if (h, w) != (SPR, SPR):
            a = box_downsample(a, SPR, SPR)
        return a.tobytes()
    centre = sprite("radar_centre")
    north  = sprite("radar_north")

    # b476: radarRingPlane (64x64) - the SA in-plane artificial-horizon overlay: a translucent
    # ground half-disc + a white horizon bar with an upward nose-notch, drawn rotated by the plane's
    # dark green (keeping the white line white), so our flight radar gets the REAL SA line shape.
    RINGSZ = 64
    rw, rh, rrgba = hud["radarringplane"]
    ring = np.frombuffer(rrgba, np.uint8).reshape(rh, rw, 4).copy()
    if (rh, rw) != (RINGSZ, RINGSZ):
        ring = box_downsample(ring, RINGSZ, RINGSZ)
    _lum = ring[:, :, :3].max(axis=2)
    _grd = (ring[:, :, 3] > 0) & (_lum < 180)        # dark ground pixels (the horizon line is ~white)
    ring[_grd, 0] = 0x20; ring[_grd, 1] = 0x66; ring[_grd, 2] = 0x20   # dark green
    ring_bytes = ring.tobytes()

    # all radar_* blip sprites, in BLIP_ORDER (== the runtime BLIP_* enum)
    blips = bytearray()
    missing = 0
    for nm in BLIP_ORDER:
        key = "radar_" + nm
        if key in hud:
            blips += sprite(key)
        else:
            blips += bytes(SPR * SPR * 4)          # missing -> transparent
            missing += 1
            print("  ! missing blip sprite:", key)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "radar.bin")
    with open(out, "wb") as f:
        f.write(b"RDR5")   # b476: +radarRingPlane block (ringW,ringH in header, block after north)
        f.write(struct.pack("<7I", ATLAS, ATLAS, SPR, SPR, len(BLIP_ORDER), RINGSZ, RINGSZ))
        f.write(atlas.tobytes())
        f.write(centre)
        f.write(north)
        f.write(ring_bytes)
        f.write(bytes(blips))
    print("radar.bin: %d/%d tiles -> %dx%d atlas + centre/north + %d blips (%d missing) %dpx (%.2f MB) -> %s"
          % (ok, NX*NY, ATLAS, ATLAS, len(BLIP_ORDER), missing, SPR, os.path.getsize(out)/1e6, out))


if __name__ == "__main__":
    main()
