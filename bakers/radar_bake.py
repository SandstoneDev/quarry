#!/usr/bin/env python3
"""radar_bake.py - bake the source game radar map into per-tile 4-bit palettized textures
(radar.bin) for the PSP port's minimap, at native resolution.

SA radar (CRadar): the world map is a 12x12 grid of 128x128 tiles, each covering
500 world units, the grid spanning -3000..3000 on both axes. Tile (x,y) texture is
the TXD "radarNN" with NN = y*12 + x, in gta3.img. North = +Y = up; the tile grid's
row y=0 is the northern (+Y) edge.

Each tile is packed alone (tools/radar_palette.py: pack_tile), NOT stitched into one
atlas and squeezed down to the PSP's 512-texel texture limit. Measured across all 144
tiles, colours per tile are min 1 / median 14 / max 15, so every tile fits an exact
4-bit palette at its native 128x128 - no downsample, no lost detail. The runtime draws
a disc-fan per visible tile whose UVs come from the inverse radar transform -> a clean
rotating circular minimap built from several small textured draws instead of one big one.

geo-ref: tile (tx,ty) covers world X in [-3000+tx*500, -3000+(tx+1)*500],
 world Y in [3000-(ty+1)*500, 3000-ty*500]

Also packs the real HUD radar sprites from models/hud.txd: radar_centre (player
marker) + radar_north (the N marker) at 16x16, and radardisc (the minimap outline
the game itself uses) at 32x32.

radar.bin (RDR6):
 'RDR6'
 u32 nTiles, tileW, tileH, gridW, gridH (144, 128, 128, 12, 12)
 u32 spriteW, nBlips, discW (16, len(BLIP_ORDER), 32)
 u32 ringW, ringH, reserved[5] (64, 64, 0,0,0,0,0)
 per tile, in row-major (tx,ty) grid order, nTiles records:
 u16 nColours (0 if the tile is absent from the disc)
 nColours * RGBA8888 (palette, only if nColours > 0)
 (tileW/2)*tileH bytes (4-bit indices, 2px/byte, low nibble = even x; only if nColours > 0)
 spriteW*spriteW RGBA8888 (radar_centre)
 spriteW*spriteW RGBA8888 (radar_north)
 discW*discW RGBA8888 (radardisc)
 ringW*ringH RGBA8888 (radarRingPlane, green-recolored ground + white horizon)
 nBlips * spriteW*spriteW RGBA8888 (all radar_* blip sprites, in BLIP_ORDER)
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

    # Native tiles, one palette each. No stitching, no downsample: the old path squeezed
    # 1536x1536 into 512x512 because that is the PSP's texture limit, and a third of the
    # detail went with it. Tiles stay separate, so each keeps its own size and palette.
    import radar_palette
    tiles = []                       # (clut, packed) per grid cell, None where absent
    ok = 0
    for nn in range(NX*NY):
        name = "radar%02d.txd" % nn          # IMG: radar00..radar143 (min 2-digit pad)
        if name.lower() not in have:
            tiles.append(None); continue
        d = sa_txd.decode(im.extract(name))
        if not d:
            tiles.append(None); continue
        w, h, rgba = next(iter(d.values()))
        tile = np.frombuffer(rgba, np.uint8).reshape(h, w, 4).copy()
        if (h, w) != (TILE, TILE):
            tile = box_downsample(tile, TILE, TILE)
        tile[:, :, 3] = 255                  # opaque, as the atlas was
        tiles.append(radar_palette.pack_tile(tile[:TILE, :TILE]))
        ok += 1

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

    # The minimap outline the game itself uses. It is a QUARTER arc, white with alpha,
    # which the original mirrors across both axes to close the ring - which is why our
    # procedural five-pixel ring never matched it.
    DISC = 32
    dw, dh, drgba = hud["radardisc"]
    disc = np.frombuffer(drgba, np.uint8).reshape(dh, dw, 4)
    if (dh, dw) != (DISC, DISC):
        disc = box_downsample(disc, DISC, DISC)
    disc_bytes = disc.tobytes()

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
        f.write(b"RDR6")     # tiles at native resolution, 4-bit with a palette each
        f.write(struct.pack("<8I", len(tiles), TILE, TILE, NX, NY,
                            SPR, len(BLIP_ORDER), DISC))
        f.write(struct.pack("<7I", RINGSZ, RINGSZ, 0, 0, 0, 0, 0))   # ring + reserve
        for t in tiles:
            if t is None:
                f.write(struct.pack("<H", 0))          # absent tile: no palette, no pixels
                continue
            clut, packed = t
            f.write(struct.pack("<H", len(clut)))
            f.write(clut.tobytes())
            f.write(packed.tobytes())
        f.write(centre)
        f.write(north)
        f.write(disc_bytes)
        f.write(ring_bytes)
        f.write(bytes(blips))
    print("radar.bin RDR6: %d/%d tiles at %dx%d native + centre/north/disc + %d blips (%d missing) (%.2f MB) -> %s"
          % (ok, NX*NY, TILE, TILE, len(BLIP_ORDER), missing, os.path.getsize(out)/1e6, out))


if __name__ == "__main__":
    main()
