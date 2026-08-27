#!/usr/bin/env python3
"""radio_bake.py - extract the source game radio MUSIC tracks into per-track OGGs + a manifest
for the PSP port's Radio.c streaming player.

Ground truth (all from the retail PC game + the reference sources):
 * XOR key + position-keyed cipher -> CAEStreamTransformer (key mod-16 by byte position)
 * AUDIO/CONFIG/STRMPAKS.DAT (16-byte names) -> packId -> archive filename
 * AUDIO/CONFIG/TRAKLKUP.DAT (12-byte records) -> trackId -> {packId, offset, size}
 * each track on disk = [8068-byte tTrackInfo header][encrypted OGG payload of `size`]
 * gRadioMusicTracks (RadioStreamsPC.h) -> which track IDs are a station's music
 * VehicleAudioSettings (the reference sources) -> per-model default station

v1 = MUSIC ONLY (no DJ intro/outro/advert/ident). Each station's music tracks are
written as plain (decrypted) OGGs; the engine streams them with stb_vorbis file mode.

Usage:
 python tools/radio_bake.py --game "" \
 --out assets_build/radio [--stations CR,MH,CO]
"""
import argparse, os, struct, sys

KEY = bytes.fromhex("ea3ac4a19aa814f348b0d7239de8fff1")
HDR = 0x1F84  # sizeof(tTrackInfo) - skipped to reach the OGG payload

# station index -> (eRadioID, ascii display name, music track IDs). The stream-archive
# CODE (CH/CO/...) is NOT hardcoded: it is resolved per-track from TRAKLKUP+STRMPAKS so
# it is ground-truth, not a guess.
STATIONS = [
    (1,  "Playback FM",        [231,238,245,252,259,266,273,280,287,294,301,308]),
    (2,  "K-Rose",             [365,372,379,386,393,400,407,414,421,428,435,442,449,456,463]),
    (3,  "K-DST",              [513,520,527,534,541,547,554,561,564,570,577,584,591,598,605,612,619]),
    (4,  "Bounce FM",          [827,834,841,848,855,862,869,876,883,890,897,904,911,918,925,932,939]),
    (5,  "SF-UR",              [981,986,991,996,1001,1006,1011,1016,1021,1026,1031,1036,1041,1046,1051,1056]),
    (6,  "Radio Los Santos",   [1111,1118,1125,1132,1139,1146,1152,1158,1164,1169,1176,1183,1190,1195,1202,1208]),
    (7,  "Radio X",            [1259,1266,1273,1280,1287,1294,1301,1308,1315,1322,1329,1336,1343,1350,1357]),
    (8,  "CSR 103.9",          [1399,1406,1413,1420,1427,1434,1441,1448,1455,1462,1469,1476,1483]),
    (9,  "K-Jah West",         [1542,1549,1556,1563,1570,1577,1584,1591,1598,1605,1612,1619,1625,1632,1639,1645]),
    (10, "Master Sounds 98.3", [1706,1713,1720,1727,1734,1741,1748,1754,1761,1768,1775,1782,1787,1790,1796,1803,1809,1814]),
    (11, "WCTR",               [1829,1832,1835,1838,1841,1844,1847,1850,1853,1856,1859,1862,1865,1868,1871,1874,1877,1880,1883,1886,1889,1892,1895,1898,1901,1904,1907,1910,1913,1916,1919]),
]

def load_lookups(game):
    cfg = os.path.join(game, "audio", "CONFIG")
    sp = open(os.path.join(cfg, "StrmPaks.dat"), "rb").read()
    packs = [sp[i:i+16].split(b"\0")[0].decode("latin1") for i in range(0, len(sp), 16)]
    tl = open(os.path.join(cfg, "TrakLkup.dat"), "rb").read()
    lut = []
    for i in range(0, len(tl), 12):
        pid = tl[i]
        off, size = struct.unpack_from("<II", tl, i + 4)
        lut.append((pid, off, size))
    return packs, lut

def ogg_duration_samples(buf):
    """total PCM samples = granulepos of the last OGG page."""
    i = buf.rfind(b"OggS")
    if i < 0:
        return 0
    return struct.unpack_from("<Q", buf, i + 6)[0]

def extract_track(game, packs, lut, tid):
    pid, off, size = lut[tid]
    path = os.path.join(game, "audio", "streams", packs[pid])
    with open(path, "rb") as f:
        f.seek(off + HDR)
        enc = f.read(size)
    base = off + HDR
    dec = bytes(b ^ KEY[(base + i) & 15] for i, b in enumerate(enc))
    return packs[pid], dec

def parse_vehicle_defaults(reversed_root):
    """model id (400+row) -> eRadioID (or -1) from VehicleAudioSettings.h column 11."""
    import re
    p = os.path.join(reversed_root, "source", "game_sa", "Audio", "entities",
                     "AEVehicleAudioEntity.VehicleAudioSettings.h")
    if not os.path.isfile(p):
        return {}
    ids = {"RADIO_EMERGENCY_AA":0,"RADIO_CLASSIC_HIP_HOP":1,"RADIO_COUNTRY":2,"RADIO_CLASSIC_ROCK":3,
           "RADIO_DISCO_FUNK":4,"RADIO_HOUSE_CLASSICS":5,"RADIO_MODERN_HIP_HOP":6,"RADIO_MODERN_ROCK":7,
           "RADIO_NEW_JACK_SWING":8,"RADIO_REGGAE":9,"RADIO_RARE_GROOVE":10,"RADIO_TALK":11,
           "RADIO_USER_TRACKS":12,"RADIO_OFF":13}
    out, row = {}, 0
    for line in open(p, encoding="latin1"):
        s = line.strip()
        if not s.startswith("{ AE_"):
            continue
        m = re.search(r"\b(RADIO_[A-Z_]+)\b", s)
        if m:
            out[400 + row] = ids.get(m.group(1), 13)
        row += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="source game install root")
    ap.add_argument("--out", required=True, help="output dir (…/audio/radio is created)")
    ap.add_argument("--reversed", default="")
    ap.add_argument("--stations", default="all", help="comma codes/names filter, or 'all'")
    args = ap.parse_args()

    packs, lut = load_lookups(args.game)
    outroot = os.path.join(args.out, "audio", "radio")
    os.makedirs(outroot, exist_ok=True)

    want = None if args.stations == "all" else set(args.stations.upper().split(","))
    baked = []
    for sidx, (radioId, name, tracks) in enumerate(STATIONS):
        # resolve station's archive CODE from its first track (ground truth)
        code = packs[lut[tracks[0]][0]]
        if want and code.upper() not in want and name.upper() not in want:
            continue
        sdir = os.path.join(outroot, code)
        os.makedirs(sdir, exist_ok=True)
        durs, total = [], 0
        for n, tid in enumerate(tracks):
            _, dec = extract_track(args.game, packs, lut, tid)
            with open(os.path.join(sdir, "%02d.ogg" % n), "wb") as f:
                f.write(dec)
            ns = ogg_duration_samples(dec)
            durs.append(ns)
            total += len(dec)
        baked.append((code, radioId, name, durs))
        print("  %-3s %-20s %2d tracks  %6.1f MB  %5.0fs" %
              (code, name, len(tracks), total/1e6, sum(durs)/32000.0))

    # manifest: 'RADI' v1 nStations [ code[4] radioId u8 nTracks u16 durMs[u32]* ]
    # then model table: nModels u16, then u8 station per model from 400
    veh = parse_vehicle_defaults(args.reversed)
    man = bytearray()
    man += b"RADI" + struct.pack("<II", 1, len(baked))
    for code, radioId, name, durs in baked:
        c = (code.encode("latin1") + b"\0\0\0\0")[:4]
        man += c + struct.pack("<BH", radioId, len(durs))
        nm = (name.encode("latin1")[:31] + b"\0" * 32)[:32]
        man += nm
        for ns in durs:
            man += struct.pack("<I", int(ns * 1000 / 32000))  # duration ms @32kHz
    nmodels = (max(veh) - 400 + 1) if veh else 0
    man += struct.pack("<H", nmodels)
    for m in range(400, 400 + nmodels):
        man += struct.pack("<B", veh.get(m, 13) & 0xFF)   # 13 = RADIO_OFF default
    with open(os.path.join(outroot, "radio.bin"), "wb") as f:
        f.write(man)
    print("manifest: %d stations, %d model defaults, %d bytes" % (len(baked), nmodels, len(man)))

if __name__ == "__main__":
    main()
