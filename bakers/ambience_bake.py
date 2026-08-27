#!/usr/bin/env python3
"""ambience_bake - SA venue ambience -> data/amb/ for the PSP port.

SA plays STREAMED ambience tracks per AUDIO ZONE (data/maps/Audiozon.ipl 'auzo'
rows; CAEAmbienceTrackManager::UpdateAmbienceTrackAndVolume maps zoneId->track,
0x4D6E60). Most zones are venue interiors (bar chatter, restaurant hum, ammunation) - exactly what the port's interiors lack.

This bake:
 1) parses Audiozon.ipl auzo rows (name, zoneId),
 2) maps zoneId -> stream track via the 0x4D6E60 switch (radio zones skipped),
 3) extracts each unique track from AUDIO/STREAMS via stream_extract,
 4) transcodes to mono 22050 Hz OGG (ffmpeg; ambience quality, small files),
 5) writes data/amb/amb_t<track>.ogg + ambzones.bin v2 (see AMBZONES FORMAT below).

AMBZONES FORMAT v2. v1 carried only { name, track } pairs, which is enough for the
interior path (the runtime matches its own interior name against the zone names) and
useless for everything else: SA picks a zone by asking which auzo VOLUME contains the
player, and v1 threw the geometry away. 117 of the 149 auzo rows are interiors, but
the other 32 are outdoors - the Santa Maria beach sphere, the dam, Area 69, Toreno's
ranch - and not one of them could ever fire without the boxes and spheres.

 char magic[4] 'AMBZ'
 u16 version 2
 u16 nNamed, nBoxes, nSpheres
 nNamed x { char name[16]; u16 track } 18 B
 nBoxes x { char name[8]; float min[3], max[3]; u16 track; u8 active; u8 volMode } 36 B
 nSpheres x { char name[8]; float centre[3], radius; u16 track; u8 active; u8 volMode } 28 B

The volumes carry their NAME because the game switches zones on and off by it:
main.scm has 17 SWITCH_AUDIO_ZONE (0x0917) calls, and they name exactly the four
zones that ship inactive - BEACH, AWARDS, LOWRIDE, MADDOGL. Those four are mission
ambiences, switched on for a scene and off after it, which is why they are the ones
the disc leaves off.

`track` is NO_TRACK for a zone the game has no ambience for. Those rows are kept
rather than dropped on purpose: SA takes the FIRST active volume containing the
player and plays silence if that one has no track, so dropping them would let a
later, tracked zone win where the original stays quiet.

`active` is the auzo row's third field, which is a FLAG and not a shape selector
despite what the column looks like (CFileLoader::LoadAudioZone: 9 fields = box,
7 = sphere, and `flags == 1` is what makes the zone live). Four rows ship inactive
and are switched on by script.
"""
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_extract
# NB: imageio_ffmpeg is imported lazily inside main so a missing ffmpeg SOFT-SKIPS the
# audio transcode instead of crashing at import (the ambzones map is still written).

# SA_ROOT env override: Quarry points this at the extracted disc. Default = PC dev loop.
# Windows' case-insensitive fs maps '/data/maps' onto the PS2 disc's 'DATA/MAPS'.
SA_ROOT = os.environ.get("SA_ROOT", "")
AUZO = SA_ROOT + "/data/maps/Audiozon.ipl"
DEFAULT_OUTS = [
    "",
    "",
    "",
]

# ambzones.bin v2
AMBZ_VERSION = 3
NO_TRACK = 0xFFFF

# Volume rules, out of the same switch that gives the tracks. Nearly every zone is a
# flat -6 dB; exactly three behave differently, so they are enumerated rather than
# inferred from the zone's shape (zone 13 is a sphere too, and IS flat).
VOL_FLAT = 0        # -6 dB
VOL_BY_DIST = 1     # GetDistanceAttenuation(|zoneCentre - camera| * 0.2) - 6
VOL_BY_CAMZ = 2     # camZ <= 1372 ? -6 - 9*(1372 - camZ): -6
ZONE_VOLMODE = {5: VOL_BY_DIST, 10: VOL_BY_DIST, 4: VOL_BY_CAMZ}

# CAEAmbienceTrackManager::UpdateAmbienceTrackAndVolume 0x4D6E60: zoneId -> track.
# Radio-station zones (30, 52..62) are intentionally absent.
ZONE_TRACK = {
    4: 143, 5: 140, 8: 165, 10: 139, 12: 168, 13: 157, 15: 164, 17: 146,
    19: 138, 20: 136, 21: 135, 23: 148, 24: 159, 25: 158, 26: 154,
    28: 147, 29: 147, 34: 162, 36: 155, 37: 144, 39: 163, 41: 169,
    44: 152, 48: 137, 50: 173, 51: 156, 64: 151, 66: 170, 67: 171,
}


class AuZone(object):
    """One auzo volume. `shape` is 'box' or 'sphere'; `geom` is (min, max) or
 (centre, radius)."""
    __slots__ = ("name", "zone_id", "active", "shape", "geom")

    def __init__(self, name, zone_id, active, shape, geom):
        self.name, self.zone_id, self.active = name, zone_id, active
        self.shape, self.geom = shape, geom

    @property
    def track(self):
        return ZONE_TRACK.get(self.zone_id, NO_TRACK)

    @property
    def vol_mode(self):
        return ZONE_VOLMODE.get(self.zone_id, VOL_FLAT)

    def min_z(self):
        if self.shape == "sphere":
            return self.geom[0][2]
        return min(self.geom[0][2], self.geom[1][2])


def parse_auzo():
    """-> [AuZone, ...] in file order, which is the order SA searches them in."""
    rows = []
    inz = False
    for ln in open(AUZO, encoding="latin-1"):
        s = ln.strip()
        if s == "auzo": inz = True; continue
        if s == "end": inz = False; continue
        if not inz or not s or s.startswith("#"): continue
        parts = [t.strip() for t in s.split(",")]
        if len(parts) < 3: continue
        try:
            zid = int(parts[1])
            active = int(parts[2]) == 1
        except ValueError:
            continue
        # The SHAPE comes from the FIELD COUNT, exactly as CFileLoader::LoadAudioZone
        # does it - the third column is the active flag, not a shape selector, and
        # reading it as one turns every inactive box into a sphere.
        try:
            if len(parts) >= 9:
                v = [float(t) for t in parts[3:9]]
                rows.append(AuZone(parts[0], zid, active, "box", (v[:3], v[3:])))
            elif len(parts) >= 7:
                v = [float(t) for t in parts[3:7]]
                rows.append(AuZone(parts[0], zid, active, "sphere", (v[:3], v[3])))
        except ValueError:
            continue
    return rows


def _name8(name):
    """The 8 bytes CAudioZones keeps per volume (tAudioZoneData::m_szName), upper-cased
 so the runtime's compare against a script's zone name needs no locale."""
    n = name.upper().encode("latin1")[:8]
    return n + bytes(8 - len(n))


def pack_ambzones(zones):
    """-> ambzones.bin v2 bytes. The named entries keep v1's contents and order so the
 interior-by-name path is untouched; the geometry is appended after them."""
    named = [z for z in zones if z.track != NO_TRACK]
    boxes = [z for z in zones if z.shape == "box"]
    spheres = [z for z in zones if z.shape == "sphere"]

    buf = bytearray(b"AMBZ")
    buf += struct.pack("<4H", AMBZ_VERSION, len(named), len(boxes), len(spheres))
    for z in named:
        n = z.name.upper().encode("latin1")[:15]
        buf += n + bytes(16 - len(n)) + struct.pack("<H", z.track)
    for z in boxes:
        lo, hi = z.geom
        buf += _name8(z.name)
        buf += struct.pack("<6f", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        buf += struct.pack("<HBB", z.track, 1 if z.active else 0, z.vol_mode)
    for z in spheres:
        c, r = z.geom
        buf += _name8(z.name)
        buf += struct.pack("<4f", c[0], c[1], c[2], r)
        buf += struct.pack("<HBB", z.track, 1 if z.active else 0, z.vol_mode)
    return bytes(buf)


def main(out_dir=None):
    """Bake venue ambience. out_dir (Quarry) -> <out_dir>/audio/amb/; else the dev list.
 NON-FATAL: always returns 0 - a missing ffmpeg or non-OGG (PS2 VAG) stream soft-skips
 the audio while still writing the stdlib-only ambzones.bin map."""
    if not os.path.isfile(AUZO):
        print("  ambience SKIPPED: Audiozon.ipl not found at", AUZO)
        return 0
    zones = parse_auzo()
    mapped = [z for z in zones if z.track != NO_TRACK]
    tracks = sorted({z.track for z in mapped})
    n_box = sum(1 for z in zones if z.shape == "box")
    n_sph = len(zones) - n_box
    # Outdoors is the half that has never been reachable: the interior rows sit in SA's
    # separate interior world above z 900, and only the by-name path could ever fire.
    outdoor = sum(1 for z in mapped if z.min_z() < 900.0)
    print(f"auzo rows={len(zones)} ({n_box} boxes, {n_sph} spheres) mapped={len(mapped)} "
          f"unique tracks={len(tracks)} outdoor-with-track={outdoor}")

    outs = [os.path.join(out_dir, "audio", "amb")] if out_dir else DEFAULT_OUTS
    for d in outs:
        try: os.makedirs(d, exist_ok=True)
        except OSError: pass

    # ambzones.bin is pure stdlib + disc-agnostic -> always write it, even when the
    # audio transcode below soft-skips.
    buf = pack_ambzones(zones)
    for d in outs:
        try: open(os.path.join(d, "ambzones.bin"), "wb").write(buf)
        except OSError: pass
    print(f"ambzones.bin v{AMBZ_VERSION}: {len(mapped)} named, {n_box} boxes, "
          f"{n_sph} spheres, {len(buf)} bytes")

    # DEP GATE: the per-track OGG transcode needs ffmpeg (imageio_ffmpeg, heavy). Soft-skip
    # if it is unavailable - never crash the section.
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print("  ambience audio SKIPPED: imageio_ffmpeg unavailable (%s)" % e)
        return 0

    tmp = os.path.join(outs[0], "_raw")
    os.makedirs(tmp, exist_ok=True)
    ok = 0
    for t in tracks:
        try:
            raw = os.path.join(tmp, f"t{t}.ogg")
            if not os.path.exists(raw):
                ogg, _pak = stream_extract.extract_ogg(t)
                if ogg[:4] != b"OggS":            # PS2 streams are VAG, not OGG -> phase 4 radio pass
                    raise RuntimeError(f"stream {t} is not OGG (PS2 VAG stream?)")
                open(raw, "wb").write(ogg)
            out0 = os.path.join(outs[0], f"amb_t{t}.ogg")
            subprocess.run([ff, "-y", "-i", raw, "-ac", "1", "-ar", "22050",
                            "-q:a", "1", out0], capture_output=True)
            data = open(out0, "rb").read()
            for d in outs[1:]:
                try: open(os.path.join(d, f"amb_t{t}.ogg"), "wb").write(data)
                except OSError: pass
            print(f"  track {t}: {len(data)//1024}KB")
            ok += 1
        except Exception as e:
            print(f"  track {t}: SKIPPED ({e})")
    print(f"ambience audio: {ok}/{len(tracks)} track(s) transcoded")
    return 0


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
