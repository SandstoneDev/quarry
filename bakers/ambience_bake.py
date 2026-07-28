#!/usr/bin/env python3
"""ambience_bake - SA venue ambience -> data/amb/ for the PSP port.

SA plays STREAMED ambience tracks per AUDIO ZONE (data/maps/Audiozon.ipl 'auzo'
rows; CAEAmbienceTrackManager::UpdateAmbienceTrackAndVolume maps zoneId->track,
0x4D6E60). Most zones are venue interiors (bar chatter, restaurant hum, ammunation)
-- exactly what the port's interiors lack.

This bake:
  1) parses Audiozon.ipl auzo rows (name, zoneId),
  2) maps zoneId -> stream track via the 0x4D6E60 switch (radio zones skipped),
  3) extracts each unique track from AUDIO/STREAMS via stream_extract,
  4) transcodes to mono 22050 Hz OGG (ffmpeg; ambience quality, small files),
  5) writes data/amb/amb_t<track>.ogg + ambzones.bin ('AMBZ' u16 n; n x
     { char name[16]; u16 trackId }) - the runtime matches its interior name
     against zone names case-insensitively.
"""
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_extract
# NB: imageio_ffmpeg is imported lazily inside main() so a missing ffmpeg SOFT-SKIPS the
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

# CAEAmbienceTrackManager::UpdateAmbienceTrackAndVolume 0x4D6E60: zoneId -> track.
# Radio-station zones (30, 52..62) are intentionally absent.
ZONE_TRACK = {
    4: 143, 5: 140, 8: 165, 10: 139, 12: 168, 13: 157, 15: 164, 17: 146,
    19: 138, 20: 136, 21: 135, 23: 148, 24: 159, 25: 158, 26: 154,
    28: 147, 29: 147, 34: 162, 36: 155, 37: 144, 39: 163, 41: 169,
    44: 152, 48: 137, 50: 173, 51: 156, 64: 151, 66: 170, 67: 171,
}


def parse_auzo():
    rows = []
    inz = False
    for ln in open(AUZO, encoding="latin-1"):
        s = ln.strip()
        if s == "auzo": inz = True; continue
        if s == "end": inz = False; continue
        if not inz or not s or s.startswith("#"): continue
        parts = [t.strip() for t in s.split(",")]
        if len(parts) < 3: continue
        try: zid = int(parts[1])
        except ValueError: continue
        rows.append((parts[0], zid))
    return rows


def main(out_dir=None):
    """Bake venue ambience. out_dir (Quarry) -> <out_dir>/audio/amb/; else the dev list.
    NON-FATAL: always returns 0 - a missing ffmpeg or non-OGG (PS2 VAG) stream soft-skips
    the audio while still writing the stdlib-only ambzones.bin map."""
    if not os.path.isfile(AUZO):
        print("  ambience SKIPPED: Audiozon.ipl not found at", AUZO)
        return 0
    rows = parse_auzo()
    mapped = [(nm, zid, ZONE_TRACK[zid]) for nm, zid in rows if zid in ZONE_TRACK]
    tracks = sorted({t for _, _, t in mapped})
    print(f"auzo rows={len(rows)} mapped={len(mapped)} unique tracks={len(tracks)}")

    outs = [os.path.join(out_dir, "audio", "amb")] if out_dir else DEFAULT_OUTS
    for d in outs:
        try: os.makedirs(d, exist_ok=True)
        except OSError: pass

    # ambzones.bin (zone name -> track id) is pure stdlib + disc-agnostic -> always write it.
    buf = bytearray(b"AMBZ")
    buf += struct.pack("<H", len(mapped))
    for nm, _, t in mapped:
        n = nm.upper().encode("latin1")[:15]
        n += b"\x00" * (16 - len(n))
        buf += n + struct.pack("<H", t)
    for d in outs:
        try: open(os.path.join(d, "ambzones.bin"), "wb").write(bytes(buf))
        except OSError: pass
    print(f"ambzones.bin: {len(mapped)} zones")

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
                if ogg[:4] != b"OggS":
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
