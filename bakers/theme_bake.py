#!/usr/bin/env python3
"""theme_bake - the source game loading theme -> data/theme.at3 (ATRAC3 for sceAtrac/ME).

The PSP plays ATRAC3 natively on the Media Engine (near-zero CPU), which is lighter
than decoding OGG on the Allegrep with stb_vorbis. So we transcode the de-obfuscated
stream track to AT3 offline:

 track OGG (stream_extract: TRAKLKUP/STRMPAKS + ADF de-XOR + skip 0x1F84)
 -> WAV (soundfile decode, linear-resample 32k -> 44100 = the PSP AT3 rate)
 -> AT3 (psp_at3tool.exe -e -wholeloop - ATRAC3, whole-track loop for seamless repeat)
 -> data/theme.at3 (+ the .ogg too, as a stb_vorbis fallback the runtime auto-selects)

Run: python theme_bake.py [trackId] # default 184 (BEATS, ~77 s) - the theme candidate
"""
import os
import sys
import subprocess

import numpy as np
import soundfile as sf

import stream_extract

AT3TOOL = os.environ.get("AT3TOOL", "")
AT3_RATE = 44100
AT3_KBPS = 105

DEPLOY = [
    "",
    "",
    "",
]


def bake(trackId):
    ogg, pak = stream_extract.extract_ogg(trackId)
    open("_theme.ogg", "wb").write(ogg)
    d, r = sf.read("_theme.ogg")
    if d.ndim == 1:
        d = d[:, None]
    # linear resample to 44100 (AT3 PSP rate)
    n = d.shape[0]
    nn = int(n * AT3_RATE / r)
    x = np.linspace(0, n - 1, nn)
    xi = x.astype(int)
    fr = (x - xi)[:, None]
    xi2 = np.minimum(xi + 1, n - 1)
    out = d[xi] * (1 - fr) + d[xi2] * fr
    sf.write("_theme.wav", out, AT3_RATE, subtype="PCM_16")

    subprocess.run([AT3TOOL, "-e", "-br", str(AT3_KBPS), "-wholeloop", "_theme.wav", "_theme.at3"],
                   check=True)
    at3 = open("_theme.at3", "rb").read()
    print("track %d (%s): %.1fs -> theme.at3 %.0f KB (%d kbps, whole-loop)"
          % (trackId, pak, nn / float(AT3_RATE), len(at3) / 1024.0, AT3_KBPS))

    n_dep = 0
    for dd in DEPLOY:
        if os.path.isdir(dd):
            os.makedirs(os.path.join(dd, "audio"), exist_ok=True)
            open(os.path.join(dd, "audio", "theme.at3"), "wb").write(at3)
            open(os.path.join(dd, "audio", "theme.ogg"), "wb").write(ogg)   # stb_vorbis fallback
            n_dep += 1
            print("  ->", dd)
    if not n_dep:
        open("theme.at3", "wb").write(at3)
        print("  no deploy dir; wrote ./theme.at3")
    for t in ("_theme.ogg", "_theme.wav", "_theme.at3"):
        try: os.remove(t)
        except OSError: pass


if __name__ == "__main__":
    bake(int(sys.argv[1]) if len(sys.argv) > 1 else 184)
