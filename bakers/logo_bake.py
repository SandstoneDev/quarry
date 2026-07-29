#!/usr/bin/env python3
"""logo_bake.py - movies/Logo.mpg -> data/logo.bin ('VLG1' frame sequence).

The PSP has no generic MPEG-1 decoder (sceMpeg is PMF-only), so the intro video
is baked offline into a CLUT8 frame sequence the boot player STREAMS from the
Memory Stick frame by frame (BootLogo.c):

 header : 'VLG1', u16 nframes, u16 fps, u16 w, u16 h
 frame : clut[256] RGBA8888 (1024 B) + w*h CLUT8 texels (row-major)

Content is 256x192 (4:3 like the 640x480 source; 256 = pow2 buffer width; the
player uploads rows into a 256x256 texture and maps v 0..192). 12 fps halves
the 25 fps source - smooth enough for the R* logo. ~8 s -> ~100 frames x 49 KB
~= 5 MB on the stick, ~590 KB/s streamed (well under MS read speed).

Usage: python logo_bake.py # bake + deploy everywhere reachable
"""
import struct
import subprocess

import numpy as np
import imageio_ffmpeg
from PIL import Image

FF  = imageio_ffmpeg.get_ffmpeg_exe()
MOV = ""
W, H = 256, 192
SRC_W, SRC_H, SRC_FPS = 640, 480, 25.0
FPS_OUT = 12
DEPLOY = [
    "",
    "",
    "",
]


def main():
    raw = subprocess.run(
        [FF, "-y", "-i", MOV, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    fsz = SRC_W * SRC_H * 3
    total = len(raw) // fsz
    print("source frames:", total)

    frames = []
    lums = []
    step = SRC_FPS / FPS_OUT
    t = 0.0
    while int(t) < total:
        i = int(t)
        img = np.frombuffer(raw, np.uint8, fsz, i * fsz).reshape(SRC_H, SRC_W, 3)
        lums.append(float(img.mean()))
        im = Image.fromarray(img).resize((W, H), Image.BILINEAR)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))
        t += step

    # TRIM the black lead-in/tail: the source opens with ~2.5 s of pure black
    # (read as "the intro is a black screen"). Keep a short beat around the
    # first/last lit frames; the mid-video black transition stays.
    lit = [i for i, l in enumerate(lums) if l > 1.5]
    if lit:
        a = max(0, lit[0] - 3)
        b = min(len(frames), lit[-1] + 7)
        frames = frames[a:b]
        print("trimmed black lead/tail: %d..%d of %d" % (a, b, len(lums)))

    print("baked frames:", len(frames), "@", FPS_OUT, "fps")
    buf = bytearray(b"VLG1")
    buf += struct.pack("<4H", len(frames), FPS_OUT, W, H)
    for im in frames:
        pal = im.getpalette() or []
        pal += [0] * (768 - len(pal))
        clut = bytearray()
        for i in range(256):
            r, g, b = pal[i*3], pal[i*3+1], pal[i*3+2]
            clut += struct.pack("<I", 0xFF000000 | (b << 16) | (g << 8) | r)
        buf += clut
        buf += im.tobytes()

    for p in DEPLOY:
        try:
            open(p, "wb").write(bytes(buf))
            print("wrote %s (%d KB)" % (p, len(buf) // 1024))
        except OSError as e:
            print("skip %s: %s" % (p, e))


if __name__ == "__main__":
    main()
