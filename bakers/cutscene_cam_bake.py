"""cutscene_cam_bake - parse a the source game cutscene camera track (.dat from cuts.img)
into a compact binary the runtime CCutsceneCam samples on the cutscene clock.

The .dat is CSV text, four tracks in order, each: a count line then <count> rows.
Row = "time_f, (value,inTan,outTan) x N,"  - N=1 for scalar tracks, N=3 for vec3.
  1) FOV     scalar (degrees)         e.g. 85..28
  2) ROLL    scalar (degrees)
  3) POS     vec3   (camera position, RELATIVE to the .cut `offset`)
  4) TARGET  vec3   (look-at point,   RELATIVE to the .cut `offset`)
Times are seconds; the four tracks share the cutscene clock (intro1a ~100.7s).
Stray ';' / blank terminator lines can appear between tracks - skip them.

We keep only the VALUE column of each triple (linear interp at runtime; the source
tangents are close to linear for these scenes and a Catmull pass can come later).

Output  data/cutscene/<name>_cam.bin:
  'CCAM' | u32 offX,offY,offZ (f32, the .cut world offset) | 4 tracks
  track: u16 dim (1 or 3) | u16 nkeys | nkeys*(f32 time + dim*f32 value)
"""
import os, sys, struct, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "")
from sa_img import SaImg

# SA_ROOT env override: Quarry points this at the extracted PS2 disc. This baker is a pure
# TEXT/CSV parse of cuts.img (intro1a.cut offset + intro1a.dat camera track) - no model
# codec - so it produces real output on both the PS2 disc and the PC dev loop.
SA_ROOT  = os.environ.get("SA_ROOT", "")
CUTS_IMG = SA_ROOT + "/anim/cuts.img"
NAME     = "intro1a"
OUT      = NAME + "_cam.bin"
DEPLOY   = []

def _num(tok):
    return float(tok.strip().rstrip("f"))

def parse_cut_offset(cut_text):
    for ln in cut_text.splitlines():
        p = ln.split()
        if len(p) == 4 and p[0] == "offset":
            return (float(p[1]), float(p[2]), float(p[3]))
    return (0.0, 0.0, 0.0)

def parse_dat(text):
    # tokenised rows, skipping blank / non-data terminator lines
    rows = []
    for ln in text.replace("\r", "").split("\n"):
        s = ln.strip().rstrip(",").strip()
        if not s or s == ";":
            continue
        rows.append(s)
    tracks = []
    i = 0
    while len(tracks) < 4 and i < len(rows):
        # a count line is a bare integer
        if not re.fullmatch(r"\d+", rows[i]):
            i += 1; continue
        n = int(rows[i]); i += 1
        keys = []
        for _ in range(n):
            if i >= len(rows): break
            vals = [_num(t) for t in rows[i].split(",") if t.strip()]
            i += 1
            if not vals: continue
            t = vals[0]
            rest = vals[1:]
            # each component is a triple laid out as [value..][inTan..][outTan..]:
            # scalar row = val,in,out (3) -> value is rest[0:1]; vec3 row = xyz,xyz,xyz
            # (9) -> value is rest[0:3]. So the VALUE block is the first len(rest)//3.
            n = len(rest) // 3 if len(rest) % 3 == 0 and len(rest) >= 3 else len(rest)
            comp = rest[0:n]
            keys.append((t, comp))
        dim = len(keys[0][1]) if keys else 1
        tracks.append((dim, keys))
    return tracks  # [FOV, ROLL, POS, TARGET]

def pack(offset, tracks):
    buf = bytearray(b"CCAM")
    buf += struct.pack("<3f", *offset)
    for dim, keys in tracks:
        buf += struct.pack("<HH", dim, len(keys))
        for t, comp in keys:
            buf += struct.pack("<f", t)
            for c in comp[:dim]:
                buf += struct.pack("<f", c)
    return bytes(buf)

def main():
    # argv[1] = explicit output path (Quarry passes <OutDir>/cutscene/intro1a_cam.bin). When
    # given we write ONLY there and skip the dev-loop memstick mirror.
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    quarry = len(sys.argv) > 1

    img = SaImg(CUTS_IMG)
    cut = img.extract(NAME + ".cut").rstrip(b"\x00").decode("ascii", "replace")
    dat = img.extract(NAME + ".dat").rstrip(b"\x00").decode("ascii", "replace")
    off = parse_cut_offset(cut)
    tr = parse_dat(dat)
    names = ["FOV", "ROLL", "POS", "TARGET"]
    print("offset", off)
    for nm, (dim, keys) in zip(names, tr):
        span = (keys[0][0], keys[-1][0]) if keys else (0, 0)
        print("  %-7s dim=%d keys=%d t[%.2f..%.2f]" % (nm, dim, len(keys), span[0], span[1]))
    blob = pack(off, tr)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(blob)
    if quarry:
        print("=== %s_cam.bin: %d bytes -> %s ===" % (NAME, len(blob), out))
        return
    n = 0
    for d in DEPLOY:
        if os.path.isdir(os.path.dirname(os.path.dirname(d))):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            open(d, "wb").write(blob); n += 1
    print("=== %s_cam.bin: %d bytes, deployed to %d dir(s) ===" % (NAME, len(blob), n))

if __name__ == "__main__":
    main()
