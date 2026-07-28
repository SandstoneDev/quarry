#!/usr/bin/env python3
"""cutaudio_bake.py - bake the intro1a cutscene's AUDIO + SUBTITLES for the PSP port.

Two outputs into data/cutscene/:
  * intro1a.ogg       - the premixed cutscene stream (voices + ambience + score), decrypted
                         from audio/streams/CUTSCENE track 703 (INTRO1A) via the SAME XOR path
                         the radio uses. 32 kHz stereo, ~100.7 s. Streamed at runtime by
                         CutsceneAudio.c (stb_vorbis worker, no loop).
  * intro1a_subs.bin  - 'CSUB' + u16 count + [u32 startMs, u32 durMs, u16 len, char[len]]...
                         parsed from anim/cuts.img : intro1a.cut TEXT section (start,dur,gxtKey)
                         with the key resolved against text/american.gxt table INTRO1 using the
                         SA GXT key hash (CRC-32-IEEE, no final XOR, uppercased). ~ tokens stripped.

Ground truth: the reference notes CutSceneStreamsPC.h (INTRO1A=703), CutsceneMgr.cpp (TEXT section),
Core/KeyGen.h (GetUppercaseKey = CRC-32 0xEDB88320, init 0xFFFFFFFF, no final invert).
"""
import os, sys, struct, zlib
sys.path.insert(0, "")
sys.path.insert(0, "")
import radio_bake
from sa_img import SaImg

# SA_ROOT env override: Quarry points this at the extracted PS2 disc. cuts.img (subtitle
# TEXT) + american.gxt (subtitle strings) are codec-free and produce real output on PS2.
# The audio (CUTSCENE stream track 703) is a PC XOR-OGG container on the PC copy but PS2-VAG
# on the disc -> the audio path SOFT-SKIPS on PS2 (see main), subtitles still bake.
GAME    = os.environ.get("SA_ROOT", "")
CUTS    = GAME + r"/anim/cuts.img"
GXT     = GAME + r"/text/american.gxt"
CUTNAME = "intro1a"
GXTTAB  = "INTRO1"     # per-episode subtitle table holding the INT1A** keys
TRACKID = 703          # INTRO1A in the CUTSCENE stream pack
OUTDIR  = ""
DEPLOY  = [
    "",
    "",
    "",
]


def sa_gxt_hash(s):
    """CKeyGen::GetUppercaseKey - CRC-32-IEEE (poly 0xEDB88320, init 0xFFFFFFFF) with NO
    final inversion, over the uppercased key. zlib.crc32 applies the final ^0xFFFFFFFF, so undo it."""
    return zlib.crc32(s.upper().encode("latin1")) ^ 0xFFFFFFFF


def load_gxt_table(name):
    """Return {gxtHash: string} for one american.gxt table (8-bit ANSI strings)."""
    g = open(GXT, "rb").read()
    tp = g.find(b"TABL")
    tsz = struct.unpack_from("<I", g, tp + 4)[0]
    eb = tp + 8
    off = None
    for i in range(0, tsz, 12):
        nm = g[eb + i:eb + i + 8].split(b"\0")[0].decode("latin1")
        if nm == name:
            off = struct.unpack_from("<I", g, eb + i + 8)[0]
            break
    if off is None:
        raise KeyError("gxt table %r not found" % name)
    p = off
    if g[p:p + 4] != b"TKEY":     # non-MAIN tables lead with an 8-byte name
        p += 8
    assert g[p:p + 4] == b"TKEY", g[p:p + 8]
    ksz = struct.unpack_from("<I", g, p + 4)[0]
    kb = p + 8
    td = kb + ksz
    assert g[td:td + 4] == b"TDAT", g[td:td + 4]
    db = td + 8
    out = {}
    for i in range(0, ksz, 8):    # each entry = (u32 tdatOffset, u32 keyHash)
        o = struct.unpack_from("<I", g, kb + i)[0]
        h = struct.unpack_from("<I", g, kb + i + 4)[0]
        e = db + o
        while e < len(g) and g[e] != 0:
            e += 1
        out[h] = g[db + o:e].decode("latin1", "replace")
    return out


def strip_tokens(s):
    """Drop GXT format tokens ~X~ (colour/newline/etc); ~n~ -> space. Keep printable ASCII."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "~":
            j = s.find("~", i + 1)
            if j >= 0:
                out.append(" ")     # a token boundary is a soft space (covers ~n~)
                i = j + 1
                continue
        c = s[i]
        out.append(c if 32 <= ord(c) < 127 else " ")
        i += 1
    # collapse runs of spaces
    return " ".join("".join(out).split())


def parse_cut_text(raw):
    """intro1a.cut TEXT section -> [(startMs, durMs, KEY)] (plain-text .cut)."""
    txt = raw.split(b"\x00")[0].decode("latin1", "replace")
    insec, subs = False, []
    for line in txt.replace("\r", "").split("\n"):
        s = line.strip()
        if s.startswith("text"):
            insec = True; continue
        if s.startswith("end"):
            insec = False; continue
        if insec and s:
            parts = s.replace(",", " ").split()
            if len(parts) >= 3:
                subs.append((int(parts[0]), int(parts[1]), parts[2].upper()))
    return subs


def _extract_cutscene_ogg():
    """Decrypt CUTSCENE stream track 703 with the PC XOR path and return the OGG bytes, or
    None if the payload isn't a PC OGG (PS2 disc = VAG). Uses radio_bake's KEY/HDR constants
    (radio_bake itself is left untouched) and tolerates the PS2 '.PAK' filename suffix that
    the extensionless PC stream files lack."""
    packs, lut = radio_bake.load_lookups(GAME)
    pid, off, size = lut[TRACKID]
    spath = os.path.join(GAME, "audio", "streams", packs[pid])
    if not os.path.isfile(spath) and os.path.isfile(spath + ".PAK"):
        spath += ".PAK"                     # PS2 stream files carry a .PAK suffix
    with open(spath, "rb") as f:
        f.seek(off + radio_bake.HDR)
        enc = f.read(size)
    base = off + radio_bake.HDR
    dec = bytes(b ^ radio_bake.KEY[(base + i) & 15] for i, b in enumerate(enc))
    return (packs[pid], dec) if dec[:4] == b"OggS" else (packs[pid], None)


def main():
    # argv[1] = explicit output DIRECTORY (Quarry passes <OutDir>/cutscene). cutaudio emits
    # two files (intro1a.ogg + intro1a_subs.bin); when given we write ONLY there.
    outdir = sys.argv[1] if len(sys.argv) > 1 else OUTDIR
    quarry = len(sys.argv) > 1
    os.makedirs(outdir, exist_ok=True)

    # ---- 1) audio (SOFT-SKIP, like ambience_bake) ----
    # The PS2 CUTSCENE stream is VAG, not the PC XOR-OGG container, so the PC path can't make
    # a usable intro1a.ogg. Defer it non-fatally - the cutscene still plays without the audio
    # track for this pass. (Any error here is also swallowed: audio never aborts the step.)
    dec = None
    try:
        print("=== audio: CUTSCENE track %d (%s) ===" % (TRACKID, CUTNAME.upper()))
        pack, dec = _extract_cutscene_ogg()
        if dec is not None:
            samp = radio_bake.ogg_duration_samples(dec)
            print("  pack=%s ogg=%d bytes samples=%d (~%.1fs)" % (pack, len(dec), samp, samp / 32000.0))
            open(os.path.join(outdir, CUTNAME + ".ogg"), "wb").write(dec)
        else:
            print("  !! CUTSCENE stream is not PC-OGG (PS2 VAG) - audio deferred; subtitles still bake")
    except Exception as e:
        print("  !! audio deferred (non-fatal): %s" % e)

    # ---- 2) subtitles ----
    print("=== subtitles: %s.cut TEXT x %s ===" % (CUTNAME, GXTTAB))
    raw = SaImg(CUTS).extract(CUTNAME + ".cut")
    subs = parse_cut_text(raw)
    tab = load_gxt_table(GXTTAB)
    rows = []
    for st, du, k in subs:
        s = tab.get(sa_gxt_hash(k))
        if s is None:
            print("  !! MISS key=%s (skipped)" % k); continue
        s = strip_tokens(s)
        rows.append((st, du, s))
        print("  t=%6d dur=%5d %s | %s" % (st, du, k, s))
    buf = bytearray(b"CSUB")
    buf += struct.pack("<H", len(rows))
    for st, du, s in rows:
        b = s.encode("latin1", "replace")
        buf += struct.pack("<IIH", st, du, len(b)) + b
    subs_path = os.path.join(outdir, CUTNAME + "_subs.bin")
    open(subs_path, "wb").write(buf)
    print("  %d subtitle rows, %d bytes" % (len(rows), len(buf)))

    if quarry:
        print("=== cutaudio: subs=%d rows, audio=%s -> %s ===" %
              (len(rows), "baked" if dec is not None else "deferred (PS2 VAG)", outdir))
        return

    # ---- deploy ----
    n = 0
    for d in DEPLOY:
        parent = os.path.dirname(os.path.dirname(os.path.dirname(d)))  # .../SA_PSP
        if not os.path.isdir(os.path.dirname(parent)):
            continue
        try:
            os.makedirs(d, exist_ok=True)
            if dec is not None:
                open(os.path.join(d, CUTNAME + ".ogg"), "wb").write(dec)
            open(os.path.join(d, CUTNAME + "_subs.bin"), "wb").write(buf)
            n += 1
        except OSError as e:
            print("  deploy skip %s (%s)" % (d, e))
    print("=== deployed to %d dir(s) ===" % n)


if __name__ == "__main__":
    main()
