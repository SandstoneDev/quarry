#!/usr/bin/env python3
"""radio_ps2_bake - the PS2 disc's radio, adverts and cutscene audio -> data/audio.

Extracts every element of every stream pack as raw SPU ADPCM, exactly as it sits on
the disc, plus a manifest describing what each element is. Nothing is decoded and
nothing is re-encoded: the PSP decodes ADPCM cheaply, so the bake is pure I/O and
the result is SMALLER than the OGG set it replaces (a 55-second song is 1.5 MB of
ADPCM against 2.8 MB of OGG).

See tools/sa_ps2_stream.py for the pack format. Element classes come from duration,
which separates them cleanly: measured on CH, exactly 12 elements run 60 seconds or
longer and the known-good build lists exactly 12 songs for that station. Talk
stations are all long, so a pack with no short elements is marked talk throughout.

Output:
 <out>/<CODE>/NNN.adp 'ADP3' u16 channels, u16 rate, u32 bytesPerCh,
 u32 samplesPerCh, then interleaved ADPCM (0x2000 blocks)
 <out>/radio.bin 'RAD2' manifest (below)
 <out>/../cutscene/NNN.adp when --cutscene is given

radio.bin:
 'RAD2' u32 version=2, u32 nStations
 station: char code[4], u8 radioId, u8 flags, u16 nElems, char name[32]
 elem[nElems]: u8 kind (0 music, 1 talk, 2 ident), u8 pad,
 u16 rate, u32 bytesPerCh, u32 samplesPerCh
 Element N of station CODE is <CODE>/NNN.adp; bytesPerCh*2 is the file size.

Usage: radio_ps2_bake.py <audio-dir> <out-dir> [--cutscene <dir>] [--elf <file>]
 [--ambience <dir>] [--intro <file> <seconds>] [--names <file>]
 --elf reads the station identifiers out of the game executable, which is where
 the disc names its own stations; --names overrides with `CODE=Display Name`
 lines. Without either, a station shows its pack code.
"""
import os
import struct
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import sa_ps2_stream as S

MUSIC, TALK, IDENT = 0, 1, 2
LONG_S = 60.0            # at or above this an element is a full track
IDENT_S = 20.0           # below this it is a station ident or a one-line drop
SHARED = ("ADVERTS",)    # packs that are not a dial station but feed every station
SKIP = ("AA", "AMBIENCE", "BEATS", "CUTSCENE")   # not radio; cutscene has its own flag


def classify(dur, pack_has_short):
    if dur >= LONG_S:
        return MUSIC if pack_has_short else TALK
    return IDENT if dur < IDENT_S else TALK


ADP_MAGIC = b"ADP3"

def copy_element(fsrc, off, nbytes_total, dst, rate=0, channels=2, samples=0,
                 blk_off=None, blk_size=None):
    """Write one element out as ADP3: a 16-byte header, then the channels interleaved
 at the runtime's refill window (see the loop below).

 blk_off/blk_size come from sa_ps2_stream.read_header and say where this element's
 audio blocks sit inside a period. They are not optional in practice - the default
 is the two-channel layout a CUTSCENE-shaped element has - because a radio element's
 audio does NOT start at the top of the period: two 750 Hz sub-streams occupy the
 first 0x1000 bytes of it. Reading from +0 mixed 0xF80 bytes of that low-rate
 material into every left-channel chunk, played 32x too fast, once every 4.78 s.
 """
    channels = max(1, channels)
    per_ch = nbytes_total // channels
    if blk_off is None or blk_size is None:
        _sz, _of = S.block_layout([(0, rate or 24000)] * channels)
        blk_off, blk_size = _of, _sz
    fsrc.seek(off + S.DATA)                       # S.DATA == S.HDR == 0x1F84, no "+4"
    # Whole periods only: a block is addressed at its own offset inside the period, so
    # a partial read would cut the last one. The element itself is stored rounded to
    # whole periods (IOPAudio.irx.text 0x4C20 divides by 0x21000 and multiplies back),
    # so this never reaches past it - verified over all 1922 elements.
    periods = (per_ch + blk_size[0] - 1) // blk_size[0]
    raw = fsrc.read(periods * S.STREAM_PERIOD)
    chans = [S.channel_bytes(raw, blk_off[c], blk_size[c], per_ch) for c in range(channels)]
    short = [c for c in range(channels) if len(chans[c]) < per_ch]
    if short:
        # Never write a file whose header promises more than its body holds: the runtime
        # trusts bytesPerCh to locate every window. Say so loudly instead.
        print("  WARNING: %s channel(s) %s came back short (%d of %d bytes) - the element "
              "is unplayable past that point" % (os.path.basename(dst), short,
                                                 min(len(chans[c]) for c in short), per_ch))
    written = 0
    with open(dst, "wb") as o:
        o.write(ADP_MAGIC + struct.pack("<HHII", channels, rate, per_ch, samples))
        # Interleave at exactly the runtime's refill window so one sequential read
        # covers every channel. Flat channels (ADP2) looked tidier but put them
        # bytesPerCh apart, which turned each refill into two DISTANT seeks: on
        # hardware the card fell from 6.6 ms to 25.3 ms per request, the world
        # streamer starved and CJ's house never loaded.
        for pos in range(0, per_ch, S.WINDOW):
            take = min(S.WINDOW, per_ch - pos)
            for c in range(channels):
                o.write(chans[c][pos:pos + take])
                written += take
    return written


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    audio, out = sys.argv[1], sys.argv[2]
    cut_dir = None
    amb_dir = None
    intro = None
    # A run that only wants the ambience or the cutscene voice must not touch the
    # stations. Without this it walked every pack again: the ambience run rewrote
    # radio.bin with ZERO stations (only step order saved it) and the cutscene run
    # re-copied ~1.5 GB of station audio into a scratch dir nobody reads.
    stations_wanted = True
    intro_subs = None
    elf_path = None
    names = {}
    a = 3
    while a < len(sys.argv):
        if sys.argv[a] == "--cutscene" and a + 1 < len(sys.argv):
            cut_dir = sys.argv[a + 1]; a += 2
        elif sys.argv[a] == "--elf" and a + 1 < len(sys.argv):
            elf_path = sys.argv[a + 1]; a += 2
        elif sys.argv[a] == "--ambience" and a + 1 < len(sys.argv):
            amb_dir = sys.argv[a + 1]; a += 2
        elif sys.argv[a] == "--intro" and a + 2 < len(sys.argv):
            intro = (sys.argv[a + 1], float(sys.argv[a + 2])); a += 3
        elif sys.argv[a] == "--no-stations":
            stations_wanted = False; a += 1
        elif sys.argv[a] == "--intro-subs" and a + 1 < len(sys.argv):
            intro_subs = sys.argv[a + 1]; a += 2
        elif sys.argv[a] == "--names" and a + 1 < len(sys.argv):
            for line in open(sys.argv[a + 1], encoding="utf-8", errors="replace"):
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    names[k.strip().upper()] = v.strip()
            a += 2
        else:
            a += 1

    if elf_path:
        # The executable lists its stations as radio_<name>, in the same order the
        # station packs appear. Verified against a known-good build: all eleven line
        # up. Taking them from here keeps station identity on the user's disc rather
        # than baked into this tool.
        import re
        blob = open(elf_path, "rb").read()
        seen = []
        for m in re.findall(rb"radio_[A-Za-z]+", blob):
            nm = m.decode()
            if nm not in seen:
                seen.append(nm)
        codes = [p for p in S.load_index(audio)[0] if p not in SKIP and p not in SHARED]
        for code, ident in zip(codes, seen):
            names.setdefault(code.upper(), ident[len("radio_"):])
        print("  station names from the executable: %d" % min(len(codes), len(seen)))

    packs, tracks = S.load_index(audio)

    # Ambience is addressed differently from everything else here. ambzones.bin maps a
    # zone to a GLOBAL track id (135..173, all inside AMBIENCE.PAK), not to a position
    # within the pack, so these files are named by that id and the zone table needs no
    # rebake. Without them the venue ambience loads its 131 zones and plays nothing.
    if amb_dir:
        os.makedirs(amb_dir, exist_ok=True)
        pid = packs.index("AMBIENCE") if "AMBIENCE" in packs else -1
        if pid >= 0:
            src = open(os.path.join(audio, "STREAMS", "AMBIENCE.PAK"), "rb")
            n = mb = 0
            for tid, (p, off, size) in enumerate(tracks):
                if p != pid:
                    continue
                try:
                    h = S.read_header(audio, packs, tracks, tid)
                except ValueError:
                    continue
                mb += copy_element(src, h["offset"], h["bytes_per_ch"] * h["channels"],
                                   os.path.join(amb_dir, "amb_t%d.adp" % tid),
                                   h["rate"], h["channels"], h["samples"],
                                   h["blk_off"], h["blk_size"])
                n += 1
            src.close()
            print("  ambience: %d tracks -> %s (%.1f MB)" % (n, amb_dir, mb / 1048576.0))

    # The intro's voice track has no name on the disc, only a duration: pick the
    # CUTSCENE element closest to the take the engine plays. Matching on length is
    # what identified it in the first place - 100.4 s against a documented 100.7 --
    # and it keeps working if the element order ever shifts.
    if intro:
        out_path, want_s = intro
        pid = packs.index("CUTSCENE") if "CUTSCENE" in packs else -1
        best = None
        # The subtitles identify the take. Length does not: the CUTSCENE pack holds 141
        # elements with closely spaced durations, and taking the one nearest the
        # animation's 100.7 s picked a different scene entirely (100.4 s, and audibly
        # the wrong dialogue). The subtitle file says exactly when someone speaks, so
        # the right element is the one loud inside those windows and quiet between --
        # it beat the runner-up 2.12 to 1.68 on the disc checked. Length stays as the
        # fallback for a run with no subtitles.
        if pid >= 0 and intro_subs and os.path.isfile(intro_subs):
            wins = S.subtitle_windows(intro_subs)
            tid, sc = S.match_by_subtitles(audio, packs, tracks, "CUTSCENE", wins)
            if tid is not None:
                h = S.read_header(audio, packs, tracks, tid)
                best = (0.0, tid, h, h["samples"] / float(h["rate"] or 1))
                print("  cutscene voice: matched %d subtitle line(s), element %d, score %.2f"
                      % (len(wins), tid, sc))
        if best is None and pid >= 0:
            for tid, (p, off, size) in enumerate(tracks):
                if p != pid:
                    continue
                try:
                    h = S.read_header(audio, packs, tracks, tid)
                except ValueError:
                    continue
                d = h["samples"] / float(h["rate"] or 1)
                if best is None or abs(d - want_s) < best[0]:
                    best = (abs(d - want_s), tid, h, d)
        if best:
            _, tid, h, d = best
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(h["path"], "rb") as src:
                copy_element(src, h["offset"], h["bytes_per_ch"] * h["channels"],
                             out_path, h["rate"], h["channels"], h["samples"],
                             h["blk_off"], h["blk_size"])
            print("  cutscene voice: element %d, %.1f s (wanted %.1f) -> %s"
                  % (tid, d, want_s, os.path.basename(out_path)))
        else:
            print("  cutscene voice: no CUTSCENE elements readable")

    by_pack = {}
    for tid, (pid, off, size) in enumerate(tracks):
        by_pack.setdefault(pid, []).append((tid, off, size))

    os.makedirs(out, exist_ok=True)
    stations = []
    total_bytes = 0

    for pid, name in enumerate(packs) if stations_wanted else []:
        is_cut = (name == "CUTSCENE")
        if name in SKIP and not (is_cut and cut_dir):
            continue
        elems = by_pack.get(pid, [])
        if not elems:
            continue
        # The track table covers every pack on the disc, but a run that only wants the
        # ambience or the cutscene voice is given just those.PAK files. Skip a pack that
        # was not staged instead of dying on it: without this an ambience-only run reports
        # failure even though it already wrote its tracks.
        if not S.pack_available(audio, name):
            print("  %s: pack not staged, skipped" % name)
            continue

        heads = []
        for tid, off, size in elems:
            try:
                heads.append((tid, S.read_header(audio, packs, tracks, tid)))
            except (ValueError, OSError):
                continue
        if not heads:
            continue
        durs = [h["samples"] / float(h["rate"] or 1) for _t, h in heads]
        has_short = any(d < LONG_S for d in durs)

        dst_dir = cut_dir if is_cut else os.path.join(out, name)
        os.makedirs(dst_dir, exist_ok=True)
        src = open(os.path.join(audio, "STREAMS", name + ".PAK"), "rb")

        recs = []
        for i, ((tid, h), dur) in enumerate(zip(heads, durs)):
            n = copy_element(src, h["offset"], h["bytes_per_ch"] * h["channels"],
                             os.path.join(dst_dir, "%03d.adp" % i),
                             h["rate"], h["channels"], h["samples"],
                             h["blk_off"], h["blk_size"])
            total_bytes += n
            recs.append((classify(dur, has_short), h["rate"],
                         h["bytes_per_ch"], h["samples"]))
        src.close()

        if is_cut:
            print("  cutscene: %d elements -> %s" % (len(recs), dst_dir))
            continue

        shared = name in SHARED
        radio_id = 0 if shared else sum(1 for st in stations if not st[2]) + 1
        stations.append((name, radio_id, 1 if shared else 0,
                         names.get(name, name), recs))
        kinds = [0, 0, 0]
        for k, _r, _b, _s in recs:
            kinds[k] += 1
        print("  %-8s %3d elements  music %3d  talk %3d  ident %3d"
              % (name, len(recs), kinds[MUSIC], kinds[TALK], kinds[IDENT]))

    if not stations_wanted:      # ambience/voice run: leave the dial manifest alone
        return

    blob = bytearray(struct.pack("<4sII", b"RAD2", 2, len(stations)))
    for code, rid, flags, disp, recs in stations:
        blob += struct.pack("<4sBBH32s", code.encode()[:4].ljust(4, b"\0"),
                            rid, flags, len(recs), disp.encode()[:32].ljust(32, b"\0"))
        for kind, rate, nb, ns in recs:
            blob += struct.pack("<BBHII", kind, 0, rate, nb, ns)
    open(os.path.join(out, "radio.bin"), "wb").write(bytes(blob))
    print("radio.bin: %d stations, %d bytes; audio extracted %.1f MB"
          % (len(stations), len(blob), total_bytes / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
