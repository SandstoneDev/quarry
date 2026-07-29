#!/usr/bin/env python3
"""stream_extract - pull a the source game streaming-audio track out of AUDIO/STREAMS -> .ogg.

Streams (radio / cutscene / theme) live in XOR-obfuscated ("ADF") pak containers under
AUDIO/STREAMS/<pak>. A track is located by TRAKLKUP.DAT + STRMPAKS.DAT; its OGG Vorbis
data starts 0x1F84 bytes into the track region (a tTrackInfo beat-grid header is skipped)
and is de-obfuscated by a 16-byte rolling XOR keyed on the absolute file offset.
See docs/gta_sa/12_audio/audio_streaming.md.

 TRAKLKUP.DAT - tTrackLookup[N], 0xC each: {u8 pakIdx; u8 pad[3]; u32 offset; u32 length}
 STRMPAKS.DAT - StreamPack[M], 0x10 each: char name[16]
 AUDIO/STREAMS/<pak> ogg = file[offset+0x1F84 : offset+0x1F84+length], then ADF de-XOR:
 out[i] = enc[i] ^ KEY[(offset+0x1F84+i) & 0xF] (absolute file offset)

Usage:
 python stream_extract.py list # list paks / track counts
 python stream_extract.py info <trackId> # extract to a temp + report OggS/rate/dur/comment
 python stream_extract.py <trackId> out.ogg # extract one track to out.ogg
"""
import os
import sys
import struct

# SA_ROOT env override: Quarry points this at the extracted disc. Default = PC dev loop.
# ('/streams' maps onto the PS2 disc's 'STREAMS' via the case-insensitive Windows fs.)
SA  = os.environ.get("SA_ROOT", "") + "/audio"
CFG = SA + "/CONFIG"
STR = SA + "/streams"
HDR = 0x1F84
KEY = bytes([0xEA,0x3A,0xC4,0xA1,0x9A,0xA8,0x14,0xF3,0x48,0xB0,0xD7,0x23,0x9D,0xE8,0xFF,0xF1])


def load_cfg():
    sp = open(CFG + "/StrmPaks.dat", "rb").read()
    tl = open(CFG + "/TrakLkup.dat", "rb").read()
    paks = [sp[i*0x10:i*0x10+0x10].split(b"\x00", 1)[0].decode("latin1")
            for i in range(len(sp)//0x10)]
    tracks = [struct.unpack_from("<BxxxII", tl, t*12) for t in range(len(tl)//12)]
    return paks, tracks


def extract_ogg(trackId):
    """Return the de-obfuscated OGG bytes for trackId."""
    paks, tracks = load_cfg()
    pakIdx, off, length = tracks[trackId]
    pak = paks[pakIdx]
    start = off + HDR
    with open(os.path.join(STR, pak), "rb") as f:
        f.seek(start)
        enc = bytearray(f.read(length))
    for i in range(len(enc)):
        enc[i] ^= KEY[(start + i) & 0xF]
    return bytes(enc), pak


# ---- minimal OGG/Vorbis inspection (no external lib) ----

def ogg_pages(data):
    """Yield (granulePos, headerType, payload) for each OGG page."""
    o, n = 0, len(data)
    while o + 27 <= n and data[o:o+4] == b"OggS":
        htype = data[o+5]
        gran  = struct.unpack_from("<q", data, o+6)[0]
        nseg  = data[o+26]
        segs  = data[o+27:o+27+nseg]
        psize = sum(segs)
        body  = data[o+27+nseg:o+27+nseg+psize]
        yield gran, htype, body
        o += 27 + nseg + psize


def inspect(data):
    """Return (ok, rate, channels, durationSec, vendor) for OGG bytes."""
    if data[:4] != b"OggS":
        return (False, 0, 0, 0.0, "")
    rate = ch = 0
    vendor = ""
    last_gran = 0
    for gran, htype, body in ogg_pages(data):
        if gran > 0:
            last_gran = gran
        if body[:7] == b"\x01vorbis":          # identification header
            ch   = body[11]
            rate = struct.unpack_from("<I", body, 12)[0]
        elif body[:7] == b"\x03vorbis":         # comment header
            p = 7
            vlen = struct.unpack_from("<I", body, p)[0]; p += 4
            vendor = body[p:p+vlen].decode("latin1", "replace"); p += vlen
    dur = (last_gran / rate) if rate else 0.0
    return (True, rate, ch, dur, vendor)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    paks, tracks = load_cfg()

    if cmd == "list":
        print("paks:", paks)
        print("tracks:", len(tracks))
    elif cmd == "info":
        tid = int(sys.argv[2])
        ogg, pak = extract_ogg(tid)
        ok, rate, ch, dur, vendor = inspect(ogg)
        print("trk%d pak=%s bytes=%d OggS=%s rate=%d ch=%d dur=%.1fs vendor=%r"
              % (tid, pak, len(ogg), ok, rate, ch, dur, vendor))
    else:
        tid = int(cmd)
        out = sys.argv[2] if len(sys.argv) > 2 else "track_%d.ogg" % tid
        ogg, pak = extract_ogg(tid)
        ok, rate, ch, dur, vendor = inspect(ogg)
        open(out, "wb").write(ogg)
        print("wrote %s (%d B) pak=%s OggS=%s rate=%d ch=%d dur=%.1fs"
              % (out, len(ogg), pak, ok, rate, ch, dur))
