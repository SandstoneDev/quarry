#!/usr/bin/env python3
"""sa_spin - reduce a the source game IFP clip to a runtime SPIN descriptor.

SA's moving map props (the LV rotating signs, windmills, the A51 radar) are
CAnimatedBuilding: an IDE `anim` row names an anim BLOCK, the block holds one
animation named after the model's DFF, and each of its SEQUENCES drives the
clump CHILD FRAME whose node name matches the sequence name. The clips are
tiny and dumb - ANP3 keyType 3 (compressed, ROTATION ONLY), 4..37 keys, no
translation channel anywhere - so every one of them collapses to

 {axis, mode, rate_deg_per_sec, amplitude_deg}

and the IFP itself never has to ship (the engine spins the split model from
this descriptor; see the region_X_Y.spin sidecar in sa_export_pmap).

Playback contract the descriptor encodes (phase = rate_deg_per_sec * t):
 mode 0 SPIN angle = phase (unbounded, wrap 360; amplitude unused)
 mode 1 SWING angle = amplitude * sin(radians(phase))

A clip that does not collapse - more than one rotation axis, a translation
channel (keyType 2/4), no time span - returns None and the exporter leaves that
atomic static rather than guessing.

Run directly to dump every map anim block:
 python sa_spin.py <extractedIsoRoot>
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_ifp

AXIS_NAME = ("X", "Y", "Z")
MODE_SPIN, MODE_SWING = 0, 1
MODE_NAME = ("spin", "swing")

_ROT_ONLY = (1, 3)          # ANP3 key types with no translation channel
# a net turn this close to a full circle is a continuous spin, not a swing
FULL_TURN_DEG = 300.0
# a swing smaller than this is invisible - do not spend a model split on it
MIN_SWING_DEG = 0.5
# the off-axis quat components must stay this far below the driving one
AXIS_PURITY = 0.02


def _keys(seq):
    """[(qx, qy, qz, qw, t_seconds)] with the on-disk conjugate undone."""
    kf, st, kt = seq["kf"], seq["stride"], seq["keyType"]
    out = []
    for i in range(seq["numFrames"]):
        if kt == 3:
            qx, qy, qz, qw, t = struct.unpack_from("<5h", kf, i * st)
            out.append((-qx / 4096.0, -qy / 4096.0, -qz / 4096.0, qw / 4096.0, t / 60.0))
        elif kt == 1:
            qx, qy, qz, qw, t = struct.unpack_from("<5f", kf, i * st)
            out.append((-qx, -qy, -qz, qw, t))
    return out


def reduce_sequence(seq):
    """One IFP sequence -> (axis, mode, rate_deg_per_sec, amplitude_deg), or None."""
    if seq["keyType"] not in _ROT_ONLY or seq["numFrames"] < 2:
        return None
    ks = _keys(seq)
    if len(ks) < 2:
        return None
    period = ks[-1][4] - ks[0][4]
    if period <= 1e-3:
        return None

    # single driving axis? (the vector part must live on one component)
    mag = [sum(abs(k[a]) for k in ks) for a in range(3)]
    axis = max(range(3), key=lambda a: mag[a])
    if mag[axis] <= 1e-6:
        return None
    if sum(mag) - mag[axis] > AXIS_PURITY * mag[axis]:
        return None                      # tumbling / multi-axis: not our shape

    # continuous angle about that axis; the quaternion double cover repeats every
    # 720 deg of theta, so unwrap on that period to follow a multi-turn clip.
    thetas = []
    prev = 0.0
    for k in ks:
        raw = 2.0 * math.degrees(math.atan2(k[axis], k[3]))
        while raw - prev > 360.0:
            raw -= 720.0
        while raw - prev < -360.0:
            raw += 720.0
        thetas.append(raw)
        prev = raw

    net = thetas[-1] - thetas[0]
    if abs(net) >= FULL_TURN_DEG:
        return (axis, MODE_SPIN, net / period, 0.0)
    amp = max(abs(t) for t in thetas)     # deviation from the rest pose
    if amp < MIN_SWING_DEG:
        return None
    return (axis, MODE_SWING, 360.0 / period, amp)


def block_spins(ifp_blob, dff_name):
    """{frame_name_lower: descriptor} for one model out of its anim block."""
    try:
        pkg = sa_ifp.decode(ifp_blob)
    except Exception:
        return {}
    want = dff_name.lower()
    out = {}
    for a in pkg["anims"]:
        if a["name"].lower() != want:
            continue
        for s in a["seqs"]:
            d = reduce_sequence(s)
            if d:
                out[s["name"].lower()] = d
        break
    return out


def make_resolver(img, verbose=False):
    """Build the `spin_resolver(anim_block, dff_name)` callback sa_export_pmap
 takes: reads <animBlock>.ifp out of gta3.img (cached) and reduces the clip."""
    have = set(n.lower() for n in img.names())
    cache = {}

    def resolve(anim_block, dff_name):
        key = (anim_block or "").lower() + ".ifp"
        if key not in cache:
            cache[key] = img.extract(key) if key in have else None
        blob = cache[key]
        if blob is None:
            return {}
        spins = block_spins(blob, dff_name)
        if verbose and spins:
            for nm, (ax, mode, rate, amp) in sorted(spins.items()):
                print("  spin %-24s %s %-5s %+8.3f deg/s amp %.1f"
                      % (nm, AXIS_NAME[ax], MODE_NAME[mode], rate, amp))
        return spins

    return resolve


def _main(argv):
    root = argv[0] if argv else os.environ.get("SA_ROOT", "")
    if not root:
        print(__doc__)
        return 2
    sys.path.insert(0, os.environ.get("GVCS_ROOT", ""))
    from gvcslib import sa_ide
    from gvcslib.sa_img import SaImg
    img = SaImg(os.environ.get("SA_GTA3_IMG", root + "/models/gta3.img"))
    ide = sa_ide.parse_maps(root + "/DATA")
    have = set(n.lower() for n in img.names())
    n_ok = n_no = 0
    for mid, d in sorted(ide.items()):
        if d.section != "anim":
            continue
        key = (d.anim_block or "").lower() + ".ifp"
        if key not in have:
            print("%-6d %-24s block %-12s NO IFP" % (mid, d.dff, d.anim_block))
            continue
        pkg = sa_ifp.decode(img.extract(key))
        clip = next((a for a in pkg["anims"] if a["name"].lower() == d.dff.lower()), None)
        if not clip:
            print("%-6d %-24s block %-12s no clip" % (mid, d.dff, d.anim_block))
            continue
        for s in clip["seqs"]:
            r = reduce_sequence(s)
            if r:
                n_ok += 1
                ax, mode, rate, amp = r
                print("%-6d %-24s %-24s %s %-5s %+8.3f deg/s amp %5.1f  (type%d n%d)"
                      % (mid, d.dff, s["name"], AXIS_NAME[ax], MODE_NAME[mode],
                         rate, amp, s["keyType"], s["numFrames"]))
            else:
                n_no += 1
                print("%-6d %-24s %-24s  - not reducible (type%d n%d)"
                      % (mid, d.dff, s["name"], s["keyType"], s["numFrames"]))
    print("reduced %d sequences, %d rejected" % (n_ok, n_no))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
