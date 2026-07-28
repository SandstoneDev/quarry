"""Shared deploy helper: mirror a baked asset to EVERY build's data/ dir.

Single-file bake tools historically wrote only to the PPSSPP memstick, so a re-bake
never reached the real PSP on F: (you had to copy by hand). `mirror(path, data)` takes
one memstick data path and also writes the F: + local-deploy equivalents (mkdir on
demand). Import from any tool run as `python tools/<x>.py` (tools/ is on sys.path).
"""
import os

# (from, to) prefix rewrites: the memstick data root -> the other build data roots.
_MEMSTICK = ""
_MIRRORS = [
    "",
    "",
]


def targets(path):
    """The memstick path plus its device-side deploy mirrors (only if the path is under the
    memstick data root; otherwise just the path itself)."""
    p = path.replace("\\", "/")
    out = [p]
    if _MEMSTICK in p:
        out += [p.replace(_MEMSTICK, m) for m in _MIRRORS]
    return out


def mirror(path, data):
    """Write `data` (bytes) to `path` and its mirror build dirs. mkdir on demand; a target
    whose parent dir does not exist yet is created. Returns how many files were written."""
    n = 0
    for t in targets(path):
        try:
            d = os.path.dirname(t)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(t, "wb") as f:
                f.write(data)
            n += 1
        except OSError:
            pass
    return n


def mirror_copy(src_path, dst_path):
    """Copy an already-written file `src_path` to `dst_path` and its mirror build dirs."""
    with open(src_path, "rb") as f:
        return mirror(dst_path, f.read())
