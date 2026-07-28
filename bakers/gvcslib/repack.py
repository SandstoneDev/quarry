"""Repack / build pipeline for the source console title (PSP) assets.

This module is the *write* side of the codec library: given decoded structures
(or raw payloads) it rebuilds engine-loadable container files and re-assembles
the streaming ``*.IMG`` archives.

Three things live here:

``rebuild_container(payload_bytes, *, compress=None)``
    Wrap an (inflated) container payload back into an on-disk file. zlib
    containers (``LVZ`` magic ``DLRW``, ``DTZ`` magic ``GTAG``) are re-deflated;
    raw containers (``XTX`` magic ``tex``, ``IMG`` cells) are returned verbatim.
    The byte-exact contract for zlib containers is at the *inflated payload*
    level: ``inflate(rebuild_container(p)) == p`` (the compressed bytes need not
    match the original disc image, only round-trip losslessly and stay
    engine-loadable).

``repack_img(blobs, *, total_size=None, align=0x800)``
    Re-assemble an IMG from a list of ``(offset, bytes)`` cell blobs. Each blob
    is written at its original byte offset; the engine streams 2 KiB
    (``0x800``) sectors, so offsets are expected to be page aligned, but the
    function honours whatever offsets it is given (sub-page tightly-packed
    chunks included) and simply lays each blob down at its absolute offset.
    Inter-blob gaps are zero-filled. This reproduces the original IMG byte ranges
    exactly when fed the blobs read back from those same offsets.

``open(path)`` / ``EditableContainer``
    A small edit-round-trip helper: ``c = repack.open(path)`` returns an
    :class:`EditableContainer` whose ``.payload`` you can mutate (or
    ``.replace_payload(new)``), then ``c.write(out_path)`` re-encodes the file
    with the correct compression for that container kind.

Everything reuses :mod:`gvcslib.container` for the header semantics, and the
IMG TOC is taken from :mod:`gvcslib.lvz` (streaming descriptors + texture-grid
cells, each an ``(offset, size)`` blob).
"""
from __future__ import annotations

import builtins
import zlib
from typing import Iterable, List, Optional, Sequence, Tuple

from .container import (
    Container,
    HDR_SIZE,
    MAGIC_DLRW,
    MAGIC_GTAG,
    MAGIC_TEX,
)

PAGE = 0x800  # 2 KiB streaming sector

# Container magics that ship zlib-wrapped on disc and must be re-deflated.
_ZLIB_MAGICS = frozenset((MAGIC_DLRW, MAGIC_GTAG))
# Container magics stored raw (uncompressed) on disc.
_RAW_MAGICS = frozenset((MAGIC_TEX,))


# --------------------------------------------------------------------------- #
# container rebuild                                                           #
# --------------------------------------------------------------------------- #
def _payload_magic(payload: bytes) -> int:
    if len(payload) < 4:
        raise ValueError("payload too short to contain a container magic")
    return int.from_bytes(payload[0:4], "little")


def container_is_zlib(payload: bytes) -> bool:
    """True if a payload with this magic ships zlib-compressed on disc."""
    return _payload_magic(payload) in _ZLIB_MAGICS


def rebuild_container(payload_bytes, *, compress: Optional[bool] = None,
                      level: int = 9) -> bytes:
    """Wrap an inflated container payload back into an engine-loadable file.

    ``payload_bytes`` is the *inflated* payload (it begins with the 0x20-byte
    container header). The compression decision is made from the magic:

      * ``DLRW`` (LVZ) and ``GTAG`` (DTZ) -> zlib-deflate.
      * ``tex`` (XTX) and anything else considered raw -> returned verbatim.

    Pass ``compress=True``/``False`` to override the magic-based choice.

    Contract: for zlib containers ``zlib.decompress(rebuild_container(p)) == p``
    byte-for-byte; for raw containers ``rebuild_container(p) == p``.
    """
    payload = bytes(payload_bytes)
    if compress is None:
        compress = container_is_zlib(payload)
    if compress:
        return zlib.compress(payload, level)
    return payload


# --------------------------------------------------------------------------- #
# IMG sector repack                                                           #
# --------------------------------------------------------------------------- #
def align_up(value: int, align: int = PAGE) -> int:
    """Round ``value`` up to the next multiple of ``align``."""
    if align <= 0:
        return value
    rem = value % align
    return value if rem == 0 else value + (align - rem)


def repack_img(blobs: Sequence[Tuple[int, bytes]], *,
               total_size: Optional[int] = None,
               align: int = PAGE) -> bytes:
    """Re-assemble an IMG from ``(offset, bytes)`` cell blobs.

    Each blob is laid down at its absolute ``offset``. Gaps between blobs are
    zero-filled. ``align`` documents the engine's 2 KiB streaming-sector
    expectation: the final buffer is rounded up to a multiple of ``align`` (and
    so is ``total_size`` if you pass one). Offsets themselves are written
    verbatim - the engine streams whole sectors, so well-formed IMGs keep blobs
    page aligned, but tightly-packed sub-page blobs round-trip unchanged too.

    When fed the blobs read back from an original IMG at their TOC offsets, the
    rebuilt buffer reproduces those byte ranges exactly.
    """
    items: List[Tuple[int, bytes]] = [(int(off), bytes(b)) for off, b in blobs]
    end = 0
    for off, b in items:
        if off < 0:
            raise ValueError(f"negative blob offset {off}")
        end = max(end, off + len(b))

    if total_size is not None:
        if total_size < end:
            raise ValueError(
                f"total_size {total_size} smaller than required end {end}")
        end = total_size

    end = align_up(end, align)
    out = bytearray(end)
    for off, b in items:
        out[off:off + len(b)] = b
    return bytes(out)


def img_toc(lvz) -> List[Tuple[int, int]]:
    """Return the IMG table-of-contents as a sorted, de-duplicated list of
    ``(img_offset, size)`` blobs from an :class:`gvcslib.lvz.Lvz`.

    The TOC is the union of the streaming descriptors and texture-grid cells;
    each names a byte range in the sibling ``*.IMG``. Duplicate ``(offset,
    size)`` entries (the engine can reference the same chunk from several grid
    cells) are collapsed.
    """
    seen = set()
    toc: List[Tuple[int, int]] = []
    for d in getattr(lvz, "streaming_descriptors", ()):  # StreamDescriptor
        key = (d.img_offset, d.read_size)
        if d.read_size and key not in seen:
            seen.add(key)
            toc.append(key)
    for t in getattr(lvz, "tex_grid", ()):               # TexCell
        key = (t.img_offset, t.size)
        if t.size and key not in seen:
            seen.add(key)
            toc.append(key)
    toc.sort()
    return toc


def read_img_blobs(img_path: str, toc: Iterable[Tuple[int, int]]
                   ) -> List[Tuple[int, bytes]]:
    """Read ``(offset, size)`` ranges out of an IMG file on disk.

    Returns ``(offset, bytes)`` blobs suitable for :func:`repack_img`. Reads
    only the requested ranges (the IMGs are hundreds of MB) via ``seek``.
    """
    blobs: List[Tuple[int, bytes]] = []
    with builtins.open(img_path, "rb") as f:
        for off, size in toc:
            f.seek(off)
            data = f.read(size)
            if len(data) != size:
                raise ValueError(
                    f"short read at 0x{off:x}: wanted {size}, got {len(data)}")
            blobs.append((off, data))
    return blobs


def repack_img_from_lvz(lvz, img_path: str) -> Tuple[bytes, List[Tuple[int, int]]]:
    """Read every TOC blob from ``img_path`` and repack them at the same offsets.

    Returns ``(rebuilt_region_bytes, toc)`` where ``rebuilt_region_bytes`` is a
    buffer covering offset 0 .. (last blob end, page-rounded) with each blob at
    its original offset. The covered ranges match the original IMG byte-exact.
    """
    toc = img_toc(lvz)
    blobs = read_img_blobs(img_path, toc)
    rebuilt = repack_img(blobs)
    return rebuilt, toc


# --------------------------------------------------------------------------- #
# edit round-trip helper                                                      #
# --------------------------------------------------------------------------- #
class EditableContainer:
    """A loaded container you can edit and re-write.

    ``payload`` is the inflated payload (mutable via :meth:`replace_payload`).
    :meth:`write` re-encodes with the correct compression for the container
    kind (zlib for LVZ/DTZ, raw for XTX), preserving the inflated payload
    byte-exact.
    """

    def __init__(self, payload: bytes, *, source_was_compressed: Optional[bool] = None):
        self._payload = bytes(payload)
        self.container = Container(self._payload)
        if source_was_compressed is None:
            source_was_compressed = container_is_zlib(self._payload)
        self.compressed = source_was_compressed

    @classmethod
    def from_file(cls, path: str) -> "EditableContainer":
        with builtins.open(path, "rb") as f:
            data = f.read()
        compressed = data[:2] in (b"\x78\xda", b"\x78\x9c", b"\x78\x01")
        payload = zlib.decompress(data) if compressed else data
        return cls(payload, source_was_compressed=compressed)

    @property
    def payload(self) -> bytes:
        return self._payload

    @payload.setter
    def payload(self, new_payload) -> None:
        self.replace_payload(new_payload)

    def replace_payload(self, new_payload) -> None:
        """Swap in a new inflated payload and re-parse its header."""
        self._payload = bytes(new_payload)
        self.container = Container(self._payload)

    def build(self) -> bytes:
        """Return the on-disk file bytes (compressed/raw per container kind)."""
        return rebuild_container(self._payload, compress=self.compressed)

    def write(self, path: str) -> int:
        """Write the (re-encoded) file to ``path``; return bytes written."""
        data = self.build()
        with builtins.open(path, "wb") as f:
            f.write(data)
        return len(data)


def open(path: str) -> EditableContainer:  # noqa: A001 - intentional API name
    """Open a container file for editing. See :class:`EditableContainer`."""
    return EditableContainer.from_file(path)
