"""Shared memory frame transport for inter-process depth streaming.

Protocol overview
-----------------
A single named shared-memory block contains:

  - A 64-byte **header** at byte offset 0.
  - Two **frame slots** (double buffer) immediately following the header.

Header layout (little-endian, packed struct)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offset  Size  Type    Field
------  ----  ------  ----
 0       4    uint32  magic       (b"RMLX" = 0x584C4D52)
 4       4    uint32  version     (currently 1)
 8       8    uint64  seq         monotonically increasing write counter
16       4    uint32  width
20       4    uint32  height
24       4    uint32  channels
28       4    uint32  dtype_code  (numpy dtype item-size in bytes, 1=uint8, 2=uint16, 4=float32)
32       8    uint64  timestamp_ns  nanoseconds since epoch at last write
40       4    uint32  data_offset   byte offset of slot 0 from start of shm
44       4    uint32  data_size     total size of ONE frame slot in bytes
48      16    bytes   reserved

Total: 64 bytes.

Seqlock protocol
~~~~~~~~~~~~~~~~~
Writers increment ``seq`` by 1 **before** writing (seq becomes odd →
"write in progress"), then increment it again after writing (seq becomes
even → "write complete").  Readers spin until seq is even, note the
value, copy the active slot, then verify seq has not changed.

Double-buffer slot selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Active slot (which readers use): ``seq % 2`` determines which slot the
**previous** completed write landed in.  Specifically, the slot written
in the most recently completed write is ``(seq // 2) % 2``.  The writer
always writes to ``(seq // 2 + 1) % 2`` (the slot that is *not* the
current read slot) before publishing the new seq.

This guarantees that a reader can copy the active slot without the
writer touching it during the copy — as long as the reader copies the
slot consistent with the seq value it read before and after.

Memory layout
~~~~~~~~~~~~~
::

    [0      .. 63]    Header (64 bytes)
    [64     .. 64+frame_bytes-1]   Slot 0
    [64+frame_bytes .. 64+2*frame_bytes-1]  Slot 1

Total shm size: 64 + 2 * frame_bytes bytes.

Thread / signal safety
~~~~~~~~~~~~~~~~~~~~~~
``ShmFrameWriter.write`` and ``ShmFrameReader.read`` are **not**
re-entrant from the same process — use one writer and one reader per
named block.  Cross-process safety is provided by the seqlock.

Limitations
~~~~~~~~~~~
- No blocking wait: ``read()`` returns immediately, returning the last
  frame and its sequence number.  Consumers should poll and compare seq.
- Maximum frame size: limited only by available shared memory (typically
  several GB on macOS).
- The shared memory block is unlinked when the *writer* calls
  ``close(unlink=True)`` (default).  Readers must call ``close()``
  before or after the writer; either order is safe.
"""

from __future__ import annotations

import atexit
import mmap
import struct
import time
from multiprocessing import shared_memory

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC = 0x584C4D52          # b"RMLX" as uint32 little-endian
_VERSION = 1
_HEADER_SIZE = 64            # bytes, must be a multiple of 8 for alignment
# Single authoritative struct format used by both _pack_header and _unpack_header.
# Fields: magic(I) version(I) seq(Q) width(I) height(I) channels(I)
#         dtype_code(I) timestamp_ns(Q) data_offset(I) data_size(I) reserved(16s)
# Sizes:  4 + 4 + 8 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + 16 = 64 bytes.
# All integer fields are unsigned (I/Q) — data_size was incorrectly signed (i) before.
_HEADER_FMT = "<IIQIIIIQII16s"

# Dtype code → numpy dtype mapping
_CODE_TO_DTYPE: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.float32),
    8: np.dtype(np.float64),
}
_DTYPE_TO_CODE: dict[np.dtype, int] = {v: k for k, v in _CODE_TO_DTYPE.items()}


def _dtype_code(dtype: np.dtype) -> int:
    """Return the protocol dtype code for a numpy dtype."""
    dt = np.dtype(dtype)
    code = _DTYPE_TO_CODE.get(dt)
    if code is None:
        raise ValueError(
            f"Unsupported dtype {dt}. Supported: uint8, uint16, float32, float64."
        )
    return code


def _frame_bytes(width: int, height: int, channels: int, dtype: np.dtype) -> int:
    return width * height * channels * np.dtype(dtype).itemsize


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------


def _pack_header(
    seq: int,
    width: int,
    height: int,
    channels: int,
    dtype_code: int,
    timestamp_ns: int,
    data_offset: int,
    data_size: int,
) -> bytes:
    """Pack a 64-byte header.

    The struct format ``<IIQIIIIQII B16s`` breaks down as:
      I  magic (4)
      I  version (4)
      Q  seq (8)
      I  width (4)
      I  height (4)
      I  channels (4)
      I  dtype_code (4)
      Q  timestamp_ns (8)
      I  data_offset (4)
      I  data_size (4)
      B  reserved_byte (1)
      16s reserved_blob (16)
    Total = 4+4+8+4+4+4+4+8+4+4+1+16 = 65 bytes — we need exactly 64.

    Correct layout using pad bytes:
      I  magic (4)
      I  version (4)
      Q  seq (8)
      I  width (4)
      I  height (4)
      I  channels (4)
      I  dtype_code (4)
      Q  timestamp_ns (8)
      I  data_offset (4)
      I  data_size (4)
      16s reserved (16)
    Total = 4+4+8+4+4+4+4+8+4+4+16 = 64 bytes.
    """
    assert struct.calcsize(_HEADER_FMT) == _HEADER_SIZE, (
        f"Header struct size {struct.calcsize(_HEADER_FMT)} != {_HEADER_SIZE}"
    )
    return struct.pack(
        _HEADER_FMT,
        _MAGIC,
        _VERSION,
        seq,
        width,
        height,
        channels,
        dtype_code,
        timestamp_ns,
        data_offset,
        data_size,
        b"\x00" * 16,
    )


def _unpack_header(buf: bytes) -> dict:
    """Unpack a 64-byte header buffer into a dict."""
    (
        magic,
        version,
        seq,
        width,
        height,
        channels,
        dtype_code,
        timestamp_ns,
        data_offset,
        data_size,
        _reserved,
    ) = struct.unpack_from(_HEADER_FMT, buf, 0)

    if magic != _MAGIC:
        raise ValueError(
            f"Bad magic: expected 0x{_MAGIC:08X}, got 0x{magic:08X}. "
            "Is this a realsense-mlx shm block?"
        )
    if version != _VERSION:
        raise ValueError(f"Unsupported protocol version {version} (expected {_VERSION})")

    return {
        "seq": seq,
        "width": width,
        "height": height,
        "channels": channels,
        "dtype_code": dtype_code,
        "timestamp_ns": timestamp_ns,
        "data_offset": data_offset,
        "data_size": data_size,
    }


# ---------------------------------------------------------------------------
# ShmFrameWriter
# ---------------------------------------------------------------------------


class ShmFrameWriter:
    """Write frames to a named shared-memory block.

    The block is created (or re-created) on construction.  If a block
    with the same name already exists it is unlinked first so the size
    is always correct.

    Parameters
    ----------
    name     : POSIX shm name (without leading ``/``).
    width    : Frame width in pixels.
    height   : Frame height in pixels.
    channels : Number of channels (1 for depth, 3 for RGB).
    dtype    : Element dtype — ``np.uint16`` (default), ``np.uint8``,
               ``np.float32``, or ``np.float64``.

    Examples
    --------
    >>> import numpy as np
    >>> writer = ShmFrameWriter("test_depth", 64, 48, dtype=np.uint16)
    >>> frame = np.zeros((48, 64), dtype=np.uint16)
    >>> writer.write(frame)
    >>> writer.close()
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        channels: int = 1,
        dtype: type | np.dtype = np.uint16,
    ) -> None:
        if width <= 0 or height <= 0 or channels <= 0:
            raise ValueError(
                f"width, height, and channels must all be > 0; "
                f"got width={width}, height={height}, channels={channels}"
            )
        self._name = name
        self._width = width
        self._height = height
        self._channels = channels
        self._dtype = np.dtype(dtype)
        self._dtype_code = _dtype_code(self._dtype)
        self._frame_bytes = _frame_bytes(width, height, channels, self._dtype)

        # Total shm size: header + 2 slots
        self._shm_size = _HEADER_SIZE + 2 * self._frame_bytes

        # Try to clean up any stale block with the same name
        try:
            old = shared_memory.SharedMemory(name=name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self._shm = shared_memory.SharedMemory(
            name=name,
            create=True,
            size=self._shm_size,
        )
        self._buf: mmap.mmap = self._shm._mmap  # type: ignore[attr-defined]

        # Initialise header with seq=0 (no writes yet)
        header = _pack_header(
            seq=0,
            width=width,
            height=height,
            channels=channels,
            dtype_code=self._dtype_code,
            timestamp_ns=0,
            data_offset=_HEADER_SIZE,
            data_size=self._frame_bytes,
        )
        self._buf.seek(0)
        self._buf.write(header)
        self._buf.flush()

        self._seq: int = 0
        self._closed: bool = False

        # Register atexit handler so the shm block is cleaned up if the
        # process exits without an explicit close() call.
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, frame: np.ndarray) -> int:
        """Write ``frame`` to the double buffer and publish it.

        Uses a seqlock: seq becomes odd before the write and even after.
        Readers spin-wait until seq is even.

        Parameters
        ----------
        frame : numpy array with shape ``(height, width)`` (channels==1)
                or ``(height, width, channels)``.

        Returns
        -------
        int
            The new (even) sequence number after this write.

        Raises
        ------
        RuntimeError
            If ``close()`` has been called.
        ValueError
            If ``frame`` shape or dtype does not match the writer config.
        """
        if self._closed:
            raise RuntimeError("ShmFrameWriter is closed")

        self._validate_frame(frame)

        # Determine which slot to write (the one NOT currently being read)
        # After the write, this slot becomes the active read slot.
        next_seq = self._seq + 2
        write_slot = (next_seq // 2) % 2
        slot_offset = _HEADER_SIZE + write_slot * self._frame_bytes

        # --- Seqlock: announce write in progress (odd seq) ---
        odd_seq = self._seq + 1
        self._write_seq(odd_seq)

        # --- Write frame data ---
        ts_ns = time.time_ns()
        flat = frame.reshape(-1).view(np.uint8)  # raw bytes
        self._buf.seek(slot_offset)
        self._buf.write(flat.tobytes())

        # --- Update header fields (seq still odd → readers won't read yet) ---
        header = _pack_header(
            seq=odd_seq,
            width=self._width,
            height=self._height,
            channels=self._channels,
            dtype_code=self._dtype_code,
            timestamp_ns=ts_ns,
            data_offset=_HEADER_SIZE,
            data_size=self._frame_bytes,
        )
        self._buf.seek(0)
        self._buf.write(header)

        # --- Seqlock: publish (even seq) ---
        self._seq = next_seq
        self._write_seq(self._seq)
        self._buf.flush()

        return self._seq

    def close(self, unlink: bool = True) -> None:
        """Release the shared-memory mapping.

        Parameters
        ----------
        unlink : If True (default) the underlying shm block is deleted
                 from the OS name-space.  Set to False if you want
                 another process to continue reading after this writer
                 exits.
        """
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        if unlink:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def seq(self) -> int:
        """Current (latest published) sequence number."""
        return self._seq

    @property
    def frame_shape(self) -> tuple[int, ...]:
        if self._channels == 1:
            return (self._height, self._width)
        return (self._height, self._width, self._channels)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_seq(self, seq: int) -> None:
        """Atomically write seq to the seq field in the header (offset 8)."""
        # struct.pack_into is not atomic in Python but for same-process
        # correctness this is fine; cross-process readers use the seqlock
        # protocol rather than relying on atomic stores.
        struct.pack_into("<Q", self._buf, 8, seq)

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame.dtype != self._dtype:
            raise ValueError(
                f"Frame dtype {frame.dtype} does not match writer dtype {self._dtype}"
            )
        expected_shape = self.frame_shape
        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame shape {frame.shape} does not match expected {expected_shape}"
            )

    def __repr__(self) -> str:
        return (
            f"ShmFrameWriter(name={self._name!r}, "
            f"{self._width}x{self._height}x{self._channels}, "
            f"dtype={self._dtype}, seq={self._seq})"
        )

    def __enter__(self) -> "ShmFrameWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Ensure the shm block is released when the writer is garbage collected."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ShmFrameReader
# ---------------------------------------------------------------------------


class ShmFrameReader:
    """Read frames from a shared-memory block created by :class:`ShmFrameWriter`.

    Parameters
    ----------
    name : POSIX shm name (must match the writer).

    Examples
    --------
    >>> reader = ShmFrameReader("test_depth")
    >>> frame, seq = reader.read()
    >>> reader.close()
    """

    _SPIN_MAX = 1000  # maximum spins before giving up on a consistent read

    def __init__(self, name: str) -> None:
        self._name = name
        self._shm = shared_memory.SharedMemory(name=name, create=False)
        self._buf: mmap.mmap = self._shm._mmap  # type: ignore[attr-defined]
        self._closed: bool = False

        # Parse header to extract frame geometry
        self._buf.seek(0)
        header_bytes = self._buf.read(_HEADER_SIZE)
        info = _unpack_header(header_bytes)

        self._width = info["width"]
        self._height = info["height"]
        self._channels = info["channels"]
        self._dtype = _CODE_TO_DTYPE.get(info["dtype_code"])
        if self._dtype is None:
            raise ValueError(
                f"Unknown dtype code {info['dtype_code']} in shm header"
            )
        self._frame_bytes = info["data_size"]

        # Validate that the shm block is large enough to hold the declared
        # geometry: header + 2 frame slots (double buffer).
        required_size = _HEADER_SIZE + 2 * self._frame_bytes
        actual_size = self._shm.size
        if actual_size < required_size:
            raise ValueError(
                f"shm block '{name}' is too small: size={actual_size} bytes, "
                f"need at least {required_size} bytes "
                f"(header={_HEADER_SIZE} + 2 * frame={self._frame_bytes})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> tuple[np.ndarray, int]:
        """Read the latest complete frame.

        Implements the seqlock read protocol:
        1. Spin until seq is even (no write in progress).
        2. Record seq_before.
        3. Determine active slot and copy frame bytes.
        4. Read seq_after.  If seq_after != seq_before, retry.

        Returns
        -------
        frame : numpy array with the shape and dtype from the header.
        seq   : Even sequence number; 0 means no frame has been written yet.

        Raises
        ------
        RuntimeError
            If ``close()`` has been called or if a consistent read cannot
            be obtained after ``_SPIN_MAX`` retries.
        """
        if self._closed:
            raise RuntimeError("ShmFrameReader is closed")

        for attempt in range(self._SPIN_MAX):
            # After 10 failed spin attempts, sleep 100 µs to avoid busy-
            # burning the CPU while the writer is in a long critical section.
            if attempt >= 10:
                time.sleep(100e-6)

            seq_before = self._read_seq()

            # Wait for an even sequence (writer not in progress)
            if seq_before % 2 != 0:
                continue

            if seq_before == 0:
                # No writes yet — return zeros
                shape = self._frame_shape()
                return np.zeros(shape, dtype=self._dtype), 0

            # Active read slot: (seq_before // 2 - 1) % 2
            # The completed write that produced seq_before wrote into slot
            # (seq_before // 2) % 2.
            active_slot = (seq_before // 2) % 2
            slot_offset = _HEADER_SIZE + active_slot * self._frame_bytes

            self._buf.seek(slot_offset)
            raw = self._buf.read(self._frame_bytes)

            seq_after = self._read_seq()

            if seq_after == seq_before:
                # Consistent read
                frame = np.frombuffer(raw, dtype=self._dtype).reshape(
                    self._frame_shape()
                ).copy()  # copy so the array owns its memory
                return frame, seq_before

            # Sequence changed during read — retry
            continue

        raise RuntimeError(
            f"Could not obtain a consistent read after {self._SPIN_MAX} attempts. "
            "The writer may be writing too fast or the shm block is corrupt."
        )

    def read_header(self) -> dict:
        """Return the current header fields as a dict.

        Useful for introspection without reading a full frame.
        """
        if self._closed:
            raise RuntimeError("ShmFrameReader is closed")
        self._buf.seek(0)
        raw = self._buf.read(_HEADER_SIZE)
        return _unpack_header(raw)

    def close(self) -> None:
        """Release the shared-memory mapping (does not unlink)."""
        if self._closed:
            return
        self._closed = True
        self._shm.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_seq(self) -> int:
        """Read the sequence counter from the header (offset 8)."""
        self._buf.seek(8)
        raw = self._buf.read(8)
        return struct.unpack_from("<Q", raw, 0)[0]

    def _frame_shape(self) -> tuple[int, ...]:
        if self._channels == 1:
            return (self._height, self._width)
        return (self._height, self._width, self._channels)

    def __repr__(self) -> str:
        return (
            f"ShmFrameReader(name={self._name!r}, "
            f"{self._width}x{self._height}x{self._channels}, "
            f"dtype={self._dtype})"
        )

    def __enter__(self) -> "ShmFrameReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
