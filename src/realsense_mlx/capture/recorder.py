"""Frame recording and playback for offline testing.

Saves depth/colour frames captured from a RealSense camera to disk as NumPy
``.npz`` archives, together with a JSON metadata sidecar that contains camera
intrinsics and recording configuration.  Recorded sessions can be replayed
frame-by-frame using :class:`FramePlayer`, enabling deterministic offline
processing without a physical camera.

Layout on disk
--------------
::

    <output_dir>/
        metadata.json          — intrinsics, depth_scale, frame_count, timestamps
        frame_000000.npz       — depth (and optionally colour) for frame 0
        frame_000001.npz
        ...

Each ``.npz`` file stores:

``depth``
    ``uint16`` array, shape ``(H, W)``.
``color`` *(optional)*
    ``uint8`` array, shape ``(H, W, 3)``.  Key absent when colour was not
    recorded.
``timestamp``
    Scalar ``float64``.

Example
-------
::

    from realsense_mlx.capture.recorder import FrameRecorder, FramePlayer

    # --- Recording ---
    recorder = FrameRecorder("/tmp/recording_001")
    recorder.start(depth_intrinsics, color_intrinsics, depth_scale=0.001)
    for depth, color, ts in capture_loop():
        recorder.add_frame(depth, color, ts)
    summary = recorder.stop()

    # --- Playback ---
    with FramePlayer("/tmp/recording_001") as player:
        print(player.frame_count, player.depth_scale)
        for depth, color, ts in player:
            process(depth)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.intrinsics import CameraIntrinsics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _intrinsics_to_dict(intr: CameraIntrinsics) -> dict:
    """Serialise :class:`CameraIntrinsics` to a plain JSON-safe dict."""
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy,
        "model": intr.model,
        "coeffs": list(intr.coeffs),
    }


def _intrinsics_from_dict(d: dict) -> CameraIntrinsics:
    """Reconstruct :class:`CameraIntrinsics` from a plain dict."""
    return CameraIntrinsics(
        width=int(d["width"]),
        height=int(d["height"]),
        ppx=float(d["ppx"]),
        ppy=float(d["ppy"]),
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        model=str(d["model"]),
        coeffs=[float(c) for c in d["coeffs"]],
    )


def _frame_filename(index: int) -> str:
    """Return the filename for the *index*-th frame (zero-padded to 6 digits)."""
    return f"frame_{index:06d}.npz"


# ---------------------------------------------------------------------------
# FrameRecorder
# ---------------------------------------------------------------------------


class FrameRecorder:
    """Record depth/colour frames to disk for offline testing.

    Saves frames as compressed NumPy ``.npz`` files with a JSON metadata
    sidecar (intrinsics, depth_scale, per-frame timestamps).

    Parameters
    ----------
    output_dir:
        Destination directory path (string or path-like).  Created if it
        does not already exist.

    Raises
    ------
    RuntimeError
        If :meth:`add_frame` or :meth:`stop` is called before :meth:`start`.
    """

    def __init__(self, output_dir: str) -> None:
        self._dir = Path(output_dir)
        self._depth_intrinsics: CameraIntrinsics | None = None
        self._color_intrinsics: CameraIntrinsics | None = None
        self._depth_scale: float = 0.001
        self._frame_index: int = 0
        self._timestamps: list[float] = []
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        depth_intrinsics: CameraIntrinsics,
        color_intrinsics: CameraIntrinsics | None = None,
        depth_scale: float = 0.001,
    ) -> None:
        """Begin a recording session.

        Creates the output directory (including parents) and writes a partial
        ``metadata.json`` that will be finalised on :meth:`stop`.

        Parameters
        ----------
        depth_intrinsics:
            Intrinsic parameters of the depth stream.
        color_intrinsics:
            Optional intrinsic parameters of the colour stream.
        depth_scale:
            Metres per raw depth count.  Defaults to ``0.001`` (1 mm/count),
            which is typical for D-series cameras.

        Raises
        ------
        RuntimeError
            If a recording session is already active.
        """
        if self._started:
            raise RuntimeError(
                "Recording session already active.  Call stop() first."
            )

        self._dir.mkdir(parents=True, exist_ok=True)
        self._depth_intrinsics = depth_intrinsics
        self._color_intrinsics = color_intrinsics
        self._depth_scale = float(depth_scale)
        self._frame_index = 0
        self._timestamps = []
        self._started = True

    def add_frame(
        self,
        depth: mx.array,
        color: mx.array | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Persist a single frame to disk.

        Parameters
        ----------
        depth:
            ``(H, W)`` ``uint16`` depth array.
        color:
            Optional ``(H, W, 3)`` ``uint8`` colour array.
        timestamp:
            Frame capture time in any consistent unit (e.g. milliseconds from
            the RealSense SDK, or ``time.monotonic()``).  Defaults to the
            current wall-clock time if ``None``.

        Raises
        ------
        RuntimeError
            If :meth:`start` has not been called.
        ValueError
            If *depth* does not have 2 dimensions, or *color* does not have
            3 dimensions / a last axis of size 3.
        """
        if not self._started:
            raise RuntimeError(
                "Recording not started.  Call start() before add_frame()."
            )

        # Convert MLX arrays to NumPy for persistence.
        depth_np = np.array(depth, copy=False)
        if depth_np.ndim != 2:
            raise ValueError(
                f"depth must be a 2-D array (H, W), got ndim={depth_np.ndim}"
            )

        ts = float(timestamp) if timestamp is not None else time.monotonic()
        self._timestamps.append(ts)

        arrays: dict[str, np.ndarray] = {
            "depth": depth_np,
            "timestamp": np.float64(ts),
        }

        if color is not None:
            color_np = np.array(color, copy=False)
            if color_np.ndim != 3 or color_np.shape[2] != 3:
                raise ValueError(
                    f"color must be a 3-D array (H, W, 3), got shape={color_np.shape}"
                )
            arrays["color"] = color_np

        filepath = self._dir / _frame_filename(self._frame_index)
        np.savez_compressed(str(filepath), **arrays)
        self._frame_index += 1

    def stop(self) -> dict:
        """Finalise the recording.

        Writes the complete ``metadata.json`` (including final frame count and
        timestamps) and marks the session as inactive.

        Returns
        -------
        dict
            A summary with keys:

            ``output_dir``
                Absolute path of the recording directory (str).
            ``frame_count``
                Number of frames recorded.
            ``depth_scale``
                Depth scale value used.
            ``duration_s``
                Wall-clock duration in seconds from first to last timestamp
                (``0.0`` for a single-frame recording).

        Raises
        ------
        RuntimeError
            If called before :meth:`start`.
        """
        if not self._started:
            raise RuntimeError(
                "No active recording session.  Call start() first."
            )

        metadata: dict = {
            "frame_count": self._frame_index,
            "depth_scale": self._depth_scale,
            "timestamps": self._timestamps,
            "depth_intrinsics": (
                _intrinsics_to_dict(self._depth_intrinsics)
                if self._depth_intrinsics is not None
                else None
            ),
            "color_intrinsics": (
                _intrinsics_to_dict(self._color_intrinsics)
                if self._color_intrinsics is not None
                else None
            ),
        }

        metadata_path = self._dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        duration = (
            self._timestamps[-1] - self._timestamps[0]
            if len(self._timestamps) >= 2
            else 0.0
        )

        summary = {
            "output_dir": str(self._dir.resolve()),
            "frame_count": self._frame_index,
            "depth_scale": self._depth_scale,
            "duration_s": duration,
        }

        # Reset state.
        self._started = False
        self._frame_index = 0
        self._timestamps = []

        return summary

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames written so far in the current (or most recent) session."""
        return self._frame_index

    @property
    def is_recording(self) -> bool:
        """``True`` while a session is active."""
        return self._started

    def __repr__(self) -> str:
        state = "recording" if self._started else "idle"
        return f"FrameRecorder(dir={self._dir!r}, state={state}, frames={self._frame_index})"


# ---------------------------------------------------------------------------
# FramePlayer
# ---------------------------------------------------------------------------


class FramePlayer:
    """Play back a recording produced by :class:`FrameRecorder`.

    Provides sequential frame access via :meth:`next_frame`, random-access
    seeking via :meth:`seek`, iterator protocol, and context-manager support.

    Parameters
    ----------
    recording_dir:
        Path to the directory created by :class:`FrameRecorder`.

    Raises
    ------
    FileNotFoundError
        If *recording_dir* or its ``metadata.json`` does not exist.

    Examples
    --------
    ::

        with FramePlayer("/tmp/recording_001") as player:
            print(f"{player.frame_count} frames, scale={player.depth_scale}")
            for depth, color, ts in player:
                process(depth)
    """

    def __init__(self, recording_dir: str) -> None:
        self._dir = Path(recording_dir)
        self._metadata: dict = {}
        self._cursor: int = 0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Load recording metadata from ``metadata.json``.

        Must be called before any frame access.  Idempotent — calling
        ``open()`` on an already-open player resets the cursor to frame 0.

        Raises
        ------
        FileNotFoundError
            If the recording directory or metadata file is missing.
        ValueError
            If ``metadata.json`` is malformed (missing required keys).
        """
        metadata_path = self._dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Recording metadata not found: {metadata_path}"
            )

        raw = json.loads(metadata_path.read_text())

        # Validate required keys.
        for key in ("frame_count", "depth_scale", "depth_intrinsics"):
            if key not in raw:
                raise ValueError(
                    f"metadata.json is missing required key '{key}'."
                )

        self._metadata = raw
        self._cursor = 0
        self._opened = True

    def close(self) -> None:
        """Release resources and mark the player as closed."""
        self._opened = False
        self._cursor = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Depth-stream :class:`CameraIntrinsics` from the recording.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return _intrinsics_from_dict(self._metadata["depth_intrinsics"])

    @property
    def color_intrinsics(self) -> CameraIntrinsics | None:
        """Colour-stream :class:`CameraIntrinsics`, or ``None`` if not recorded."""
        self._require_open()
        d = self._metadata.get("color_intrinsics")
        return _intrinsics_from_dict(d) if d is not None else None

    @property
    def depth_scale(self) -> float:
        """Metres per raw depth count as stored in the recording.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return float(self._metadata["depth_scale"])

    @property
    def frame_count(self) -> int:
        """Total number of frames in the recording.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return int(self._metadata["frame_count"])

    @property
    def current_index(self) -> int:
        """Index of the *next* frame that :meth:`next_frame` will return."""
        return self._cursor

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def next_frame(self) -> tuple[mx.array, mx.array | None, float]:
        """Return the next frame and advance the internal cursor.

        Returns
        -------
        tuple[mx.array, mx.array | None, float]
            ``(depth, color_or_None, timestamp)`` where:

            - *depth*: ``(H, W)`` ``uint16`` MLX array.
            - *color*: ``(H, W, 3)`` ``uint8`` MLX array, or ``None`` if
              colour was not recorded.
            - *timestamp*: Original capture timestamp (float).

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        StopIteration
            If no more frames are available.
        FileNotFoundError
            If the frame file is missing on disk.
        """
        self._require_open()

        if self._cursor >= self.frame_count:
            raise StopIteration(
                f"No more frames (cursor={self._cursor}, "
                f"frame_count={self.frame_count})."
            )

        frame_path = self._dir / _frame_filename(self._cursor)
        if not frame_path.exists():
            raise FileNotFoundError(
                f"Frame file missing: {frame_path}"
            )

        data = np.load(str(frame_path))

        depth = mx.array(data["depth"])
        color: mx.array | None = (
            mx.array(data["color"]) if "color" in data else None
        )
        timestamp = float(data["timestamp"])

        self._cursor += 1
        return depth, color, timestamp

    def has_frames(self) -> bool:
        """Return ``True`` if there are unread frames remaining.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return self._cursor < self.frame_count

    def seek(self, frame_index: int) -> None:
        """Move the cursor to the given frame index.

        Parameters
        ----------
        frame_index:
            Zero-based frame index to seek to.  Clamped to
            ``[0, frame_count]``.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        ValueError
            If *frame_index* is negative.
        """
        self._require_open()

        if frame_index < 0:
            raise ValueError(
                f"frame_index must be >= 0, got {frame_index}."
            )

        self._cursor = min(frame_index, self.frame_count)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array | None, float]]:
        """Iterate over all remaining frames from the current cursor position."""
        while self.has_frames():
            yield self.next_frame()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "FramePlayer":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError(
                "FramePlayer is not open.  Call open() (or use as context manager) first."
            )

    def __repr__(self) -> str:
        if self._opened:
            return (
                f"FramePlayer(dir={self._dir!r}, "
                f"cursor={self._cursor}/{self.frame_count})"
            )
        return f"FramePlayer(dir={self._dir!r}, closed)"
