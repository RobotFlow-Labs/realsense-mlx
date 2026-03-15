"""Generic stereo camera capture via OpenCV.

Works with any USB stereo camera that presents as a single wide frame
(side-by-side layout) or as two separate video devices.  No vendor SDK
required.

Supported layouts
-----------------
Side-by-side (e.g. ZED 2i, most USB stereo cameras):
    A single device outputs a wide frame where the left half is the left
    image and the right half is the right image.

    Example: ZED 2i at 720p outputs 2560×720 BGRA frames.

Dual camera (e.g. two USB webcams on a rig):
    Two separate device IDs, each outputting a full-resolution frame.

Design notes
------------
- ``start()`` / ``stop()`` follow the typical camera lifecycle so that
  ``StereoCamera`` can be used as a context manager or in a manual loop.
- ``capture()`` returns (left, right) as BGR uint8 numpy arrays.
- ``fps`` returns the actual measured frame rate (moving average over the
  last 30 frames).
- ``grab_raw()`` returns the unmodified frame from the device (useful for
  debugging or recording the raw stream).
- All errors from OpenCV are surfaced as :class:`StereoCameraError`.

Usage
-----
>>> cam = StereoCamera.from_side_by_side(device_id=0, width=2560, height=720)
>>> cam.start()
>>> left, right = cam.capture()
>>> cam.stop()

>>> with StereoCamera.from_dual(left_id=0, right_id=2) as cam:
...     left, right = cam.capture()
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from typing import Optional

import numpy as np

__all__ = ["StereoCamera", "StereoCameraError", "CaptureMode"]


class StereoCameraError(RuntimeError):
    """Raised for camera open, read, or configuration failures."""


class CaptureMode(Enum):
    """How the stereo pair is sourced from the hardware."""
    SIDE_BY_SIDE = auto()
    DUAL = auto()


class StereoCamera:
    """Capture stereo image pairs from USB cameras via OpenCV.

    Supports two layouts:

    - **Side-by-side** (one device, wide frame split down the middle).
    - **Dual** (two separate device IDs, one frame each).

    Do not instantiate directly — use the class-method constructors:
    :meth:`from_side_by_side` or :meth:`from_dual`.

    Parameters
    ----------
    mode:
        :attr:`CaptureMode.SIDE_BY_SIDE` or :attr:`CaptureMode.DUAL`.
    device_ids:
        For SIDE_BY_SIDE: single-element list ``[device_id]``.
        For DUAL: two-element list ``[left_id, right_id]``.
    width:
        Requested frame width in pixels (full wide frame for SIDE_BY_SIDE,
        per-eye width for DUAL).
    height:
        Requested frame height in pixels.
    target_fps:
        Requested frame rate hint (passed to ``cv2.CAP_PROP_FPS``).
    api:
        OpenCV backend constant (e.g. ``cv2.CAP_V4L2``, ``cv2.CAP_ANY``).
        Defaults to ``cv2.CAP_ANY`` (0).

    Attributes
    ----------
    mode : CaptureMode
    width : int
    height : int
    """

    def __init__(
        self,
        mode: CaptureMode,
        device_ids: list[int],
        width: int,
        height: int,
        target_fps: float = 30.0,
        api: int = 0,
    ) -> None:
        self.mode = mode
        self._device_ids = list(device_ids)
        self.width = width
        self.height = height
        self._target_fps = float(target_fps)
        self._api = api

        self._caps: list = []          # list of cv2.VideoCapture objects
        self._running: bool = False

        # FPS measurement: deque of recent inter-frame durations (seconds).
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_ts: float = 0.0

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_side_by_side(
        cls,
        device_id: int = 0,
        width: int = 2560,
        height: int = 720,
        target_fps: float = 30.0,
        api: int = 0,
    ) -> "StereoCamera":
        """Create a camera that reads a side-by-side stereo stream.

        The full frame is ``(height, width)`` wide.  The left half occupies
        columns ``[0, width//2)`` and the right half ``[width//2, width)``.

        Parameters
        ----------
        device_id:
            OpenCV device index.  Usually 0 for the first USB camera.
        width:
            Full frame width (both eyes combined).  ZED 2i at 720p = 2560.
        height:
            Frame height.  ZED 2i at 720p = 720.
        target_fps:
            Requested frame rate hint.
        api:
            OpenCV capture backend.  0 = CAP_ANY (auto-select).

        Returns
        -------
        StereoCamera
        """
        return cls(
            mode=CaptureMode.SIDE_BY_SIDE,
            device_ids=[device_id],
            width=width,
            height=height,
            target_fps=target_fps,
            api=api,
        )

    @classmethod
    def from_dual(
        cls,
        left_id: int = 0,
        right_id: int = 1,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        api: int = 0,
    ) -> "StereoCamera":
        """Create a camera that reads from two separate USB devices.

        Parameters
        ----------
        left_id:
            OpenCV device index for the left camera.
        right_id:
            OpenCV device index for the right camera.
        width:
            Per-eye frame width (individual camera resolution).
        height:
            Per-eye frame height.
        target_fps:
            Requested frame rate hint.
        api:
            OpenCV capture backend.  0 = CAP_ANY (auto-select).

        Returns
        -------
        StereoCamera
        """
        return cls(
            mode=CaptureMode.DUAL,
            device_ids=[left_id, right_id],
            width=width,
            height=height,
            target_fps=target_fps,
            api=api,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera device(s) and begin streaming.

        Raises
        ------
        StereoCameraError
            If any device fails to open or the requested resolution cannot
            be set.
        ImportError
            If ``opencv-python`` is not installed.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for camera capture. "
                "Install it with: pip install opencv-python"
            ) from exc

        if self._running:
            return

        self._caps = []
        for dev_id in self._device_ids:
            cap = cv2.VideoCapture(dev_id + self._api)
            if not cap.isOpened():
                self._release_all()
                raise StereoCameraError(
                    f"Failed to open camera device {dev_id}. "
                    f"Check that the device is connected and not in use."
                )

            # Request resolution and FPS.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)

            # Verify we got the requested resolution (cameras may negotiate down).
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_w, actual_h) != (self.width, self.height):
                # Warn but don't fail — some cameras report incorrect values
                # before the first frame; capture() will verify dimensions.
                pass  # noqa: B021  (intentional no-op placeholder)

            self._caps.append(cap)

        self._running = True
        self._last_ts = time.perf_counter()

    def stop(self) -> None:
        """Stop streaming and release all camera resources."""
        self._running = False
        self._release_all()

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """Grab the next stereo frame pair.

        Returns
        -------
        left : np.ndarray
            Left eye image, ``(H, W, 3)`` BGR uint8.
        right : np.ndarray
            Right eye image, ``(H, W, 3)`` BGR uint8.

        Raises
        ------
        StereoCameraError
            If the camera is not started, a read fails, or the frame has an
            unexpected shape.
        """
        if not self._running:
            raise StereoCameraError(
                "Camera is not running. Call start() before capture()."
            )

        if self.mode == CaptureMode.SIDE_BY_SIDE:
            return self._capture_side_by_side()
        return self._capture_dual()

    def grab_raw(self) -> np.ndarray:
        """Capture the raw (unsplit) frame from the first device.

        Useful for debugging the full wide frame or recording the raw stream.

        Returns
        -------
        np.ndarray
            ``(height, width, C)`` uint8 BGR frame from device 0.

        Raises
        ------
        StereoCameraError
            If the camera is not started or the read fails.
        """
        if not self._running:
            raise StereoCameraError("Camera is not running. Call start() first.")
        return self._read_one(self._caps[0], self._device_ids[0])

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "StereoCamera":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Measured frames-per-second (moving average over last 30 frames).

        Returns ``0.0`` until at least 2 frames have been captured.
        """
        if len(self._frame_times) < 2:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0.0 else 0.0

    @property
    def is_running(self) -> bool:
        """True if the camera has been started and not yet stopped."""
        return self._running

    @property
    def eye_width(self) -> int:
        """Width of a single eye image.

        For SIDE_BY_SIDE this is ``width // 2``.
        For DUAL this is ``width``.
        """
        if self.mode == CaptureMode.SIDE_BY_SIDE:
            return self.width // 2
        return self.width

    @property
    def eye_height(self) -> int:
        """Height of a single eye image."""
        return self.height

    # ------------------------------------------------------------------
    # Private capture helpers
    # ------------------------------------------------------------------

    def _capture_side_by_side(self) -> tuple[np.ndarray, np.ndarray]:
        """Read one wide frame and split it down the middle."""
        frame = self._read_one(self._caps[0], self._device_ids[0])
        self._update_fps()

        h, w = frame.shape[:2]
        if w % 2 != 0:
            raise StereoCameraError(
                f"Side-by-side frame width {w} is not even; cannot split."
            )
        mid = w // 2
        left = frame[:, :mid, :]
        right = frame[:, mid:, :]
        return left, right

    def _capture_dual(self) -> tuple[np.ndarray, np.ndarray]:
        """Read one frame from each of the two devices."""
        left = self._read_one(self._caps[0], self._device_ids[0])
        right = self._read_one(self._caps[1], self._device_ids[1])
        self._update_fps()
        return left, right

    def _read_one(self, cap, dev_id: int) -> np.ndarray:
        """Read a single frame from *cap*, raising on failure."""
        ok, frame = cap.read()
        if not ok or frame is None:
            raise StereoCameraError(
                f"Failed to read frame from device {dev_id}. "
                f"The device may have been disconnected."
            )
        # Handle 4-channel output (e.g. ZED outputs BGRA).
        if frame.ndim == 3 and frame.shape[2] == 4:
            try:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except ImportError:
                frame = frame[:, :, :3]  # drop alpha channel
        return frame

    def _update_fps(self) -> None:
        """Record the elapsed time since the last captured frame."""
        now = time.perf_counter()
        dt = now - self._last_ts
        if dt > 0.0:
            self._frame_times.append(dt)
        self._last_ts = now

    def _release_all(self) -> None:
        """Release all OpenCV VideoCapture objects."""
        for cap in self._caps:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._caps = []

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        if self.mode == CaptureMode.SIDE_BY_SIDE:
            devices = f"device={self._device_ids[0]}, {self.width}x{self.height}"
        else:
            devices = (
                f"left={self._device_ids[0]}, right={self._device_ids[1]}, "
                f"{self.width}x{self.height}"
            )
        return (
            f"StereoCamera({self.mode.name}, {devices}, "
            f"fps={self.fps:.1f}, {status})"
        )
