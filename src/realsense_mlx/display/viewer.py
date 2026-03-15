"""Display viewer for RealSense MLX streams.

Renders depth and colour frames using OpenCV (``opencv-python``).
OpenCV is an optional dependency; this module raises a clear
``ImportError`` when it is absent rather than failing on package import.

Colour conversion note
----------------------
MLX colourizers produce **RGB** output (R in channel-0).
OpenCV's ``imshow`` expects **BGR**.  Every ``show*`` method performs
the channel swap automatically so callers can pass raw colorizer output
without a manual conversion step.

Usage
-----
::

    from realsense_mlx.display import RealsenseViewer

    with RealsenseViewer(title="Depth Demo") as viewer:
        while viewer.is_open():
            colored = colorizer.colorize(pipeline.process(depth))
            viewer.show_depth(colored)

Entry-point
-----------
``rs-mlx-viewer`` (defined in ``pyproject.toml``) calls :func:`main`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import mlx.core as mx

try:
    import cv2  # type: ignore[import]
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_CV2_INSTALL_HINT = (
    "opencv-python is required for the display viewer.\n"
    "Install it with:  uv pip install opencv-python\n"
    "or:               pip install opencv-python"
)


def _mx_to_bgr_uint8(frame: mx.array) -> np.ndarray:
    """Convert an MLX frame to a BGR uint8 numpy array suitable for cv2.imshow.

    Handles:
    - (H, W, 3) RGB  → BGR swap
    - (H, W, 3) BGR  → no swap (caller's responsibility; we assume RGB from
      the realsense-mlx colorizers)
    - (H, W)         → replicated to 3-channel grayscale BGR

    Parameters
    ----------
    frame:
        An ``mx.array`` that has been evaluated (or will be evaluated here).

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 BGR array.

    Raises
    ------
    ValueError
        If frame has an unsupported shape or dtype.
    """
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2-D or 3-D frame, got shape {frame.shape}"
        )

    mx.eval(frame)
    arr = np.array(frame, copy=False)

    if arr.dtype != np.uint8:
        # Gracefully coerce common types: float [0,1] → uint8; uint16 → rescale
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            # uint16 or int32: scale to 8-bit by dividing out max
            arr_max = arr.max()
            if arr_max > 0:
                arr = (arr.astype(np.float32) * 255.0 / float(arr_max)).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        # Grayscale → replicate to BGR
        bgr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 3:
        # Assume RGB from MLX colourizers → swap to BGR for OpenCV
        bgr = arr[:, :, ::-1]
    elif arr.shape[2] == 4:
        # RGBA → drop alpha, RGB→BGR
        bgr = arr[:, :, :3][:, :, ::-1]
    else:
        raise ValueError(
            f"Unsupported channel count {arr.shape[2]}; expected 1, 3, or 4."
        )

    return np.ascontiguousarray(bgr)


class RealsenseViewer:
    """Simple display viewer for depth and colour streams.

    Uses OpenCV's ``imshow`` / ``waitKey`` event loop.  The viewer
    opens a window on the first ``show*`` call.  Press ``q``, ``Q``, or
    ``Esc`` in the window, or close the window, to stop the loop
    (``is_open()`` will return ``False``).

    Parameters
    ----------
    title:
        Window title string.  Default ``"RealSense MLX"``.
    width:
        Window width hint in pixels.  Frames are resized to fit if their
        natural width differs, unless ``auto_resize=False``.  Default 1280.
    height:
        Window height hint in pixels.  Default 480.
    auto_resize:
        When ``True`` (default), frames are rescaled to the configured
        ``(width, height)`` before display.  Set to ``False`` to show
        frames at their native resolution.
    wait_ms:
        ``cv2.waitKey`` interval in milliseconds.  Lower values give
        smoother display but consume more CPU.  Default 1.

    Raises
    ------
    ImportError
        If ``opencv-python`` is not installed.

    Examples
    --------
    ::

        viewer = RealsenseViewer(title="My Demo", width=640, height=480)
        while viewer.is_open():
            viewer.show_depth(colorized_depth)        # (H, W, 3) uint8 RGB

        # Side-by-side: pass two equal-height frames
        viewer.show_side_by_side(colorized_depth, color)
    """

    def __init__(
        self,
        title: str = "RealSense MLX",
        width: int = 1280,
        height: int = 480,
        auto_resize: bool = True,
        wait_ms: int = 1,
    ) -> None:
        if not HAS_CV2:
            raise ImportError(_CV2_INSTALL_HINT)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got ({width}, {height})"
            )
        self.title = title
        self.width = width
        self.height = height
        self.auto_resize = auto_resize
        self.wait_ms = max(1, int(wait_ms))
        self._open = True
        self._window_created = False

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------

    def show(self, frame: mx.array) -> None:
        """Display a single RGB frame.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 RGB array (or 2-D grayscale).
            The RGB→BGR conversion is applied automatically.
        """
        if not self._open:
            return
        bgr = _mx_to_bgr_uint8(frame)
        bgr = self._maybe_resize(bgr)
        self._ensure_window()
        cv2.imshow(self.title, bgr)
        self._poll_events()

    def show_depth(self, depth_colorized: mx.array) -> None:
        """Display a colourized depth frame.

        Parameters
        ----------
        depth_colorized:
            ``(H, W, 3)`` uint8 RGB output from
            :meth:`~realsense_mlx.filters.colorizer.DepthColorizer.colorize`.
        """
        self.show(depth_colorized)

    def show_side_by_side(
        self,
        left: mx.array,
        right: mx.array,
        gap_px: int = 4,
    ) -> None:
        """Display two frames side-by-side in a single window.

        Both frames are rescaled to the same height (``self.height``)
        before concatenation.  A thin black gap separates them.

        Parameters
        ----------
        left:
            ``(H, W, 3)`` or ``(H, W)`` left frame (e.g. colorized depth).
        right:
            ``(H, W, 3)`` or ``(H, W)`` right frame (e.g. colour camera).
        gap_px:
            Width of the black separator bar in pixels.  Default 4.
        """
        if not self._open:
            return

        bgr_l = _mx_to_bgr_uint8(left)
        bgr_r = _mx_to_bgr_uint8(right)

        # Rescale both to self.height, preserving aspect ratio
        h_target = self.height

        def _scale_to_height(img: np.ndarray, h: int) -> np.ndarray:
            ih, iw = img.shape[:2]
            if ih == h:
                return img
            scale = h / ih
            new_w = max(1, int(iw * scale))
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)

        bgr_l = _scale_to_height(bgr_l, h_target)
        bgr_r = _scale_to_height(bgr_r, h_target)

        # Black gap separator
        gap = np.zeros((h_target, gap_px, 3), dtype=np.uint8)
        combined = np.concatenate([bgr_l, gap, bgr_r], axis=1)

        self._ensure_window()
        cv2.imshow(self.title, combined)
        self._poll_events()

    def show_grid(
        self,
        frames: list[mx.array],
        cols: int = 2,
        gap_px: int = 4,
    ) -> None:
        """Display multiple frames in a grid layout.

        Parameters
        ----------
        frames:
            List of ``mx.array`` frames (RGB or grayscale).  Up to
            ``cols * rows`` frames are displayed; extra entries are ignored.
        cols:
            Number of columns in the grid.  Default 2.
        gap_px:
            Gap in pixels between cells.  Default 4.
        """
        if not self._open or not frames:
            return

        import math
        rows = math.ceil(len(frames) / cols)
        cell_h = self.height // rows
        cell_w = self.width // cols

        gap_v = np.zeros((gap_px, self.width, 3), dtype=np.uint8)
        row_imgs = []

        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(frames):
                    bgr = _mx_to_bgr_uint8(frames[idx])
                    bgr = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                else:
                    bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_cells.append(bgr)
                if c < cols - 1:
                    row_cells.append(np.zeros((cell_h, gap_px, 3), dtype=np.uint8))
            row_imgs.append(np.concatenate(row_cells, axis=1))
            if r < rows - 1:
                row_imgs.append(gap_v)

        combined = np.concatenate(row_imgs, axis=0)
        self._ensure_window()
        cv2.imshow(self.title, combined)
        self._poll_events()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        """Return ``True`` while the viewer window is open and the user has
        not pressed the quit key.
        """
        return self._open

    def close(self) -> None:
        """Close the viewer window and release OpenCV resources."""
        self._open = False
        if self._window_created:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
            self._window_created = False

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "RealsenseViewer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self._open else "closed"
        return (
            f"RealsenseViewer(title={self.title!r}, "
            f"{self.width}x{self.height}, {state})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_window(self) -> None:
        """Create the OpenCV named window on first use."""
        if not self._window_created:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)
            self._window_created = True

    def _maybe_resize(self, bgr: np.ndarray) -> np.ndarray:
        """Optionally rescale *bgr* to ``(self.height, self.width)``."""
        if not self.auto_resize:
            return bgr
        h, w = bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            return cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return bgr

    def _poll_events(self) -> None:
        """Process pending OpenCV events and check for quit signals.

        Sets ``self._open = False`` when:
        - The user presses ``q``, ``Q``, or ``Esc``.
        - The window has been closed (``cv2.getWindowProperty`` returns -1).
        """
        key = cv2.waitKey(self.wait_ms) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # 27 = Esc
            self._open = False
            return

        # Detect window close via the X button
        try:
            prop = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
            if prop < 1.0:
                self._open = False
        except Exception:
            self._open = False


# ---------------------------------------------------------------------------
# Entry-point for ``rs-mlx-viewer`` console script
# ---------------------------------------------------------------------------


def main() -> None:
    """Minimal live viewer entry-point.

    Launches a live RealSense depth viewer.  Requires both
    ``pyrealsense2`` and ``opencv-python``.  Prints a helpful message if
    either is absent.

    This is the target of the ``rs-mlx-viewer`` console script defined in
    ``pyproject.toml``.
    """
    print("rs-mlx-viewer: starting live depth viewer")
    print("Press 'q' or Esc in the window to quit.\n")

    try:
        from realsense_mlx.capture import RealsenseCapture, CaptureConfig
    except ImportError as exc:
        print(f"Error: {exc}")
        return

    from realsense_mlx.filters import DepthPipeline
    from realsense_mlx.filters.colorizer import DepthColorizer

    cfg = CaptureConfig(width=640, height=480, fps=30, enable_color=True)
    pipeline = DepthPipeline()
    colorizer = DepthColorizer(colormap="jet", equalize=True)

    try:
        capture = RealsenseCapture(cfg)
    except ImportError as exc:
        print(f"Error: {exc}")
        return

    try:
        capture.start()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    viewer = RealsenseViewer(title="RealSense MLX Viewer", width=1280, height=480)

    frame_count = 0
    t0 = time.perf_counter()

    try:
        with viewer:
            while viewer.is_open():
                frames = capture.get_frames()
                if frames.depth is not None and frames.color is not None:
                    processed = pipeline.process(frames.depth)
                    colored = colorizer.colorize(processed)
                    viewer.show_side_by_side(colored, frames.color)
                    frame_count += 1
                    if frame_count % 60 == 0:
                        elapsed = time.perf_counter() - t0
                        fps = frame_count / elapsed
                        print(f"  {frame_count} frames  |  {fps:.1f} fps", end="\r")
    finally:
        capture.stop()
        print(f"\nDone. {frame_count} frames captured.")
