"""Capture pipeline wrapping pyrealsense2 with MLX processing.

Bridges the Intel RealSense SDK (pyrealsense2) to the realsense-mlx
MLX processing pipeline.  pyrealsense2 is an optional dependency — no
official ARM64 macOS wheels exist, so users must build from source.
This module raises an ImportError with actionable instructions when the
library is absent, rather than failing at import time of the whole package.

Typical usage
-------------
::

    from realsense_mlx.capture import RealsenseCapture, CaptureConfig
    from realsense_mlx.filters import DepthPipeline
    from realsense_mlx.filters.colorizer import DepthColorizer

    cfg = CaptureConfig(width=640, height=480, fps=30)
    capture = RealsenseCapture(cfg)
    pipeline = DepthPipeline()
    colorizer = DepthColorizer()

    capture.start()
    try:
        while True:
            frames = capture.get_frames()
            if frames.depth is not None:
                processed = pipeline.process(frames.depth)
                colored = colorizer.colorize(processed)
    finally:
        capture.stop()
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

try:
    import pyrealsense2 as rs  # type: ignore[import]
    HAS_RS2 = True
except ImportError:
    HAS_RS2 = False

_RS2_BUILD_INSTRUCTIONS = """\
pyrealsense2 is not installed.  No ARM64 macOS wheel is available on PyPI;
you must build from the librealsense source tree:

    git clone https://github.com/IntelRealSense/librealsense.git
    cd librealsense
    mkdir build && cd build
    cmake .. \\
        -DBUILD_PYTHON_BINDINGS=ON \\
        -DPYTHON_EXECUTABLE=$(which python3) \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DBUILD_EXAMPLES=OFF
    make -j$(sysctl -n hw.ncpu)
    sudo make install

Then add the generated .so to your PYTHONPATH or install it into the venv:
    cp wrappers/python/pyrealsense2*.so $(python3 -c "import site; print(site.getsitepackages()[0])")
"""


class CaptureConfig:
    """Camera stream configuration for :class:`RealsenseCapture`.

    Parameters
    ----------
    width:
        Requested frame width in pixels.  Default 640.
    height:
        Requested frame height in pixels.  Default 480.
    fps:
        Target frame-rate.  Default 30.
    enable_depth:
        Enable the depth stream.  Default ``True``.
    enable_color:
        Enable the RGB colour stream.  Default ``True``.
    enable_ir:
        Enable the infrared stream.  Default ``False``.

    Notes
    -----
    Not all width/height/fps combinations are supported by every RealSense
    model.  The SDK will raise an error on ``pipeline.start()`` if an
    unsupported combination is requested.  Common safe choices are
    640x480@30, 848x480@30, 1280x720@30.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,
        enable_ir: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got ({width}, {height})"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if not (enable_depth or enable_color or enable_ir):
            raise ValueError("At least one stream must be enabled.")
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.enable_ir = enable_ir

    def __repr__(self) -> str:
        streams = []
        if self.enable_depth:
            streams.append("depth")
        if self.enable_color:
            streams.append("color")
        if self.enable_ir:
            streams.append("ir")
        return (
            f"CaptureConfig({self.width}x{self.height}@{self.fps}fps, "
            f"streams={streams})"
        )


class CapturedFrames:
    """Container for a single captured frame-set as MLX arrays.

    All array members may be ``None`` if the corresponding stream was
    not enabled in :class:`CaptureConfig` or if the frame was dropped.

    Attributes
    ----------
    depth:
        ``(H, W)`` uint16 raw depth counts.  Multiply by
        :attr:`RealsenseCapture.depth_scale` to get metres.
    color:
        ``(H, W, 3)`` uint8 RGB image.
    infrared:
        ``(H, W)`` uint8 infrared image.
    timestamp:
        Frame timestamp in milliseconds (as reported by the SDK).
    frame_number:
        Monotonically increasing frame counter from the SDK.
    """

    def __init__(self) -> None:
        self.depth: mx.array | None = None      # (H, W) uint16
        self.color: mx.array | None = None      # (H, W, 3) uint8
        self.infrared: mx.array | None = None   # (H, W) uint8
        self.timestamp: float = 0.0
        self.frame_number: int = 0

    def __repr__(self) -> str:
        parts = []
        if self.depth is not None:
            parts.append(f"depth={self.depth.shape}")
        if self.color is not None:
            parts.append(f"color={self.color.shape}")
        if self.infrared is not None:
            parts.append(f"ir={self.infrared.shape}")
        parts.append(f"frame={self.frame_number}")
        return f"CapturedFrames({', '.join(parts)})"


class RealsenseCapture:
    """Capture frames from an Intel RealSense camera as MLX arrays.

    Wraps ``pyrealsense2.pipeline`` and converts each frame to an
    ``mx.array`` so it can be passed directly to :class:`DepthPipeline`,
    :class:`DepthColorizer`, or any other realsense-mlx processor.

    Parameters
    ----------
    config:
        :class:`CaptureConfig` instance.  Defaults to
        ``CaptureConfig()`` (640x480@30fps, depth+color).

    Raises
    ------
    ImportError
        If ``pyrealsense2`` is not installed.

    Examples
    --------
    ::

        from realsense_mlx.capture import RealsenseCapture, CaptureConfig
        from realsense_mlx.filters import DepthPipeline
        from realsense_mlx.filters.colorizer import DepthColorizer

        capture = RealsenseCapture(CaptureConfig())
        pipeline = DepthPipeline()
        colorizer = DepthColorizer()

        with capture:
            for _ in range(100):
                frames = capture.get_frames()
                if frames.depth is not None:
                    out = colorizer.colorize(pipeline.process(frames.depth))

    Notes
    -----
    - ``start()`` / ``stop()`` or the context-manager protocol must be used;
      ``get_frames()`` raises ``RuntimeError`` when the pipeline is not running.
    - Depth scale is populated from the active device on ``start()``.
    - Camera intrinsics are extracted on ``start()`` and cached.
    """

    def __init__(self, config: CaptureConfig | None = None) -> None:
        if not HAS_RS2:
            raise ImportError(_RS2_BUILD_INSTRUCTIONS)

        self.config = config or CaptureConfig()
        self._pipeline: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None
        self._intrinsics: dict[str, object] = {}   # stream name -> CameraIntrinsics
        self._depth_scale: float = 0.001           # default D435 scale (1 mm)
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Configure and start the RealSense pipeline.

        Populates :attr:`depth_scale` and caches intrinsics / extrinsics
        from the active stream profiles.

        Raises
        ------
        RuntimeError
            If the pipeline is already running.
        RuntimeError
            If no RealSense device is connected.
        """
        if self._running:
            raise RuntimeError("RealsenseCapture is already running.  Call stop() first.")

        rs_config = rs.config()
        cfg = self.config

        if cfg.enable_depth:
            rs_config.enable_stream(
                rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
            )
        if cfg.enable_color:
            rs_config.enable_stream(
                rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, cfg.fps
            )
        if cfg.enable_ir:
            rs_config.enable_stream(
                rs.stream.infrared, 1, cfg.width, cfg.height, rs.format.y8, cfg.fps
            )

        self._pipeline = rs.pipeline()
        try:
            self._profile = self._pipeline.start(rs_config)
        except RuntimeError as exc:
            self._pipeline = None
            self._profile = None
            raise RuntimeError(
                f"Failed to start RealSense pipeline: {exc}\n"
                "Is a RealSense camera connected and not in use by another process?"
            ) from exc

        self._running = True
        self._extract_metadata()

    def stop(self) -> None:
        """Stop the RealSense pipeline and release hardware resources.

        Safe to call even if the pipeline is not running.
        """
        if self._pipeline is not None and self._running:
            self._pipeline.stop()
        self._running = False
        self._pipeline = None
        self._profile = None

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def get_frames(self) -> CapturedFrames:
        """Block until the next frame-set arrives and return it as MLX arrays.

        Returns
        -------
        CapturedFrames
            Container with depth/color/ir as ``mx.array`` instances.

        Raises
        ------
        RuntimeError
            If the pipeline has not been started.
        """
        if not self._running or self._pipeline is None:
            raise RuntimeError(
                "Pipeline is not running.  Call start() before get_frames()."
            )

        frameset = self._pipeline.wait_for_frames()
        result = CapturedFrames()
        result.timestamp = frameset.get_timestamp()
        result.frame_number = frameset.get_frame_number()

        if self.config.enable_depth:
            depth_frame = frameset.get_depth_frame()
            if depth_frame:
                depth_np = np.asanyarray(depth_frame.get_data())
                result.depth = mx.array(depth_np)

        if self.config.enable_color:
            color_frame = frameset.get_color_frame()
            if color_frame:
                color_np = np.asanyarray(color_frame.get_data())
                result.color = mx.array(color_np)

        if self.config.enable_ir:
            ir_frame = frameset.get_infrared_frame(1)
            if ir_frame:
                ir_np = np.asanyarray(ir_frame.get_data())
                result.infrared = mx.array(ir_np)

        return result

    # ------------------------------------------------------------------
    # Intrinsics / extrinsics accessors
    # ------------------------------------------------------------------

    def get_depth_intrinsics(self):
        """Return :class:`~realsense_mlx.geometry.intrinsics.CameraIntrinsics` for the depth stream.

        Returns
        -------
        CameraIntrinsics or None
            ``None`` if the depth stream is not enabled or ``start()`` has
            not been called.
        """
        return self._intrinsics.get("depth")

    def get_color_intrinsics(self):
        """Return :class:`~realsense_mlx.geometry.intrinsics.CameraIntrinsics` for the colour stream.

        Returns
        -------
        CameraIntrinsics or None
        """
        return self._intrinsics.get("color")

    def get_depth_to_color_extrinsics(self):
        """Return :class:`~realsense_mlx.geometry.intrinsics.CameraExtrinsics` from depth to colour.

        Returns
        -------
        CameraExtrinsics or None
        """
        return self._intrinsics.get("depth_to_color_extrinsics")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def depth_scale(self) -> float:
        """Metres per raw depth count (typically 0.001 for D-series cameras)."""
        return self._depth_scale

    @property
    def is_running(self) -> bool:
        """True if the pipeline is actively streaming."""
        return self._running

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "RealsenseCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"RealsenseCapture({self.config!r}, state={state})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metadata(self) -> None:
        """Populate depth_scale and intrinsics from the active pipeline profile.

        Called once immediately after ``pipeline.start()``.  Imports
        ``CameraIntrinsics`` and ``CameraExtrinsics`` lazily so that the
        capture module can be imported without the geometry module being
        a hard dependency at module load time.
        """
        from realsense_mlx.geometry.intrinsics import (
            CameraIntrinsics,
            CameraExtrinsics,
        )

        if self._profile is None:
            return

        # ---- Depth scale ------------------------------------------------
        if self.config.enable_depth:
            depth_sensor = (
                self._profile
                .get_device()
                .first_depth_sensor()
            )
            self._depth_scale = float(depth_sensor.get_depth_scale())

        # ---- Depth intrinsics -------------------------------------------
        if self.config.enable_depth:
            try:
                depth_stream = (
                    self._profile
                    .get_stream(rs.stream.depth)
                    .as_video_stream_profile()
                )
                self._intrinsics["depth"] = CameraIntrinsics.from_rs2(
                    depth_stream.get_intrinsics()
                )
            except Exception:
                pass  # non-critical: callers must check for None

        # ---- Colour intrinsics ------------------------------------------
        if self.config.enable_color:
            try:
                color_stream = (
                    self._profile
                    .get_stream(rs.stream.color)
                    .as_video_stream_profile()
                )
                self._intrinsics["color"] = CameraIntrinsics.from_rs2(
                    color_stream.get_intrinsics()
                )
            except Exception:
                pass

        # ---- Depth → Colour extrinsics ----------------------------------
        if self.config.enable_depth and self.config.enable_color:
            try:
                depth_stream = (
                    self._profile
                    .get_stream(rs.stream.depth)
                    .as_video_stream_profile()
                )
                color_stream = (
                    self._profile
                    .get_stream(rs.stream.color)
                    .as_video_stream_profile()
                )
                rs_ext = depth_stream.get_extrinsics_to(color_stream)
                self._intrinsics["depth_to_color_extrinsics"] = (
                    CameraExtrinsics.from_rs2(rs_ext)
                )
            except Exception:
                pass
