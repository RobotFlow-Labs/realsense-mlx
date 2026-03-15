"""Capture module — wraps pyrealsense2 and converts frames to MLX arrays.

pyrealsense2 is an optional dependency (no ARM64 macOS wheels).
Importing this module is safe even without pyrealsense2 installed;
the ImportError is deferred until :class:`RealsenseCapture` is
instantiated.

Example
-------
::

    from realsense_mlx.capture import RealsenseCapture, CaptureConfig, CapturedFrames

    cfg = CaptureConfig(width=848, height=480, fps=30, enable_ir=False)
    with RealsenseCapture(cfg) as cap:
        frames = cap.get_frames()          # CapturedFrames
        depth_mx = frames.depth            # mx.array (H, W) uint16
        color_mx = frames.color            # mx.array (H, W, 3) uint8
"""

from realsense_mlx.capture.pipeline import (
    CaptureConfig,
    CapturedFrames,
    RealsenseCapture,
)

__all__ = [
    "CaptureConfig",
    "CapturedFrames",
    "RealsenseCapture",
]
