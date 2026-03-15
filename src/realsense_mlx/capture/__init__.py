"""Capture module — wraps pyrealsense2 and converts frames to MLX arrays.

pyrealsense2 is an optional dependency (no ARM64 macOS wheels).
Importing this module is safe even without pyrealsense2 installed;
the ImportError is deferred until :class:`RealsenseCapture` or
:class:`MultiCameraCapture` is instantiated.

Example — single camera
-----------------------
::

    from realsense_mlx.capture import RealsenseCapture, CaptureConfig, CapturedFrames

    cfg = CaptureConfig(width=848, height=480, fps=30, enable_ir=False)
    with RealsenseCapture(cfg) as cap:
        frames = cap.get_frames()          # CapturedFrames
        depth_mx = frames.depth            # mx.array (H, W) uint16
        color_mx = frames.color            # mx.array (H, W, 3) uint8

Example — multiple cameras
--------------------------
::

    from realsense_mlx.capture import MultiCameraCapture

    with MultiCameraCapture() as mc:
        frames_dict = mc.get_frames()      # dict[serial → CapturedFrames]

Example — record and replay
----------------------------
::

    from realsense_mlx.capture import FrameRecorder, FramePlayer

    recorder = FrameRecorder("/tmp/recording_001")
    recorder.start(depth_intrinsics)
    recorder.add_frame(depth, color, timestamp)
    recorder.stop()

    with FramePlayer("/tmp/recording_001") as player:
        for depth, color, ts in player:
            process(depth)
"""

from realsense_mlx.capture.pipeline import (
    CaptureConfig,
    CapturedFrames,
    RealsenseCapture,
)
from realsense_mlx.capture.multi_camera import MultiCameraCapture
from realsense_mlx.capture.recorder import FrameRecorder, FramePlayer

__all__ = [
    "CaptureConfig",
    "CapturedFrames",
    "RealsenseCapture",
    "MultiCameraCapture",
    "FrameRecorder",
    "FramePlayer",
]
