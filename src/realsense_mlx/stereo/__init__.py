"""MLX-accelerated stereo depth estimation.

Works with ANY stereo camera (ZED, OAK-D, custom rigs) that provides
left/right image pairs.  No vendor SDK required — uses OpenCV for
capture and MLX for post-processing.

Stereo matching uses OpenCV SGBM as the matching backend, then applies
the full MLX filter pipeline for quality enhancement.

Example
-------
>>> from realsense_mlx.stereo import StereoDepthEstimator, StereoCamera
>>> cam = StereoCamera.from_side_by_side(device_id=0, width=2560, height=720)
>>> estimator = StereoDepthEstimator(baseline_mm=120.0, focal_px=700.0)
>>> cam.start()
>>> left, right = cam.capture()
>>> depth = estimator.compute(left, right)
>>> cam.stop()
"""

from realsense_mlx.stereo.depth import StereoDepthEstimator
from realsense_mlx.stereo.camera import StereoCamera

__all__ = ["StereoDepthEstimator", "StereoCamera"]
