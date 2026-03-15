"""Geometry module — camera models, distortion, point cloud, alignment.

Public API
----------
CameraIntrinsics
    Pinhole camera parameters with optional distortion coefficients.
CameraExtrinsics
    Rigid-body transform between two camera frames.
PointCloudGenerator
    MLX-accelerated depth-to-point-cloud deprojection.
Aligner
    Depth-colour frame alignment via 3-D reprojection.
"""

from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator
from realsense_mlx.geometry.align import Aligner

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "PointCloudGenerator",
    "Aligner",
]
