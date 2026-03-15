"""Geometry module — camera models, distortion, point cloud, alignment, meshing.

Public API
----------
CameraIntrinsics
    Pinhole camera parameters with optional distortion coefficients.
CameraExtrinsics
    Rigid-body transform between two camera frames.
PointCloudGenerator
    MLX-accelerated depth-to-point-cloud deprojection with PLY/OBJ export
    and per-vertex normal computation.
Aligner
    Depth-colour frame alignment via 3-D reprojection.
DepthMeshGenerator
    Vectorised triangle-mesh generation from organised depth point clouds.
"""

from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator
from realsense_mlx.geometry.align import Aligner
from realsense_mlx.geometry.mesh import DepthMeshGenerator

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "PointCloudGenerator",
    "Aligner",
    "DepthMeshGenerator",
]
