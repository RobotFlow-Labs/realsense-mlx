"""Robotics-specific processing modules for realsense-mlx.

Provides utilities tailored for robot perception pipelines running on
Apple Silicon with MLX acceleration:

- :mod:`~realsense_mlx.robotics.occupancy` — 2D bird's-eye occupancy
  grids from depth frames (suitable for A*/RRT path planning).
- :mod:`~realsense_mlx.robotics.obstacles` — near-field obstacle
  detection with bounding boxes and free-path width estimation.

Quick start
-----------
>>> from realsense_mlx.robotics import OccupancyGridGenerator, ObstacleDetector
>>> from realsense_mlx.geometry.intrinsics import CameraIntrinsics
>>> import mlx.core as mx
>>>
>>> intr = CameraIntrinsics(640, 480, 318.8, 239.5, 383.7, 383.7)
>>> depth = mx.full((480, 640), 1000, dtype=mx.uint16)
>>>
>>> occ_gen = OccupancyGridGenerator(grid_size=(200, 200), cell_size_m=0.05)
>>> result = occ_gen.generate(depth, intr, depth_scale=0.001)
>>> result.grid.shape
(200, 200)
>>>
>>> detector = ObstacleDetector(min_distance_m=0.2, max_distance_m=3.0)
>>> obs = detector.detect(depth, intr, depth_scale=0.001)
>>> obs.closest_distance_m
1.0
"""

from realsense_mlx.robotics.occupancy import (
    OccupancyGrid,
    OccupancyGridGenerator,
    FREE,
    OCCUPIED,
    UNKNOWN,
)
from realsense_mlx.robotics.obstacles import (
    ObstacleDetector,
    ObstacleResult,
)

__all__ = [
    # Occupancy grid
    "OccupancyGridGenerator",
    "OccupancyGrid",
    "FREE",
    "OCCUPIED",
    "UNKNOWN",
    # Obstacle detection
    "ObstacleDetector",
    "ObstacleResult",
]
