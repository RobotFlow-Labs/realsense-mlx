"""Depth processing filters for realsense-mlx.

All five RS2-equivalent post-processing filters plus the pipeline combiner
are exposed at this level.

Example
-------
>>> from realsense_mlx.filters import DepthPipeline, PipelineConfig
>>> from realsense_mlx.filters import (
...     DecimationFilter,
...     DisparityTransform,
...     SpatialFilter,
...     TemporalFilter,
...     HoleFillingFilter,
... )
"""

from realsense_mlx.filters.decimation import DecimationFilter
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
from realsense_mlx.filters.spatial import SpatialFilter
from realsense_mlx.filters.temporal import TemporalFilter

__all__ = [
    "DecimationFilter",
    "DisparityTransform",
    "HoleFillingFilter",
    "SpatialFilter",
    "TemporalFilter",
    "PipelineConfig",
    "DepthPipeline",
]
