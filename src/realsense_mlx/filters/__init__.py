"""Depth processing filters for realsense-mlx.

All five RS2-equivalent post-processing filters, the pipeline combiner,
the joint bilateral filter, and the high-quality depth enhancer are
exposed at this level.

Example
-------
>>> from realsense_mlx.filters import DepthPipeline, PipelineConfig
>>> from realsense_mlx.filters import (
...     DecimationFilter,
...     DisparityTransform,
...     SpatialFilter,
...     TemporalFilter,
...     HoleFillingFilter,
...     BilateralFilter,
...     DepthEnhancer,
...     DepthEnhancerConfig,
... )
"""

from realsense_mlx.filters.bilateral import BilateralFilter
from realsense_mlx.filters.decimation import DecimationFilter
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.filters.enhancement import DepthEnhancer, DepthEnhancerConfig
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
from realsense_mlx.filters.spatial import SpatialFilter
from realsense_mlx.filters.temporal import TemporalFilter

__all__ = [
    "BilateralFilter",
    "DecimationFilter",
    "DepthEnhancer",
    "DepthEnhancerConfig",
    "DisparityTransform",
    "HoleFillingFilter",
    "SpatialFilter",
    "TemporalFilter",
    "PipelineConfig",
    "DepthPipeline",
]
