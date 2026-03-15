"""Standard RealSense post-processing pipeline on MLX.

This module wires all five depth filters into the canonical RS2 SDK
post-processing order and exposes a single ``DepthPipeline`` class that
can be dropped into any capture loop.

Processing order
----------------
1. Decimation       — reduce resolution (less data for subsequent filters)
2. Depth→Disparity  — convert to disparity space for better filter response
3. Spatial          — edge-preserving smoothing in disparity space
4. Temporal         — EMA smoothing across frames in disparity space
5. Disparity→Depth  — convert back to depth space
6. Hole Filling     — fill remaining invalid pixels

Working in disparity space for steps 3–4 is the same approach used by the
RS2 SDK: the bilateral weights are more uniform in disparity space and
the filter produces better results near depth discontinuities.

Usage
-----
>>> import mlx.core as mx, numpy as np
>>> from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
>>>
>>> cfg = PipelineConfig(decimation_scale=2, temporal_alpha=0.4)
>>> pipeline = DepthPipeline(cfg)
>>>
>>> for raw_depth_uint16 in stream:
...     processed = pipeline.process(mx.array(raw_depth_uint16))

Temporal state
--------------
The ``TemporalFilter`` inside ``DepthPipeline`` accumulates state across
calls to ``process()``.  Call ``pipeline.reset()`` to clear it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from realsense_mlx.filters.decimation import DecimationFilter
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.spatial import SpatialFilter
from realsense_mlx.filters.temporal import TemporalFilter

__all__ = ["PipelineConfig", "DepthPipeline"]


@dataclass
class PipelineConfig:
    """Configuration for :class:`DepthPipeline`.

    All parameters mirror the corresponding RS2 SDK filter options.

    Attributes
    ----------
    decimation_scale:
        Integer downsampling factor [1, 8].  1 = no decimation.
    spatial_alpha:
        Spatial filter smoothing strength [0, 1].
    spatial_delta:
        Spatial filter depth-difference sensitivity (disparity units).
    spatial_iterations:
        Number of spatial filter passes [1, 5].
    temporal_alpha:
        Temporal EMA weight for current frame [0, 1].
    temporal_delta:
        Temporal filter change threshold.
    temporal_persistence:
        Minimum valid-frame count in last 8 frames [0, 8].
    hole_fill_mode:
        Hole filling mode (0=left, 1=farthest, 2=nearest).
    baseline_mm:
        Stereo baseline in millimetres.
    focal_px:
        Horizontal focal length in pixels.
    depth_units:
        Metres per depth count (rs2_option.DEPTH_UNITS).
    """

    decimation_scale: int = 2
    spatial_alpha: float = 0.5
    spatial_delta: float = 20.0
    spatial_iterations: int = 2
    enable_spatial: bool = True
    temporal_alpha: float = 0.4
    temporal_delta: float = 20.0
    temporal_persistence: int = 3
    enable_temporal: bool = True
    hole_fill_mode: int = 1       # HoleFillingFilter.FARTHEST
    enable_hole_fill: bool = True
    baseline_mm: float = 50.0
    focal_px: float = 383.7
    depth_units: float = 0.001


class DepthPipeline:
    """Standard RealSense post-processing pipeline on MLX.

    Parameters
    ----------
    config:
        :class:`PipelineConfig` instance.  Defaults to ``PipelineConfig()``
        (RS2 SDK defaults).

    Examples
    --------
    >>> pipeline = DepthPipeline()
    >>> out = pipeline.process(mx.array(raw_uint16_frame))
    >>> pipeline.reset()   # clear temporal state
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._build_filters()

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_filters(self) -> None:
        """Instantiate all filter objects from ``self.config``."""
        cfg = self.config

        self._decimation = DecimationFilter(scale=cfg.decimation_scale)

        self._depth_to_disp = DisparityTransform(
            baseline_mm=cfg.baseline_mm,
            focal_px=cfg.focal_px,
            depth_units=cfg.depth_units,
            to_disparity=True,
        )

        self._spatial = SpatialFilter(
            alpha=cfg.spatial_alpha,
            delta=cfg.spatial_delta,
            iterations=cfg.spatial_iterations,
            hole_fill=0,  # hole fill handled separately at end of pipeline
        )

        self._temporal = TemporalFilter(
            alpha=cfg.temporal_alpha,
            delta=cfg.temporal_delta,
            persistence=cfg.temporal_persistence,
        )

        self._disp_to_depth = DisparityTransform(
            baseline_mm=cfg.baseline_mm,
            focal_px=cfg.focal_px,
            depth_units=cfg.depth_units,
            to_disparity=False,
        )

        self._hole_fill = HoleFillingFilter(mode=cfg.hole_fill_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array) -> mx.array:
        """Run the full post-processing pipeline on one depth frame.

        Parameters
        ----------
        depth:
            ``(H, W)`` uint16 depth frame as returned by the RealSense SDK.
            Zero values are treated as invalid measurements.

        Returns
        -------
        mx.array
            Post-processed depth frame (uint16), potentially at reduced
            resolution if ``decimation_scale > 1``.
        """
        # Step 1: Decimate
        frame = self._decimation.process(depth)

        # Step 2: Depth → Disparity
        frame = self._depth_to_disp.process(frame)

        # Step 3: Spatial filter (in disparity space)
        if self.config.enable_spatial:
            frame = self._spatial.process(frame)

        # Step 4: Temporal filter (in disparity space)
        if self.config.enable_temporal:
            frame = self._temporal.process(frame)

        # Step 5: Disparity → Depth
        frame = self._disp_to_depth.process(frame)

        # Step 6: Hole filling
        if self.config.enable_hole_fill:
            frame = self._hole_fill.process(frame)

        mx.eval(frame)
        return frame

    def reset(self) -> None:
        """Reset stateful components (temporal filter history).

        Call this after a scene cut, resolution change, or when
        restarting a depth stream.
        """
        self._temporal.reset()

    def reconfigure(self, config: PipelineConfig) -> None:
        """Replace the current configuration and rebuild all filters.

        This also resets temporal state as the new parameters may have
        different units/scales.

        Parameters
        ----------
        config:
            New :class:`PipelineConfig`.
        """
        self.config = config
        self._build_filters()

    # ------------------------------------------------------------------
    # Properties for inspection / testing
    # ------------------------------------------------------------------

    @property
    def decimation(self) -> DecimationFilter:
        return self._decimation

    @property
    def spatial(self) -> SpatialFilter:
        return self._spatial

    @property
    def temporal(self) -> TemporalFilter:
        return self._temporal

    @property
    def hole_fill(self) -> HoleFillingFilter:
        return self._hole_fill

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DepthPipeline(config={self.config!r})"
