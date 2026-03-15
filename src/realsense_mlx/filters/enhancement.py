"""High-quality depth post-processing combining bilateral and temporal filters.

This module provides :class:`DepthEnhancer`, a composable pipeline that
produces better-quality depth than the standard RS2 post-processing because
it uses the colour/IR image as a guide for the bilateral filter, aligning
smoothing boundaries to real image edges rather than noisy depth edges.

Pipeline
--------
1. **Threshold** – zero out pixels outside [min_depth, max_depth] in metres.
2. **Bilateral** – joint bilateral filter using the guide image (or standard
   bilateral on depth alone when no guide is provided).
3. **Temporal** – EMA smoothing with persistence gating across frames.
4. **Hole-fill** – fill remaining zero pixels from valid neighbours.

Each stage can be individually disabled via constructor flags.

Usage
-----
>>> import mlx.core as mx, numpy as np
>>> from realsense_mlx.filters.enhancement import DepthEnhancer
>>>
>>> enhancer = DepthEnhancer(min_depth=0.3, max_depth=5.0)
>>>
>>> # Without a guide image (standard bilateral on depth itself).
>>> raw = mx.array(np.random.randint(300, 5000, (480, 640), dtype=np.uint16))
>>> out = enhancer.process(raw)
>>>
>>> # With an IR guide image for better edge preservation.
>>> ir_guide = mx.array(np.random.randint(0, 255, (480, 640), dtype=np.uint8))
>>> out = enhancer.process(raw, guide=ir_guide)
>>> enhancer.reset()  # clear temporal state

Temporal state
--------------
:class:`DepthEnhancer` maintains state across calls via the internal
:class:`~realsense_mlx.filters.temporal.TemporalFilter`.  Call
:meth:`reset` when switching streams, changing resolution, or after a
significant scene change.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from realsense_mlx.filters.bilateral import BilateralFilter
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.temporal import TemporalFilter

__all__ = ["DepthEnhancerConfig", "DepthEnhancer"]


@dataclass
class DepthEnhancerConfig:
    """Configuration for :class:`DepthEnhancer`.

    Attributes
    ----------
    min_depth : float
        Minimum valid depth in **metres**.  Pixels shallower than this
        are zeroed before filtering.  Default 0.1 m.
    max_depth : float
        Maximum valid depth in **metres**.  Pixels deeper than this are
        zeroed.  Default 10.0 m.
    depth_units : float
        Metres per depth count (same as ``rs2_option.DEPTH_UNITS``).
        Default 0.001 (1 mm per count, standard RealSense).
    bilateral_sigma_spatial : float
        Spatial sigma for the bilateral filter in pixels.  Default 5.0.
    bilateral_sigma_range : float
        Range sigma for the bilateral filter (guide image units,
        typically 0–255 for IR/colour guides).  Default 30.0.
    bilateral_kernel_size : int
        Bilateral filter window size (odd).  Default 5.
    bilateral_n_bins : int
        Number of range quantisation bins.  Default 8.
    temporal_alpha : float
        EMA weight for the current frame [0, 1].  Default 0.4.
    temporal_delta : float
        Large-change threshold to prevent EMA ghosting.  Default 20.0.
    temporal_persistence : int
        Minimum valid frames in last 8 required to keep a pixel.
        0 = disabled.  Default 3.
    hole_fill_mode : int
        Hole-fill mode: 0=fill-from-left, 1=farthest (default), 2=nearest.
    enable_bilateral : bool
        Whether to apply the bilateral filter stage.  Default True.
    enable_temporal : bool
        Whether to apply the temporal filter stage.  Default True.
    enable_hole_fill : bool
        Whether to apply the hole-fill stage.  Default True.
    """

    min_depth: float = 0.1
    max_depth: float = 10.0
    depth_units: float = 0.001

    # Bilateral parameters.
    bilateral_sigma_spatial: float = 5.0
    bilateral_sigma_range: float = 30.0
    bilateral_kernel_size: int = 5
    bilateral_n_bins: int = 8

    # Temporal parameters.
    temporal_alpha: float = 0.4
    temporal_delta: float = 20.0
    temporal_persistence: int = 3

    # Hole fill.
    hole_fill_mode: int = HoleFillingFilter.FARTHEST

    # Stage toggles.
    enable_bilateral: bool = True
    enable_temporal: bool = True
    enable_hole_fill: bool = True


class DepthEnhancer:
    """High-quality depth post-processing combining multiple filters.

    Pipeline: threshold → bilateral → temporal → hole-fill

    This produces better results than the standard RS2 pipeline because
    the bilateral filter uses the colour/IR guide image — real image
    edges are much cleaner than the noisy depth edges that the spatial
    filter uses.

    Parameters
    ----------
    min_depth : float
        Minimum valid depth in metres.  Default 0.1.
    max_depth : float
        Maximum valid depth in metres.  Default 10.0.
    depth_units : float
        Metres per depth count.  Default 0.001 (1 mm).
    bilateral_sigma_spatial : float
        Spatial sigma for bilateral filter.  Default 5.0.
    bilateral_sigma_range : float
        Range sigma for bilateral filter.  Default 30.0.
    bilateral_kernel_size : int
        Bilateral filter window size (odd integer).  Default 5.
    bilateral_n_bins : int
        Bilateral range quantisation bins.  Default 8.
    temporal_alpha : float
        Temporal EMA weight for current frame.  Default 0.4.
    temporal_delta : float
        Large-change threshold.  Default 20.0.
    temporal_persistence : int
        Minimum valid appearances in last 8 frames.  Default 3.
    hole_fill_mode : int
        Hole-fill mode (0/1/2).  Default 1 (farthest).
    enable_bilateral : bool
        Toggle bilateral stage.  Default True.
    enable_temporal : bool
        Toggle temporal stage.  Default True.
    enable_hole_fill : bool
        Toggle hole-fill stage.  Default True.
    config : DepthEnhancerConfig | None
        If provided, overrides all individual keyword arguments.

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> enhancer = DepthEnhancer(min_depth=0.3, max_depth=4.0)
    >>> raw = mx.array(np.full((48, 64), 1500, dtype=np.uint16))
    >>> out = enhancer.process(raw)
    >>> out.shape
    (48, 64)
    """

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        depth_units: float = 0.001,
        bilateral_sigma_spatial: float = 5.0,
        bilateral_sigma_range: float = 30.0,
        bilateral_kernel_size: int = 5,
        bilateral_n_bins: int = 8,
        temporal_alpha: float = 0.4,
        temporal_delta: float = 20.0,
        temporal_persistence: int = 3,
        hole_fill_mode: int = HoleFillingFilter.FARTHEST,
        enable_bilateral: bool = True,
        enable_temporal: bool = True,
        enable_hole_fill: bool = True,
        config: DepthEnhancerConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = DepthEnhancerConfig(
                min_depth=min_depth,
                max_depth=max_depth,
                depth_units=depth_units,
                bilateral_sigma_spatial=bilateral_sigma_spatial,
                bilateral_sigma_range=bilateral_sigma_range,
                bilateral_kernel_size=bilateral_kernel_size,
                bilateral_n_bins=bilateral_n_bins,
                temporal_alpha=temporal_alpha,
                temporal_delta=temporal_delta,
                temporal_persistence=temporal_persistence,
                hole_fill_mode=hole_fill_mode,
                enable_bilateral=enable_bilateral,
                enable_temporal=enable_temporal,
                enable_hole_fill=enable_hole_fill,
            )
        self._build_filters()

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_filters(self) -> None:
        """Instantiate all sub-filters from ``self.config``."""
        cfg = self.config

        self._bilateral = BilateralFilter(
            sigma_spatial=cfg.bilateral_sigma_spatial,
            sigma_range=cfg.bilateral_sigma_range,
            kernel_size=cfg.bilateral_kernel_size,
            n_bins=cfg.bilateral_n_bins,
        )

        self._temporal = TemporalFilter(
            alpha=cfg.temporal_alpha,
            delta=cfg.temporal_delta,
            persistence=cfg.temporal_persistence,
        )

        self._hole_fill = HoleFillingFilter(mode=cfg.hole_fill_mode)

        # Pre-compute threshold counts from metric distances.
        # depth_counts = metres / depth_units
        du = cfg.depth_units
        self._min_count: float = cfg.min_depth / du if du > 0.0 else 0.0
        self._max_count: float = cfg.max_depth / du if du > 0.0 else float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        depth: mx.array,
        guide: mx.array | None = None,
    ) -> mx.array:
        """Apply the full enhancement pipeline to one depth frame.

        Parameters
        ----------
        depth:
            ``(H, W)`` uint16 or float32 depth array.  Zero = invalid.
        guide:
            Optional guide image for edge-aware bilateral filtering.
            Shape ``(H, W)`` or ``(H, W, C)``.  May be uint8 or float32.
            If ``None``, the bilateral filter uses depth as its own guide
            (standard bilateral filter).

        Returns
        -------
        mx.array
            Enhanced depth array, same shape and dtype as ``depth``.

        Raises
        ------
        ValueError
            If ``depth`` is not a 2-D array.
        """
        if depth.ndim != 2:
            raise ValueError(
                f"DepthEnhancer expects 2-D (H, W) depth array, got shape {depth.shape}"
            )
        if depth.shape[0] == 0 or depth.shape[1] == 0:
            return depth

        orig_dtype = depth.dtype
        frame = depth.astype(mx.float32)

        # ------------------------------------------------------------------
        # Stage 1: Threshold — zero out out-of-range pixels.
        # ------------------------------------------------------------------
        in_range = (frame >= self._min_count) & (frame <= self._max_count)
        frame = mx.where(in_range, frame, mx.array(0.0, dtype=mx.float32))
        mx.eval(frame)

        # ------------------------------------------------------------------
        # Stage 2: Bilateral filter.
        # ------------------------------------------------------------------
        if self.config.enable_bilateral:
            frame = self._bilateral.process(frame, guide)

        # ------------------------------------------------------------------
        # Stage 3: Temporal filter.
        # ------------------------------------------------------------------
        if self.config.enable_temporal:
            frame = self._temporal.process(frame)

        # ------------------------------------------------------------------
        # Stage 4: Hole-fill.
        # ------------------------------------------------------------------
        if self.config.enable_hole_fill:
            frame = self._hole_fill.process(frame)

        # Clamp and cast back to original dtype.
        if orig_dtype == mx.uint16:
            frame = mx.clip(frame, 0.0, 65535.0)

        mx.eval(frame)
        return frame.astype(orig_dtype)

    def reset(self) -> None:
        """Clear temporal filter state.

        Call after a scene cut, resolution change, or when restarting the
        depth stream.  The bilateral and hole-fill stages are stateless and
        do not need resetting.
        """
        self._temporal.reset()

    def reconfigure(self, config: DepthEnhancerConfig) -> None:
        """Replace configuration and rebuild all sub-filters.

        Resets temporal state because the new parameters may use different
        units or scales.

        Parameters
        ----------
        config:
            New :class:`DepthEnhancerConfig`.
        """
        self.config = config
        self._build_filters()

    # ------------------------------------------------------------------
    # Properties for inspection / testing
    # ------------------------------------------------------------------

    @property
    def bilateral(self) -> BilateralFilter:
        """The :class:`BilateralFilter` sub-filter."""
        return self._bilateral

    @property
    def temporal(self) -> TemporalFilter:
        """The :class:`TemporalFilter` sub-filter."""
        return self._temporal

    @property
    def hole_fill(self) -> HoleFillingFilter:
        """The :class:`HoleFillingFilter` sub-filter."""
        return self._hole_fill

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"DepthEnhancer("
            f"min_depth={cfg.min_depth}, max_depth={cfg.max_depth}, "
            f"bilateral={cfg.enable_bilateral}, temporal={cfg.enable_temporal}, "
            f"hole_fill={cfg.enable_hole_fill})"
        )
