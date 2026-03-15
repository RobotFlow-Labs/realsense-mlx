"""Tests for DepthPipeline and PipelineConfig.

Covers:
- Default config construction.
- Custom PipelineConfig propagates to sub-filters.
- process() returns uint16 with correct shape.
- Decimation scale 1 (no shape change), scale 2 (half resolution).
- Temporal state accumulates: pixel values converge over repeated frames.
- reset() clears temporal state.
- reconfigure() rebuilds all filters.
- All-zero frame propagates as zeros.
- Pipeline with ramp pattern (gradient is plausible after processing).
- Filter attribute access (decimation, spatial, temporal, hole_fill).
- Repr contains 'DepthPipeline'.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


def _make_depth(value: int = 1000, shape=(48, 64), dtype=np.uint16) -> mx.array:
    return mx.array(np.full(shape, value, dtype=dtype))


def _make_ramp_depth(shape=(48, 64), lo=500, hi=3000) -> mx.array:
    """Horizontal ramp from lo to hi across width."""
    row = np.linspace(lo, hi, shape[1], dtype=np.float32).astype(np.uint16)
    frame = np.tile(row, (shape[0], 1))
    return mx.array(frame)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_config(self):
        p = DepthPipeline()
        assert isinstance(p.config, PipelineConfig)

    def test_custom_config_stored(self):
        cfg = PipelineConfig(decimation_scale=4, temporal_alpha=0.6)
        p = DepthPipeline(cfg)
        assert p.config.decimation_scale == 4
        assert p.config.temporal_alpha == 0.6

    def test_filter_attributes_accessible(self):
        p = DepthPipeline()
        from realsense_mlx.filters.decimation import DecimationFilter
        from realsense_mlx.filters.spatial import SpatialFilter
        from realsense_mlx.filters.temporal import TemporalFilter
        from realsense_mlx.filters.hole_filling import HoleFillingFilter
        assert isinstance(p.decimation, DecimationFilter)
        assert isinstance(p.spatial, SpatialFilter)
        assert isinstance(p.temporal, TemporalFilter)
        assert isinstance(p.hole_fill, HoleFillingFilter)

    def test_repr(self):
        assert "DepthPipeline" in repr(DepthPipeline())


# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.decimation_scale == 2
        assert cfg.spatial_alpha == 0.5
        assert cfg.spatial_delta == 20.0
        assert cfg.spatial_iterations == 2
        assert cfg.temporal_alpha == 0.4
        assert cfg.temporal_delta == 20.0
        assert cfg.temporal_persistence == 3
        assert cfg.hole_fill_mode == 1
        assert cfg.baseline_mm == 50.0
        assert cfg.focal_px == pytest.approx(383.7, rel=1e-5)
        assert cfg.depth_units == pytest.approx(0.001, rel=1e-9)


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_output_dtype_uint16(self):
        p = DepthPipeline(PipelineConfig(decimation_scale=1,
                                         temporal_persistence=0))
        inp = _make_depth(1000)
        out = p.process(inp)
        assert out.dtype == mx.uint16

    def test_shape_with_scale1(self):
        cfg = PipelineConfig(decimation_scale=1, temporal_persistence=0)
        p = DepthPipeline(cfg)
        inp = _make_depth(1000, shape=(48, 64))
        out = p.process(inp)
        # With scale=1 the shape should be (48, 64).
        assert out.shape == (48, 64)

    def test_shape_with_scale2(self):
        cfg = PipelineConfig(decimation_scale=2, temporal_persistence=0)
        p = DepthPipeline(cfg)
        inp = _make_depth(1000, shape=(48, 64))
        out = p.process(inp)
        assert out.shape == (24, 32)

    def test_shape_with_scale4(self):
        cfg = PipelineConfig(decimation_scale=4, temporal_persistence=0)
        p = DepthPipeline(cfg)
        inp = _make_depth(1000, shape=(48, 64))
        out = p.process(inp)
        assert out.shape == (12, 16)


# ---------------------------------------------------------------------------
# All-zero frame
# ---------------------------------------------------------------------------

class TestAllZero:
    def test_all_zero_input_returns_zero(self):
        cfg = PipelineConfig(decimation_scale=1, temporal_persistence=0)
        p = DepthPipeline(cfg)
        inp = mx.array(np.zeros((16, 16), dtype=np.uint16))
        out = _np(p.process(inp))
        assert np.all(out == 0)


# ---------------------------------------------------------------------------
# Temporal state accumulation
# ---------------------------------------------------------------------------

class TestTemporalAccumulation:
    def test_temporal_converges_over_frames(self):
        """Feed 20 identical frames; output should converge to input value."""
        cfg = PipelineConfig(
            decimation_scale=1,
            temporal_alpha=0.4,
            temporal_delta=10000.0,
            temporal_persistence=0,
            spatial_iterations=1,
            hole_fill_mode=0,
        )
        p = DepthPipeline(cfg)
        val = 1000
        for _ in range(20):
            out = _np(p.process(_make_depth(val, shape=(16, 16))))

        # Valid pixels (non-zero) should be close to original.
        valid = out > 0
        if valid.any():
            mean_val = np.mean(out[valid].astype(np.float32))
            assert abs(mean_val - val) < 50, (
                f"Mean output {mean_val:.1f} should be near {val}"
            )

    def test_state_accumulates_across_calls(self):
        cfg = PipelineConfig(decimation_scale=1, temporal_persistence=1)
        p = DepthPipeline(cfg)
        inp = _make_depth(1000, shape=(8, 8))
        # First call: state initialised.
        p.process(inp)
        assert p.temporal._prev_frame is not None


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_temporal_state(self):
        p = DepthPipeline(PipelineConfig(decimation_scale=1, temporal_persistence=0))
        inp = _make_depth(1000, shape=(8, 8))
        p.process(inp)
        p.process(inp)
        p.reset()
        assert p.temporal._prev_frame is None
        assert p.temporal._history is None

    def test_processing_after_reset_works(self):
        p = DepthPipeline(PipelineConfig(decimation_scale=1, temporal_persistence=0))
        inp = _make_depth(1000, shape=(16, 16))
        p.process(inp)
        p.reset()
        out = _np(p.process(inp))
        assert out.shape == (16, 16)
        assert out.dtype == np.uint16


# ---------------------------------------------------------------------------
# reconfigure()
# ---------------------------------------------------------------------------

class TestReconfigure:
    def test_reconfigure_rebuilds_filters(self):
        p = DepthPipeline(PipelineConfig(decimation_scale=1))
        inp = _make_depth(1000, shape=(48, 64))
        p.process(inp)
        # Reconfigure to scale=2.
        new_cfg = PipelineConfig(decimation_scale=2, temporal_persistence=0)
        p.reconfigure(new_cfg)
        assert p.decimation.scale == 2
        # State should be reset.
        assert p.temporal._prev_frame is None

    def test_output_shape_changes_after_reconfigure(self):
        p = DepthPipeline(PipelineConfig(decimation_scale=1, temporal_persistence=0))
        inp = _make_depth(1000, shape=(48, 64))
        out1 = p.process(inp)
        assert out1.shape == (48, 64)

        p.reconfigure(PipelineConfig(decimation_scale=2, temporal_persistence=0))
        out2 = p.process(inp)
        assert out2.shape == (24, 32)


# ---------------------------------------------------------------------------
# Ramp pattern
# ---------------------------------------------------------------------------

class TestRampPattern:
    def test_ramp_gradient_direction_preserved(self):
        """After full pipeline, far pixels should generally remain farther
        than near pixels (gradient direction preserved)."""
        cfg = PipelineConfig(
            decimation_scale=2,
            temporal_persistence=0,
            spatial_iterations=1,
        )
        p = DepthPipeline(cfg)
        ramp = _make_ramp_depth(shape=(48, 64), lo=500, hi=3000)
        for _ in range(5):
            out = _np(p.process(ramp))

        valid = out > 0
        if valid.any():
            # Left column mean should be less than right column mean.
            left_col = out[:, 0]
            right_col = out[:, -1]
            valid_left  = left_col[left_col > 0]
            valid_right = right_col[right_col > 0]
            if len(valid_left) and len(valid_right):
                assert np.mean(valid_left) < np.mean(valid_right), (
                    "Gradient direction should be preserved after pipeline"
                )


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------

class TestImports:
    def test_filter_exports_from_package(self):
        from realsense_mlx.filters import (
            DecimationFilter,
            DisparityTransform,
            HoleFillingFilter,
            SpatialFilter,
            TemporalFilter,
            PipelineConfig,
            DepthPipeline,
        )
        assert DepthPipeline is not None

    def test_top_level_lazy_import(self):
        import realsense_mlx as rsmlx
        assert rsmlx.DepthPipeline is not None
