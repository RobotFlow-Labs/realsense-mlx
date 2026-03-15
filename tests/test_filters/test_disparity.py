"""Tests for DisparityTransform filter.

Covers:
- Basic depth→disparity conversion correctness.
- Inverse (disparity→depth) round-trip accuracy.
- Zero (invalid) pixel handling in both directions.
- All-zero frames.
- Single-pixel frames.
- Max uint16 value.
- Zero depth_units guard (d2d_factor=0).
- dtype contracts: float32 out of depth→disparity, uint16 out of disp→depth.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.disparity import DisparityTransform

# ---------------------------------------------------------------------------
# Shared camera-like parameters (D435 ballpark).
# ---------------------------------------------------------------------------
BASELINE_MM = 50.0
FOCAL_PX = 383.7
DEPTH_UNITS = 0.001  # 1 mm per count
D2D = BASELINE_MM * FOCAL_PX * 32.0 / DEPTH_UNITS  # expected factor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def d2d_transform() -> DisparityTransform:
    return DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, to_disparity=True)


@pytest.fixture
def d2depth_transform() -> DisparityTransform:
    return DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, to_disparity=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Disparity factor
# ---------------------------------------------------------------------------

class TestD2DFactor:
    def test_factor_computed_correctly(self):
        t = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS)
        expected = BASELINE_MM * FOCAL_PX * 32.0 / DEPTH_UNITS
        assert math.isclose(t.d2d_factor, expected, rel_tol=1e-9)

    def test_zero_depth_units_gives_zero_factor(self):
        t = DisparityTransform(BASELINE_MM, FOCAL_PX, depth_units=0.0)
        assert t.d2d_factor == 0.0


# ---------------------------------------------------------------------------
# depth → disparity
# ---------------------------------------------------------------------------

class TestDepthToDisparity:
    def test_output_dtype_is_float32(self, d2d_transform):
        depth = mx.array(np.array([[1000, 2000]], dtype=np.uint16))
        out = d2d_transform.process(depth)
        assert out.dtype == mx.float32

    def test_output_shape_preserved(self, d2d_transform):
        depth = mx.array(np.ones((48, 64), dtype=np.uint16) * 1000)
        out = d2d_transform.process(depth)
        assert out.shape == (48, 64)

    def test_disparity_formula(self, d2d_transform):
        depth_val = 1000  # depth counts
        expected_disp = D2D / depth_val
        depth = mx.array(np.array([[depth_val]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert abs(out[0, 0] - expected_disp) < 0.1

    def test_zero_depth_gives_zero_disparity(self, d2d_transform):
        depth = mx.array(np.array([[0, 500]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] > 0.0

    def test_all_zeros_frame(self, d2d_transform):
        depth = mx.array(np.zeros((10, 10), dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert np.all(out == 0.0)

    def test_single_pixel_valid(self, d2d_transform):
        depth = mx.array(np.array([[800]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert out[0, 0] == pytest.approx(D2D / 800, rel=1e-5)

    def test_single_pixel_zero(self, d2d_transform):
        depth = mx.array(np.array([[0]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert out[0, 0] == pytest.approx(0.0)

    def test_larger_depth_gives_smaller_disparity(self, d2d_transform):
        depth = mx.array(np.array([[500, 2000]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert out[0, 0] > out[0, 1]

    def test_max_uint16(self, d2d_transform):
        depth = mx.array(np.array([[65535]], dtype=np.uint16))
        out = _np(d2d_transform.process(depth))
        assert out[0, 0] > 0.0
        assert np.isfinite(out[0, 0])


# ---------------------------------------------------------------------------
# disparity → depth
# ---------------------------------------------------------------------------

class TestDisparityToDepth:
    def test_output_dtype_is_uint16(self, d2depth_transform):
        disp = mx.array(np.array([[1000.0, 500.0]], dtype=np.float32))
        out = d2depth_transform.process(disp)
        assert out.dtype == mx.uint16

    def test_output_shape_preserved(self, d2depth_transform):
        disp = mx.array(np.ones((48, 64), dtype=np.float32) * 500.0)
        out = d2depth_transform.process(disp)
        assert out.shape == (48, 64)

    def test_zero_disparity_gives_zero_depth(self, d2depth_transform):
        disp = mx.array(np.array([[0.0, 500.0]], dtype=np.float32))
        out = _np(d2depth_transform.process(disp))
        assert out[0, 0] == 0
        assert out[0, 1] > 0

    def test_all_zeros_disparity(self, d2depth_transform):
        disp = mx.array(np.zeros((8, 8), dtype=np.float32))
        out = _np(d2depth_transform.process(disp))
        assert np.all(out == 0)


# ---------------------------------------------------------------------------
# Round-trip accuracy
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip_close(self):
        """depth → disparity → depth should reconstruct within ±2 counts."""
        rng = np.random.default_rng(42)
        depths = rng.integers(200, 8000, size=(30, 40), dtype=np.uint16)
        # Introduce some zeros (invalid pixels).
        depths[0:5, 0:5] = 0

        fwd = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, True)
        bwd = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, False)

        disp = fwd.process(mx.array(depths))
        reconstructed = _np(bwd.process(disp))

        # Valid pixels only.
        valid = depths > 0
        diff = np.abs(reconstructed[valid].astype(np.int32) - depths[valid].astype(np.int32))
        assert np.max(diff) <= 2, f"Max round-trip error {np.max(diff)} counts > 2"

    def test_invalid_pixels_remain_zero_after_round_trip(self):
        depths = np.array([[0, 1000, 0, 2000]], dtype=np.uint16)
        fwd = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, True)
        bwd = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, False)

        disp = fwd.process(mx.array(depths))
        reconstructed = _np(bwd.process(disp))
        assert reconstructed[0, 0] == 0
        assert reconstructed[0, 2] == 0

    def test_repr(self):
        t = DisparityTransform(BASELINE_MM, FOCAL_PX, DEPTH_UNITS, True)
        r = repr(t)
        assert "DisparityTransform" in r
        assert "depth→disparity" in r
