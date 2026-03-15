"""Tests for DepthStats — quality metrics on depth frames.

Coverage
--------
* All-valid frame → valid_ratio = 1.0
* All-zero frame  → valid_ratio = 0.0, hole_count = H*W
* Frame with rectangular hole → correct hole_count
* Statistics (min, max, mean, std, median) are computed on valid pixels only
* edge_pixel_count: flat wall has 0 edges; frame with step has edges
* compare: identical frames → RMSE = 0, PSNR = inf, ssim_approx = 1.0
* compare: different frames → RMSE > 0
* compare: holes_filled / holes_created counting
* compare: numpy arrays accepted (no MLX required)
* compare: mismatched shapes raise ValueError
* DepthStats importable from utils package-level
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.utils.depth_stats import DepthStats
from realsense_mlx.utils import DepthStats as DepthStatsFromPackage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(value: int, H: int = 48, W: int = 64) -> mx.array:
    return mx.full((H, W), value, dtype=mx.uint16)


def _np_flat(value: int, H: int = 48, W: int = 64) -> np.ndarray:
    return np.full((H, W), value, dtype=np.uint16)


# ---------------------------------------------------------------------------
# 1. compute — all-valid frame
# ---------------------------------------------------------------------------


class TestComputeAllValid:
    def test_valid_ratio_is_one(self):
        stats = DepthStats.compute(_flat(1000))
        assert stats["valid_ratio"] == pytest.approx(1.0)

    def test_hole_count_zero(self):
        stats = DepthStats.compute(_flat(1000))
        assert stats["hole_count"] == 0

    def test_hole_ratio_zero(self):
        stats = DepthStats.compute(_flat(1000))
        assert stats["hole_ratio"] == pytest.approx(0.0)

    def test_min_max_mean_equal_for_flat_wall(self):
        """All pixels at 1000 counts * 0.001 = 1.0 m."""
        stats = DepthStats.compute(_flat(1000), depth_scale=0.001)
        assert stats["min_m"] == pytest.approx(1.0, abs=1e-6)
        assert stats["max_m"] == pytest.approx(1.0, abs=1e-6)
        assert stats["mean_m"] == pytest.approx(1.0, abs=1e-6)
        assert stats["std_m"] == pytest.approx(0.0, abs=1e-6)
        assert stats["median_m"] == pytest.approx(1.0, abs=1e-6)

    def test_flat_wall_has_no_edges(self):
        """A flat wall at constant depth should have zero edge pixels."""
        stats = DepthStats.compute(_flat(1000), depth_scale=0.001)
        assert stats["edge_pixel_count"] == 0

    def test_all_keys_present(self):
        stats = DepthStats.compute(_flat(500))
        expected_keys = {
            "valid_ratio", "min_m", "max_m", "mean_m", "std_m",
            "median_m", "hole_count", "hole_ratio", "edge_pixel_count",
        }
        assert expected_keys == set(stats.keys())


# ---------------------------------------------------------------------------
# 2. compute — all-zero frame
# ---------------------------------------------------------------------------


class TestComputeAllZero:
    def test_valid_ratio_is_zero(self):
        stats = DepthStats.compute(_flat(0))
        assert stats["valid_ratio"] == pytest.approx(0.0)

    def test_hole_count_equals_total(self):
        H, W = 48, 64
        stats = DepthStats.compute(_flat(0, H, W))
        assert stats["hole_count"] == H * W

    def test_statistical_values_are_none(self):
        stats = DepthStats.compute(_flat(0))
        for key in ("min_m", "max_m", "mean_m", "std_m", "median_m"):
            assert stats[key] is None, f"Expected None for {key}, got {stats[key]}"

    def test_edge_count_zero_for_all_zero(self):
        """An all-zero frame has no depth edges (no finite-difference transitions)."""
        stats = DepthStats.compute(_flat(0))
        assert stats["edge_pixel_count"] == 0


# ---------------------------------------------------------------------------
# 3. compute — frame with holes
# ---------------------------------------------------------------------------


class TestComputeWithHoles:
    def test_rectangular_hole_count(self):
        """Rectangular hole rows 10:20, cols 20:40 → 10*20 = 200 hole pixels."""
        H, W = 48, 64
        depth_np = np.full((H, W), 1000, dtype=np.uint16)
        depth_np[10:20, 20:40] = 0
        hole_pixels = 10 * 20
        stats = DepthStats.compute(mx.array(depth_np))
        assert stats["hole_count"] == hole_pixels

    def test_valid_ratio_correct_with_holes(self):
        H, W = 48, 64
        depth_np = np.full((H, W), 500, dtype=np.uint16)
        depth_np[:H // 2, :] = 0   # top half zeroed
        expected_valid_ratio = 0.5
        stats = DepthStats.compute(mx.array(depth_np))
        assert stats["valid_ratio"] == pytest.approx(expected_valid_ratio, abs=1e-6)

    def test_stats_computed_only_over_valid(self):
        """Mean/std should ignore zero-depth pixels."""
        H, W = 10, 10
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[5:, :] = 2000   # bottom half at 2.0 m
        stats = DepthStats.compute(mx.array(depth_np), depth_scale=0.001)
        assert stats["mean_m"] == pytest.approx(2.0, abs=1e-5)
        assert stats["std_m"] == pytest.approx(0.0, abs=1e-5)

    def test_edge_pixels_at_depth_step(self):
        """A sharp depth step should produce non-zero edge_pixel_count."""
        H, W = 10, 10
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[:, :W // 2] = 1000   # left half at 1 m
        depth_np[:, W // 2:] = 3000   # right half at 3 m — 2 m step >> 5 mm
        stats = DepthStats.compute(mx.array(depth_np), depth_scale=0.001)
        assert stats["edge_pixel_count"] > 0

    def test_numpy_input_accepted(self):
        """DepthStats.compute should accept a plain NumPy array."""
        stats = DepthStats.compute(_np_flat(1000))
        assert stats["valid_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. compare — identical frames
# ---------------------------------------------------------------------------


class TestCompareIdentical:
    def test_rmse_is_zero(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_mae_is_zero(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert result["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_psnr_is_inf(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert math.isinf(result["psnr"])

    def test_ssim_approx_is_one(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert result["ssim_approx"] == pytest.approx(1.0, abs=1e-6)

    def test_no_holes_filled_or_created(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert result["holes_filled"] == 0
        assert result["holes_created"] == 0

    def test_smoothness_improvement_zero(self):
        """Identical frames → std_before == std_after → improvement == 0."""
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        assert result["smoothness_improvement"] == pytest.approx(0.0, abs=1e-6)

    def test_all_keys_present(self):
        d = _flat(1000)
        result = DepthStats.compare(d, d)
        expected_keys = {
            "rmse", "mae", "psnr", "ssim_approx",
            "holes_filled", "holes_created", "smoothness_improvement",
        }
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# 5. compare — different frames
# ---------------------------------------------------------------------------


class TestCompareDifferent:
    def test_rmse_positive(self):
        """Two frames with different depth values should yield RMSE > 0."""
        before = _flat(1000)
        after = _flat(1100)
        result = DepthStats.compare(before, after, depth_scale=0.001)
        assert result["rmse"] > 0.0

    def test_rmse_correct_value(self):
        """Single-pixel check: before=1000, after=1100, scale=0.001 → diff=0.1 m."""
        H, W = 4, 4
        before = mx.full((H, W), 1000, dtype=mx.uint16)
        after = mx.full((H, W), 1100, dtype=mx.uint16)
        result = DepthStats.compare(before, after, depth_scale=0.001)
        # All pixels differ by exactly 0.1 m → RMSE = MAE = 0.1 m
        assert result["rmse"] == pytest.approx(0.1, abs=1e-5)
        assert result["mae"] == pytest.approx(0.1, abs=1e-5)

    def test_psnr_finite_when_rmse_positive(self):
        before = _flat(1000)
        after = _flat(1100)
        result = DepthStats.compare(before, after, depth_scale=0.001)
        assert math.isfinite(result["psnr"])
        assert result["psnr"] > 0.0

    def test_holes_filled_count(self):
        """Pixels that were 0 in before but non-zero in after should be counted."""
        H, W = 10, 10
        before_np = np.full((H, W), 0, dtype=np.uint16)
        after_np = np.full((H, W), 1000, dtype=np.uint16)
        # Entire frame filled
        result = DepthStats.compare(mx.array(before_np), mx.array(after_np))
        assert result["holes_filled"] == H * W
        assert result["holes_created"] == 0

    def test_holes_created_count(self):
        H, W = 10, 10
        before_np = np.full((H, W), 1000, dtype=np.uint16)
        after_np = np.full((H, W), 0, dtype=np.uint16)
        result = DepthStats.compare(mx.array(before_np), mx.array(after_np))
        assert result["holes_created"] == H * W
        assert result["holes_filled"] == 0

    def test_partial_hole_filling(self):
        H, W = 10, 10
        before_np = np.full((H, W), 500, dtype=np.uint16)
        before_np[:5, :] = 0   # top half = holes
        after_np = before_np.copy()
        after_np[:5, :] = 500  # holes filled in top half
        result = DepthStats.compare(mx.array(before_np), mx.array(after_np))
        assert result["holes_filled"] == 5 * W
        assert result["holes_created"] == 0

    def test_smoothness_improvement_positive_after_smoothing(self):
        """After smoothing (lower std), improvement should be > 0."""
        rng = np.random.default_rng(42)
        H, W = 20, 20
        # Noisy depth
        before_np = rng.integers(800, 1200, (H, W), dtype=np.uint16)
        # Smoother: all pixels at 1000
        after_np = np.full((H, W), 1000, dtype=np.uint16)
        result = DepthStats.compare(
            mx.array(before_np), mx.array(after_np), depth_scale=0.001
        )
        assert result["smoothness_improvement"] > 0.0

    def test_numpy_arrays_accepted(self):
        before = _np_flat(1000)
        after = _np_flat(1050)
        result = DepthStats.compare(before, after, depth_scale=0.001)
        assert result["rmse"] > 0.0

    def test_shape_mismatch_raises(self):
        before = mx.full((10, 10), 1000, dtype=mx.uint16)
        after = mx.full((10, 12), 1000, dtype=mx.uint16)
        with pytest.raises(ValueError, match="shape"):
            DepthStats.compare(before, after)


# ---------------------------------------------------------------------------
# 6. Package-level import
# ---------------------------------------------------------------------------


def test_depth_stats_importable_from_utils():
    """DepthStats must be exported from realsense_mlx.utils."""
    assert DepthStatsFromPackage is DepthStats
