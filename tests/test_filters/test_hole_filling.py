"""Tests for HoleFillingFilter.

Covers all three modes:
- FILL_FROM_LEFT (0): left-propagation, row-independent, stops at first valid.
- FARTHEST (1): 4-connected neighbour max, only fills invalid pixels.
- NEAREST (2): 4-connected neighbour min (non-zero), only fills invalid pixels.

Edge cases:
- All-zero frame (no valid source pixels).
- All-valid frame (no change expected).
- Single-pixel frames.
- Frame with isolated holes vs large regions.
- dtype preservation (uint16 and float32).
- Invalid mode raises ValueError.
- Metal vs Python equivalence for FILL_FROM_LEFT.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.hole_filling import HoleFillingFilter


def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[True, False], ids=["metal", "python"])
def fill_left(request) -> HoleFillingFilter:
    return HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT, use_metal=request.param)


@pytest.fixture
def fill_farthest() -> HoleFillingFilter:
    return HoleFillingFilter(mode=HoleFillingFilter.FARTHEST)


@pytest.fixture
def fill_nearest() -> HoleFillingFilter:
    return HoleFillingFilter(mode=HoleFillingFilter.NEAREST)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_mode_is_farthest(self):
        f = HoleFillingFilter()
        assert f.mode == HoleFillingFilter.FARTHEST

    def test_default_use_metal_true(self):
        f = HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT)
        assert f.use_metal is True

    def test_use_metal_false_stored(self):
        f = HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT, use_metal=False)
        assert f.use_metal is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown hole-fill mode"):
            HoleFillingFilter(mode=99)

    def test_repr_includes_mode_name(self):
        f = HoleFillingFilter(mode=HoleFillingFilter.NEAREST)
        assert "NEAREST" in repr(f)

    def test_repr_includes_use_metal(self):
        f = HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT, use_metal=False)
        assert "use_metal=False" in repr(f)


# ---------------------------------------------------------------------------
# FILL_FROM_LEFT — parametrized over Metal and Python backends
# ---------------------------------------------------------------------------

class TestFillFromLeft:
    def test_propagates_valid_value_rightward(self, fill_left):
        row = np.array([[1000, 0, 0, 0]], dtype=np.uint16)
        out = _np(fill_left.process(mx.array(row)))
        assert out[0, 1] == 1000
        assert out[0, 2] == 1000
        assert out[0, 3] == 1000

    def test_does_not_propagate_leftward(self, fill_left):
        row = np.array([[0, 0, 0, 2000]], dtype=np.uint16)
        out = _np(fill_left.process(mx.array(row)))
        # First three pixels have no valid source to the left.
        assert out[0, 0] == 0
        assert out[0, 1] == 0
        assert out[0, 2] == 0
        assert out[0, 3] == 2000

    def test_stops_and_resumes_at_new_valid(self, fill_left):
        row = np.array([[500, 0, 2000, 0]], dtype=np.uint16)
        out = _np(fill_left.process(mx.array(row)))
        assert out[0, 0] == 500
        assert out[0, 1] == 500   # propagated from 500
        assert out[0, 2] == 2000  # new valid value
        assert out[0, 3] == 2000  # propagated from 2000

    def test_leading_zeros_remain_zero(self, fill_left):
        row = np.array([[0, 0, 1000]], dtype=np.uint16)
        out = _np(fill_left.process(mx.array(row)))
        assert out[0, 0] == 0
        assert out[0, 1] == 0

    def test_all_zeros_stays_zero(self, fill_left):
        frame = np.zeros((8, 8), dtype=np.uint16)
        out = _np(fill_left.process(mx.array(frame)))
        assert np.all(out == 0)

    def test_all_valid_unchanged(self, fill_left):
        rng = np.random.default_rng(0)
        frame = rng.integers(500, 3000, (10, 10), dtype=np.uint16)
        out = _np(fill_left.process(mx.array(frame)))
        # Valid pixels should be very close to original (float rounding ±1).
        diff = np.abs(out.astype(np.int32) - frame.astype(np.int32))
        assert np.max(diff) <= 1

    def test_output_dtype_uint16(self, fill_left):
        frame = mx.array(np.array([[1000, 0]], dtype=np.uint16))
        out = fill_left.process(frame)
        assert out.dtype == mx.uint16

    def test_output_dtype_float32(self, fill_left):
        frame = mx.array(np.array([[1000.0, 0.0]], dtype=np.float32))
        out = fill_left.process(frame)
        assert out.dtype == mx.float32

    def test_single_pixel(self, fill_left):
        out = _np(fill_left.process(mx.array(np.array([[500]], dtype=np.uint16))))
        assert out[0, 0] == 500

    def test_multi_row_independent(self, fill_left):
        frame = np.array([[1000, 0, 0],
                          [0,    0, 2000]], dtype=np.uint16)
        out = _np(fill_left.process(mx.array(frame)))
        # Row 0: 1000 propagates right
        assert out[0, 1] == 1000
        assert out[0, 2] == 1000
        # Row 1: nothing propagates left, last pixel valid
        assert out[1, 0] == 0
        assert out[1, 1] == 0
        assert out[1, 2] == 2000


# ---------------------------------------------------------------------------
# Metal vs Python equivalence for FILL_FROM_LEFT
# ---------------------------------------------------------------------------

class TestFillFromLeftMetalPythonEquivalence:
    """Verify that the Metal kernel produces bit-identical results to the
    pure-Python MLX fallback across a range of input patterns."""

    def _both(self, frame_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = mx.array(frame_np)
        metal_out = _np(
            HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT, use_metal=True)
            .process(arr)
        )
        python_out = _np(
            HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT, use_metal=False)
            .process(arr)
        )
        return metal_out, python_out

    def test_simple_row(self):
        frame = np.array([[1000, 0, 0, 2000, 0]], dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)

    def test_all_zeros(self):
        frame = np.zeros((8, 16), dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)

    def test_all_valid(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(100, 5000, (12, 20), dtype=np.uint16)
        m, p = self._both(frame)
        # Both paths go through float32 and back; allow ±1 rounding difference.
        np.testing.assert_array_equal(m, p)

    def test_sparse_valid_pixels(self):
        rng = np.random.default_rng(7)
        frame = np.zeros((16, 32), dtype=np.uint16)
        # Place valid pixels at ~20% of positions.
        mask = rng.random((16, 32)) > 0.8
        frame[mask] = rng.integers(200, 4000, mask.sum(), dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)

    def test_float32_input(self):
        rng = np.random.default_rng(99)
        frame = np.zeros((10, 15), dtype=np.float32)
        mask = rng.random((10, 15)) > 0.6
        frame[mask] = rng.random(mask.sum()).astype(np.float32) * 3000.0 + 100.0
        m, p = self._both(frame)
        np.testing.assert_allclose(m, p, atol=1e-4)

    def test_large_frame(self):
        """Regression: ensure graph-growth workaround (eval every 64 cols)
        in the Python path and Metal path produce identical results."""
        rng = np.random.default_rng(123)
        frame = np.zeros((32, 128), dtype=np.uint16)
        mask = rng.random((32, 128)) > 0.7
        frame[mask] = rng.integers(500, 3000, mask.sum(), dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)

    def test_single_row_wide(self):
        """Single row — each thread processes exactly one row."""
        frame = np.array([[0, 0, 500, 0, 0, 1000, 0]], dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)

    def test_single_column(self):
        """Single column — nothing to propagate, output equals input."""
        frame = np.array([[1000], [0], [500], [0]], dtype=np.uint16)
        m, p = self._both(frame)
        np.testing.assert_array_equal(m, p)


# ---------------------------------------------------------------------------
# FARTHEST (max neighbour)
# ---------------------------------------------------------------------------

class TestFillFarthest:
    def test_fills_hole_with_max_neighbour(self, fill_farthest):
        # Pattern: [100, 0, 500] — hole at [0,1], neighbours are 100 and 500.
        row = np.array([[100, 0, 500]], dtype=np.uint16)
        out = _np(fill_farthest.process(mx.array(row)))
        assert out[0, 1] == 500  # farthest = max = 500

    def test_does_not_modify_valid_pixels(self, fill_farthest):
        rng = np.random.default_rng(7)
        frame = rng.integers(200, 5000, (12, 12), dtype=np.uint16)
        out = _np(fill_farthest.process(mx.array(frame)))
        diff = np.abs(out.astype(np.int32) - frame.astype(np.int32))
        assert np.max(diff) <= 1  # float rounding only

    def test_all_zeros_stays_zero(self, fill_farthest):
        out = _np(fill_farthest.process(mx.array(np.zeros((6, 6), dtype=np.uint16))))
        assert np.all(out == 0)

    def test_single_valid_surrounded_by_holes(self, fill_farthest):
        frame = np.zeros((3, 3), dtype=np.uint16)
        frame[1, 1] = 1000  # centre valid
        out = _np(fill_farthest.process(mx.array(frame)))
        # Neighbours of centre get filled with max neighbour = 1000.
        assert out[0, 1] == 1000
        assert out[1, 0] == 1000
        assert out[1, 2] == 1000
        assert out[2, 1] == 1000
        # Original valid pixel must not change.
        assert out[1, 1] == 1000

    def test_output_shape_preserved(self, fill_farthest):
        frame = mx.array(np.zeros((48, 64), dtype=np.uint16))
        out = fill_farthest.process(frame)
        assert out.shape == (48, 64)


# ---------------------------------------------------------------------------
# NEAREST (min non-zero neighbour)
# ---------------------------------------------------------------------------

class TestFillNearest:
    def test_fills_hole_with_min_nonzero_neighbour(self, fill_nearest):
        row = np.array([[2000, 0, 500]], dtype=np.uint16)
        out = _np(fill_nearest.process(mx.array(row)))
        assert out[0, 1] == 500  # nearest = min non-zero = 500

    def test_does_not_use_zero_as_minimum(self, fill_nearest):
        # [0, 0, 1000, 0, 0] — hole in middle; only 1000 is valid.
        row = np.array([[0, 0, 1000, 0, 0]], dtype=np.uint16)
        out = _np(fill_nearest.process(mx.array(row)))
        # col 1 gets min(0-sentinel, 1000) = 1000 (or 0 if no valid neighbour)
        # col 3 gets 1000 from left neighbour
        assert out[0, 2] == 1000
        assert out[0, 3] == 1000

    def test_isolated_hole_no_valid_neighbour(self, fill_nearest):
        # All zeros except one isolated group — island with gap between.
        frame = np.zeros((3, 5), dtype=np.uint16)
        frame[1, 0] = 500
        # [1,1] hole — only left neighbour valid.
        out = _np(fill_nearest.process(mx.array(frame)))
        assert out[1, 1] == 500

    def test_all_zeros_stays_zero(self, fill_nearest):
        out = _np(fill_nearest.process(mx.array(np.zeros((5, 5), dtype=np.uint16))))
        assert np.all(out == 0)

    def test_output_dtype_preserved(self, fill_nearest):
        frame = mx.array(np.array([[1000, 0]], dtype=np.uint16))
        assert fill_nearest.process(frame).dtype == mx.uint16
        frame_f = mx.array(np.array([[1000.0, 0.0]], dtype=np.float32))
        assert fill_nearest.process(frame_f).dtype == mx.float32


# ---------------------------------------------------------------------------
# Single pixel edge cases (all modes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [0, 1, 2])
class TestSinglePixel:
    def test_valid_single_pixel_unchanged(self, mode):
        f = HoleFillingFilter(mode=mode)
        frame = mx.array(np.array([[1000]], dtype=np.uint16))
        out = _np(f.process(frame))
        assert out[0, 0] == pytest.approx(1000, abs=1)

    def test_zero_single_pixel_stays_zero(self, mode):
        f = HoleFillingFilter(mode=mode)
        frame = mx.array(np.array([[0]], dtype=np.uint16))
        out = _np(f.process(frame))
        assert out[0, 0] == 0
