"""Tests for DecimationFilter.

Covers:
- Scale 1 (no-op).
- Scale 2 and 3 (median path).
- Scale 4+ (valid-mean path).
- Output shape formula: (H // scale, W // scale).
- Cropping to multiples of scale.
- dtype preservation.
- All-zero tiles yield zero output.
- Single-pixel per tile (trivial median/mean).
- Scale clamping to [1, 8].
- Partial-tile cropping correctness.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.decimation import DecimationFilter


def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scale2() -> DecimationFilter:
    return DecimationFilter(scale=2)


@pytest.fixture
def scale4() -> DecimationFilter:
    return DecimationFilter(scale=4)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_scale(self):
        assert DecimationFilter().scale == 2

    def test_clamps_below_1(self):
        assert DecimationFilter(scale=0).scale == 1

    def test_clamps_above_8(self):
        assert DecimationFilter(scale=99).scale == 8

    def test_repr(self):
        assert "DecimationFilter(scale=3)" in repr(DecimationFilter(scale=3))


# ---------------------------------------------------------------------------
# Scale 1 — no-op
# ---------------------------------------------------------------------------

class TestScaleOne:
    def test_returns_same_array(self):
        f = DecimationFilter(scale=1)
        data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        inp = mx.array(data)
        out = f.process(inp)
        np.testing.assert_array_equal(_np(out), data)

    def test_shape_unchanged(self):
        f = DecimationFilter(scale=1)
        inp = mx.array(np.ones((48, 64), dtype=np.uint16))
        assert f.process(inp).shape == (48, 64)


# ---------------------------------------------------------------------------
# Scale 2 — median path
# ---------------------------------------------------------------------------

class TestScale2:
    def test_output_shape(self, scale2):
        inp = mx.array(np.ones((48, 64), dtype=np.uint16))
        out = scale2.process(inp)
        assert out.shape == (24, 32)

    def test_uniform_tile_median_equals_value(self, scale2):
        data = np.full((4, 4), 1000, dtype=np.uint16)
        out = _np(scale2.process(mx.array(data)))
        np.testing.assert_allclose(out, 1000, atol=1)

    def test_known_tile_median(self, scale2):
        # 2×2 tile: [100, 200, 300, 400] → sorted: 100,200,300,400 → median=250
        data = np.array([[100, 200],
                         [300, 400]], dtype=np.uint16)
        out = _np(scale2.process(mx.array(data)))
        assert out.shape == (1, 1)
        assert abs(int(out[0, 0]) - 250) <= 1  # median of 4 values

    def test_dtype_uint16_preserved(self, scale2):
        inp = mx.array(np.ones((4, 4), dtype=np.uint16) * 500)
        assert scale2.process(inp).dtype == mx.uint16

    def test_dtype_float32_preserved(self, scale2):
        inp = mx.array(np.ones((4, 4), dtype=np.float32) * 500.0)
        assert scale2.process(inp).dtype == mx.float32

    def test_non_multiple_height_crops(self, scale2):
        # 5 rows → cropped to 4 → output 2 rows
        inp = mx.array(np.ones((5, 4), dtype=np.uint16) * 500)
        out = scale2.process(inp)
        assert out.shape == (2, 2)

    def test_non_multiple_width_crops(self, scale2):
        # 3 cols → cropped to 2 → output 1 col
        inp = mx.array(np.ones((4, 3), dtype=np.uint16) * 500)
        out = scale2.process(inp)
        assert out.shape == (2, 1)

    def test_all_zero_tiles(self, scale2):
        inp = mx.array(np.zeros((4, 4), dtype=np.uint16))
        out = _np(scale2.process(inp))
        assert np.all(out == 0)


# ---------------------------------------------------------------------------
# Scale 3 — still uses median path
# ---------------------------------------------------------------------------

class TestScale3:
    def test_output_shape(self):
        f = DecimationFilter(scale=3)
        inp = mx.array(np.ones((9, 9), dtype=np.uint16))
        assert f.process(inp).shape == (3, 3)

    def test_uniform_tile(self):
        f = DecimationFilter(scale=3)
        data = np.full((9, 9), 750, dtype=np.uint16)
        out = _np(f.process(mx.array(data)))
        np.testing.assert_allclose(out, 750, atol=1)

    def test_single_tile(self):
        f = DecimationFilter(scale=3)
        # 3×3 tile with known median = 5.
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.uint16)
        out = _np(f.process(mx.array(data)))
        assert out.shape == (1, 1)
        assert abs(int(out[0, 0]) - 5) <= 1


# ---------------------------------------------------------------------------
# Scale 4+ — valid-mean path
# ---------------------------------------------------------------------------

class TestScale4:
    def test_output_shape(self, scale4):
        inp = mx.array(np.ones((16, 16), dtype=np.uint16))
        assert scale4.process(inp).shape == (4, 4)

    def test_uniform_tile_mean(self, scale4):
        data = np.full((8, 8), 2000, dtype=np.uint16)
        out = _np(scale4.process(mx.array(data)))
        np.testing.assert_allclose(out, 2000, atol=1)

    def test_tile_with_zeros_excluded_from_mean(self, scale4):
        # Frame: 4 rows × 8 cols, split into two 4×4 tiles.
        # Left tile (cols 0-3): all 1000 → valid mean = 1000.
        # Right tile (cols 4-7): all zeros → no valid pixels → output = 0.
        data = np.zeros((4, 8), dtype=np.uint16)
        data[:, :4] = 1000  # left tile fully valid
        out = _np(scale4.process(mx.array(data)))
        assert out.shape == (1, 2)
        assert abs(int(out[0, 0]) - 1000) <= 1  # left tile: all 1000
        assert int(out[0, 1]) == 0               # right tile: all zero → 0

    def test_mixed_valid_tile_mean_correct(self, scale4):
        # 4×4 tile where half pixels are 1000 and half are 2000 → mean = 1500.
        data = np.zeros((4, 4), dtype=np.uint16)
        data[:2, :] = 1000
        data[2:, :] = 2000
        out = _np(scale4.process(mx.array(data)))
        assert out.shape == (1, 1)
        assert abs(int(out[0, 0]) - 1500) <= 1

    def test_all_zero_tile_gives_zero_output(self, scale4):
        data = np.zeros((4, 4), dtype=np.uint16)
        out = _np(scale4.process(mx.array(data)))
        assert np.all(out == 0)

    def test_dtype_preserved(self, scale4):
        inp = mx.array(np.ones((8, 8), dtype=np.uint16) * 300)
        assert scale4.process(inp).dtype == mx.uint16


# ---------------------------------------------------------------------------
# Ramp pattern — gradual depth gradient
# ---------------------------------------------------------------------------

class TestRampPattern:
    def test_scale2_preserves_gradient_direction(self):
        f = DecimationFilter(scale=2)
        # Horizontal ramp: increasing left to right.
        row = np.arange(0, 200, 2, dtype=np.uint16).reshape(1, -1)
        row = np.tile(row, (2, 1))  # 2×100 frame
        out = _np(f.process(mx.array(row)))
        # Output should be monotonically increasing.
        assert out.shape[1] > 1
        assert np.all(np.diff(out[0]) >= 0)
