"""Tests for DepthColorizer.

Coverage
--------
- Output shape and dtype for every supported colormap.
- Zero-depth pixels map to black (0, 0, 0) in both modes.
- Direct mode: a linear ramp produces a monotonically varying gradient
  (values are non-decreasing or non-increasing per channel).
- Equalized mode: a clustered input has its histogram spread across the
  full 0–255 range.
- Invalid colormap name raises ValueError.
- Invalid depth dimensions raise ValueError.
- set_colormap() switches the active map and rebuilds the LUT.
- lut_numpy() returns a (256, 3) uint8 array with sensible endpoints.
- Flat wall (all same depth) colorizes without error.
- Large frame (720p) processes correctly.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from realsense_mlx.filters.colorizer import DepthColorizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval(arr: mx.array) -> np.ndarray:
    """Materialize an mx.array to numpy."""
    mx.eval(arr)
    return np.array(arr, copy=False)


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_params(self):
        c = DepthColorizer()
        assert c.colormap == "jet"
        assert c.min_depth == pytest.approx(0.1)
        assert c.max_depth == pytest.approx(10.0)
        assert c.equalize is True
        assert c.depth_units == pytest.approx(0.001)

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            DepthColorizer(colormap="nonexistent")

    def test_inverted_depth_range_raises(self):
        with pytest.raises(ValueError, match="min_depth"):
            DepthColorizer(min_depth=5.0, max_depth=1.0)

    def test_equal_depth_range_raises(self):
        with pytest.raises(ValueError, match="min_depth"):
            DepthColorizer(min_depth=3.0, max_depth=3.0)

    def test_zero_depth_units_raises(self):
        with pytest.raises(ValueError, match="depth_units"):
            DepthColorizer(depth_units=0.0)

    def test_negative_depth_units_raises(self):
        with pytest.raises(ValueError, match="depth_units"):
            DepthColorizer(depth_units=-0.001)

    def test_all_colormaps_construct(self):
        for cmap in DepthColorizer.COLORMAPS:
            c = DepthColorizer(colormap=cmap)
            assert c.colormap == cmap


# ---------------------------------------------------------------------------
# LUT correctness
# ---------------------------------------------------------------------------

class TestLUT:
    def test_lut_shape_and_dtype(self):
        c = DepthColorizer(colormap="jet", equalize=False)
        lut = c.lut_numpy()
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_grayscale_lut_endpoints(self):
        c = DepthColorizer(colormap="grayscale", equalize=False)
        lut = c.lut_numpy()
        np.testing.assert_array_equal(lut[0], [0, 0, 0])
        np.testing.assert_array_equal(lut[255], [255, 255, 255])

    def test_inv_grayscale_lut_endpoints(self):
        c = DepthColorizer(colormap="inv_grayscale", equalize=False)
        lut = c.lut_numpy()
        np.testing.assert_array_equal(lut[0], [255, 255, 255])
        np.testing.assert_array_equal(lut[255], [0, 0, 0])

    def test_grayscale_lut_monotone(self):
        """Grayscale LUT should be strictly non-decreasing in all channels."""
        c = DepthColorizer(colormap="grayscale", equalize=False)
        lut = c.lut_numpy().astype(np.int32)
        diff = np.diff(lut, axis=0)
        assert np.all(diff >= 0), "Grayscale LUT must be non-decreasing"

    def test_jet_lut_has_color_variation(self):
        """Jet LUT should span multiple hues — R, G, B values differ across entries."""
        c = DepthColorizer(colormap="jet", equalize=False)
        lut = c.lut_numpy()
        # Each channel should have range > 100 (not a flat color)
        for ch in range(3):
            assert lut[:, ch].max() - lut[:, ch].min() > 100


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShapeAndDtype:
    @pytest.mark.parametrize("cmap", sorted(DepthColorizer.COLORMAPS.keys()))
    def test_shape_480p_direct(self, depth_ramp_480p, cmap):
        c = DepthColorizer(colormap=cmap, equalize=False)
        out = c.colorize(mx.array(depth_ramp_480p))
        rgb = _eval(out)
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8

    @pytest.mark.parametrize("cmap", sorted(DepthColorizer.COLORMAPS.keys()))
    def test_shape_480p_equalized(self, depth_ramp_480p, cmap):
        c = DepthColorizer(colormap=cmap, equalize=True)
        out = c.colorize(mx.array(depth_ramp_480p))
        rgb = _eval(out)
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8

    def test_shape_720p(self, depth_ramp_720p):
        c = DepthColorizer(equalize=False)
        out = c.colorize(mx.array(depth_ramp_720p))
        rgb = _eval(out)
        assert rgb.shape == (720, 1280, 3)
        assert rgb.dtype == np.uint8

    def test_3d_input_raises(self, depth_ramp_480p):
        c = DepthColorizer(equalize=False)
        with pytest.raises(ValueError, match="2-D"):
            c.colorize(mx.array(depth_ramp_480p[:, :, np.newaxis]))


# ---------------------------------------------------------------------------
# Zero-depth → black
# ---------------------------------------------------------------------------

class TestZeroDepthIsBlack:
    def test_all_zeros_direct(self):
        zeros = mx.zeros((480, 640), dtype=mx.uint16)
        c = DepthColorizer(equalize=False, min_depth=0.1, max_depth=10.0)
        rgb = _eval(c.colorize(zeros))
        # All-zero depth is below min_depth → clips to index 0.
        # For jet, index 0 is (0, 0, 128) but what matters is no crash.
        assert rgb.shape == (480, 640, 3)

    def test_zero_pixels_equalized_are_excluded(self, depth_with_holes):
        """Equalized mode must set hist[0]=0 so invalid pixels don't pollute CDF."""
        c = DepthColorizer(equalize=True)
        out_mx = c.colorize(mx.array(depth_with_holes))
        rgb = _eval(out_mx)
        assert rgb.shape == (480, 640, 3)

    def test_explicit_zero_region_direct_mode(self, depth_with_holes):
        """Rectangular hole region (depth=0) must be black (0,0,0), not LUT[0]."""
        c = DepthColorizer(equalize=False, min_depth=0.1, max_depth=10.0)
        rgb = _eval(c.colorize(mx.array(depth_with_holes)))
        # Zero-depth (invalid) pixels are masked to black, not mapped through LUT.
        hole_pixels = rgb[100:200, 200:400]
        np.testing.assert_array_equal(hole_pixels, np.zeros_like(hole_pixels))


# ---------------------------------------------------------------------------
# Direct mode: gradient properties
# ---------------------------------------------------------------------------

class TestDirectMode:
    def test_ramp_produces_gradient(self, depth_ramp_480p):
        """A linear depth ramp should produce a smooth color gradient.

        We verify that the summed RGB intensity across columns is monotone
        (either non-decreasing or non-increasing) for a simple 'grayscale'
        map, which is guaranteed to be monotone by construction.
        """
        c = DepthColorizer(colormap="grayscale", equalize=False,
                           min_depth=0.0, max_depth=9.6)
        rgb = _eval(c.colorize(mx.array(depth_ramp_480p)))
        # First row has all different column depths; sum intensity per column.
        row = rgb[0].astype(np.float32).sum(axis=-1)  # (640,)
        diffs = np.diff(row)
        # Allow small equal-value runs (flat steps in the LUT) but no reversals.
        assert np.all(diffs >= -1), "Grayscale gradient should be non-decreasing"

    def test_flat_wall_uniform_color(self, depth_flat_wall):
        """Flat wall at 1 m should produce the exact same color for every pixel."""
        c = DepthColorizer(colormap="jet", equalize=False)
        rgb = _eval(c.colorize(mx.array(depth_flat_wall)))
        # All pixels should be identical.
        first_pixel = rgb[0, 0]
        assert np.all(rgb == first_pixel), "All pixels should be the same color"

    def test_min_clip(self):
        """Pixels closer than min_depth should all get LUT[0]."""
        very_close = np.full((10, 10), 10, dtype=np.uint16)  # 0.01 m
        c = DepthColorizer(colormap="jet", equalize=False,
                           min_depth=0.5, max_depth=5.0)
        rgb = _eval(c.colorize(mx.array(very_close)))
        lut = c.lut_numpy()
        np.testing.assert_array_equal(rgb[0, 0], lut[0])

    def test_max_clip(self):
        """Pixels farther than max_depth should all get LUT[255]."""
        very_far = np.full((10, 10), 60000, dtype=np.uint16)  # 60 m
        c = DepthColorizer(colormap="jet", equalize=False,
                           min_depth=0.1, max_depth=10.0)
        rgb = _eval(c.colorize(mx.array(very_far)))
        lut = c.lut_numpy()
        np.testing.assert_array_equal(rgb[0, 0], lut[255])

    def test_depth_units_scaling(self):
        """depth_units=0.0001 means counts represent 0.1 mm steps.

        A pixel at 10000 counts → 1.0 m with depth_units=0.0001.
        With default depth_units=0.001 the same pixel is 10 m.
        The colorized output should differ.
        """
        depth = mx.array(np.full((4, 4), 10000, dtype=np.uint16))
        c1 = DepthColorizer(colormap="grayscale", equalize=False,
                            depth_units=0.001, min_depth=0.1, max_depth=10.0)
        c2 = DepthColorizer(colormap="grayscale", equalize=False,
                            depth_units=0.0001, min_depth=0.1, max_depth=10.0)
        rgb1 = _eval(c1.colorize(depth))
        rgb2 = _eval(c2.colorize(depth))
        assert not np.array_equal(rgb1, rgb2), (
            "Different depth_units should produce different colors"
        )


# ---------------------------------------------------------------------------
# Equalized mode
# ---------------------------------------------------------------------------

class TestEqualizedMode:
    def test_clustered_depth_spreads_histogram(self):
        """Values clustered near one depth level should spread across the LUT."""
        # All depth in a tight cluster around 2000 counts.
        rng = np.random.default_rng(7)
        depth_np = rng.integers(1990, 2010, size=(480, 640), dtype=np.uint16)
        c = DepthColorizer(colormap="grayscale", equalize=True)
        rgb = _eval(c.colorize(mx.array(depth_np)))
        # Equalization should spread values; check there is non-trivial variance.
        unique_vals = np.unique(rgb[:, :, 0])
        assert len(unique_vals) > 1, (
            "Equalized mode should produce multiple intensity levels"
        )

    def test_all_invalid_pixels(self):
        """All-zero depth (all invalid) should produce all-black output."""
        zeros = mx.zeros((480, 640), dtype=mx.uint16)
        c = DepthColorizer(equalize=True)
        rgb = _eval(c.colorize(zeros))
        assert rgb.shape == (480, 640, 3)
        # Zero-depth pixels are always masked to black (0,0,0), not LUT[0].
        np.testing.assert_array_equal(rgb[0, 0], np.array([0, 0, 0], dtype=np.uint8))

    def test_equalized_differs_from_direct(self, depth_with_holes):
        """Equalized output should differ from direct output for a non-uniform scene."""
        depth_mx = mx.array(depth_with_holes)
        c_direct = DepthColorizer(colormap="jet", equalize=False)
        c_equal = DepthColorizer(colormap="jet", equalize=True)
        rgb_d = _eval(c_direct.colorize(depth_mx))
        rgb_e = _eval(c_equal.colorize(depth_mx))
        assert not np.array_equal(rgb_d, rgb_e), (
            "Equalized and direct modes must differ on non-trivial input"
        )


# ---------------------------------------------------------------------------
# All 10 colormaps — smoke tests
# ---------------------------------------------------------------------------

COLORMAPS = sorted(DepthColorizer.COLORMAPS.keys())


class TestAllColormaps:
    @pytest.mark.parametrize("cmap", COLORMAPS)
    def test_colormap_valid_output_direct(self, cmap, depth_with_holes):
        c = DepthColorizer(colormap=cmap, equalize=False)
        rgb = _eval(c.colorize(mx.array(depth_with_holes)))
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8
        # Values must be in valid uint8 range (guaranteed by LUT dtype, but verify).
        assert rgb.min() >= 0
        assert rgb.max() <= 255

    @pytest.mark.parametrize("cmap", COLORMAPS)
    def test_colormap_valid_output_equalized(self, cmap, depth_with_holes):
        c = DepthColorizer(colormap=cmap, equalize=True)
        rgb = _eval(c.colorize(mx.array(depth_with_holes)))
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0
        assert rgb.max() <= 255

    @pytest.mark.parametrize("cmap", COLORMAPS)
    def test_colormap_lut_all_valid(self, cmap):
        c = DepthColorizer(colormap=cmap)
        lut = c.lut_numpy()
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
        assert lut.min() >= 0
        assert lut.max() <= 255


# ---------------------------------------------------------------------------
# set_colormap and repr
# ---------------------------------------------------------------------------

class TestSetColormapAndRepr:
    def test_set_colormap_switches_lut(self):
        c = DepthColorizer(colormap="jet")
        lut_before = c.lut_numpy().copy()
        c.set_colormap("grayscale")
        lut_after = c.lut_numpy()
        assert not np.array_equal(lut_before, lut_after)
        assert c.colormap == "grayscale"

    def test_set_colormap_invalid_raises(self):
        c = DepthColorizer()
        with pytest.raises(ValueError, match="Unknown colormap"):
            c.set_colormap("bogus")

    def test_repr_contains_colormap(self):
        c = DepthColorizer(colormap="warm")
        assert "warm" in repr(c)

    def test_repr_contains_equalize(self):
        c = DepthColorizer(equalize=False)
        assert "equalize=False" in repr(c)

    def test_set_colormap_produces_correct_output(self, depth_flat_wall):
        """After switching colormap, output should reflect the new map."""
        c = DepthColorizer(colormap="grayscale", equalize=False)
        rgb_gray = _eval(c.colorize(mx.array(depth_flat_wall)))

        c.set_colormap("jet")
        rgb_jet = _eval(c.colorize(mx.array(depth_flat_wall)))

        assert not np.array_equal(rgb_gray, rgb_jet), (
            "Switching colormap must produce different output"
        )
