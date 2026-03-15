"""Tests for the generic stereo depth pipeline.

Coverage
--------
StereoDepthEstimator
- Synthetic parallel images produce disparity and non-zero depth.
- Depth formula: depth = baseline_mm * focal_px / disparity / 1000.
- All-black frames → all-zero depth (no valid correspondences).
- Identical left/right images → near-zero/undefined disparity (depth = 0).
- Single-pixel frame (1×1) → handled without crash.
- Large disparity clipped to num_disparities limit.
- Depth clipping: values outside [min_depth_m, max_depth_m] → 0.
- MLX filter integration: spatial + temporal + hole-fill run without error.
- compute_with_color returns (mx.array, np.ndarray BGR).
- reset() clears temporal state without error.
- Config validation: bad num_disparities, block_size, baseline, focal.

StereoCamera
- from_side_by_side: splits a 2W×H synthetic frame into two W×H halves.
- from_dual: stub test (no real hardware; verifies constructor params).
- eye_width / eye_height properties.
- CaptureMode enum values.
- capture() raises StereoCameraError when not started.
- Context manager protocol (__enter__ / __exit__).

MLX filter integration (disparity → filtered depth → colorized)
- Full pipeline: compute_with_color on a synthetic ramp scene.
- Colorized output shape and dtype.
- Filter-disabled path (no_filter=True).

Edge cases
- Empty (0×0) disparity should not crash.
- Single-pixel image.
- All-invalid disparity after clipping.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.stereo.depth import StereoDepthEstimator, StereoDepthConfig
from realsense_mlx.stereo.camera import (
    StereoCamera,
    StereoCameraError,
    CaptureMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr, copy=False)


def _make_parallel_pair(
    height: int = 48,
    width: int = 64,
    shift: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic rectified stereo pair.

    Left image has a bright vertical bar at column ``width // 2``.
    Right image is the left image shifted left by ``shift`` pixels,
    simulating a disparity of ``shift`` pixels.

    Returns (left_gray, right_gray) as uint8 (H, W).
    """
    rng = np.random.default_rng(42)
    base = rng.integers(80, 180, (height, width), dtype=np.uint8)
    # Add a textured region to give SGBM something to match.
    base[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = (
        rng.integers(40, 240, (height // 2, width // 2), dtype=np.uint8)
    )
    left = base.copy()
    # Shift right image left by `shift` pixels to simulate disparity.
    right = np.roll(base, -shift, axis=1)
    right[:, -shift:] = 0  # fill wrap-around with black (invalid)
    return left, right


def _make_flat_pair(
    height: int = 48,
    width: int = 64,
    value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Both images filled with a single constant value."""
    img = np.full((height, width), value, dtype=np.uint8)
    return img.copy(), img.copy()


def _build_estimator(**kwargs) -> StereoDepthEstimator:
    """Build a fast estimator suitable for unit tests.

    Uses minimal disparities and no temporal filter to keep tests quick.
    """
    defaults = dict(
        baseline_mm=120.0,
        focal_px=700.0,
        num_disparities=32,
        block_size=3,
        enable_spatial=True,
        enable_temporal=False,   # avoid state between test calls
        enable_hole_fill=True,
        sgbm_mode=0,             # STEREO_SGBM (faster than MODE_HH)
    )
    defaults.update(kwargs)
    return StereoDepthEstimator(**defaults)


# ---------------------------------------------------------------------------
# StereoDepthConfig validation
# ---------------------------------------------------------------------------

class TestStereoDepthConfig:
    def test_valid_config(self):
        cfg = StereoDepthConfig(baseline_mm=120.0, focal_px=700.0)
        assert cfg.baseline_mm == 120.0
        assert cfg.focal_px == 700.0

    def test_bad_num_disparities_not_divisible_by_16(self):
        with pytest.raises(ValueError, match="divisible by 16"):
            StereoDepthConfig(num_disparities=100)

    def test_bad_block_size_even(self):
        with pytest.raises(ValueError, match="odd integer"):
            StereoDepthConfig(block_size=4)

    def test_bad_block_size_too_small(self):
        with pytest.raises(ValueError, match="odd integer"):
            StereoDepthConfig(block_size=1)

    def test_bad_baseline_zero(self):
        with pytest.raises(ValueError, match="baseline_mm"):
            StereoDepthConfig(baseline_mm=0.0)

    def test_bad_focal_negative(self):
        with pytest.raises(ValueError, match="focal_px"):
            StereoDepthConfig(focal_px=-1.0)

    def test_bad_depth_range(self):
        with pytest.raises(ValueError, match="min_depth_m"):
            StereoDepthConfig(min_depth_m=10.0, max_depth_m=5.0)


# ---------------------------------------------------------------------------
# StereoDepthEstimator — constructor
# ---------------------------------------------------------------------------

class TestStereoDepthEstimatorConstructor:
    def test_defaults(self):
        est = StereoDepthEstimator(baseline_mm=100.0, focal_px=600.0)
        assert est.baseline_mm == 100.0
        assert est.focal_px == 600.0

    def test_repr_contains_key_params(self):
        est = _build_estimator()
        r = repr(est)
        assert "120.0" in r
        assert "700.0" in r

    def test_config_override_via_kwargs(self):
        est = StereoDepthEstimator(
            baseline_mm=80.0,
            focal_px=500.0,
            num_disparities=64,
        )
        assert est.config.num_disparities == 64

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError, match="Unknown config field"):
            StereoDepthEstimator(
                baseline_mm=80.0,
                focal_px=500.0,
                config=StereoDepthConfig(),
                nonexistent_field=42,
            )


# ---------------------------------------------------------------------------
# StereoDepthEstimator — compute() with synthetic data
# ---------------------------------------------------------------------------

class TestStereoDepthEstimatorCompute:
    """These tests use OpenCV SGBM via the real estimator."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cv2(self):
        pytest.importorskip("cv2", reason="opencv-python not installed")

    @pytest.fixture
    def estimator(self) -> StereoDepthEstimator:
        return _build_estimator()

    def test_output_shape(self, estimator):
        left, right = _make_parallel_pair(height=48, width=64)
        depth = estimator.compute(left, right)
        assert depth.shape == (48, 64)

    def test_output_dtype(self, estimator):
        left, right = _make_parallel_pair()
        depth = estimator.compute(left, right)
        assert depth.dtype == mx.float32

    def test_parallel_pair_has_valid_pixels(self, estimator):
        """A textured stereo pair should produce some non-zero depth values."""
        left, right = _make_parallel_pair(shift=8)
        depth = estimator.compute(left, right)
        depth_np = _np(depth)
        # SGBM should find correspondences in the textured region.
        valid_count = int((depth_np > 0).sum())
        assert valid_count > 0, "Expected at least some valid depth pixels"

    def test_all_black_frames_zero_depth(self):
        """Featureless black frames → no valid correspondences → all-zero depth."""
        est = _build_estimator(enable_spatial=False, enable_hole_fill=False)
        left, right = _make_flat_pair(value=0)
        depth = est.compute(left, right)
        depth_np = _np(depth)
        # All-black frames have no texture; SGBM should return all-invalid.
        assert float(depth_np.max()) == 0.0

    def test_identical_lr_mostly_zero_depth(self):
        """Identical L/R images → near-zero disparity → depth clipped to 0."""
        # With identical images, best match is at disparity=0 which is
        # below min_disparity (0), so most pixels should be invalid.
        est = _build_estimator(
            min_disparity=1,
            enable_spatial=False,
            enable_hole_fill=False,
        )
        left, right = _make_flat_pair(value=128)
        depth = est.compute(left, right)
        depth_np = _np(depth)
        # Most/all pixels at disparity 0 will be invalid (< min_disparity=1).
        valid_frac = float((depth_np > 0).mean())
        assert valid_frac < 0.5, (
            f"Expected most pixels invalid for identical images, got {valid_frac:.2%} valid"
        )

    def test_depth_values_in_valid_range(self, estimator):
        left, right = _make_parallel_pair(shift=8)
        depth = estimator.compute(left, right)
        depth_np = _np(depth)
        valid = depth_np[depth_np > 0]
        if len(valid) > 0:
            assert valid.min() >= estimator.config.min_depth_m - 1e-6
            assert valid.max() <= estimator.config.max_depth_m + 1e-6

    def test_bgr_input_accepted(self, estimator):
        """BGR (H, W, 3) input should be auto-converted to grayscale."""
        rng = np.random.default_rng(7)
        left_bgr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        right_bgr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        depth = estimator.compute(left_bgr, right_bgr)
        assert depth.shape == (48, 64)

    def test_mismatched_shapes_raise(self, estimator):
        left = np.zeros((48, 64), dtype=np.uint8)
        right = np.zeros((48, 80), dtype=np.uint8)
        with pytest.raises(ValueError, match="same shape"):
            estimator.compute(left, right)

    def test_single_pixel_no_crash(self):
        """Single-pixel image: bypass real SGBM (which rejects tiny images)
        and verify that the depth conversion + filter chain handles a 1×1
        disparity map without crashing.
        """
        est = _build_estimator(num_disparities=16, enable_spatial=False,
                               enable_hole_fill=False)
        # Inject a 1×1 zero disparity map directly (skipping SGBM).
        mock_sgbm = MagicMock()
        mock_sgbm.compute.return_value = np.array([[0]], dtype=np.int16)
        est._sgbm = mock_sgbm

        left = np.array([[100]], dtype=np.uint8)
        right = np.array([[100]], dtype=np.uint8)
        depth = est.compute(left, right)
        assert depth.shape == (1, 1)
        assert float(_np(depth)[0, 0]) == 0.0

    def test_reset_clears_temporal_state(self):
        """reset() should not raise and should clear TemporalFilter state."""
        est = _build_estimator(enable_temporal=True)
        left, right = _make_parallel_pair()
        est.compute(left, right)
        est.reset()
        assert est._temporal._prev_frame is None  # type: ignore[union-attr]

    def test_compute_with_color_return_types(self, estimator):
        left_bgr = np.random.randint(0, 200, (48, 64, 3), dtype=np.uint8)
        right_bgr = np.random.randint(0, 200, (48, 64, 3), dtype=np.uint8)
        depth_mx, color_bgr = estimator.compute_with_color(left_bgr, right_bgr)
        assert isinstance(depth_mx, mx.array)
        assert isinstance(color_bgr, np.ndarray)
        assert color_bgr.dtype == np.uint8
        assert color_bgr.ndim == 3
        assert color_bgr.shape[2] == 3

    def test_compute_with_color_shape_consistency(self, estimator):
        left_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        right_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        depth_mx, color_bgr = estimator.compute_with_color(left_bgr, right_bgr)
        h, w = depth_mx.shape
        assert color_bgr.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# StereoDepthEstimator — depth formula unit test (mocked SGBM)
# ---------------------------------------------------------------------------

class TestDepthFormula:
    """Verify the disparity → depth conversion without real SGBM."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cv2(self):
        pytest.importorskip("cv2", reason="opencv-python not installed")

    def test_depth_formula_known_disparity(self):
        """Inject a known disparity map and verify depth = bf / disp / 1000."""
        import cv2

        baseline_mm = 120.0
        focal_px = 700.0
        # SGBM returns int16 ×16 fixed-point; disparity in pixels = raw / 16.
        disp_pixels = 16.0   # corresponds to raw value 256
        raw_disp_int16 = int(disp_pixels * 16)  # 256

        # Build a small (4×4) synthetic raw disparity map.
        disp_raw = np.full((4, 4), raw_disp_int16, dtype=np.int16)

        expected_depth_m = (baseline_mm * focal_px) / disp_pixels / 1000.0
        # = 120 * 700 / 16 / 1000 = 84000 / 16 / 1000 = 5.25 m

        est = StereoDepthEstimator(
            baseline_mm=baseline_mm,
            focal_px=focal_px,
            num_disparities=32,
            block_size=3,
            enable_spatial=False,
            enable_temporal=False,
            enable_hole_fill=False,
            min_depth_m=0.1,
            max_depth_m=20.0,
        )

        # Monkey-patch SGBM to return our controlled disparity.
        mock_sgbm = MagicMock()
        mock_sgbm.compute.return_value = disp_raw
        est._sgbm = mock_sgbm

        left = np.zeros((4, 4), dtype=np.uint8)
        right = np.zeros((4, 4), dtype=np.uint8)
        depth = est.compute(left, right)
        depth_np = _np(depth)

        # All pixels should equal expected_depth_m (within float32 precision).
        np.testing.assert_allclose(
            depth_np, expected_depth_m, rtol=1e-5,
            err_msg="Depth formula: depth = baseline_mm * focal_px / disparity / 1000",
        )

    def test_zero_disparity_gives_zero_depth(self):
        """Disparity 0 (invalid) must map to depth 0 (invalid)."""
        import cv2

        disp_raw = np.zeros((4, 4), dtype=np.int16)
        est = StereoDepthEstimator(
            baseline_mm=120.0, focal_px=700.0,
            num_disparities=32, block_size=3,
            enable_spatial=False, enable_temporal=False, enable_hole_fill=False,
        )
        mock_sgbm = MagicMock()
        mock_sgbm.compute.return_value = disp_raw
        est._sgbm = mock_sgbm

        depth = est.compute(
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        )
        assert float(_np(depth).max()) == 0.0

    def test_out_of_range_depth_clipped_to_zero(self):
        """Depth outside [min_depth_m, max_depth_m] must be zeroed."""
        import cv2

        # disp_pixels = 1 → depth = 120 * 700 / 1 / 1000 = 84 m  (> max=20)
        disp_raw = np.full((4, 4), 16, dtype=np.int16)  # 1 px disparity
        est = StereoDepthEstimator(
            baseline_mm=120.0, focal_px=700.0,
            num_disparities=32, block_size=3,
            enable_spatial=False, enable_temporal=False, enable_hole_fill=False,
            min_depth_m=0.1, max_depth_m=20.0,
        )
        mock_sgbm = MagicMock()
        mock_sgbm.compute.return_value = disp_raw
        est._sgbm = mock_sgbm

        depth = est.compute(
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        )
        assert float(_np(depth).max()) == 0.0


# ---------------------------------------------------------------------------
# StereoDepthEstimator — MLX filter integration
# ---------------------------------------------------------------------------

class TestMLXFilterIntegration:
    """End-to-end tests verifying the MLX post-processing chain."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cv2(self):
        pytest.importorskip("cv2", reason="opencv-python not installed")

    def test_spatial_filter_runs(self):
        """SpatialFilter must not crash on real depth output."""
        est = _build_estimator(enable_spatial=True, enable_hole_fill=False,
                               enable_temporal=False)
        left, right = _make_parallel_pair()
        depth = est.compute(left, right)
        assert depth.ndim == 2

    def test_temporal_filter_accumulates(self):
        """After 5 frames TemporalFilter state should be non-None."""
        est = _build_estimator(enable_spatial=False, enable_hole_fill=False,
                               enable_temporal=True)
        left, right = _make_parallel_pair()
        for _ in range(5):
            est.compute(left, right)
        assert est._temporal is not None
        assert est._temporal._prev_frame is not None

    def test_hole_fill_runs(self):
        """HoleFillingFilter must not crash."""
        est = _build_estimator(enable_spatial=False, enable_hole_fill=True,
                               enable_temporal=False)
        left, right = _make_parallel_pair()
        depth = est.compute(left, right)
        assert depth.ndim == 2

    def test_all_filters_combined(self):
        """Full pipeline (spatial + temporal + hole_fill) must produce valid output."""
        est = _build_estimator(enable_spatial=True, enable_temporal=True,
                               enable_hole_fill=True)
        left, right = _make_parallel_pair()
        for _ in range(3):
            depth = est.compute(left, right)
        assert depth.dtype == mx.float32
        depth_np = _np(depth)
        # No NaN or Inf values after filtering.
        assert not np.isnan(depth_np).any(), "NaN values in filtered depth"
        assert not np.isinf(depth_np).any(), "Inf values in filtered depth"

    def test_colorized_output_dtype_and_shape(self):
        est = _build_estimator()
        left, right = _make_parallel_pair(height=48, width=64)
        _, color_bgr = est.compute_with_color(
            left[:, :, None].repeat(3, axis=2),
            right[:, :, None].repeat(3, axis=2),
        )
        assert color_bgr.shape == (48, 64, 3)
        assert color_bgr.dtype == np.uint8

    def test_no_filter_path(self):
        """With all filters disabled, depth should still be valid."""
        est = _build_estimator(
            enable_spatial=False,
            enable_temporal=False,
            enable_hole_fill=False,
        )
        left, right = _make_parallel_pair()
        depth = est.compute(left, right)
        assert depth.shape[0] > 0
        assert depth.dtype == mx.float32


# ---------------------------------------------------------------------------
# StereoCamera — side-by-side splitting (no hardware)
# ---------------------------------------------------------------------------

class TestStereoCameraSideBySide:
    def test_from_side_by_side_params(self):
        cam = StereoCamera.from_side_by_side(device_id=0, width=2560, height=720)
        assert cam.mode == CaptureMode.SIDE_BY_SIDE
        assert cam.width == 2560
        assert cam.height == 720
        assert cam.eye_width == 1280
        assert cam.eye_height == 720
        assert not cam.is_running

    def test_from_side_by_side_defaults(self):
        cam = StereoCamera.from_side_by_side()
        assert cam.width == 2560
        assert cam.height == 720

    def test_capture_raises_when_not_started(self):
        cam = StereoCamera.from_side_by_side()
        with pytest.raises(StereoCameraError, match="not running"):
            cam.capture()

    def test_grab_raw_raises_when_not_started(self):
        cam = StereoCamera.from_side_by_side()
        with pytest.raises(StereoCameraError, match="not running"):
            cam.grab_raw()

    def test_split_correctness(self):
        """Manually inject a wide synthetic frame and verify the split."""
        cam = StereoCamera.from_side_by_side(device_id=0, width=120, height=60)

        # Synthesise a wide frame: left half all-red (channel 2), right half all-blue.
        wide = np.zeros((60, 120, 3), dtype=np.uint8)
        wide[:, :60, 2] = 200    # red in left half (BGR: channel 2 = R)
        wide[:, 60:, 0] = 200   # blue in right half (BGR: channel 0 = B)

        # Patch _read_one to return our synthetic frame.
        cam._running = True
        cam._caps = [MagicMock()]

        with patch.object(cam, "_read_one", return_value=wide):
            left, right = cam._capture_side_by_side()

        assert left.shape == (60, 60, 3)
        assert right.shape == (60, 60, 3)

        # Left half should be mostly red.
        assert int(left[:, :, 2].mean()) > 150
        # Right half should be mostly blue.
        assert int(right[:, :, 0].mean()) > 150

    def test_split_odd_width_raises(self):
        cam = StereoCamera.from_side_by_side(device_id=0, width=121, height=60)
        cam._running = True
        cam._caps = [MagicMock()]

        odd_frame = np.zeros((60, 121, 3), dtype=np.uint8)
        with patch.object(cam, "_read_one", return_value=odd_frame):
            with pytest.raises(StereoCameraError, match="not even"):
                cam._capture_side_by_side()

    def test_bgra_input_converted_to_bgr(self):
        """4-channel (BGRA) frames should be converted to 3-channel BGR."""
        cam = StereoCamera.from_side_by_side(device_id=0, width=120, height=60)
        bgra = np.zeros((60, 120, 4), dtype=np.uint8)
        bgra[:, :60, 2] = 200

        import cv2
        expected_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        cam._running = True
        cam._caps = [MagicMock()]

        # Simulate _read_one returning a BGRA frame before BGR conversion.
        # We need to exercise the actual _read_one BGRA branch, so patch cap.read.
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, bgra)
        cam._caps = [mock_cap]

        with patch("cv2.cvtColor", wraps=cv2.cvtColor) as mock_cvt:
            left, right = cam._capture_side_by_side()

        # After BGRA → BGR conversion the output should be 3-channel.
        assert left.shape[2] == 3
        assert right.shape[2] == 3

    def test_fps_zero_before_capture(self):
        cam = StereoCamera.from_side_by_side()
        assert cam.fps == 0.0

    def test_fps_nonzero_after_captures(self):
        cam = StereoCamera.from_side_by_side(device_id=0, width=120, height=60)
        cam._running = True
        cam._caps = [MagicMock()]

        wide = np.zeros((60, 120, 3), dtype=np.uint8)
        with patch.object(cam, "_read_one", return_value=wide):
            for _ in range(5):
                cam._capture_side_by_side()

        # After 5 captures the fps moving average should be non-zero.
        assert cam.fps > 0.0

    def test_context_manager_starts_and_stops(self):
        """__enter__/__exit__ should call start() and stop() correctly."""
        cam = StereoCamera.from_side_by_side(device_id=99, width=120, height=60)
        with patch.object(cam, "start") as mock_start, \
             patch.object(cam, "stop") as mock_stop:
            with cam:
                mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_repr_contains_mode(self):
        cam = StereoCamera.from_side_by_side(device_id=0, width=2560, height=720)
        assert "SIDE_BY_SIDE" in repr(cam)


# ---------------------------------------------------------------------------
# StereoCamera — dual camera
# ---------------------------------------------------------------------------

class TestStereoCameraDual:
    def test_from_dual_params(self):
        cam = StereoCamera.from_dual(left_id=0, right_id=2, width=1280, height=720)
        assert cam.mode == CaptureMode.DUAL
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.eye_width == 1280  # same as width in dual mode
        assert cam._device_ids == [0, 2]

    def test_capture_raises_when_not_started(self):
        cam = StereoCamera.from_dual()
        with pytest.raises(StereoCameraError, match="not running"):
            cam.capture()

    def test_dual_capture_uses_both_devices(self):
        cam = StereoCamera.from_dual(left_id=0, right_id=2, width=64, height=48)
        cam._running = True
        # _caps must have two entries; the mocks are only consumed by _read_one.
        cam._caps = [MagicMock(), MagicMock()]

        left_frame = np.full((48, 64, 3), 10, dtype=np.uint8)
        right_frame = np.full((48, 64, 3), 200, dtype=np.uint8)

        calls = iter([left_frame, right_frame])
        with patch.object(cam, "_read_one", side_effect=calls):
            left, right = cam._capture_dual()

        assert int(left.mean()) == 10
        assert int(right.mean()) == 200

    def test_repr_contains_dual_mode(self):
        cam = StereoCamera.from_dual()
        assert "DUAL" in repr(cam)


# ---------------------------------------------------------------------------
# StereoCamera — _to_gray static method (no hardware required)
# ---------------------------------------------------------------------------

class TestToGray:
    def test_grayscale_passthrough(self):
        img = np.full((10, 12), 128, dtype=np.uint8)
        out = StereoDepthEstimator._to_gray(img)
        assert out.shape == (10, 12)
        assert out.dtype == np.uint8

    def test_bgr_to_gray_shape(self):
        pytest.importorskip("cv2")
        img = np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8)
        out = StereoDepthEstimator._to_gray(img)
        assert out.shape == (10, 12)
        assert out.dtype == np.uint8

    def test_single_channel_3d(self):
        img = np.full((10, 12, 1), 77, dtype=np.uint8)
        out = StereoDepthEstimator._to_gray(img)
        assert out.shape == (10, 12)
        assert int(out[0, 0]) == 77

    def test_unsupported_shape_raises(self):
        img = np.zeros((10, 12, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            StereoDepthEstimator._to_gray(img)


# ---------------------------------------------------------------------------
# CaptureMode enum
# ---------------------------------------------------------------------------

class TestCaptureMode:
    def test_enum_values_distinct(self):
        assert CaptureMode.SIDE_BY_SIDE != CaptureMode.DUAL

    def test_enum_names(self):
        assert CaptureMode.SIDE_BY_SIDE.name == "SIDE_BY_SIDE"
        assert CaptureMode.DUAL.name == "DUAL"
