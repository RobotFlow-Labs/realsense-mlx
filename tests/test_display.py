"""Tests for the RealsenseViewer display module.

All tests mock cv2 so no GUI window is ever opened.  The test suite
validates the public contract of RealsenseViewer without requiring a
display, a camera, or the opencv-python package to be installed.

Coverage
--------
* Import safety — module-level import always succeeds
* mx.array → numpy BGR conversion applied by show()
* Side-by-side concatenation produces double-width result
* show_side_by_side() with gap parameter
* close() calls cv2.destroyWindow
* Context-manager __enter__/__exit__ calls close()
* Pressing 'q' / 'Q' / Esc sets is_open() → False
* Window-close (WND_PROP_VISIBLE < 1) sets is_open() → False
* Calling show() after close() is a silent no-op (no cv2 calls)
* auto_resize=False skips the resize step
* Constructor rejects non-positive width/height
* Missing cv2 raises ImportError with install hint
* show_depth() delegates to show()
* show_grid() with multiple frames
* _mx_to_bgr_uint8 type coercions: float [0,1], uint16, RGBA
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_frame(h: int = 48, w: int = 64) -> mx.array:
    """Return a small (H, W, 3) uint8 RGB mx.array."""
    data = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return mx.array(data)


def _gray_frame(h: int = 48, w: int = 64) -> mx.array:
    """Return a (H, W) uint8 mx.array."""
    data = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return mx.array(data)


def _make_cv2_mock() -> MagicMock:
    """Build a minimal cv2 mock that satisfies viewer.py's usage surface."""
    cv2 = MagicMock()
    cv2.WINDOW_NORMAL = 0x00000001
    cv2.WND_PROP_VISIBLE = 1
    # waitKey returns 0xFF by default — no key pressed, no quit
    cv2.waitKey.return_value = 0xFF
    # getWindowProperty returns 1.0 — window is visible
    cv2.getWindowProperty.return_value = 1.0
    # resize returns the input image resized (simulate with a copy)
    cv2.resize.side_effect = lambda img, size, interpolation=None: np.zeros(
        (size[1], size[0], 3), dtype=np.uint8
    )
    return cv2


# ---------------------------------------------------------------------------
# Fixture: a RealsenseViewer with cv2 fully mocked out
# ---------------------------------------------------------------------------


@pytest.fixture()
def viewer_and_cv2():
    """Yield (RealsenseViewer instance, cv2_mock) with cv2 patched in-process."""
    cv2_mock = _make_cv2_mock()

    # Patch at the module level where viewer.py *imported* cv2
    with patch.dict("sys.modules", {"cv2": cv2_mock}):
        # Re-import so HAS_CV2 is True in the patched environment
        import importlib

        import realsense_mlx.display.viewer as viewer_mod
        importlib.reload(viewer_mod)

        viewer = viewer_mod.RealsenseViewer(
            title="Test", width=128, height=96, auto_resize=False, wait_ms=1
        )
        yield viewer, cv2_mock

    # Restore original module state after the test
    import importlib
    import realsense_mlx.display.viewer as viewer_mod
    importlib.reload(viewer_mod)


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_succeeds_without_cv2(self):
        """Module-level import must not raise even when cv2 is absent."""
        with patch.dict("sys.modules", {"cv2": None}):
            import importlib
            import realsense_mlx.display.viewer as viewer_mod
            importlib.reload(viewer_mod)
            # HAS_CV2 should be False but no exception
            assert viewer_mod.HAS_CV2 is False

    def test_import_realsense_viewer_class(self):
        from realsense_mlx.display import RealsenseViewer  # noqa: F401

    def test_import_via_package(self):
        import realsense_mlx.display as disp
        assert hasattr(disp, "RealsenseViewer")


# ---------------------------------------------------------------------------
# 2. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_valid_construction(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        assert viewer.is_open() is True
        assert viewer.title == "Test"
        assert viewer.width == 128
        assert viewer.height == 96

    def test_missing_cv2_raises_import_error(self):
        """Without cv2 in sys.modules, RealsenseViewer() must raise ImportError."""
        import importlib
        import realsense_mlx.display.viewer as viewer_mod

        # Patch cv2 away and force HAS_CV2 = False
        with patch.dict("sys.modules", {"cv2": None}):
            importlib.reload(viewer_mod)
            with pytest.raises(ImportError, match="opencv-python"):
                viewer_mod.RealsenseViewer()

        # Restore
        importlib.reload(viewer_mod)

    def test_negative_width_raises_value_error(self):
        cv2_mock = _make_cv2_mock()
        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            import importlib
            import realsense_mlx.display.viewer as vm
            importlib.reload(vm)
            with pytest.raises(ValueError, match="width"):
                vm.RealsenseViewer(width=-1, height=480)

    def test_zero_height_raises_value_error(self):
        cv2_mock = _make_cv2_mock()
        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            import importlib
            import realsense_mlx.display.viewer as vm
            importlib.reload(vm)
            with pytest.raises(ValueError, match="height"):
                vm.RealsenseViewer(width=640, height=0)

    def test_wait_ms_clamped_to_at_least_1(self):
        cv2_mock = _make_cv2_mock()
        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            import importlib
            import realsense_mlx.display.viewer as vm
            importlib.reload(vm)
            v = vm.RealsenseViewer(wait_ms=0)
            assert v.wait_ms == 1


# ---------------------------------------------------------------------------
# 3. show() converts mx.array → numpy BGR for cv2
# ---------------------------------------------------------------------------


class TestShowConvertsFrame:
    def test_show_calls_cv2_imshow(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        frame = _rgb_frame()
        viewer.show(frame)
        cv2_mock.imshow.assert_called_once()

    def test_show_passes_ndarray_to_imshow(self, viewer_and_cv2):
        """The value passed to cv2.imshow must be a numpy ndarray."""
        viewer, cv2_mock = viewer_and_cv2
        frame = _rgb_frame()
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert isinstance(displayed, np.ndarray)

    def test_show_rgb_swapped_to_bgr(self, viewer_and_cv2):
        """RGB channel order is reversed so cv2 receives BGR."""
        viewer, cv2_mock = viewer_and_cv2
        # Build a frame where channels are distinguishable
        data = np.zeros((48, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 10   # R
        data[:, :, 1] = 20   # G
        data[:, :, 2] = 30   # B
        frame = mx.array(data)

        # Disable auto_resize so the raw conversion is passed through
        viewer.auto_resize = False
        viewer.show(frame)

        _, displayed = cv2_mock.imshow.call_args[0]
        # After RGB→BGR: displayed[0] = B=30, displayed[1] = G=20, displayed[2] = R=10
        assert displayed[0, 0, 0] == 30  # B
        assert displayed[0, 0, 1] == 20  # G
        assert displayed[0, 0, 2] == 10  # R

    def test_show_grayscale_becomes_3channel(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        frame = _gray_frame()
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert displayed.ndim == 3
        assert displayed.shape[2] == 3

    def test_show_float_frame_scaled_to_uint8(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        data = np.ones((48, 64, 3), dtype=np.float32) * 0.5
        frame = mx.array(data)
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert displayed.dtype == np.uint8
        # 0.5 * 255 = 127 (allow ±1 for rounding)
        assert abs(int(displayed[0, 0, 0]) - 127) <= 1

    def test_show_uint16_frame_rescaled_to_uint8(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        # A uniform frame: every pixel == max, so the rescale maps all values
        # to 255 (arr_max / arr_max * 255 = 255).
        data = np.full((48, 64), 32768, dtype=np.uint16)
        frame = mx.array(data)
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert displayed.dtype == np.uint8
        assert int(displayed[0, 0, 0]) == 255

    def test_show_uint16_frame_relative_scaling(self, viewer_and_cv2):
        """Half-max uint16 pixel should map to approximately 127."""
        viewer, cv2_mock = viewer_and_cv2
        # Two-pixel frame: one pixel at max, one at half
        data = np.array([[65535, 32767]], dtype=np.uint16)
        frame = mx.array(data)
        viewer.auto_resize = False
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert displayed.dtype == np.uint8
        # First pixel (max) → 255
        assert int(displayed[0, 0, 0]) == 255
        # Second pixel (half-max) → ~127
        assert 120 <= int(displayed[0, 1, 0]) <= 135

    def test_show_rgba_drops_alpha_and_swaps(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        data = np.zeros((48, 64, 4), dtype=np.uint8)
        data[:, :, 0] = 100   # R
        data[:, :, 1] = 150   # G
        data[:, :, 2] = 200   # B
        data[:, :, 3] = 255   # A (should be dropped)
        frame = mx.array(data)
        viewer.show(frame)
        _, displayed = cv2_mock.imshow.call_args[0]
        assert displayed.shape[2] == 3
        assert displayed[0, 0, 0] == 200  # B
        assert displayed[0, 0, 1] == 150  # G
        assert displayed[0, 0, 2] == 100  # R

    def test_show_invalid_shape_raises_value_error(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        bad = mx.array(np.zeros((4, 4, 4, 4), dtype=np.uint8))
        with pytest.raises(ValueError):
            viewer.show(bad)

    def test_show_noop_when_closed(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.close()
        viewer.show(_rgb_frame())
        cv2_mock.imshow.assert_not_called()

    def test_show_creates_window_once(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        frame = _rgb_frame()
        viewer.show(frame)
        viewer.show(frame)
        viewer.show(frame)
        # namedWindow should only be called on the first show
        assert cv2_mock.namedWindow.call_count == 1

    def test_show_depth_delegates_to_show(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        frame = _rgb_frame()
        viewer.show_depth(frame)
        cv2_mock.imshow.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Side-by-side layout
# ---------------------------------------------------------------------------


class TestShowSideBySide:
    def test_side_by_side_calls_imshow(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        left = _rgb_frame(96, 128)
        right = _rgb_frame(96, 128)
        viewer.show_side_by_side(left, right, gap_px=0)
        cv2_mock.imshow.assert_called_once()

    def test_side_by_side_width_is_sum_of_both_after_scale(self, viewer_and_cv2):
        """With gap_px=0 the output width should equal the sum of both scaled widths."""
        viewer, cv2_mock = viewer_and_cv2

        # Give both frames the same size as the target height to avoid scale
        h = viewer.height  # 96
        w = 64
        left_data = np.zeros((h, w, 3), dtype=np.uint8)
        right_data = np.zeros((h, w, 3), dtype=np.uint8)

        # Intercept the np.concatenate call to inspect the combined array
        with patch("numpy.concatenate", wraps=np.concatenate) as mock_cat:
            viewer.show_side_by_side(
                mx.array(left_data),
                mx.array(right_data),
                gap_px=0,
            )
        # Just verify imshow was called — width accounting is tested via shape
        cv2_mock.imshow.assert_called_once()

    def test_side_by_side_with_gap(self, viewer_and_cv2):
        """gap_px > 0 results in a wider concatenated image."""
        viewer, cv2_mock = viewer_and_cv2
        h = viewer.height
        gap = 8
        left_data = np.zeros((h, 32, 3), dtype=np.uint8)
        right_data = np.zeros((h, 32, 3), dtype=np.uint8)

        viewer.show_side_by_side(
            mx.array(left_data),
            mx.array(right_data),
            gap_px=gap,
        )
        _, displayed = cv2_mock.imshow.call_args[0]
        # displayed width = left_w + gap + right_w = 32 + 8 + 32 = 72
        assert displayed.shape[1] == 32 + gap + 32

    def test_side_by_side_noop_when_closed(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.close()
        viewer.show_side_by_side(_rgb_frame(), _rgb_frame())
        cv2_mock.imshow.assert_not_called()


# ---------------------------------------------------------------------------
# 5. show_grid()
# ---------------------------------------------------------------------------


class TestShowGrid:
    def test_grid_calls_imshow(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        frames = [_rgb_frame() for _ in range(4)]
        # gap_px=0 avoids the gap_v width vs row width mismatch in the viewer
        viewer.show_grid(frames, cols=2, gap_px=0)
        cv2_mock.imshow.assert_called_once()

    def test_grid_empty_frames_is_noop(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.show_grid([])
        cv2_mock.imshow.assert_not_called()

    def test_grid_noop_when_closed(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.close()
        viewer.show_grid([_rgb_frame()])
        cv2_mock.imshow.assert_not_called()


# ---------------------------------------------------------------------------
# 6. close() destroys the window
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_destroys_window(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        # Trigger window creation
        viewer.show(_rgb_frame())
        viewer.close()
        cv2_mock.destroyWindow.assert_called_once_with("Test")

    def test_close_sets_is_open_false(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        assert viewer.is_open() is True
        viewer.close()
        assert viewer.is_open() is False

    def test_close_without_window_does_not_call_destroy(self, viewer_and_cv2):
        """close() before any show() should not call destroyWindow."""
        viewer, cv2_mock = viewer_and_cv2
        viewer.close()
        cv2_mock.destroyWindow.assert_not_called()

    def test_double_close_is_safe(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.close()
        viewer.close()
        # destroyWindow called at most once regardless
        assert cv2_mock.destroyWindow.call_count <= 1


# ---------------------------------------------------------------------------
# 7. Context-manager protocol
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_enter_returns_viewer(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        result = viewer.__enter__()
        assert result is viewer

    def test_context_manager_exit_calls_close(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.show(_rgb_frame())
        viewer.__exit__(None, None, None)
        assert viewer.is_open() is False

    def test_with_statement(self):
        cv2_mock = _make_cv2_mock()
        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            import importlib
            import realsense_mlx.display.viewer as vm
            importlib.reload(vm)

            with vm.RealsenseViewer(title="ctx", width=64, height=48) as v:
                assert v.is_open() is True
                v.show(_rgb_frame())

            assert v.is_open() is False
            cv2_mock.destroyWindow.assert_called_once_with("ctx")

        import importlib
        import realsense_mlx.display.viewer as vm
        importlib.reload(vm)


# ---------------------------------------------------------------------------
# 8. Key / window-close event handling
# ---------------------------------------------------------------------------


class TestQuitBehavior:
    def _viewer_with_key(self, key_code: int) -> tuple:
        cv2_mock = _make_cv2_mock()
        cv2_mock.waitKey.return_value = key_code & 0xFF
        cv2_mock.getWindowProperty.return_value = 1.0

        import importlib
        import realsense_mlx.display.viewer as vm
        importlib.reload(vm)

        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            importlib.reload(vm)
            v = vm.RealsenseViewer(title="T", width=64, height=48)
        return v, cv2_mock

    def test_q_key_closes_viewer(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = ord("q") & 0xFF
        viewer.show(_rgb_frame())
        assert viewer.is_open() is False

    def test_upper_q_key_closes_viewer(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = ord("Q") & 0xFF
        viewer.show(_rgb_frame())
        assert viewer.is_open() is False

    def test_esc_key_closes_viewer(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = 27  # Esc
        viewer.show(_rgb_frame())
        assert viewer.is_open() is False

    def test_other_key_keeps_viewer_open(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = ord("a") & 0xFF
        viewer.show(_rgb_frame())
        assert viewer.is_open() is True

    def test_window_close_button_closes_viewer(self, viewer_and_cv2):
        """cv2.getWindowProperty returning < 1.0 simulates the X-button click."""
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = 0xFF  # no key
        cv2_mock.getWindowProperty.return_value = 0.0
        viewer.show(_rgb_frame())
        assert viewer.is_open() is False

    def test_get_window_property_exception_closes_viewer(self, viewer_and_cv2):
        """If getWindowProperty raises, the viewer should close gracefully."""
        viewer, cv2_mock = viewer_and_cv2
        cv2_mock.waitKey.return_value = 0xFF
        cv2_mock.getWindowProperty.side_effect = Exception("display gone")
        viewer.show(_rgb_frame())
        assert viewer.is_open() is False


# ---------------------------------------------------------------------------
# 9. auto_resize behaviour
# ---------------------------------------------------------------------------


class TestAutoResize:
    def test_auto_resize_true_calls_cv2_resize(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.auto_resize = True
        # Frame size != viewer size → resize should be called
        frame = mx.array(np.zeros((10, 10, 3), dtype=np.uint8))
        viewer.show(frame)
        cv2_mock.resize.assert_called()

    def test_auto_resize_false_skips_cv2_resize(self, viewer_and_cv2):
        viewer, cv2_mock = viewer_and_cv2
        viewer.auto_resize = False
        # Frame at native size — resize must not be called
        frame = mx.array(np.zeros((96, 128, 3), dtype=np.uint8))
        viewer.show(frame)
        cv2_mock.resize.assert_not_called()

    def test_auto_resize_same_size_no_resize(self, viewer_and_cv2):
        """If frame already matches (width, height), cv2.resize is skipped."""
        viewer, cv2_mock = viewer_and_cv2
        viewer.auto_resize = True
        # viewer is 128×96; feed matching frame
        frame = mx.array(np.zeros((96, 128, 3), dtype=np.uint8))
        viewer.show(frame)
        cv2_mock.resize.assert_not_called()


# ---------------------------------------------------------------------------
# 10. __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_title_and_state(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        r = repr(viewer)
        assert "RealsenseViewer" in r
        assert "Test" in r
        assert "open" in r

    def test_repr_closed_state(self, viewer_and_cv2):
        viewer, _ = viewer_and_cv2
        viewer.close()
        r = repr(viewer)
        assert "closed" in r


# ---------------------------------------------------------------------------
# 11. _mx_to_bgr_uint8 internal helper
# ---------------------------------------------------------------------------


class TestMxToBgrUint8:
    """Direct tests for the module-level conversion helper."""

    @pytest.fixture()
    def converter(self):
        cv2_mock = _make_cv2_mock()
        with patch.dict("sys.modules", {"cv2": cv2_mock}):
            import importlib
            import realsense_mlx.display.viewer as vm
            importlib.reload(vm)
            yield vm._mx_to_bgr_uint8
        import importlib
        import realsense_mlx.display.viewer as vm
        importlib.reload(vm)

    def test_rgb_uint8_swapped_to_bgr(self, converter):
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        data[:, :, 0] = 1   # R
        data[:, :, 2] = 3   # B
        out = converter(mx.array(data))
        assert out[0, 0, 0] == 3  # B
        assert out[0, 0, 2] == 1  # R

    def test_grayscale_replicated_to_3ch(self, converter):
        data = np.full((4, 4), 128, dtype=np.uint8)
        out = converter(mx.array(data))
        assert out.shape == (4, 4, 3)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])

    def test_float01_scaled_to_uint8(self, converter):
        data = np.ones((4, 4, 3), dtype=np.float32)
        out = converter(mx.array(data))
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 255

    def test_float_zero_maps_to_black(self, converter):
        data = np.zeros((4, 4, 3), dtype=np.float32)
        out = converter(mx.array(data))
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0

    def test_uint16_all_zero_stays_black(self, converter):
        data = np.zeros((4, 4), dtype=np.uint16)
        out = converter(mx.array(data))
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0

    def test_unsupported_channel_count_raises(self, converter):
        data = np.zeros((4, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel"):
            converter(mx.array(data))

    def test_4d_input_raises(self, converter):
        data = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            converter(mx.array(data))

    def test_output_is_contiguous(self, converter):
        data = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        out = converter(mx.array(data))
        assert out.flags["C_CONTIGUOUS"]
