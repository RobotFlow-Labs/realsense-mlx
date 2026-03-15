"""Tests for the capture pipeline (mock-based — no physical camera required).

All tests use ``unittest.mock`` to avoid importing pyrealsense2, which has
no official ARM64 macOS wheel.  The strategy:

1. ``CaptureConfig`` and ``CapturedFrames`` are pure-Python dataclasses with
   no pyrealsense2 dependency — they can be tested directly.
2. ``RealsenseCapture`` checks ``HAS_RS2`` at instantiation time, so we patch
   ``realsense_mlx.capture.pipeline.HAS_RS2`` to control whether the guard
   fires, and supply a fake ``rs`` module where the RS2 API surface is needed.
"""

from __future__ import annotations

import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CaptureConfig — validation tests (no RS2 dependency)
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_default_values(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        cfg = CaptureConfig()
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.fps == 30
        assert cfg.enable_depth is True
        assert cfg.enable_color is True
        assert cfg.enable_ir is False
        assert cfg.timeout_ms == 5000

    def test_custom_values(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        cfg = CaptureConfig(
            width=1280,
            height=720,
            fps=15,
            enable_depth=True,
            enable_color=False,
            enable_ir=True,
            timeout_ms=2000,
        )
        assert cfg.width == 1280
        assert cfg.height == 720
        assert cfg.fps == 15
        assert cfg.enable_color is False
        assert cfg.enable_ir is True
        assert cfg.timeout_ms == 2000

    def test_invalid_width(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="width and height must be positive"):
            CaptureConfig(width=0, height=480)

    def test_invalid_height(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="width and height must be positive"):
            CaptureConfig(width=640, height=-1)

    def test_invalid_fps(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="fps must be positive"):
            CaptureConfig(fps=0)

    def test_invalid_timeout_ms(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            CaptureConfig(timeout_ms=0)

    def test_negative_timeout_ms(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            CaptureConfig(timeout_ms=-100)

    def test_no_streams_enabled(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        with pytest.raises(ValueError, match="At least one stream must be enabled"):
            CaptureConfig(enable_depth=False, enable_color=False, enable_ir=False)

    def test_repr_contains_dimensions(self):
        from realsense_mlx.capture.pipeline import CaptureConfig

        cfg = CaptureConfig(width=848, height=480, fps=60)
        r = repr(cfg)
        assert "848x480" in r
        assert "60fps" in r

    def test_ir_only_config(self):
        """A depth=False, color=False, ir=True config must be accepted."""
        from realsense_mlx.capture.pipeline import CaptureConfig

        cfg = CaptureConfig(enable_depth=False, enable_color=False, enable_ir=True)
        assert cfg.enable_ir is True


# ---------------------------------------------------------------------------
# CapturedFrames — container tests (no RS2 dependency)
# ---------------------------------------------------------------------------


class TestCapturedFrames:
    def test_default_state(self):
        from realsense_mlx.capture.pipeline import CapturedFrames

        f = CapturedFrames()
        assert f.depth is None
        assert f.color is None
        assert f.infrared is None
        assert f.timestamp == 0.0
        assert f.frame_number == 0

    def test_repr_empty(self):
        from realsense_mlx.capture.pipeline import CapturedFrames

        r = repr(CapturedFrames())
        assert "frame=0" in r

    def test_repr_with_arrays(self):
        from realsense_mlx.capture.pipeline import CapturedFrames

        f = CapturedFrames()
        f.depth = mx.zeros((480, 640), dtype=mx.uint16)
        f.frame_number = 42
        r = repr(f)
        assert "depth=" in r
        assert "frame=42" in r

    def test_assign_color_and_ir(self):
        from realsense_mlx.capture.pipeline import CapturedFrames

        f = CapturedFrames()
        f.color = mx.zeros((480, 640, 3), dtype=mx.uint8)
        f.infrared = mx.zeros((480, 640), dtype=mx.uint8)
        assert f.color is not None
        assert f.infrared is not None


# ---------------------------------------------------------------------------
# RealsenseCapture — ImportError path (pyrealsense2 absent)
# ---------------------------------------------------------------------------


class TestRealsenseCaptureImportError:
    """Verify that a missing pyrealsense2 raises an ImportError with build
    instructions, rather than a generic AttributeError or silent failure."""

    def test_raises_import_error_when_rs2_missing(self):
        """Patch HAS_RS2=False so the guard fires without touching the real
        pyrealsense2 import."""
        with patch("realsense_mlx.capture.pipeline.HAS_RS2", False):
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig

            with pytest.raises(ImportError) as exc_info:
                RealsenseCapture(CaptureConfig())

            msg = str(exc_info.value)
            # The error message must contain actionable build instructions.
            assert "pyrealsense2" in msg
            assert "cmake" in msg.lower() or "build" in msg.lower()

    def test_import_error_message_mentions_pythonpath(self):
        """The build instructions should tell the user how to make the .so
        discoverable."""
        with patch("realsense_mlx.capture.pipeline.HAS_RS2", False):
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig

            with pytest.raises(ImportError) as exc_info:
                RealsenseCapture(CaptureConfig())

            assert "PYTHONPATH" in str(exc_info.value) or "site" in str(exc_info.value)


# ---------------------------------------------------------------------------
# RealsenseCapture — behaviour with a mocked pyrealsense2
# ---------------------------------------------------------------------------


def _make_fake_rs2_module() -> types.ModuleType:
    """Build a minimal fake pyrealsense2 module sufficient for the tests."""
    rs = types.ModuleType("pyrealsense2")

    # Enums / constants accessed during config
    stream = MagicMock()
    stream.depth = "depth"
    stream.color = "color"
    stream.infrared = "infrared"
    rs.stream = stream

    format_mock = MagicMock()
    format_mock.z16 = "z16"
    format_mock.rgb8 = "rgb8"
    format_mock.y8 = "y8"
    rs.format = format_mock

    # rs.config — records enable_stream calls
    rs.config = MagicMock

    # rs.pipeline — the main object
    fake_pipeline = MagicMock()
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    rs.pipeline = fake_pipeline_cls

    return rs, fake_pipeline


class TestRealsenseCaptureLifecycle:
    """Tests for start/stop/context-manager using a fully mocked RS2 SDK."""

    def _patch_rs2(self):
        """Return a context manager that injects a fake rs module."""
        rs_mod, fake_pipeline = _make_fake_rs2_module()
        return rs_mod, fake_pipeline, patch.multiple(
            "realsense_mlx.capture.pipeline",
            HAS_RS2=True,
            rs=rs_mod,
        )

    def test_not_running_before_start(self):
        rs_mod, fake_pipeline, patches = self._patch_rs2()
        with patches:
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig())
            assert capture.is_running is False

    def test_double_start_raises(self):
        rs_mod, fake_pipeline, patches = self._patch_rs2()

        # Provide a fake profile so _extract_metadata doesn't crash.
        fake_profile = MagicMock()
        fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = Exception("mock")
        fake_pipeline.start.return_value = fake_profile

        with patches:
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig())
            capture.start()
            with pytest.raises(RuntimeError, match="already running"):
                capture.start()
            capture.stop()

    def test_stop_when_not_running_is_safe(self):
        rs_mod, fake_pipeline, patches = self._patch_rs2()
        with patches:
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig())
            capture.stop()  # must not raise
            assert capture.is_running is False

    def test_get_frames_before_start_raises(self):
        rs_mod, fake_pipeline, patches = self._patch_rs2()
        with patches:
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig())
            with pytest.raises(RuntimeError, match="not running"):
                capture.get_frames()

    def test_context_manager_starts_and_stops(self):
        rs_mod, fake_pipeline, patches = self._patch_rs2()

        fake_profile = MagicMock()
        fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = Exception("mock")
        fake_pipeline.start.return_value = fake_profile

        with patches:
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig())
            with capture:
                assert capture.is_running is True
            assert capture.is_running is False


# ---------------------------------------------------------------------------
# RealsenseCapture — timeout / frame-drop handling
# ---------------------------------------------------------------------------


class TestWaitForFramesTimeout:
    """Verify that a RuntimeError from wait_for_frames is caught and a
    CapturedFrames with all-None fields is returned (not an exception)."""

    def test_timeout_returns_empty_captured_frames(self):
        rs_mod, fake_pipeline = _make_fake_rs2_module()

        fake_profile = MagicMock()
        fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = Exception("mock")
        fake_pipeline.start.return_value = fake_profile

        # Simulate a timeout from the RS2 SDK
        fake_pipeline.wait_for_frames.side_effect = RuntimeError("Frame didn't arrive within 5000")

        with patch.multiple("realsense_mlx.capture.pipeline", HAS_RS2=True, rs=rs_mod):  # type: ignore[arg-type]
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig, CapturedFrames

            capture = RealsenseCapture(CaptureConfig(timeout_ms=5000))
            capture._pipeline = fake_pipeline
            capture._running = True

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = capture.get_frames()

            assert isinstance(result, CapturedFrames)
            assert result.depth is None
            assert result.color is None
            assert result.infrared is None

            # A RuntimeWarning must have been issued
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "timed out" in str(w[0].message).lower()

    def test_timeout_ms_forwarded_to_sdk(self):
        """Confirm the configured timeout_ms is passed to wait_for_frames."""
        rs_mod, fake_pipeline = _make_fake_rs2_module()

        fake_profile = MagicMock()
        fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = Exception("mock")
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_extrinsics_to.side_effect = Exception("mock")
        fake_pipeline.start.return_value = fake_profile

        # Return a valid (fake) frameset on the first call
        fake_frameset = MagicMock()
        fake_frameset.get_timestamp.return_value = 12345.0
        fake_frameset.get_frame_number.return_value = 1
        fake_frameset.get_depth_frame.return_value = None
        fake_frameset.get_color_frame.return_value = None
        fake_pipeline.wait_for_frames.return_value = fake_frameset

        with patch.multiple("realsense_mlx.capture.pipeline", HAS_RS2=True, rs=rs_mod):
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig(timeout_ms=2500))
            capture._pipeline = fake_pipeline
            capture._running = True
            capture.get_frames()

        fake_pipeline.wait_for_frames.assert_called_once_with(timeout_ms=2500)


# ---------------------------------------------------------------------------
# _extract_metadata — warning tests
# ---------------------------------------------------------------------------


class TestExtractMetadataWarnings:
    """Verify that failures inside _extract_metadata emit warnings rather than
    silently swallowing exceptions."""

    def _make_capture_with_broken_profile(self, *, depth=True, color=True):
        rs_mod, fake_pipeline = _make_fake_rs2_module()

        fake_profile = MagicMock()
        # depth scale succeeds
        fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        # all intrinsics/extrinsics calls fail
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = RuntimeError("broken")
        fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_extrinsics_to.side_effect = RuntimeError("broken")
        fake_pipeline.start.return_value = fake_profile

        return rs_mod, fake_pipeline

    def test_depth_intrinsics_failure_emits_warning(self):
        rs_mod, fake_pipeline = self._make_capture_with_broken_profile()

        with patch.multiple("realsense_mlx.capture.pipeline", HAS_RS2=True, rs=rs_mod):
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig(enable_color=False))
            capture._pipeline = fake_pipeline
            capture._profile = fake_pipeline.start.return_value
            capture._running = True

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                capture._extract_metadata()

            warning_msgs = [str(x.message) for x in w]
            assert any("depth intrinsics" in m for m in warning_msgs)

    def test_color_intrinsics_failure_emits_warning(self):
        rs_mod, fake_pipeline = self._make_capture_with_broken_profile()

        with patch.multiple("realsense_mlx.capture.pipeline", HAS_RS2=True, rs=rs_mod):
            from realsense_mlx.capture.pipeline import RealsenseCapture, CaptureConfig
            capture = RealsenseCapture(CaptureConfig(enable_depth=False))
            capture._pipeline = fake_pipeline
            capture._profile = fake_pipeline.start.return_value
            capture._running = True

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                capture._extract_metadata()

            warning_msgs = [str(x.message) for x in w]
            assert any("color intrinsics" in m for m in warning_msgs)
