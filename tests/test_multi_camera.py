"""Tests for MultiCameraCapture.

All tests are mock-based — no physical RealSense camera is required.

Test matrix
-----------
- discover() returns empty list when no cameras are connected (mock).
- discover() returns correct dicts when cameras are present.
- Instantiation raises ImportError when HAS_RS2 is False.
- start() with no cameras emits a warning and results in zero active cameras.
- start(serials=...) targets only the specified serials.
- stop() cleans up all active cameras.
- get_frames() returns dict keyed by serial.
- get_frames() on empty multi-camera returns empty dict.
- active_serials and camera_count reflect live state.
- Context-manager starts and stops cleanly.
- Repr contains serial list.
- Camera that fails to start emits warning and is skipped.
"""

from __future__ import annotations

import types
import warnings
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — fake pyrealsense2 context / device stubs
# ---------------------------------------------------------------------------


def _make_fake_device(serial: str, name: str = "Intel RealSense D435") -> MagicMock:
    """Return a fake rs2 device mock that responds to get_info()."""
    dev = MagicMock()

    def _get_info(key):
        mapping = {
            "serial_number": serial,
            "name": name,
            "firmware_version": "5.15.0.0",
            "usb_type_descriptor": "3.2",
        }
        # The key may be a string or a mock enum; use str() to normalise.
        return mapping.get(str(key), "unknown")

    dev.get_info.side_effect = _get_info
    return dev


def _make_fake_rs2_module(devices: list[MagicMock] | None = None) -> types.ModuleType:
    """Construct a minimal fake pyrealsense2 module.

    Parameters
    ----------
    devices:
        List of fake device mocks.  Defaults to an empty list.
    """
    rs = types.ModuleType("pyrealsense2")

    # context().devices must be iterable.
    ctx = MagicMock()
    ctx.devices = list(devices or [])
    rs.context = MagicMock(return_value=ctx)

    # camera_info enum-like strings (used as dict keys after str()).
    ci = MagicMock()
    ci.serial_number = "serial_number"
    ci.name = "name"
    ci.firmware_version = "firmware_version"
    ci.usb_type_descriptor = "usb_type_descriptor"
    rs.camera_info = ci

    # Pipeline / config / stream / format enums.
    stream = MagicMock()
    stream.depth = "depth"
    stream.color = "color"
    stream.infrared = "infrared"
    rs.stream = stream

    fmt = MagicMock()
    fmt.z16 = "z16"
    fmt.rgb8 = "rgb8"
    fmt.y8 = "y8"
    rs.format = fmt

    rs.config = MagicMock
    rs.pipeline = MagicMock

    return rs


def _make_started_pipeline_profile():
    """Return a fake pipeline + profile pair that survives _extract_metadata."""
    fake_profile = MagicMock()
    fake_profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = (
        0.001
    )
    fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.side_effect = Exception(
        "mock skip"
    )
    fake_profile.get_stream.return_value.as_video_stream_profile.return_value.get_extrinsics_to.side_effect = Exception(
        "mock skip"
    )
    fake_pipeline = MagicMock()
    fake_pipeline.start.return_value = fake_profile
    return fake_pipeline, fake_profile


# ---------------------------------------------------------------------------
# ImportError guard (HAS_RS2 = False)
# ---------------------------------------------------------------------------


class TestMultiCameraImportError:
    def test_raises_import_error_when_no_rs2(self):
        with patch("realsense_mlx.capture.multi_camera.HAS_RS2", False):
            from realsense_mlx.capture.multi_camera import MultiCameraCapture

            with pytest.raises(ImportError, match="pyrealsense2"):
                MultiCameraCapture()

    def test_discover_returns_empty_when_no_rs2(self):
        with patch("realsense_mlx.capture.multi_camera.HAS_RS2", False):
            from realsense_mlx.capture.multi_camera import MultiCameraCapture

            result = MultiCameraCapture.discover()
            assert result == []


# ---------------------------------------------------------------------------
# discover() — static method
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_no_devices_returns_empty_list(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                result = MultiCameraCapture.discover()

        assert result == []

    def test_single_device_returns_one_dict(self):
        dev = _make_fake_device("111111111111", "Intel RealSense D435")
        rs_mod = _make_fake_rs2_module(devices=[dev])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                result = MultiCameraCapture.discover()

        assert len(result) == 1
        assert result[0]["serial"] == "111111111111"
        assert result[0]["name"] == "Intel RealSense D435"
        assert "firmware" in result[0]
        assert "usb_type" in result[0]

    def test_two_devices_returns_two_dicts(self):
        dev1 = _make_fake_device("AAA000000000")
        dev2 = _make_fake_device("BBB000000000")
        rs_mod = _make_fake_rs2_module(devices=[dev1, dev2])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                result = MultiCameraCapture.discover()

        serials = {d["serial"] for d in result}
        assert serials == {"AAA000000000", "BBB000000000"}

    def test_device_info_failure_emits_warning_and_skips(self):
        dev = MagicMock()
        dev.get_info.side_effect = RuntimeError("device busy")
        rs_mod = _make_fake_rs2_module(devices=[dev])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = MultiCameraCapture.discover()

        assert result == []
        assert len(w) == 1

    def test_context_exception_emits_warning(self):
        rs_mod = _make_fake_rs2_module()
        rs_mod.context.side_effect = RuntimeError("context init failed")

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = MultiCameraCapture.discover()

        assert result == []
        assert len(w) == 1


# ---------------------------------------------------------------------------
# Lifecycle — start / stop
# ---------------------------------------------------------------------------


class TestMultiCameraLifecycle:
    """Use _SerialCapture's start() path with mocked pyrealsense2."""

    def _patch_for_serial_capture(self, serial: str, rs_mod):
        """Return a context manager that patches the two RS2 touch-points."""
        return patch.multiple(
            "realsense_mlx.capture.multi_camera",
            HAS_RS2=True,
        ), patch.multiple(
            "realsense_mlx.capture.pipeline",
            HAS_RS2=True,
            rs=rs_mod,
        )

    def test_start_with_no_discovered_cameras_emits_warning(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    mc.start()

        assert mc.camera_count == 0
        assert any("no cameras" in str(x.message).lower() for x in w)

    def test_start_with_explicit_serials_starts_only_those(self):
        """Patch _SerialCapture.start() directly to avoid RS2 SDK calls."""
        rs_mod = _make_fake_rs2_module(devices=[])
        fake_pipeline, fake_profile = _make_started_pipeline_profile()
        rs_mod.pipeline = MagicMock(return_value=fake_pipeline)

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                with patch("realsense_mlx.capture.pipeline.HAS_RS2", True):
                    with patch("realsense_mlx.capture.pipeline.rs", rs_mod):
                        from realsense_mlx.capture.multi_camera import (
                            MultiCameraCapture,
                            _SerialCapture,
                        )

                        # Patch _SerialCapture.start so no real pipeline is started.
                        with patch.object(_SerialCapture, "start") as mock_start:
                            mc = MultiCameraCapture()
                            mc.start(serials=["SN0001", "SN0002"])

                        assert mc.camera_count == 2
                        assert set(mc.active_serials) == {"SN0001", "SN0002"}

    def test_stop_clears_all_cameras(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                with patch("realsense_mlx.capture.pipeline.HAS_RS2", True):
                    with patch("realsense_mlx.capture.pipeline.rs", rs_mod):
                        from realsense_mlx.capture.multi_camera import (
                            MultiCameraCapture,
                            _SerialCapture,
                        )

                        with patch.object(_SerialCapture, "start"):
                            mc = MultiCameraCapture()
                            mc.start(serials=["SN0001"])
                            assert mc.camera_count == 1

                        mc.stop()
                        assert mc.camera_count == 0
                        assert mc.active_serials == []

    def test_camera_start_failure_emits_warning_and_is_skipped(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                with patch("realsense_mlx.capture.pipeline.HAS_RS2", True):
                    with patch("realsense_mlx.capture.pipeline.rs", rs_mod):
                        from realsense_mlx.capture.multi_camera import (
                            MultiCameraCapture,
                            _SerialCapture,
                        )

                        with patch.object(
                            _SerialCapture,
                            "start",
                            side_effect=RuntimeError("device busy"),
                        ):
                            mc = MultiCameraCapture()
                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                                mc.start(serials=["FAIL_SN"])

        assert mc.camera_count == 0
        assert any("FAIL_SN" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# get_frames()
# ---------------------------------------------------------------------------


class TestGetFrames:
    def test_get_frames_returns_empty_dict_when_no_cameras(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()
                result = mc.get_frames()

        assert result == {}

    def test_get_frames_returns_dict_keyed_by_serial(self):
        """Inject fake captures into _cameras and call get_frames()."""
        from realsense_mlx.capture.pipeline import CapturedFrames
        import mlx.core as mx

        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()

                # Inject two fake capture objects directly.
                fake_frames_a = CapturedFrames()
                fake_frames_a.depth = mx.zeros((480, 640), dtype=mx.uint16)
                fake_frames_a.frame_number = 1

                fake_frames_b = CapturedFrames()
                fake_frames_b.depth = mx.zeros((480, 640), dtype=mx.uint16)
                fake_frames_b.frame_number = 2

                mock_cam_a = MagicMock()
                mock_cam_a.get_frames.return_value = fake_frames_a

                mock_cam_b = MagicMock()
                mock_cam_b.get_frames.return_value = fake_frames_b

                mc._cameras["SN_A"] = mock_cam_a
                mc._cameras["SN_B"] = mock_cam_b

                result = mc.get_frames()

        assert set(result.keys()) == {"SN_A", "SN_B"}
        assert result["SN_A"].frame_number == 1
        assert result["SN_B"].frame_number == 2

    def test_get_frames_calls_get_frames_on_each_camera(self):
        rs_mod = _make_fake_rs2_module(devices=[])

        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture
                from realsense_mlx.capture.pipeline import CapturedFrames

                mc = MultiCameraCapture()

                cam_a = MagicMock()
                cam_a.get_frames.return_value = CapturedFrames()
                cam_b = MagicMock()
                cam_b.get_frames.return_value = CapturedFrames()

                mc._cameras["AAA"] = cam_a
                mc._cameras["BBB"] = cam_b

                mc.get_frames()

        cam_a.get_frames.assert_called_once()
        cam_b.get_frames.assert_called_once()


# ---------------------------------------------------------------------------
# Properties and repr
# ---------------------------------------------------------------------------


class TestProperties:
    def test_active_serials_empty_initially(self):
        rs_mod = _make_fake_rs2_module(devices=[])
        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()
                assert mc.active_serials == []
                assert mc.camera_count == 0

    def test_repr_contains_serial_list(self):
        rs_mod = _make_fake_rs2_module(devices=[])
        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture
                from realsense_mlx.capture.pipeline import CapturedFrames

                mc = MultiCameraCapture()
                mock_cam = MagicMock()
                mock_cam.get_frames.return_value = CapturedFrames()
                mc._cameras["SN_REPR"] = mock_cam

                r = repr(mc)
                assert "SN_REPR" in r

    def test_repr_contains_config(self):
        rs_mod = _make_fake_rs2_module(devices=[])
        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture
                from realsense_mlx.capture.pipeline import CaptureConfig

                mc = MultiCameraCapture(CaptureConfig(width=848, height=480))
                r = repr(mc)
                assert "848" in r


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_calls_start_and_stop(self):
        rs_mod = _make_fake_rs2_module(devices=[])
        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()
                with patch.object(mc, "start") as mock_start, patch.object(
                    mc, "stop"
                ) as mock_stop:
                    with mc:
                        mock_start.assert_called_once()
                    mock_stop.assert_called_once()

    def test_context_manager_stop_called_on_exception(self):
        rs_mod = _make_fake_rs2_module(devices=[])
        with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
            with patch("realsense_mlx.capture.multi_camera.HAS_RS2", True):
                from realsense_mlx.capture.multi_camera import MultiCameraCapture

                mc = MultiCameraCapture()
                with patch.object(mc, "start"), patch.object(mc, "stop") as mock_stop:
                    try:
                        with mc:
                            raise ValueError("boom")
                    except ValueError:
                        pass
                mock_stop.assert_called_once()
