"""Tests for the ROS2 bridge compatibility seam.

The test suite is split into two tiers:

Tier 1 — Always runs (no rclpy required)
    * Module import never raises
    * ``HAS_ROS2`` is a bool
    * ``ROS2Bridge(...)`` raises ``ImportError`` with a helpful message
      when ``HAS_ROS2 is False``

Tier 2 — Runs when rclpy is mocked (simulates rclpy presence)
    * ``ROS2Bridge`` construction creates the expected publishers
    * ``publish_depth``: depth frame dispatched to the depth publisher
    * ``publish_color``: colour frame dispatched to the colour publisher
    * ``publish_pointcloud``: point cloud dispatched to points publisher
    * ``publish_camera_info``: camera info dispatched to camera_info pub
    * ``shutdown()`` calls ``node.destroy_node()``
    * Double ``shutdown()`` is safe
    * Context-manager ``__exit__`` calls ``shutdown()``
    * Publishing after ``shutdown()`` raises ``RuntimeError``
    * ``repr()`` includes node name and state
    * Internal converters (_depth_mx_to_image_msg, etc.) are tested
      independently from the bridge class

Coverage of internal converters (always runs — no rclpy dependency)
    * _mx_to_numpy returns numpy ndarray
    * _depth_mx_to_image_msg: uint16 passthrough, float→mm conversion,
      encoding is 16UC1, step == width * 2
    * _color_mx_to_image_msg: RGB→BGR swap, grayscale→3ch, RGBA→BGR,
      encoding is bgr8, step == width * 3
    * _pointcloud_mx_to_msg: shape (N, 3), field names xyz, point_step 12
    * _camera_info_from_intrinsics: K matrix, distortion coeffs, w/h
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_rclpy() -> MagicMock:
    """Build a minimal rclpy mock that satisfies ros2_bridge.py's usage."""
    rclpy = MagicMock()

    # Node mock
    mock_node = MagicMock()
    mock_node.get_name.return_value = "test_node"
    mock_node.get_clock.return_value.now.return_value.to_msg.return_value = (
        MagicMock()  # stamp
    )
    mock_node.create_publisher.return_value = MagicMock()
    rclpy.node.Node.return_value = mock_node

    # QoSProfile
    rclpy.qos.QoSProfile.return_value = MagicMock()
    rclpy.qos.ReliabilityPolicy = MagicMock()

    return rclpy


def _make_mock_sensor_msgs() -> types.ModuleType:
    """Create a minimal sensor_msgs mock."""
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")

    class Image:
        pass

    class PointCloud2:
        pass

    class CameraInfo:
        pass

    class PointField:
        FLOAT32 = 7

        def __init__(self, name="", offset=0, datatype=0, count=1):
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count

    sensor_msgs.msg.Image = Image
    sensor_msgs.msg.PointCloud2 = PointCloud2
    sensor_msgs.msg.CameraInfo = CameraInfo
    sensor_msgs.msg.PointField = PointField

    return sensor_msgs


def _make_mock_std_msgs() -> types.ModuleType:
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")

    class Header:
        def __init__(self):
            self.stamp = None
            self.frame_id = ""

    std_msgs.msg.Header = Header
    return std_msgs


def _inject_ros2_mocks():
    """Inject rclpy and sensor_msgs mocks into sys.modules and return them."""
    mock_rclpy = _make_mock_rclpy()
    mock_sensor_msgs = _make_mock_sensor_msgs()
    mock_std_msgs = _make_mock_std_msgs()
    return {
        "rclpy": mock_rclpy,
        "rclpy.node": mock_rclpy.node,
        "rclpy.qos": mock_rclpy.qos,
        "sensor_msgs": mock_sensor_msgs,
        "sensor_msgs.msg": mock_sensor_msgs.msg,
        "std_msgs": mock_std_msgs,
        "std_msgs.msg": mock_std_msgs.msg,
    }


def _reload_bridge_with_ros2(mock_modules: dict):
    """Reload ros2_bridge with mock modules injected; return module."""
    import importlib
    with patch.dict("sys.modules", mock_modules):
        import realsense_mlx.bridges.ros2_bridge as bridge_mod
        importlib.reload(bridge_mod)
    return bridge_mod


def _get_bridge_module_no_ros2():
    """Return ros2_bridge module reloaded without rclpy (HAS_ROS2=False)."""
    import importlib
    with patch.dict("sys.modules", {"rclpy": None, "sensor_msgs": None}):
        import realsense_mlx.bridges.ros2_bridge as bridge_mod
        importlib.reload(bridge_mod)
    return bridge_mod


# Minimal intrinsics stand-in
class _FakeIntrinsics:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fx = 383.7
        self.fy = 383.7
        self.ppx = 320.0
        self.ppy = 240.0
        self.coeffs = [0.1, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Tier 1: Import safety — always runs
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_module_import_always_succeeds(self):
        """Importing ros2_bridge must never raise, even without rclpy."""
        import realsense_mlx.bridges.ros2_bridge  # noqa: F401

    def test_package_import_always_succeeds(self):
        import realsense_mlx.bridges  # noqa: F401

    def test_has_ros2_is_bool(self):
        import realsense_mlx.bridges.ros2_bridge as bm
        assert isinstance(bm.HAS_ROS2, bool)

    def test_ros2bridge_class_is_accessible(self):
        from realsense_mlx.bridges.ros2_bridge import ROS2Bridge  # noqa: F401
        from realsense_mlx.bridges import ROS2Bridge  # noqa: F401

    def test_has_ros2_exported_from_package(self):
        from realsense_mlx.bridges import HAS_ROS2
        assert isinstance(HAS_ROS2, bool)


class TestImportErrorWithoutRclpy:
    def test_instantiation_raises_import_error(self):
        """Without rclpy, ROS2Bridge() must raise ImportError."""
        import importlib
        with patch.dict("sys.modules", {"rclpy": None, "sensor_msgs": None, "std_msgs": None}):
            import realsense_mlx.bridges.ros2_bridge as bm
            importlib.reload(bm)
            assert bm.HAS_ROS2 is False
            with pytest.raises(ImportError) as exc_info:
                bm.ROS2Bridge()
            assert "rclpy" in str(exc_info.value).lower()

        # Restore
        importlib.reload(bm)

    def test_import_error_message_is_helpful(self):
        """The ImportError message should mention rclpy and installation steps."""
        import importlib
        with patch.dict("sys.modules", {"rclpy": None, "sensor_msgs": None, "std_msgs": None}):
            import realsense_mlx.bridges.ros2_bridge as bm
            importlib.reload(bm)
            with pytest.raises(ImportError) as exc_info:
                bm.ROS2Bridge()
            msg = str(exc_info.value)
            assert "rclpy" in msg
            # Should mention how to fix it
            assert "ROS2" in msg or "ros" in msg.lower() or "install" in msg.lower()
        importlib.reload(bm)

    def test_has_ros2_false_without_rclpy(self):
        import importlib
        with patch.dict("sys.modules", {"rclpy": None, "sensor_msgs": None, "std_msgs": None}):
            import realsense_mlx.bridges.ros2_bridge as bm
            importlib.reload(bm)
            assert bm.HAS_ROS2 is False
        importlib.reload(bm)


# ---------------------------------------------------------------------------
# Tier 2: Mock-based publish tests (simulates rclpy presence)
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge_env():
    """Yield (bridge_module, rclpy_mock, node_mock) with ros2 mocked."""
    import importlib

    mocks = _inject_ros2_mocks()
    rclpy_mock = mocks["rclpy"]
    node_mock = rclpy_mock.node.Node.return_value

    with patch.dict("sys.modules", mocks):
        import realsense_mlx.bridges.ros2_bridge as bm
        importlib.reload(bm)
        assert bm.HAS_ROS2 is True
        yield bm, rclpy_mock, node_mock

    # Restore
    importlib.reload(bm)


@pytest.fixture()
def bridge(bridge_env):
    """Yield a constructed ROS2Bridge (with mocked rclpy)."""
    bm, rclpy_mock, node_mock = bridge_env
    b = bm.ROS2Bridge(node_name="test_node")
    yield b
    if b.is_active:
        b.shutdown()


class TestROS2BridgeConstruction:
    def test_construction_creates_node(self, bridge_env):
        bm, rclpy_mock, node_mock = bridge_env
        bm.ROS2Bridge(node_name="my_node")
        rclpy_mock.node.Node.assert_called_with("my_node")

    def test_construction_creates_four_publishers(self, bridge_env):
        bm, _, node_mock = bridge_env
        bm.ROS2Bridge()
        assert node_mock.create_publisher.call_count == 4

    def test_topic_names_contain_prefix(self, bridge_env):
        bm, _, node_mock = bridge_env
        bm.ROS2Bridge(topic_prefix="/my_prefix")
        topics = [c.args[1] for c in node_mock.create_publisher.call_args_list]
        for t in topics:
            assert t.startswith("/my_prefix"), f"Topic {t!r} missing prefix"

    def test_default_topic_prefix(self, bridge_env):
        bm, _, node_mock = bridge_env
        bm.ROS2Bridge()
        topics = {c.args[1] for c in node_mock.create_publisher.call_args_list}
        assert "/realsense_mlx/depth" in topics
        assert "/realsense_mlx/color" in topics
        assert "/realsense_mlx/points" in topics
        assert "/realsense_mlx/camera_info" in topics

    def test_is_active_true_after_construction(self, bridge):
        assert bridge.is_active is True


class TestPublishDepth:
    def test_publish_depth_calls_publisher(self, bridge, bridge_env):
        bm, _, node_mock = bridge_env
        depth = mx.array(np.zeros((48, 64), dtype=np.uint16))
        bridge.publish_depth(depth)
        bridge._pub_depth.publish.assert_called_once()

    def test_publish_depth_message_is_image(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        depth = mx.array(np.full((48, 64), 1000, dtype=np.uint16))
        bridge.publish_depth(depth)
        msg = bridge._pub_depth.publish.call_args[0][0]
        assert msg.encoding == "16UC1"

    def test_publish_depth_height_width_correct(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        depth = mx.array(np.zeros((48, 64), dtype=np.uint16))
        bridge.publish_depth(depth)
        msg = bridge._pub_depth.publish.call_args[0][0]
        assert msg.height == 48
        assert msg.width == 64

    def test_publish_depth_step_is_width_times_2(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        depth = mx.array(np.zeros((48, 64), dtype=np.uint16))
        bridge.publish_depth(depth)
        msg = bridge._pub_depth.publish.call_args[0][0]
        assert msg.step == 64 * 2

    def test_publish_depth_float_converted_to_uint16(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        # 1.5 metres should become 1500 mm
        depth = mx.array(np.full((4, 4), 1.5, dtype=np.float32))
        bridge.publish_depth(depth)
        msg = bridge._pub_depth.publish.call_args[0][0]
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(4, 4)
        assert arr[0, 0] == 1500


class TestPublishColor:
    def test_publish_color_calls_publisher(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        color = mx.array(np.zeros((48, 64, 3), dtype=np.uint8))
        bridge.publish_color(color)
        bridge._pub_color.publish.assert_called_once()

    def test_publish_color_encoding_is_bgr8(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        color = mx.array(np.zeros((48, 64, 3), dtype=np.uint8))
        bridge.publish_color(color)
        msg = bridge._pub_color.publish.call_args[0][0]
        assert msg.encoding == "bgr8"

    def test_publish_color_rgb_swapped_to_bgr(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        data[:, :, 0] = 10   # R
        data[:, :, 1] = 20   # G
        data[:, :, 2] = 30   # B
        bridge.publish_color(mx.array(data))
        msg = bridge._pub_color.publish.call_args[0][0]
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(4, 4, 3)
        assert arr[0, 0, 0] == 30  # B
        assert arr[0, 0, 1] == 20  # G
        assert arr[0, 0, 2] == 10  # R

    def test_publish_color_step_is_width_times_3(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        color = mx.array(np.zeros((48, 64, 3), dtype=np.uint8))
        bridge.publish_color(color)
        msg = bridge._pub_color.publish.call_args[0][0]
        assert msg.step == 64 * 3


class TestPublishPointcloud:
    def test_publish_pointcloud_calls_publisher(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        pts = mx.array(np.zeros((100, 3), dtype=np.float32))
        bridge.publish_pointcloud(pts)
        bridge._pub_points.publish.assert_called_once()

    def test_publish_pointcloud_field_names(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        bridge.publish_pointcloud(pts)
        msg = bridge._pub_points.publish.call_args[0][0]
        field_names = {f.name for f in msg.fields}
        assert field_names == {"x", "y", "z"}

    def test_publish_pointcloud_point_step(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        bridge.publish_pointcloud(pts)
        msg = bridge._pub_points.publish.call_args[0][0]
        assert msg.point_step == 12  # 3 × float32

    def test_publish_pointcloud_width_equals_n_points(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        n = 250
        pts = mx.array(np.zeros((n, 3), dtype=np.float32))
        bridge.publish_pointcloud(pts)
        msg = bridge._pub_points.publish.call_args[0][0]
        assert msg.width == n

    def test_publish_pointcloud_wrong_shape_raises(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        bad = mx.array(np.zeros((10, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="3"):
            bridge.publish_pointcloud(bad)


class TestPublishCameraInfo:
    def test_publish_camera_info_calls_publisher(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        intrinsics = _FakeIntrinsics()
        bridge.publish_camera_info(intrinsics)
        bridge._pub_camera_info.publish.assert_called_once()

    def test_publish_camera_info_width_height(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        intrinsics = _FakeIntrinsics()
        bridge.publish_camera_info(intrinsics)
        msg = bridge._pub_camera_info.publish.call_args[0][0]
        assert msg.width == 640
        assert msg.height == 480

    def test_publish_camera_info_k_matrix(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        intrinsics = _FakeIntrinsics()
        bridge.publish_camera_info(intrinsics)
        msg = bridge._pub_camera_info.publish.call_args[0][0]
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        assert msg.k[0] == pytest.approx(intrinsics.fx)
        assert msg.k[4] == pytest.approx(intrinsics.fy)
        assert msg.k[2] == pytest.approx(intrinsics.ppx)
        assert msg.k[5] == pytest.approx(intrinsics.ppy)
        assert msg.k[8] == pytest.approx(1.0)

    def test_publish_camera_info_distortion_model(self, bridge, bridge_env):
        bm, _, _ = bridge_env
        intrinsics = _FakeIntrinsics()
        bridge.publish_camera_info(intrinsics)
        msg = bridge._pub_camera_info.publish.call_args[0][0]
        assert msg.distortion_model == "plumb_bob"
        assert msg.d == pytest.approx(intrinsics.coeffs)


class TestShutdown:
    def test_shutdown_destroys_node(self, bridge_env):
        bm, _, node_mock = bridge_env
        b = bm.ROS2Bridge()
        b.shutdown()
        node_mock.destroy_node.assert_called_once()

    def test_shutdown_sets_is_active_false(self, bridge_env):
        bm, _, _ = bridge_env
        b = bm.ROS2Bridge()
        assert b.is_active is True
        b.shutdown()
        assert b.is_active is False

    def test_double_shutdown_is_safe(self, bridge_env):
        bm, _, node_mock = bridge_env
        b = bm.ROS2Bridge()
        b.shutdown()
        b.shutdown()
        node_mock.destroy_node.assert_called_once()

    def test_publish_after_shutdown_raises_runtime_error(self, bridge_env):
        bm, _, _ = bridge_env
        b = bm.ROS2Bridge()
        b.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            b.publish_depth(mx.array(np.zeros((4, 4), dtype=np.uint16)))

    def test_context_manager_calls_shutdown(self, bridge_env):
        bm, _, node_mock = bridge_env
        with bm.ROS2Bridge() as b:
            assert b.is_active is True
        assert b.is_active is False
        node_mock.destroy_node.assert_called_once()


class TestRepr:
    def test_repr_active(self, bridge_env):
        bm, _, _ = bridge_env
        b = bm.ROS2Bridge(node_name="my_node")
        r = repr(b)
        assert "ROS2Bridge" in r
        assert "active" in r
        b.shutdown()

    def test_repr_closed(self, bridge_env):
        bm, _, _ = bridge_env
        b = bm.ROS2Bridge()
        b.shutdown()
        r = repr(b)
        assert "closed" in r

    def test_node_property(self, bridge_env):
        bm, _, node_mock = bridge_env
        b = bm.ROS2Bridge()
        assert b.node is node_mock
        b.shutdown()


# ---------------------------------------------------------------------------
# Converter unit tests — these run WITHOUT rclpy (direct function tests)
# ---------------------------------------------------------------------------
# We import the internal helpers directly by reloading with mocks injected
# so HAS_ROS2=True and the helpers are importable.


@pytest.fixture()
def converters(bridge_env):
    """Return the internal converter functions from the (mocked) bridge module."""
    bm, _, _ = bridge_env
    return {
        "mx_to_numpy": bm._mx_to_numpy,
        "depth_to_image": bm._depth_mx_to_image_msg,
        "color_to_image": bm._color_mx_to_image_msg,
        "pts_to_cloud": bm._pointcloud_mx_to_msg,
        "intrinsics_to_info": bm._camera_info_from_intrinsics,
        "make_header": bm._make_header,
    }


class TestMxToNumpy:
    def test_returns_ndarray(self, converters):
        arr = mx.array(np.zeros((4, 4), dtype=np.float32))
        out = converters["mx_to_numpy"](arr)
        assert isinstance(out, np.ndarray)

    def test_preserves_dtype(self, converters):
        arr = mx.array(np.ones((4, 4), dtype=np.uint16))
        out = converters["mx_to_numpy"](arr)
        assert out.dtype == np.uint16

    def test_preserves_shape(self, converters):
        arr = mx.array(np.zeros((3, 5, 2), dtype=np.float32))
        out = converters["mx_to_numpy"](arr)
        assert out.shape == (3, 5, 2)


class TestDepthConverter:
    def _header(self, bridge_env):
        bm, _, node_mock = bridge_env
        from realsense_mlx.bridges.ros2_bridge import _make_header
        return bm._make_header(node_mock)

    def test_uint16_encoding(self, converters, bridge_env):
        h = self._header(bridge_env)
        depth = mx.array(np.full((4, 4), 500, dtype=np.uint16))
        msg = converters["depth_to_image"](depth, h)
        assert msg.encoding == "16UC1"

    def test_uint16_passthrough(self, converters, bridge_env):
        h = self._header(bridge_env)
        depth = mx.array(np.full((4, 4), 1234, dtype=np.uint16))
        msg = converters["depth_to_image"](depth, h)
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(4, 4)
        assert arr[0, 0] == 1234

    def test_float_metres_to_mm(self, converters, bridge_env):
        h = self._header(bridge_env)
        depth = mx.array(np.full((4, 4), 2.5, dtype=np.float32))
        msg = converters["depth_to_image"](depth, h)
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(4, 4)
        assert arr[0, 0] == 2500

    def test_step_equals_width_times_2(self, converters, bridge_env):
        h = self._header(bridge_env)
        depth = mx.array(np.zeros((8, 16), dtype=np.uint16))
        msg = converters["depth_to_image"](depth, h)
        assert msg.step == 16 * 2

    def test_3d_array_raises(self, converters, bridge_env):
        h = self._header(bridge_env)
        depth = mx.array(np.zeros((4, 4, 3), dtype=np.uint16))
        with pytest.raises(ValueError):
            converters["depth_to_image"](depth, h)


class TestColorConverter:
    def _header(self, bridge_env):
        bm, _, node_mock = bridge_env
        return bm._make_header(node_mock)

    def test_bgr8_encoding(self, converters, bridge_env):
        h = self._header(bridge_env)
        color = mx.array(np.zeros((4, 4, 3), dtype=np.uint8))
        msg = converters["color_to_image"](color, h)
        assert msg.encoding == "bgr8"

    def test_rgb_swapped_to_bgr(self, converters, bridge_env):
        h = self._header(bridge_env)
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        data[:, :, 0] = 5   # R
        data[:, :, 2] = 50  # B
        msg = converters["color_to_image"](mx.array(data), h)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(4, 4, 3)
        assert arr[0, 0, 0] == 50  # B in BGR
        assert arr[0, 0, 2] == 5   # R in BGR

    def test_grayscale_to_3channel(self, converters, bridge_env):
        h = self._header(bridge_env)
        gray = mx.array(np.full((4, 4), 128, dtype=np.uint8))
        msg = converters["color_to_image"](gray, h)
        assert msg.step == 4 * 3

    def test_rgba_drops_alpha(self, converters, bridge_env):
        h = self._header(bridge_env)
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[:, :, 3] = 255  # alpha
        msg = converters["color_to_image"](mx.array(data), h)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(4, 4, 3)
        assert arr.shape[2] == 3

    def test_step_equals_width_times_3(self, converters, bridge_env):
        h = self._header(bridge_env)
        color = mx.array(np.zeros((8, 16, 3), dtype=np.uint8))
        msg = converters["color_to_image"](color, h)
        assert msg.step == 16 * 3

    def test_wrong_channels_raises(self, converters, bridge_env):
        h = self._header(bridge_env)
        bad = mx.array(np.zeros((4, 4, 5), dtype=np.uint8))
        with pytest.raises(ValueError):
            converters["color_to_image"](bad, h)


class TestPointcloudConverter:
    def _header(self, bridge_env):
        bm, _, node_mock = bridge_env
        return bm._make_header(node_mock)

    def test_field_names_xyz(self, converters, bridge_env):
        h = self._header(bridge_env)
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        names = {f.name for f in msg.fields}
        assert names == {"x", "y", "z"}

    def test_point_step_is_12(self, converters, bridge_env):
        h = self._header(bridge_env)
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        assert msg.point_step == 12

    def test_width_equals_n_points(self, converters, bridge_env):
        h = self._header(bridge_env)
        pts = mx.array(np.zeros((42, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        assert msg.width == 42

    def test_height_is_1(self, converters, bridge_env):
        h = self._header(bridge_env)
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        assert msg.height == 1

    def test_is_dense(self, converters, bridge_env):
        h = self._header(bridge_env)
        pts = mx.array(np.zeros((10, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        assert msg.is_dense is True

    def test_data_byte_length(self, converters, bridge_env):
        h = self._header(bridge_env)
        n = 20
        pts = mx.array(np.zeros((n, 3), dtype=np.float32))
        msg = converters["pts_to_cloud"](pts, h)
        assert len(msg.data) == n * 12

    def test_wrong_shape_raises(self, converters, bridge_env):
        h = self._header(bridge_env)
        bad = mx.array(np.zeros((10, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            converters["pts_to_cloud"](bad, h)

    def test_1d_raises(self, converters, bridge_env):
        h = self._header(bridge_env)
        bad = mx.array(np.zeros(30, dtype=np.float32))
        with pytest.raises((ValueError, Exception)):
            converters["pts_to_cloud"](bad, h)


class TestCameraInfoConverter:
    def _header(self, bridge_env):
        bm, _, node_mock = bridge_env
        return bm._make_header(node_mock)

    def test_width_height(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        assert msg.width == 640
        assert msg.height == 480

    def test_k_matrix_diagonal(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        assert msg.k[0] == pytest.approx(intr.fx)
        assert msg.k[4] == pytest.approx(intr.fy)
        assert msg.k[2] == pytest.approx(intr.ppx)
        assert msg.k[5] == pytest.approx(intr.ppy)
        assert msg.k[1] == pytest.approx(0.0)   # off-diagonal
        assert msg.k[8] == pytest.approx(1.0)

    def test_distortion_model(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        assert msg.distortion_model == "plumb_bob"

    def test_distortion_coeffs(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        assert msg.d == pytest.approx(intr.coeffs)

    def test_no_coeffs_attribute_defaults_to_zeros(self, converters, bridge_env):
        h = self._header(bridge_env)

        class MinimalIntrinsics:
            width = 320
            height = 240
            fx = fy = 200.0
            ppx = ppy = 160.0

        msg = converters["intrinsics_to_info"](MinimalIntrinsics(), h)
        assert msg.d == pytest.approx([0.0] * 5)

    def test_p_matrix_length(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        assert len(msg.p) == 12  # 3×4 row-major

    def test_r_matrix_is_identity(self, converters, bridge_env):
        h = self._header(bridge_env)
        intr = _FakeIntrinsics()
        msg = converters["intrinsics_to_info"](intr, h)
        expected_r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        assert msg.r == pytest.approx(expected_r)
