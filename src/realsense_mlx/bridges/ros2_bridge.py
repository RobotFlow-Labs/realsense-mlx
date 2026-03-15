"""ROS2 bridge for publishing MLX-processed depth data.

This is a **compatibility seam**, not a full ROS2 node implementation.
It publishes processed depth frames to ROS2 topics when ``rclpy`` is
available.  The module can always be imported safely; the bridge raises
``ImportError`` on *instantiation* if the runtime dependency is absent.

Capability-gated
----------------
If ``rclpy`` is not installed, importing this module succeeds and
``HAS_ROS2`` is set to ``False``.  ``ROS2Bridge(...)`` will raise a
clear ``ImportError`` with installation instructions.

Threading model
---------------
ROS2 spinning is **not** managed here.  The caller is responsible for
running ``rclpy.spin`` or ``rclpy.spin_once`` in a separate thread or
executor.  ``ROS2Bridge`` only creates publishers and provides
``publish_*`` helper methods.  This avoids dictating an executor model
and keeps the bridge usable in both single-threaded and multi-threaded
callers.

Topics published
----------------
+----------------------------------------------+------------------------------+
| Topic                                        | Message type                 |
+==============================================+==============================+
| ``/realsense_mlx/depth``                     | ``sensor_msgs/Image``        |
|                                              | encoding: ``16UC1``          |
+----------------------------------------------+------------------------------+
| ``/realsense_mlx/color``                     | ``sensor_msgs/Image``        |
|                                              | encoding: ``bgr8``           |
+----------------------------------------------+------------------------------+
| ``/realsense_mlx/points``                    | ``sensor_msgs/PointCloud2``  |
+----------------------------------------------+------------------------------+
| ``/realsense_mlx/camera_info``               | ``sensor_msgs/CameraInfo``   |
+----------------------------------------------+------------------------------+

All topic prefixes are configurable via the ``topic_prefix`` constructor
argument.

Limitations (known prototype gaps)
-----------------------------------
- No QoS / reliability configuration exposed yet.
- ``publish_pointcloud`` produces a flat XYZ point cloud (no intensity).
- ``publish_camera_info`` only fills the intrinsic matrix K; D, R, P are
  zeros unless the caller provides a full ``CameraIntrinsics`` object.
- This bridge does **not** call ``rclpy.init``.  The caller must
  initialise rclpy before constructing a ``ROS2Bridge``.

Example
-------
::

    import rclpy
    import mlx.core as mx
    from realsense_mlx.bridges import ROS2Bridge

    rclpy.init()
    bridge = ROS2Bridge(node_name="realsense_mlx_bridge")

    # In your capture loop:
    bridge.publish_depth(depth_mx)
    bridge.publish_color(color_mx)

    bridge.shutdown()
    rclpy.shutdown()
"""

from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------

try:
    import rclpy                                         # type: ignore[import]
    import rclpy.node                                    # type: ignore[import]
    from rclpy.qos import QoSProfile, ReliabilityPolicy  # type: ignore[import]
    from sensor_msgs.msg import CameraInfo               # type: ignore[import]
    from sensor_msgs.msg import Image                    # type: ignore[import]
    from sensor_msgs.msg import PointCloud2              # type: ignore[import]
    from sensor_msgs.msg import PointField               # type: ignore[import]
    from std_msgs.msg import Header                      # type: ignore[import]
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

_ROS2_INSTALL_HINT = (
    "rclpy is not installed.\n"
    "Install ROS2 Humble or Iron and source the setup script, then:\n"
    "    source /opt/ros/humble/setup.bash\n"
    "    pip install rclpy  # or use colcon build in your workspace"
)


# ---------------------------------------------------------------------------
# Internal helpers (no rclpy dependency)
# ---------------------------------------------------------------------------


def _mx_to_numpy(arr: mx.array) -> np.ndarray:
    """Evaluate an MLX array and return a numpy ndarray."""
    mx.eval(arr)
    return np.array(arr, copy=False)


def _make_header(node: "rclpy.node.Node", stamp=None) -> "Header":
    """Build a ROS2 Header with the node's clock or a supplied stamp."""
    header = Header()
    if stamp is not None:
        header.stamp = stamp
    else:
        header.stamp = node.get_clock().now().to_msg()
    header.frame_id = "realsense_mlx_frame"
    return header


def _depth_mx_to_image_msg(
    depth: mx.array,
    header: "Header",
) -> "Image":
    """Convert a (H, W) uint16 MLX depth array to a sensor_msgs/Image.

    The output encoding is ``16UC1`` (millimetres, standard RealSense units).
    Float arrays are converted to uint16 by multiplying by 1000 (m → mm).
    """
    arr = _mx_to_numpy(depth)

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Assume metres → convert to millimetres
        arr = (arr * 1000.0).clip(0, 65535).astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)

    if arr.ndim != 2:
        raise ValueError(
            f"depth array must be 2-D (H, W), got shape {arr.shape}"
        )

    msg = Image()
    msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = "16UC1"
    msg.is_bigendian = False
    msg.step = arr.shape[1] * 2  # 2 bytes per uint16 pixel
    msg.data = arr.tobytes()
    return msg


def _color_mx_to_image_msg(
    color: mx.array,
    header: "Header",
) -> "Image":
    """Convert a (H, W, 3) uint8 RGB MLX array to a sensor_msgs/Image (bgr8).

    MLX colourizers produce RGB; ROS2 Image convention is BGR.
    The channel swap is applied automatically.
    """
    arr = _mx_to_numpy(color)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        # Grayscale → stack to 3-channel BGR
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # RGB → BGR
        arr = arr[:, :, ::-1]
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA → BGR (drop alpha, swap)
        arr = arr[:, :, :3][:, :, ::-1]
    else:
        raise ValueError(
            f"color array must be (H, W), (H, W, 3) or (H, W, 4), got {arr.shape}"
        )

    arr = np.ascontiguousarray(arr)

    msg = Image()
    msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = arr.shape[1] * 3
    msg.data = arr.tobytes()
    return msg


def _pointcloud_mx_to_msg(
    points: mx.array,
    header: "Header",
) -> "PointCloud2":
    """Convert an (N, 3) float32 XYZ point array to sensor_msgs/PointCloud2.

    Parameters
    ----------
    points:
        ``(N, 3)`` float32 array of XYZ coordinates in metres.
    header:
        ROS2 Header with timestamp and frame_id.

    Returns
    -------
    sensor_msgs/PointCloud2
        Unordered, single-row point cloud.
    """
    arr = _mx_to_numpy(points).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {arr.shape}"
        )

    n_points = arr.shape[0]

    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12           # 3 × float32 = 12 bytes
    msg.row_step = msg.point_step * n_points
    msg.data = arr.tobytes()
    msg.is_dense = True
    return msg


def _camera_info_from_intrinsics(intrinsics, header: "Header") -> "CameraInfo":
    """Build a sensor_msgs/CameraInfo from a CameraIntrinsics-like object.

    The *intrinsics* object must have attributes: ``width``, ``height``,
    ``fx``, ``fy``, ``ppx``, ``ppy``.  The ``coeffs`` attribute (list of
    5 floats) is used for the distortion vector D if present.

    Parameters
    ----------
    intrinsics:
        A :class:`~realsense_mlx.geometry.intrinsics.CameraIntrinsics`
        instance or any object with the attributes described above.
    header:
        ROS2 Header with timestamp and frame_id.
    """
    msg = CameraInfo()
    msg.header = header
    msg.width = int(intrinsics.width)
    msg.height = int(intrinsics.height)

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)

    # Intrinsic matrix K (row-major, 3×3):
    #  [fx  0  cx]
    #  [ 0 fy  cy]
    #  [ 0  0   1]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    # Rectification matrix R = identity
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # Projection matrix P (3×4):
    #  [fx  0  cx  0]
    #  [ 0 fy  cy  0]
    #  [ 0  0   1  0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Distortion model and coefficients
    coeffs = getattr(intrinsics, "coeffs", None) or [0.0] * 5
    msg.distortion_model = "plumb_bob"
    msg.d = [float(c) for c in coeffs]

    return msg


# ---------------------------------------------------------------------------
# ROS2Bridge
# ---------------------------------------------------------------------------


class ROS2Bridge:
    """Publish MLX-processed depth / colour / point-cloud frames to ROS2.

    This class is a **compatibility seam** — a thin adapter that converts
    MLX arrays to ROS2 message types and publishes them.  It does not manage
    the ROS2 executor; the caller must spin the node separately.

    Parameters
    ----------
    node_name:
        Name for the underlying ``rclpy.node.Node``.
    topic_prefix:
        Prefix applied to all published topic names.
        Default ``"/realsense_mlx"``.
    qos_depth:
        QoS history depth for all publishers.  Default 10.

    Raises
    ------
    ImportError
        If ``rclpy`` (and the ``sensor_msgs`` package) are not installed.

    Notes
    -----
    ``rclpy.init()`` must be called before constructing this class.
    The bridge does **not** call ``rclpy.init()`` or ``rclpy.shutdown()``
    to avoid interfering with the caller's lifecycle management.

    Topics
    ------
    ``{prefix}/depth``         sensor_msgs/Image  (16UC1)
    ``{prefix}/color``         sensor_msgs/Image  (bgr8)
    ``{prefix}/points``        sensor_msgs/PointCloud2
    ``{prefix}/camera_info``   sensor_msgs/CameraInfo

    Example
    -------
    ::

        import rclpy
        from realsense_mlx.bridges import ROS2Bridge

        rclpy.init()
        bridge = ROS2Bridge()

        bridge.publish_depth(depth_mlx_array)
        bridge.publish_color(color_mlx_array)

        bridge.shutdown()
        rclpy.shutdown()
    """

    def __init__(
        self,
        node_name: str = "realsense_mlx_bridge",
        topic_prefix: str = "/realsense_mlx",
        qos_depth: int = 10,
    ) -> None:
        if not HAS_ROS2:
            raise ImportError(_ROS2_INSTALL_HINT)

        self._node = rclpy.node.Node(node_name)
        prefix = topic_prefix.rstrip("/")

        qos = QoSProfile(depth=qos_depth)

        self._pub_depth = self._node.create_publisher(
            Image, f"{prefix}/depth", qos
        )
        self._pub_color = self._node.create_publisher(
            Image, f"{prefix}/color", qos
        )
        self._pub_points = self._node.create_publisher(
            PointCloud2, f"{prefix}/points", qos
        )
        self._pub_camera_info = self._node.create_publisher(
            CameraInfo, f"{prefix}/camera_info", qos
        )

        self._closed = False

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def publish_depth(
        self,
        depth: mx.array,
        timestamp=None,
    ) -> None:
        """Publish a depth frame to ``{prefix}/depth`` (16UC1).

        Parameters
        ----------
        depth:
            ``(H, W)`` array.  uint16 values are treated as millimetres.
            float32/float64 values are treated as metres and converted.
        timestamp:
            Optional ROS2 ``Time`` message.  When ``None``, the node's
            clock is used.

        Raises
        ------
        RuntimeError
            If ``shutdown()`` has been called.
        ValueError
            If *depth* is not a 2-D array.
        """
        self._check_not_closed()
        header = _make_header(self._node, timestamp)
        msg = _depth_mx_to_image_msg(depth, header)
        self._pub_depth.publish(msg)

    def publish_color(
        self,
        color: mx.array,
        timestamp=None,
    ) -> None:
        """Publish a colour frame to ``{prefix}/color`` (bgr8).

        Parameters
        ----------
        color:
            ``(H, W, 3)`` uint8 RGB array (RGB→BGR swap applied automatically).
            ``(H, W)`` grayscale and ``(H, W, 4)`` RGBA are also accepted.
        timestamp:
            Optional ROS2 ``Time`` message.
        """
        self._check_not_closed()
        header = _make_header(self._node, timestamp)
        msg = _color_mx_to_image_msg(color, header)
        self._pub_color.publish(msg)

    def publish_pointcloud(
        self,
        points: mx.array,
        timestamp=None,
    ) -> None:
        """Publish an XYZ point cloud to ``{prefix}/points``.

        Parameters
        ----------
        points:
            ``(N, 3)`` float32 array of XYZ coordinates in metres.
        timestamp:
            Optional ROS2 ``Time`` message.

        Raises
        ------
        ValueError
            If *points* is not shape ``(N, 3)``.
        """
        self._check_not_closed()
        header = _make_header(self._node, timestamp)
        msg = _pointcloud_mx_to_msg(points, header)
        self._pub_points.publish(msg)

    def publish_camera_info(
        self,
        intrinsics,
        timestamp=None,
    ) -> None:
        """Publish camera intrinsics to ``{prefix}/camera_info``.

        Parameters
        ----------
        intrinsics:
            Any object with ``width``, ``height``, ``fx``, ``fy``,
            ``ppx``, ``ppy`` attributes (e.g.
            :class:`~realsense_mlx.geometry.intrinsics.CameraIntrinsics`).
            Optional ``coeffs`` attribute for distortion coefficients.
        timestamp:
            Optional ROS2 ``Time`` message.
        """
        self._check_not_closed()
        header = _make_header(self._node, timestamp)
        msg = _camera_info_from_intrinsics(intrinsics, header)
        self._pub_camera_info.publish(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Destroy publishers and the underlying ROS2 node.

        Safe to call multiple times.  Does **not** call ``rclpy.shutdown()``.
        """
        if self._closed:
            return
        self._closed = True
        self._node.destroy_node()

    def __enter__(self) -> "ROS2Bridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "active"
        node_name = getattr(self._node, "get_name", lambda: "?")()
        return f"ROS2Bridge(node={node_name!r}, {state})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node(self) -> "rclpy.node.Node":
        """The underlying ``rclpy.node.Node`` instance."""
        return self._node

    @property
    def is_active(self) -> bool:
        """True while the bridge has not been shut down."""
        return not self._closed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("ROS2Bridge has been shut down")
