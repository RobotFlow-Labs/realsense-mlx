"""Compatibility-seam bridges for external middleware.

Each bridge is a thin adapter that publishes MLX-processed frames to an
external bus (ROS2, ZeroMQ, etc.).  Bridges are capability-gated: importing
this package always succeeds; a concrete bridge raises ``ImportError`` on
instantiation when its runtime dependency is absent.

Available bridges
-----------------
:class:`~realsense_mlx.bridges.ros2_bridge.ROS2Bridge`
    Publish depth / colour / point-cloud frames to ROS2 topics via
    ``rclpy``.  Requires ROS2 Humble or later.

Usage
-----
::

    from realsense_mlx.bridges import ROS2Bridge

    bridge = ROS2Bridge(node_name="my_node")   # raises ImportError without rclpy
    bridge.publish_depth(depth_mx_array)
    bridge.shutdown()

Design notes
------------
These are **prototypes / compatibility seams**, not production-grade ROS2
node implementations.  They provide a stable API surface so that downstream
ROS2 packages can integrate without coupling to the internal MLX pipeline
structure.  See each bridge's docstring for limitations and threading notes.
"""

from realsense_mlx.bridges.ros2_bridge import ROS2Bridge, HAS_ROS2

__all__ = ["ROS2Bridge", "HAS_ROS2"]
