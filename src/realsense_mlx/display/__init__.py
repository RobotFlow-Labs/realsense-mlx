"""Display module — OpenCV-backed viewer for depth and colour streams.

``opencv-python`` is an optional dependency.  Importing this module is safe
even without it; the ImportError is deferred until :class:`RealsenseViewer`
is instantiated.

Example
-------
::

    from realsense_mlx.display import RealsenseViewer

    with RealsenseViewer(title="Depth Demo", width=1280, height=480) as viewer:
        while viewer.is_open():
            viewer.show_depth(colorized_depth)           # (H, W, 3) uint8 RGB
            viewer.show_side_by_side(colorized, color)   # two frames
"""

from realsense_mlx.display.viewer import RealsenseViewer

__all__ = ["RealsenseViewer"]
