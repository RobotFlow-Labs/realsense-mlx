"""MLX point cloud generator — port of ``cuda-pointcloud.cu``.

Key design decisions
--------------------
* Coordinate grids are precomputed once per intrinsics and cached as
  instance attributes.  Subsequent calls to ``generate`` perform only
  depth scaling, optional distortion correction, and broadcasting — no
  repeated ``arange`` allocation.
* All MLX arrays remain on-device throughout; ``mx.eval`` is called
  explicitly after grid construction to materialise them on the Metal
  device before the first ``generate`` call.
* Distortion correction is applied *per-grid*, not per-frame.  For
  constant-intrinsics pipelines this means distortion correction is
  amortised to zero cost after the first call.
* ``export_ply`` uses structured NumPy to write the binary PLY header
  and body in one vectorised call — no per-point Python loop.

Coordinate conventions
-----------------------
Output XYZ follows the RealSense camera coordinate frame::

    +X → right, +Y → down, +Z → into the scene

This matches ``deproject_pixel_to_point`` in ``rsutil.h``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = ["PointCloudGenerator"]

from realsense_mlx.geometry.distortion import undistort
from realsense_mlx.geometry.intrinsics import CameraIntrinsics


class PointCloudGenerator:
    """Deprojects a RealSense depth frame into a (H, W, 3) XYZ point cloud.

    Parameters
    ----------
    intrinsics  : Camera intrinsics (focal lengths, principal point, distortion).
    depth_scale : Depth unit in metres per raw uint16 count.
                  Typical value for D-series: ``0.001`` (1 mm per count).

    Examples
    --------
    >>> import mlx.core as mx
    >>> from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    >>> intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
    >>> gen = PointCloudGenerator(intr, depth_scale=0.001)
    >>> depth = mx.zeros((480, 640), dtype=mx.uint16)
    >>> pc = gen.generate(depth)
    >>> pc.shape
    (480, 640, 3)
    """

    def __init__(self, intrinsics: CameraIntrinsics, depth_scale: float) -> None:
        if depth_scale <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale}")
        self._intrinsics = intrinsics
        self._depth_scale = float(depth_scale)

        # Cached (H, W) normalized-coordinate grids — populated lazily on
        # first call to _ensure_grids().
        self._x_grid: Optional[mx.array] = None   # (W,) or (H, W) after distortion
        self._y_grid: Optional[mx.array] = None   # (H,) or (H, W) after distortion

        # Whether the cached grids have distortion already baked in
        self._grids_corrected: bool = False

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _ensure_grids(self) -> None:
        """Precompute and cache normalised coordinate grids.

        For the ``"none"`` model the grids remain 1-D vectors (broadcast
        by MLX during ``generate``).  For distortion models they are
        expanded to full (H, W) arrays with correction applied.

        This method is idempotent; calling it multiple times is safe and
        only computes once.
        """
        if self._x_grid is not None:
            return  # Already cached

        i = self._intrinsics

        # Raw 1-D normalised coordinate vectors
        x_raw = (mx.arange(i.width, dtype=mx.float32) - i.ppx) / i.fx   # (W,)
        y_raw = (mx.arange(i.height, dtype=mx.float32) - i.ppy) / i.fy  # (H,)

        if not i.has_distortion:
            # Keep 1-D — will broadcast cleanly against (H, W) depth
            self._x_grid = x_raw   # (W,)
            self._y_grid = y_raw   # (H,)
            self._grids_corrected = False
        else:
            # Expand to (H, W) for element-wise distortion correction
            # x repeats along rows, y repeats along columns
            x2d = mx.broadcast_to(x_raw[None, :], (i.height, i.width))   # (H, W)
            y2d = mx.broadcast_to(y_raw[:, None], (i.height, i.width))   # (H, W)

            x_corr, y_corr = undistort(x2d, y2d, i)
            self._x_grid = x_corr   # (H, W)
            self._y_grid = y_corr   # (H, W)
            self._grids_corrected = True

        # Materialise on Metal device before first generate() call
        mx.eval(self._x_grid, self._y_grid)

    def invalidate_cache(self) -> None:
        """Discard cached grids (e.g. after intrinsics update)."""
        self._x_grid = None
        self._y_grid = None
        self._grids_corrected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, depth: mx.array) -> mx.array:
        """Deproject a depth frame into an XYZ point cloud.

        Parameters
        ----------
        depth : (H, W) array of uint16 raw depth values.

        Returns
        -------
        mx.array
            (H, W, 3) float32 array of XYZ coordinates in metres.
            Pixels with zero depth produce ``(0, 0, 0)``.

        Raises
        ------
        ValueError
            If ``depth`` shape does not match intrinsics dimensions.
        """
        i = self._intrinsics
        expected = (i.height, i.width)
        if depth.shape != expected:
            raise ValueError(
                f"depth shape {depth.shape} does not match intrinsics "
                f"{expected} (H, W)"
            )

        self._ensure_grids()

        # Convert raw counts to metres — use float32 throughout
        z = depth.astype(mx.float32) * self._depth_scale  # (H, W)

        if not self._grids_corrected:
            # 1-D grids: x_grid is (W,), y_grid is (H,)
            # Broadcasting: x_grid[None, :] * z → (H, W)
            #               y_grid[:, None] * z → (H, W)
            X = self._x_grid[None, :] * z   # (H, W)
            Y = self._y_grid[:, None] * z   # (H, W)
        else:
            # 2-D grids: (H, W) — element-wise multiply
            X = self._x_grid * z  # (H, W)
            Y = self._y_grid * z  # (H, W)

        # Stack into (H, W, 3)
        return mx.stack([X, Y, z], axis=-1)

    def generate_with_color(
        self,
        depth: mx.array,
        color: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Deproject depth and pair with co-registered colour values.

        Assumes ``depth`` and ``color`` are already aligned (same resolution).
        Use :class:`~realsense_mlx.geometry.align.Aligner` to align first
        if they come from different sensors.

        Parameters
        ----------
        depth : (H, W) uint16 depth frame.
        color : (H, W, C) uint8 or float32 colour frame.
                ``C`` is typically 3 (RGB) or 4 (RGBA).

        Returns
        -------
        points : (H, W, 3) float32 XYZ
        colors : (H, W, C) — same dtype as input ``color``
        """
        if color.shape[:2] != (self._intrinsics.height, self._intrinsics.width):
            raise ValueError(
                f"color shape {color.shape[:2]} does not match intrinsics "
                f"({self._intrinsics.height}, {self._intrinsics.width})"
            )
        points = self.generate(depth)
        return points, color

    def export_ply(
        self,
        points: mx.array,
        path: str | Path,
        colors: Optional[mx.array] = None,
        skip_zero: bool = True,
    ) -> int:
        """Write point cloud to a binary little-endian PLY file.

        Parameters
        ----------
        points   : (H, W, 3) or (N, 3) float32 XYZ array.
        path     : Destination file path.
        colors   : Optional (H, W, 3) or (N, 3) uint8 RGB array.
        skip_zero: If True, omit points where Z==0 (invalid depth).

        Returns
        -------
        int
            Number of points written.
        """
        path = Path(path)

        # Materialise and convert to NumPy
        mx.eval(points)
        pts_np = np.array(points, copy=False).reshape(-1, 3)  # (N, 3) float32

        has_color = colors is not None
        if has_color:
            mx.eval(colors)
            col_np = np.array(colors, copy=False).reshape(-1, 3)  # (N, 3)
            # Ensure uint8
            if col_np.dtype != np.uint8:
                if np.issubdtype(col_np.dtype, np.floating):
                    col_np = np.clip(col_np * 255.0, 0, 255).astype(np.uint8)
                else:
                    col_np = col_np.astype(np.uint8)
        else:
            col_np = None

        if skip_zero:
            mask = pts_np[:, 2] != 0.0
            pts_np = pts_np[mask]
            if col_np is not None:
                col_np = col_np[mask]

        n_points = len(pts_np)

        # Build PLY header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n_points}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_color:
            header_lines += [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))

            if has_color and col_np is not None:
                # Interleave xyz (float32 x3) + rgb (uint8 x3) = 15 bytes/point
                # Use structured NumPy array for zero-copy write
                dtype = np.dtype([
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("r", np.uint8),
                    ("g", np.uint8),
                    ("b", np.uint8),
                ])
                buf = np.empty(n_points, dtype=dtype)
                buf["x"] = pts_np[:, 0]
                buf["y"] = pts_np[:, 1]
                buf["z"] = pts_np[:, 2]
                buf["r"] = col_np[:, 0]
                buf["g"] = col_np[:, 1]
                buf["b"] = col_np[:, 2]
                f.write(buf.tobytes())
            else:
                # XYZ only — write as contiguous float32 block
                f.write(pts_np.astype(np.float32, copy=False).tobytes())

        return n_points

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """The camera intrinsics used by this generator."""
        return self._intrinsics

    @property
    def depth_scale(self) -> float:
        """Depth scale in metres per raw count."""
        return self._depth_scale

    def __repr__(self) -> str:
        return (
            f"PointCloudGenerator({self._intrinsics.width}x{self._intrinsics.height}, "
            f"scale={self._depth_scale}, model='{self._intrinsics.model}', "
            f"cached={self._x_grid is not None})"
        )
