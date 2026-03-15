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

    def export_ply_mesh(
        self,
        points: mx.array,
        faces: mx.array,
        path: str | Path,
        colors: Optional[mx.array] = None,
        normals: Optional[mx.array] = None,
    ) -> int:
        """Write a triangle mesh to a binary little-endian PLY file.

        Parameters
        ----------
        points  : (N, 3) float32 vertex positions.
        faces   : (M, 3) int32 face (triangle) indices.
        path    : Destination file path.
        colors  : Optional (N, 3) uint8 or float32 per-vertex RGB colors.
        normals : Optional (N, 3) float32 per-vertex normals.

        Returns
        -------
        int
            Number of faces written.
        """
        path = Path(path)

        mx.eval(points, faces)
        pts_np = np.array(points, copy=False).reshape(-1, 3).astype(np.float32)
        faces_np = np.array(faces, copy=False).reshape(-1, 3).astype(np.int32)
        n_verts = len(pts_np)
        n_faces = len(faces_np)

        # --- optional per-vertex attributes ---
        has_color = colors is not None
        if has_color:
            mx.eval(colors)
            col_np = np.array(colors, copy=False).reshape(-1, 3)
            if col_np.dtype != np.uint8:
                if np.issubdtype(col_np.dtype, np.floating):
                    col_np = np.clip(col_np * 255.0, 0, 255).astype(np.uint8)
                else:
                    col_np = col_np.astype(np.uint8)
        else:
            col_np = None

        has_normals = normals is not None
        if has_normals:
            mx.eval(normals)
            nrm_np = np.array(normals, copy=False).reshape(-1, 3).astype(np.float32)
        else:
            nrm_np = None

        # --- header ---
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n_verts}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_normals:
            header_lines += [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        if has_color:
            header_lines += [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        header_lines += [
            f"element face {n_faces}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        header = "\n".join(header_lines) + "\n"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))

            # --- vertex body ---
            if not has_normals and not has_color:
                f.write(pts_np.tobytes())
            else:
                # Build structured dtype to interleave fields
                fields: list[tuple] = [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                ]
                if has_normals:
                    fields += [
                        ("nx", np.float32),
                        ("ny", np.float32),
                        ("nz", np.float32),
                    ]
                if has_color:
                    fields += [
                        ("r", np.uint8),
                        ("g", np.uint8),
                        ("b", np.uint8),
                    ]
                dtype = np.dtype(fields)
                buf = np.empty(n_verts, dtype=dtype)
                buf["x"] = pts_np[:, 0]
                buf["y"] = pts_np[:, 1]
                buf["z"] = pts_np[:, 2]
                if has_normals and nrm_np is not None:
                    buf["nx"] = nrm_np[:, 0]
                    buf["ny"] = nrm_np[:, 1]
                    buf["nz"] = nrm_np[:, 2]
                if has_color and col_np is not None:
                    buf["r"] = col_np[:, 0]
                    buf["g"] = col_np[:, 1]
                    buf["b"] = col_np[:, 2]
                f.write(buf.tobytes())

            # --- face body ---
            # Each face: 1 byte count (=3) + 3 x int32 = 13 bytes
            face_dtype = np.dtype([("n", np.uint8), ("v", np.int32, (3,))])
            face_buf = np.empty(n_faces, dtype=face_dtype)
            face_buf["n"] = 3
            face_buf["v"] = faces_np
            f.write(face_buf.tobytes())

        return n_faces

    def export_obj(
        self,
        points: mx.array,
        path: str | Path,
        faces: Optional[mx.array] = None,
        colors: Optional[mx.array] = None,
    ) -> int:
        """Write point cloud or mesh to Wavefront OBJ format.

        OBJ uses ASCII text.  Vertex colours, when provided, are written as
        non-standard per-vertex ``v x y z r g b`` lines (the convention used
        by MeshLab and CloudCompare).

        Parameters
        ----------
        points  : (N, 3) or (H, W, 3) float32 vertex positions.
        path    : Destination file path.
        faces   : Optional (M, 3) int32 face indices.  If omitted, only
                  ``v`` lines are written (pure point cloud).
        colors  : Optional (N, 3) or (H, W, 3) uint8 or float32 RGB.

        Returns
        -------
        int
            Number of vertices written.
        """
        path = Path(path)

        mx.eval(points)
        pts_np = np.array(points, copy=False).reshape(-1, 3).astype(np.float32)
        n_verts = len(pts_np)

        has_color = colors is not None
        if has_color:
            mx.eval(colors)
            col_np = np.array(colors, copy=False).reshape(-1, 3)
            if col_np.dtype != np.uint8:
                if np.issubdtype(col_np.dtype, np.floating):
                    col_np = np.clip(col_np * 255.0, 0, 255).astype(np.uint8)
                else:
                    col_np = col_np.astype(np.uint8)
        else:
            col_np = None

        has_faces = faces is not None
        if has_faces:
            mx.eval(faces)
            faces_np = np.array(faces, copy=False).reshape(-1, 3).astype(np.int32)
        else:
            faces_np = None

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="ascii") as f:
            f.write("# Wavefront OBJ - exported by realsense-mlx\n")
            f.write(f"# Vertices: {n_verts}\n")
            if has_faces and faces_np is not None:
                f.write(f"# Faces: {len(faces_np)}\n")

            # Vertex lines
            if has_color and col_np is not None:
                # Non-standard RGB extension: v x y z r g b (normalised 0-1)
                for i in range(n_verts):
                    x, y, z = pts_np[i]
                    r, g, b = col_np[i] / 255.0
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                for i in range(n_verts):
                    x, y, z = pts_np[i]
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            # Face lines — OBJ uses 1-based indices
            if has_faces and faces_np is not None:
                for fi in range(len(faces_np)):
                    a, b_idx, c = faces_np[fi] + 1   # 1-based
                    f.write(f"f {a} {b_idx} {c}\n")

        return n_verts

    def compute_normals(
        self,
        points: mx.array,
        faces: mx.array,
    ) -> mx.array:
        """Compute per-vertex normals from mesh faces.

        Uses face normal averaging weighted by triangle area.  Each face
        contributes its (unnormalised) cross-product normal to the three
        incident vertices; the accumulated vectors are then L2-normalised.

        Parameters
        ----------
        points : mx.array
            ``(N, 3)`` float32 vertex positions.
        faces  : mx.array
            ``(M, 3)`` int32 face indices.

        Returns
        -------
        mx.array
            ``(N, 3)`` float32 unit per-vertex normals.  Isolated vertices
            (not referenced by any face) receive the zero vector.
        """
        if faces.shape[0] == 0:
            n = points.shape[0] if points.ndim >= 1 else 0
            return mx.zeros((n, 3), dtype=mx.float32)

        mx.eval(points, faces)
        verts_np = np.array(points, copy=False).reshape(-1, 3).astype(np.float32)
        faces_np = np.array(faces, copy=False).reshape(-1, 3)

        N = verts_np.shape[0]
        p0 = verts_np[faces_np[:, 0]]
        p1 = verts_np[faces_np[:, 1]]
        p2 = verts_np[faces_np[:, 2]]

        # Cross product — magnitude proportional to triangle area
        face_normals = np.cross(p1 - p0, p2 - p0)  # (M, 3)

        normals_np = np.zeros((N, 3), dtype=np.float32)
        np.add.at(normals_np, faces_np[:, 0], face_normals)
        np.add.at(normals_np, faces_np[:, 1], face_normals)
        np.add.at(normals_np, faces_np[:, 2], face_normals)

        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        normals_np = normals_np / safe_norms

        return mx.array(normals_np)

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
