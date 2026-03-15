"""Triangle mesh generation from organised depth point clouds.

Key design decisions
--------------------
* Fully vectorised using MLX — zero Python loops over pixels.
* Two triangles are generated per quad cell in the (H-1)×(W-1) grid.
* Triangles whose vertices include any zero-depth point (invalid) are
  discarded before face indices are returned.
* Triangles where any edge length exceeds ``max_edge_length`` are treated
  as depth discontinuity artefacts and are also discarded.
* All filtering is performed in MLX before converting to NumPy, keeping
  data on-device as long as possible.
* ``compute_normals`` accumulates area-weighted face contributions into
  per-vertex normals and then L2-normalises the result — matches the
  standard Gouraud shading convention used by Open3D / librealsense viewer.

Coordinate conventions
-----------------------
Input points follow the RealSense camera frame output by
:class:`~realsense_mlx.geometry.pointcloud.PointCloudGenerator`::

    +X → right, +Y → down, +Z → into the scene
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = ["DepthMeshGenerator"]


class DepthMeshGenerator:
    """Generate triangle meshes from organised depth point clouds.

    Takes a ``(H, W, 3)`` organised point cloud and generates a triangle mesh
    by connecting neighbouring valid pixels.  Each 2×2 quad of neighbouring
    pixels yields two triangles.  Edges that span depth discontinuities
    (longer than ``max_edge_length``) or that involve zero-depth vertices are
    suppressed.

    Parameters
    ----------
    max_edge_length : float
        Maximum allowed edge length in metres.  Edges longer than this are
        considered depth discontinuities; the corresponding triangle is
        omitted entirely.

    Examples
    --------
    >>> import mlx.core as mx
    >>> gen = DepthMeshGenerator(max_edge_length=0.05)
    >>> H, W = 10, 10
    >>> # Flat plane at Z = 1.0 m (fx=fy=1, ppx=ppy=0 → X=u, Y=v, Z=1)
    >>> pts = mx.ones((H, W, 3), dtype=mx.float32)
    >>> verts, faces = gen.generate(pts)
    >>> faces.shape[0]  # 2 * (H-1) * (W-1)
    162
    """

    def __init__(self, max_edge_length: float = 0.05) -> None:
        if max_edge_length <= 0.0:
            raise ValueError(
                f"max_edge_length must be positive, got {max_edge_length}"
            )
        self.max_edge_length = float(max_edge_length)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, points: mx.array) -> tuple[mx.array, mx.array]:
        """Generate a triangle mesh from an organised point cloud.

        Parameters
        ----------
        points : mx.array
            ``(H, W, 3)`` float32 organised point cloud from
            :class:`~realsense_mlx.geometry.pointcloud.PointCloudGenerator`.
            Pixels with zero depth must have all three coordinates set to
            zero (which is the default output of ``PointCloudGenerator``).

        Returns
        -------
        vertices : mx.array
            ``(H*W, 3)`` float32 vertex positions (the flattened input grid).
        faces : mx.array
            ``(M, 3)`` int32 triangle face indices into ``vertices``.
            Only valid, short-edge triangles are included.

        Raises
        ------
        ValueError
            If ``points`` does not have shape ``(H, W, 3)``.
        """
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(
                f"points must be (H, W, 3), got {points.shape}"
            )

        H, W = points.shape[0], points.shape[1]

        # Vertex array is simply the flattened grid: (H*W, 3)
        vertices = points.reshape(-1, 3)

        # Build a flat index grid (H, W) — each cell holds its linear index
        idx = mx.arange(H * W, dtype=mx.int32).reshape(H, W)

        # ── Quad decomposition ────────────────────────────────────────────
        # For each quad (i,j)→(i+1,j+1) we emit two counter-clockwise tris:
        #   tri_a: top-left    (i,j),   bottom-left (i+1,j),  top-right (i,j+1)
        #   tri_b: bottom-left (i+1,j), bottom-right(i+1,j+1),top-right (i,j+1)
        #
        # Index grid slicing — all shapes (H-1, W-1), then flattened to (N,)
        tl = idx[:-1, :-1].reshape(-1)   # top-left
        bl = idx[1:,  :-1].reshape(-1)   # bottom-left
        tr = idx[:-1, 1:].reshape(-1)    # top-right
        br = idx[1:,  1:].reshape(-1)    # bottom-right

        # tri_a: TL, BL, TR  |  tri_b: BL, BR, TR
        tri_a = mx.stack([tl, bl, tr], axis=1)   # (N_quads, 3)
        tri_b = mx.stack([bl, br, tr], axis=1)

        faces_all = mx.concatenate([tri_a, tri_b], axis=0)   # (2*N_quads, 3)

        # Force evaluation so we can access via numpy for filtering
        mx.eval(vertices, faces_all)

        pts_np = np.array(vertices, copy=False)   # (H*W, 3) float32
        faces_np = np.array(faces_all, copy=False)  # (2*N_quads, 3) int32

        # ── Filter 1: remove triangles with any zero-depth vertex ─────────
        # A vertex is invalid when Z == 0 (PointCloudGenerator convention)
        valid_vertex = pts_np[:, 2] != 0.0           # (H*W,) bool

        v0_ok = valid_vertex[faces_np[:, 0]]
        v1_ok = valid_vertex[faces_np[:, 1]]
        v2_ok = valid_vertex[faces_np[:, 2]]
        depth_mask = v0_ok & v1_ok & v2_ok           # (M,) bool

        faces_np = faces_np[depth_mask]

        if faces_np.shape[0] == 0:
            return mx.array(pts_np), mx.zeros((0, 3), dtype=mx.int32)

        # ── Filter 2: remove triangles with any edge > max_edge_length ────
        p0 = pts_np[faces_np[:, 0]]   # (M, 3)
        p1 = pts_np[faces_np[:, 1]]
        p2 = pts_np[faces_np[:, 2]]

        # Three edge squared lengths per triangle
        e01_sq = np.sum((p1 - p0) ** 2, axis=1)
        e12_sq = np.sum((p2 - p1) ** 2, axis=1)
        e20_sq = np.sum((p0 - p2) ** 2, axis=1)

        max_sq = float(self.max_edge_length) ** 2
        edge_mask = (e01_sq <= max_sq) & (e12_sq <= max_sq) & (e20_sq <= max_sq)
        faces_np = faces_np[edge_mask]

        return mx.array(pts_np), mx.array(faces_np, dtype=mx.int32)

    def compute_normals(
        self,
        vertices: mx.array,
        faces: mx.array,
    ) -> mx.array:
        """Compute per-vertex normals from mesh faces using area-weighting.

        Each face contributes its face normal scaled by the triangle's area
        to all three of its vertices.  The accumulated vectors are then
        L2-normalised.  Isolated vertices (not referenced by any face) receive
        the zero vector.

        Parameters
        ----------
        vertices : mx.array
            ``(N, 3)`` float32 vertex positions.
        faces : mx.array
            ``(M, 3)`` int32 face indices.

        Returns
        -------
        mx.array
            ``(N, 3)`` float32 per-vertex normals (unit length, or zero for
            isolated vertices).
        """
        if faces.shape[0] == 0:
            return mx.zeros_like(vertices)

        mx.eval(vertices, faces)
        verts_np = np.array(vertices, copy=False).astype(np.float32)
        faces_np = np.array(faces, copy=False)

        N = verts_np.shape[0]

        p0 = verts_np[faces_np[:, 0]]   # (M, 3)
        p1 = verts_np[faces_np[:, 1]]
        p2 = verts_np[faces_np[:, 2]]

        # Face normals (cross product) — magnitude = 2 * triangle area
        e1 = p1 - p0
        e2 = p2 - p0
        face_normals = np.cross(e1, e2)   # (M, 3), already area-weighted

        # Accumulate into per-vertex normals
        normals_np = np.zeros((N, 3), dtype=np.float32)
        # Use np.add.at for unbuffered scatter-add
        np.add.at(normals_np, faces_np[:, 0], face_normals)
        np.add.at(normals_np, faces_np[:, 1], face_normals)
        np.add.at(normals_np, faces_np[:, 2], face_normals)

        # L2 normalise — guard against zero vectors
        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normals_np = normals_np / norms

        return mx.array(normals_np)
