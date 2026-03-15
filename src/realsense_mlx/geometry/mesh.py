"""Triangle mesh generation from organised depth point clouds.

Key design decisions
--------------------
* Fully vectorised using MLX — zero Python loops over pixels.
* Two triangles are generated per quad cell in the (H-1)×(W-1) grid.
* Triangles whose vertices include any zero-depth point (invalid) are
  discarded before face indices are returned.
* Triangles where any edge length exceeds ``max_edge_length`` are treated
  as depth discontinuity artefacts and are also discarded.
* ALL filtering masks (zero-depth check, edge-length check) are computed
  entirely on MLX using vectorised gather + arithmetic — no CPU round-trip
  for the hot path.  Only the final boolean-index compaction (scatter a
  sparse set of rows from a GPU buffer) falls back to NumPy, because MLX
  does not yet expose a first-class ``compress`` / boolean-index primitive.
  That single transfer is an index copy, not a computation.
* ``compute_normals`` computes edge vectors and cross products on MLX,
  then falls back to NumPy only for the scatter-add accumulation step
  (``np.add.at``), which has no direct MLX equivalent.

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
        pts_flat = points.reshape(-1, 3)  # (H*W, 3)

        # ── Quad decomposition ────────────────────────────────────────────
        # Build a flat index grid (H, W) — each cell holds its linear index.
        # For each quad (i,j)→(i+1,j+1) emit two counter-clockwise tris:
        #   tri_a: top-left    (i,j),   bottom-left (i+1,j),  top-right (i,j+1)
        #   tri_b: bottom-left (i+1,j), bottom-right(i+1,j+1),top-right (i,j+1)
        idx = mx.arange(H * W, dtype=mx.int32).reshape(H, W)

        tl = idx[:-1, :-1].reshape(-1)   # top-left     (N_quads,)
        bl = idx[1:,  :-1].reshape(-1)   # bottom-left
        tr = idx[:-1, 1:].reshape(-1)    # top-right
        br = idx[1:,  1:].reshape(-1)    # bottom-right

        tri_a = mx.stack([tl, bl, tr], axis=1)          # (N_quads, 3)
        tri_b = mx.stack([bl, br, tr], axis=1)
        faces = mx.concatenate([tri_a, tri_b], axis=0)  # (M, 3)

        # ── Filter 1: zero-depth check (ALL ON MLX) ───────────────────────
        # A vertex is invalid when Z == 0 (PointCloudGenerator convention).
        # Gather z-coordinate for each face corner via vectorised indexing.
        z_vals = pts_flat[:, 2]          # (H*W,)
        z0 = z_vals[faces[:, 0]]         # (M,)
        z1 = z_vals[faces[:, 1]]
        z2 = z_vals[faces[:, 2]]
        valid_z = (z0 > 0) & (z1 > 0) & (z2 > 0)

        # ── Filter 2: edge-length check (ALL ON MLX) ──────────────────────
        # Compute squared edge lengths; compare against max_edge_length².
        # Avoids a sqrt — squaring both sides preserves the inequality.
        p0 = pts_flat[faces[:, 0]]       # (M, 3)
        p1 = pts_flat[faces[:, 1]]
        p2 = pts_flat[faces[:, 2]]

        max_len_sq = self.max_edge_length ** 2
        e01_sq = mx.sum((p0 - p1) ** 2, axis=1)
        e12_sq = mx.sum((p1 - p2) ** 2, axis=1)
        e02_sq = mx.sum((p0 - p2) ** 2, axis=1)
        valid_edge = (e01_sq <= max_len_sq) & (e12_sq <= max_len_sq) & (e02_sq <= max_len_sq)

        # ── Combined mask ─────────────────────────────────────────────────
        keep = valid_z & valid_edge      # (M,) bool — computed entirely on MLX

        # Materialise mask + faces onto CPU for the compaction step.
        # MLX does not yet expose boolean-index gather (compress), so we
        # transfer the mask (cheap) and the face array (int32, small) then
        # do the row-selection on NumPy.  All expensive arithmetic stayed
        # on the GPU.
        mx.eval(keep, faces)
        keep_np   = np.array(keep,  copy=False)  # (M,) bool
        faces_np  = np.array(faces, copy=False)  # (M, 3) int32

        filtered_np = faces_np[keep_np]

        if filtered_np.shape[0] == 0:
            return pts_flat, mx.zeros((0, 3), dtype=mx.int32)

        return pts_flat, mx.array(filtered_np, dtype=mx.int32)

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

        The cross-product computation (edge vectors, cross product) is
        performed on MLX.  The scatter-add accumulation uses ``np.add.at``
        because MLX does not expose an unbuffered scatter-add primitive; the
        final normalisation is done in NumPy and returned as an MLX array.

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

        # ── Cross products on MLX ─────────────────────────────────────────
        # Gather vertex positions for each face corner.
        p0 = vertices[faces[:, 0]]   # (M, 3)
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]

        e1 = p1 - p0                 # (M, 3) — first  edge vector
        e2 = p2 - p0                 # (M, 3) — second edge vector

        # Cross product: face_normal = e1 × e2
        # magnitude = 2 * triangle_area  (area-weighted automatically)
        cx = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
        cy = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
        cz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        face_normals = mx.stack([cx, cy, cz], axis=1)  # (M, 3)

        # Materialise for scatter-add (no MLX equivalent of np.add.at)
        mx.eval(face_normals, faces)
        fn_np    = np.array(face_normals, copy=False).astype(np.float32)
        faces_np = np.array(faces,        copy=False)

        N = int(vertices.shape[0])
        normals_np = np.zeros((N, 3), dtype=np.float32)
        np.add.at(normals_np, faces_np[:, 0], fn_np)
        np.add.at(normals_np, faces_np[:, 1], fn_np)
        np.add.at(normals_np, faces_np[:, 2], fn_np)

        # L2 normalise — guard against zero vectors (isolated vertices)
        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normals_np = normals_np / norms

        return mx.array(normals_np)
