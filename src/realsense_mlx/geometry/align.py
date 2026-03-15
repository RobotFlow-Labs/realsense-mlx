"""Depth-colour alignment — port of ``cuda-align.cu``.

Two operations are provided:

a) **align_color_to_depth** — Maps colour pixels into the depth frame.
   For every depth pixel: deproject → transform → project → gather from
   colour frame.  Output has depth resolution with colour values.

b) **align_depth_to_color** — Maps depth pixels into the colour frame.
   For every depth pixel: deproject → transform → project → scatter
   depth values onto a colour-sized grid.  Where multiple depth pixels
   map to the same colour pixel, the minimum (nearest) depth wins.

Scatter-min implementation note
--------------------------------
MLX (as of 0.31) does not expose an atomic scatter-min kernel.  The
scatter-min step uses ``np.minimum.at`` on the CPU, which is equivalent
to ``atomicMin`` in CUDA.

Note: a sort-based MLX approach was considered but rejected because it
still falls back to NumPy for the within-segment reduction (MLX lacks
scan-min), making it a sort on Metal followed by the same ``np.minimum.at``
call — slower overall due to the extra Metal kernel launch overhead and
host round-trip, with no algorithmic benefit over the direct NumPy path.

Coordinate convention
---------------------
All intrinsic projections follow the RealSense camera frame::

    pixel_x = X/Z * fx + ppx
    pixel_y = Y/Z * fy + ppy
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.distortion import apply_distortion_forward, undistort
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics

__all__ = ["Aligner"]


# ---------------------------------------------------------------------------
# Scatter-min backends
# ---------------------------------------------------------------------------

def _scatter_min_numpy(
    flat_idx: np.ndarray,
    depth_v: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Scatter-min using NumPy's unbuffered in-place minimum.

    Implements the same semantics as ``atomicMin`` in CUDA: for each
    valid depth pixel projected to a colour-frame flat index, the output
    carries the minimum (nearest) depth value when multiple source pixels
    map to the same output pixel.

    MLX (as of 0.31) lacks a native scatter-min operation.  A sort-based
    MLX approach was explored but it still requires a NumPy round-trip for
    the within-segment reduction, adding Metal kernel launch overhead with
    no net benefit.  The pure NumPy path is simpler, honest, and fast
    enough for real-time use on Apple Silicon unified memory.

    Parameters
    ----------
    flat_idx  : (M,) int64 — flattened colour-frame pixel indices (valid only).
    depth_v   : (M,) uint16 — corresponding raw depth values.
    output_size : H_c * W_c — total number of output pixels.

    Returns
    -------
    (output_size,) uint16 — scattered depth; unfilled slots are 0.
    """
    aligned = np.full(output_size, fill_value=np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(aligned, flat_idx, depth_v)
    aligned[aligned == np.iinfo(np.uint16).max] = 0
    return aligned


# ---------------------------------------------------------------------------
# Internal geometry helpers (fully vectorised)
# ---------------------------------------------------------------------------


def _deproject(
    depth: mx.array,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Deproject (H_d, W_d) depth → (H_d, W_d) X, Y, Z in metres.

    Returns three (H, W) float32 arrays.
    """
    H, W = depth.shape
    i = intrinsics

    # Pixel coordinate grids
    u = mx.arange(W, dtype=mx.float32)[None, :]  # (1, W) → broadcast to (H, W)
    v = mx.arange(H, dtype=mx.float32)[:, None]  # (H, 1) → broadcast to (H, W)

    # Normalised pinhole coords
    nx = (u - i.ppx) / i.fx  # (H, W)
    ny = (v - i.ppy) / i.fy  # (H, W)

    # Apply distortion correction
    nx, ny = undistort(nx, ny, i)

    z = depth.astype(mx.float32) * depth_scale  # (H, W)
    X = nx * z
    Y = ny * z

    return X, Y, z


def _transform(
    X: mx.array,
    Y: mx.array,
    Z: mx.array,
    extrinsics: CameraExtrinsics,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apply rigid transform to (H, W) XYZ arrays.

    Implements ``p_other = R @ p_depth + t`` vectorised over all pixels.

    The rotation matrix is stored row-major in CameraExtrinsics, so::

        Xo = R[0,0]*X + R[0,1]*Y + R[0,2]*Z + t[0]
        Yo = R[1,0]*X + R[1,1]*Y + R[1,2]*Z + t[1]
        Zo = R[2,0]*X + R[2,1]*Y + R[2,2]*Z + t[2]

    This matches the CUDA ``rs2_transform_point_to_point`` which stores
    ``rotation`` column-major::

        to[0] = R[0]*f[0] + R[3]*f[1] + R[6]*f[2] + t[0]

    where R[0], R[3], R[6] are the first column → R^T row 0 (transposed).
    After :meth:`CameraExtrinsics.from_rs2` we have the standard row-major
    matrix, so element-wise multiplication is straightforward.
    """
    if extrinsics.is_identity:
        return X, Y, Z

    R = extrinsics.rotation.astype(np.float32)  # (3, 3) numpy
    t = extrinsics.translation.astype(np.float32)  # (3,) numpy

    # Convert to MLX scalars for broadcast arithmetic
    Xo = (float(R[0, 0]) * X + float(R[0, 1]) * Y + float(R[0, 2]) * Z
          + float(t[0]))
    Yo = (float(R[1, 0]) * X + float(R[1, 1]) * Y + float(R[1, 2]) * Z
          + float(t[1]))
    Zo = (float(R[2, 0]) * X + float(R[2, 1]) * Y + float(R[2, 2]) * Z
          + float(t[2]))

    return Xo, Yo, Zo


def _project(
    X: mx.array,
    Y: mx.array,
    Z: mx.array,
    intrinsics: CameraIntrinsics,
) -> tuple[mx.array, mx.array, mx.array]:
    """Project 3-D points into pixel coordinates.

    Returns (px, py, valid_mask) where valid_mask is a boolean (H, W)
    array indicating pixels with Z > 0 that project inside the frame.
    """
    i = intrinsics
    eps = 1e-6  # guard against division by zero for Z≈0

    # Safe division: where Z ≤ 0, normalised coords will be large
    # but we'll mask them out via valid_mask anyway.
    Zs = mx.maximum(Z, mx.array(eps, dtype=mx.float32))

    nx = X / Zs  # normalised (undistorted) coordinates
    ny = Y / Zs

    # Apply forward distortion (for projection into distorted image space)
    nx_d, ny_d = apply_distortion_forward(nx, ny, i)

    px = nx_d * i.fx + i.ppx  # (H, W) pixel x
    py = ny_d * i.fy + i.ppy  # (H, W) pixel y

    # Valid: Z positive and projected pixel inside the other frame
    valid = (
        (Z > 0.0)
        & (px >= 0.0) & (px < float(i.width))
        & (py >= 0.0) & (py < float(i.height))
    )

    return px, py, valid


# ---------------------------------------------------------------------------
# Bilinear gather helper
# ---------------------------------------------------------------------------


def _gather_nearest(
    image: mx.array,
    px: mx.array,
    py: mx.array,
    valid: mx.array,
) -> mx.array:
    """Nearest-neighbour gather from ``image`` at float pixel coords.

    Parameters
    ----------
    image : (H_src, W_src, C) source image.
    px    : (H_dst, W_dst) float32 x-coordinates into source.
    py    : (H_dst, W_dst) float32 y-coordinates into source.
    valid : (H_dst, W_dst) bool mask — invalid pixels write zero.

    Returns
    -------
    (H_dst, W_dst, C) gathered image, zeros where invalid.
    """
    H_src, W_src = image.shape[:2]

    # Round to nearest integer indices and clamp to valid range
    xi = mx.clip(mx.round(px).astype(mx.int32), 0, W_src - 1)  # (H_dst, W_dst)
    yi = mx.clip(mx.round(py).astype(mx.int32), 0, H_src - 1)

    # Flatten to 1-D index into (H_src * W_src)
    flat_idx = (yi * W_src + xi).astype(mx.int32)  # (H_dst, W_dst)

    H_dst, W_dst = px.shape
    flat_idx_1d = flat_idx.reshape(-1)  # (N,)

    # Reshape source image to (H_src*W_src, C) for 1-D gather
    C = image.shape[2] if image.ndim == 3 else 1
    image_flat = image.reshape(-1, C) if image.ndim == 3 else image.reshape(-1, 1)

    # Gather
    gathered = image_flat[flat_idx_1d]  # (N, C)
    gathered = gathered.reshape(H_dst, W_dst, C)

    # Zero-out invalid pixels
    # Expand valid mask for broadcasting over channels
    valid_mask = valid[:, :, None]  # (H_dst, W_dst, 1)
    zero = mx.zeros_like(gathered)
    result = mx.where(valid_mask, gathered, zero)

    if image.ndim == 2:
        result = result.squeeze(-1)

    return result


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------


class Aligner:
    """Aligns depth and colour frames via 3-D reprojection.

    Parameters
    ----------
    depth_intrinsics           : Intrinsics of the depth sensor.
    color_intrinsics           : Intrinsics of the colour sensor.
    depth_to_color_extrinsics  : Rigid transform from depth → colour frame.
    depth_scale                : Metres per raw uint16 count.

    Examples
    --------
    >>> from realsense_mlx.geometry.intrinsics import (
    ...     CameraIntrinsics, CameraExtrinsics)
    >>> d_intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
    >>> c_intr = CameraIntrinsics(1280, 720, 640.0, 360.0, 900.0, 900.0)
    >>> ext = CameraExtrinsics.identity()
    >>> aligner = Aligner(d_intr, c_intr, ext, depth_scale=0.001)
    """

    def __init__(
        self,
        depth_intrinsics: CameraIntrinsics,
        color_intrinsics: CameraIntrinsics,
        depth_to_color_extrinsics: CameraExtrinsics,
        depth_scale: float,
    ) -> None:
        if depth_scale <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale}")
        self._d_intr = depth_intrinsics
        self._c_intr = color_intrinsics
        self._ext = depth_to_color_extrinsics
        self._depth_scale = float(depth_scale)

    # ------------------------------------------------------------------
    # align_color_to_depth
    # ------------------------------------------------------------------

    def align_color_to_depth(
        self,
        depth: mx.array,
        color: mx.array,
    ) -> mx.array:
        """Map colour pixels into the depth frame.

        For every depth pixel the algorithm:

        1. Deprojects using depth intrinsics → (X, Y, Z) in depth camera space.
        2. Transforms → (X', Y', Z') in colour camera space.
        3. Projects using colour intrinsics → (px, py) colour pixel.
        4. Nearest-neighbour gathers from the colour frame.

        Parameters
        ----------
        depth : (H_d, W_d) uint16 raw depth frame.
        color : (H_c, W_c, C) uint8 or float32 colour frame.

        Returns
        -------
        mx.array
            (H_d, W_d, C) colour values aligned to the depth frame.
            Pixels that project outside the colour frame are zero.

        Raises
        ------
        ValueError
            If frame shapes do not match intrinsics.
        """
        self._check_depth_shape(depth)
        self._check_color_shape(color)

        # Step 1: deproject depth → 3-D in depth camera frame
        X, Y, Z = _deproject(depth, self._d_intr, self._depth_scale)

        # Step 2: rigid transform to colour camera frame
        Xc, Yc, Zc = _transform(X, Y, Z, self._ext)

        # Step 3: project into colour image
        px, py, valid = _project(Xc, Yc, Zc, self._c_intr)

        # Step 4: gather colour values
        aligned_color = _gather_nearest(color, px, py, valid)

        return aligned_color

    # ------------------------------------------------------------------
    # align_depth_to_color
    # ------------------------------------------------------------------

    def align_depth_to_color(
        self,
        depth: mx.array,
    ) -> mx.array:
        """Map depth values into the colour frame.

        For every depth pixel:

        1. Deproject using depth intrinsics.
        2. Transform to colour camera frame.
        3. Project to colour pixel coordinates.
        4. Scatter-min depth onto colour-sized grid (nearest depth wins
           when multiple depth pixels map to the same colour pixel).

        The scatter-min step uses ``np.minimum.at`` on the CPU as MLX
        does not expose atomic scatter operations.

        Parameters
        ----------
        depth : (H_d, W_d) uint16 raw depth frame.

        Returns
        -------
        mx.array
            (H_c, W_c) uint16 depth aligned to colour resolution.
            Pixels with no corresponding depth are zero.
        """
        self._check_depth_shape(depth)

        Hc = self._c_intr.height
        Wc = self._c_intr.width

        # Step 1: deproject
        X, Y, Z = _deproject(depth, self._d_intr, self._depth_scale)

        # Step 2: transform to colour frame
        Xc, Yc, Zc = _transform(X, Y, Z, self._ext)

        # Step 3: project into colour image
        px, py, valid = _project(Xc, Yc, Zc, self._c_intr)

        # Materialise projected coordinates and valid mask to NumPy.
        # depth is already an MLX array, evalulate once for efficient transfer.
        mx.eval(px, py, valid, depth)

        px_np = np.array(px, copy=False).reshape(-1)            # (N,) float32
        py_np = np.array(py, copy=False).reshape(-1)            # (N,) float32
        valid_np = np.array(valid, copy=False).reshape(-1)      # (N,) bool
        depth_np = np.array(depth, copy=False).reshape(-1).astype(np.uint16)  # (N,)

        # Filter to valid projections only
        sel = valid_np.astype(bool)
        if not sel.any():
            # No valid projections — return zero depth at colour resolution
            return mx.zeros((Hc, Wc), dtype=mx.uint16)

        px_v = np.clip(np.round(px_np[sel]).astype(np.int32), 0, Wc - 1)
        py_v = np.clip(np.round(py_np[sel]).astype(np.int32), 0, Hc - 1)
        depth_v = depth_np[sel]

        flat_idx = (py_v * Wc + px_v).astype(np.int64)  # (M,)

        # Scatter-min via NumPy (see module docstring for rationale).
        aligned = _scatter_min_numpy(flat_idx, depth_v, Hc * Wc)

        return mx.array(aligned.reshape(Hc, Wc))

    # ------------------------------------------------------------------
    # Alignment with subpixel accuracy (half-pixel shift variant)
    # ------------------------------------------------------------------

    def align_color_to_depth_subpixel(
        self,
        depth: mx.array,
        color: mx.array,
    ) -> mx.array:
        """Colour-to-depth alignment using the half-pixel shift technique.

        Mirrors the CUDA kernel ``kernel_transfer_pixels`` which tests two
        shifted versions of each depth pixel corner (±0.5 pixel) and fills
        a rectangle in the colour image — giving better coverage at depth
        edges.  Here we implement the same idea as two nearest-neighbour
        gather passes merged via OR of valid masks, taking the *first*
        valid sample.

        This is marginally slower than :meth:`align_color_to_depth` but
        reduces black fringe artefacts at object boundaries.
        """
        self._check_depth_shape(depth)
        self._check_color_shape(color)

        Hd, Wd = depth.shape

        u_base = mx.arange(Wd, dtype=mx.float32)[None, :]
        v_base = mx.arange(Hd, dtype=mx.float32)[:, None]

        results = []
        valids = []

        for shift in (-0.5, 0.5):
            u = u_base + shift
            v = v_base + shift

            nx = (u - self._d_intr.ppx) / self._d_intr.fx
            ny = (v - self._d_intr.ppy) / self._d_intr.fy
            nx, ny = undistort(nx, ny, self._d_intr)

            z = depth.astype(mx.float32) * self._depth_scale
            X = nx * z
            Y = ny * z

            Xc, Yc, Zc = _transform(X, Y, z, self._ext)
            px, py, valid = _project(Xc, Yc, Zc, self._c_intr)
            gathered = _gather_nearest(color, px, py, valid)
            results.append(gathered)
            valids.append(valid)

        # Prefer the negative-shift result; fill missing from positive-shift
        valid0 = valids[0][:, :, None]
        merged = mx.where(valid0, results[0], results[1])
        return merged

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _check_depth_shape(self, depth: mx.array) -> None:
        expected = (self._d_intr.height, self._d_intr.width)
        if depth.shape != expected:
            raise ValueError(
                f"depth shape {depth.shape} does not match depth intrinsics "
                f"{expected}"
            )

    def _check_color_shape(self, color: mx.array) -> None:
        Hc, Wc = self._c_intr.height, self._c_intr.width
        if color.shape[:2] != (Hc, Wc):
            raise ValueError(
                f"color shape {color.shape[:2]} does not match colour intrinsics "
                f"({Hc}, {Wc})"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def depth_intrinsics(self) -> CameraIntrinsics:
        return self._d_intr

    @property
    def color_intrinsics(self) -> CameraIntrinsics:
        return self._c_intr

    @property
    def extrinsics(self) -> CameraExtrinsics:
        return self._ext

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    def __repr__(self) -> str:
        return (
            f"Aligner("
            f"depth={self._d_intr.width}x{self._d_intr.height}, "
            f"color={self._c_intr.width}x{self._c_intr.height}, "
            f"identity_ext={self._ext.is_identity})"
        )
