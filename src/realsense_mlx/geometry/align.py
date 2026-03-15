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

Metal kernel for align_color_to_depth
--------------------------------------
When both intrinsic models are ``"none"`` (no lens distortion), the four
separate MLX operations — deproject, transform, project, gather — are fused
into a single Metal kernel launch: one thread per depth pixel reads the raw
depth value, walks the full pipeline in registers, and writes the gathered
colour directly to the output buffer.

Benefits over the MLX graph:
  * Eliminates 4 intermediate (H_d, W_d) float32 arrays (X, Y, Z, valid, px, py).
  * Reduces memory bandwidth: ~3× fewer round-trips through L2/HBM.
  * Single kernel dispatch vs ~6 MLX graph nodes.

The Metal path is the default (``use_metal=True``).  It activates only when
both depth and colour intrinsics use the ``"none"`` distortion model; any
other model falls back silently to the pure-MLX path which handles all models
via the distortion module.

Coordinate convention
---------------------
All intrinsic projections follow the RealSense camera frame::

    pixel_x = X/Z * fx + ppx
    pixel_y = Y/Z * fy + ppy
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.distortion import apply_distortion_forward, undistort
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics

__all__ = ["Aligner"]


# ---------------------------------------------------------------------------
# Metal kernel: fused deproject → transform → project → gather (pinhole only)
# ---------------------------------------------------------------------------
#
# One Metal thread per depth pixel (tid = flat index into H_d * W_d).
#
# Inputs (all flat / 1-D for simplicity with mx.fast.metal_kernel):
#   depth_u16  : (Hd * Wd,) uint16  — raw depth frame, row-major
#   color_u8   : (Hc * Wc * C,) uint8 — source colour frame, row-major
#   depth_intr : (4,) float32        — [ppx_d, ppy_d, fx_d, fy_d]
#   color_intr : (4,) float32        — [ppx_c, ppy_c, fx_c, fy_c]
#   rotation   : (9,) float32        — row-major rotation matrix R
#   translation: (3,) float32        — translation vector T
#   dims_buf   : (5,) int32          — [Hd, Wd, Hc, Wc, C]
#   scale_buf  : (1,) float32        — depth scale (metres per count)
#
# Output:
#   out_u8     : (Hd * Wd * C,) uint8 — aligned colour, zeros where invalid
#
# Algorithm per thread:
#   1. Unpack pixel (y, x) from tid.
#   2. Read depth; if 0, write zero channels and return (invalid).
#   3. Deproject (pinhole, no distortion):
#        z  = depth * scale
#        nx = (x - ppx_d) / fx_d
#        ny = (y - ppy_d) / fy_d
#        X, Y, Z = nx*z, ny*z, z
#   4. Apply rigid transform: Xc = R*[X,Y,Z]^T + T
#   5. Project to colour frame (pinhole):
#        u = Xc/Zc * fx_c + ppx_c
#        v = Yc/Zc * fy_c + ppy_c
#   6. Bounds check: 0 <= u < Wc and 0 <= v < Hc and Zc > 0
#   7. Nearest-neighbour gather: round(u), round(v)
#   8. Copy C bytes from colour to output; or zero if invalid.
# ---------------------------------------------------------------------------

_ALIGN_COLOR_TO_DEPTH_METAL_SOURCE = r"""
    uint tid = thread_position_in_grid.x;

    int Hd = dims_buf[0];
    int Wd = dims_buf[1];
    int Hc = dims_buf[2];
    int Wc = dims_buf[3];
    int C  = dims_buf[4];

    if ((int)tid >= Hd * Wd) return;

    // --- Step 1: unpack pixel coordinates ---
    int y = (int)tid / Wd;
    int x = (int)tid % Wd;

    // --- Step 2: read depth; skip invalid (zero) pixels ---
    float depth_raw = (float)depth_u16[tid];
    if (depth_raw == 0.0f) {
        int out_base = (int)tid * C;
        for (int c = 0; c < C; c++) out_u8[out_base + c] = 0;
        return;
    }

    // --- Step 3: deproject (pinhole, no distortion) ---
    float ppx_d = depth_intr[0];
    float ppy_d = depth_intr[1];
    float fx_d  = depth_intr[2];
    float fy_d  = depth_intr[3];

    float scale = scale_buf[0];
    float Z = depth_raw * scale;
    float X = ((float)x - ppx_d) / fx_d * Z;
    float Y = ((float)y - ppy_d) / fy_d * Z;

    // --- Step 4: rigid transform  Xc = R*[X,Y,Z]^T + T ---
    // rotation is row-major: rotation[row*3 + col]
    float Xc = rotation[0]*X + rotation[1]*Y + rotation[2]*Z + translation[0];
    float Yc = rotation[3]*X + rotation[4]*Y + rotation[5]*Z + translation[1];
    float Zc = rotation[6]*X + rotation[7]*Y + rotation[8]*Z + translation[2];

    // --- Step 5: project to colour frame (pinhole) ---
    float ppx_c = color_intr[0];
    float ppy_c = color_intr[1];
    float fx_c  = color_intr[2];
    float fy_c  = color_intr[3];

    int out_base = (int)tid * C;

    // --- Step 6: validity check ---
    if (Zc <= 0.0f) {
        for (int c = 0; c < C; c++) out_u8[out_base + c] = 0;
        return;
    }

    float u = Xc / Zc * fx_c + ppx_c;
    float v = Yc / Zc * fy_c + ppy_c;

    if (u < 0.0f || u >= (float)Wc || v < 0.0f || v >= (float)Hc) {
        for (int c = 0; c < C; c++) out_u8[out_base + c] = 0;
        return;
    }

    // --- Step 7: nearest-neighbour gather ---
    // Use metal::rint() (round-to-nearest-even / banker's rounding) to match
    // MLX's mx.round() semantics exactly.  Plain (int)(u + 0.5f) would use
    // round-half-up and diverge from MLX for exactly half-integer projections
    // (e.g. u=12.5 → rint gives 12, (int)(13.0) gives 13).
    // Clamp to [0, dim-1] as a safety guard for floating-point edge cases.
    int ui = (int)metal::rint(u);
    int vi = (int)metal::rint(v);
    if (ui < 0) ui = 0;
    if (ui >= Wc) ui = Wc - 1;
    if (vi < 0) vi = 0;
    if (vi >= Hc) vi = Hc - 1;

    int src_base = (vi * Wc + ui) * C;

    // --- Step 8: write output channels ---
    for (int c = 0; c < C; c++) out_u8[out_base + c] = color_u8[src_base + c];
"""


@functools.lru_cache(maxsize=1)
def _get_align_color_to_depth_kernel() -> object:
    """JIT-compile and cache the fused align_color_to_depth Metal kernel.

    Compiled once per process; subsequent calls return the cached instance.
    Thread-safe via ``functools.lru_cache``.
    """
    return mx.fast.metal_kernel(
        name="align_color_to_depth",
        input_names=[
            "depth_u16",
            "color_u8",
            "depth_intr",
            "color_intr",
            "rotation",
            "translation",
            "dims_buf",
            "scale_buf",
        ],
        output_names=["out_u8"],
        source=_ALIGN_COLOR_TO_DEPTH_METAL_SOURCE,
    )


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
    use_metal                  : When ``True`` (default), use the fused Metal
                                 GPU kernel for ``align_color_to_depth`` if
                                 both intrinsics use the ``"none"`` distortion
                                 model.  Falls back to the pure-MLX path
                                 automatically when the condition is not met
                                 (distorted intrinsics, non-Metal platform).
                                 Set to ``False`` to always use the MLX path.

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
        use_metal: bool = True,
    ) -> None:
        if depth_scale <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale}")
        self._d_intr = depth_intrinsics
        self._c_intr = color_intrinsics
        self._ext = depth_to_color_extrinsics
        self._depth_scale = float(depth_scale)
        self.use_metal = bool(use_metal)

        # Pre-build Metal-compatible parameter buffers from intrinsics/extrinsics.
        # These are constant for the lifetime of the Aligner so we cache them
        # to avoid per-frame allocation overhead.
        self._metal_depth_intr: mx.array | None = None
        self._metal_color_intr: mx.array | None = None
        self._metal_rotation: mx.array | None = None
        self._metal_translation: mx.array | None = None
        self._metal_dims: mx.array | None = None
        self._metal_scale_buf: mx.array | None = None
        if self.use_metal:
            self._build_metal_buffers()

    # ------------------------------------------------------------------
    # Metal parameter buffer construction
    # ------------------------------------------------------------------

    def _can_use_metal(self) -> bool:
        """Return True when the Metal kernel path is usable.

        The fused Metal kernel only implements the pinhole (no-distortion)
        projection model.  Any distorted intrinsics fall back to the
        vectorised MLX path which covers all models via the distortion module.
        """
        return (
            self.use_metal
            and self._d_intr.model == "none"
            and self._c_intr.model == "none"
        )

    def _build_metal_buffers(self) -> None:
        """Pre-allocate constant MLX arrays for the Metal kernel inputs.

        These are built once in ``__init__`` and reused across every frame
        to avoid per-frame Python-side allocation and MLX array construction.
        """
        d = self._d_intr
        c = self._c_intr
        R = self._ext.rotation.astype(np.float32)    # (3,3)
        t = self._ext.translation.astype(np.float32) # (3,)

        self._metal_depth_intr = mx.array(
            [d.ppx, d.ppy, d.fx, d.fy], dtype=mx.float32
        )
        self._metal_color_intr = mx.array(
            [c.ppx, c.ppy, c.fx, c.fy], dtype=mx.float32
        )
        # Row-major rotation, flattened to (9,)
        self._metal_rotation = mx.array(R.flatten(), dtype=mx.float32)
        self._metal_translation = mx.array(t, dtype=mx.float32)
        self._metal_dims = mx.array(
            [d.height, d.width, c.height, c.width, 0],  # C filled at call time
            dtype=mx.int32,
        )
        self._metal_scale_buf = mx.array([self._depth_scale], dtype=mx.float32)

    # ------------------------------------------------------------------
    # align_color_to_depth — public dispatch
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

        When ``use_metal=True`` and both intrinsics use the ``"none"``
        distortion model, the four operations are fused into a single Metal
        kernel launch (see module docstring for details).  Otherwise the
        pure-MLX pipeline is used.

        Parameters
        ----------
        depth : (H_d, W_d) uint16 raw depth frame.
        color : (H_c, W_c, C) uint8 colour frame.

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

        if self._can_use_metal():
            return self._align_color_to_depth_metal(depth, color)
        return self._align_color_to_depth_mlx(depth, color)

    # ------------------------------------------------------------------
    # Metal implementation
    # ------------------------------------------------------------------

    def _align_color_to_depth_metal(
        self,
        depth: mx.array,
        color: mx.array,
    ) -> mx.array:
        """Fused deproject→transform→project→gather Metal kernel.

        One Metal thread per depth pixel.  Requires both intrinsics to use
        the ``"none"`` distortion model (pure pinhole).

        Parameters
        ----------
        depth : (H_d, W_d) uint16
        color : (H_c, W_c, C) uint8

        Returns
        -------
        (H_d, W_d, C) uint8
        """
        Hd, Wd = depth.shape
        Hc, Wc, C = color.shape[0], color.shape[1], color.shape[2] if color.ndim == 3 else 1

        # Ensure correct dtypes for kernel inputs
        depth_u16 = depth.astype(mx.uint16).flatten()
        color_u8 = color.astype(mx.uint8).flatten()

        # Build dims buffer with the actual channel count C
        # We rebuild only the dims array (tiny) — the rest are cached constants.
        dims_buf = mx.array(
            [Hd, Wd, Hc, Wc, C], dtype=mx.int32
        )

        N = Hd * Wd
        kernel = _get_align_color_to_depth_kernel()

        outputs = kernel(
            inputs=[
                depth_u16,
                color_u8,
                self._metal_depth_intr,
                self._metal_color_intr,
                self._metal_rotation,
                self._metal_translation,
                dims_buf,
                self._metal_scale_buf,
            ],
            output_shapes=[(N * C,)],
            output_dtypes=[mx.uint8],
            grid=(N, 1, 1),
            threadgroup=(min(N, 256), 1, 1),
        )

        result = outputs[0].reshape(Hd, Wd, C)
        if color.ndim == 2:
            result = result.squeeze(-1)
        return result

    # ------------------------------------------------------------------
    # Pure-MLX implementation (all distortion models)
    # ------------------------------------------------------------------

    def _align_color_to_depth_mlx(
        self,
        depth: mx.array,
        color: mx.array,
    ) -> mx.array:
        """Pure-MLX pipeline: deproject → transform → project → gather.

        Handles all distortion models via the distortion module.  Used when
        ``use_metal=False`` or when either intrinsics has a non-``"none"``
        distortion model.
        """
        # Step 1: deproject depth → 3-D in depth camera frame
        X, Y, Z = _deproject(depth, self._d_intr, self._depth_scale)

        # Step 2: rigid transform to colour camera frame
        Xc, Yc, Zc = _transform(X, Y, Z, self._ext)

        # Step 3: project into colour image
        px, py, valid = _project(Xc, Yc, Zc, self._c_intr)

        # Step 4: gather colour values
        return _gather_nearest(color, px, py, valid)

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
        metal_active = self._can_use_metal()
        return (
            f"Aligner("
            f"depth={self._d_intr.width}x{self._d_intr.height}, "
            f"color={self._c_intr.width}x{self._c_intr.height}, "
            f"identity_ext={self._ext.is_identity}, "
            f"use_metal={self.use_metal}, "
            f"metal_active={metal_active})"
        )
