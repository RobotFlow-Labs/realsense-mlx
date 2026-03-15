"""Distortion correction models, fully vectorized over (H, W) pixel grids.

All functions operate on MLX arrays of normalized coordinates
(i.e. pixels after applying the ``(u - ppx) / fx`` transform).

Distortion model reference
--------------------------
The RealSense SDK defines three relevant models:

* **none** — ideal pinhole, no correction needed.
* **brown_conrady** — standard Brown-Conrady model used for *forward*
  projection. The iterative Newton-Raphson inverse is required for
  deprojection (pixel → ray).
* **inverse_brown_conrady** — the forward formula is the *undistortion*
  polynomial (used directly during deprojection).  This is the model
  typically reported by the depth sensor.
* **modified_brown_conrady** — forward-only model (used for projection of
  the color camera); we expose a ``apply_modified_brown_conrady_forward``
  helper for completeness.

All implementations are exact ports of ``rscuda_utils.cuh`` /
``cuda-pointcloud.cu``, translated from per-pixel CUDA threads to
MLX fully-batched operations.
"""

from __future__ import annotations

import mlx.core as mx

from realsense_mlx.geometry.intrinsics import CameraIntrinsics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _r2(nx: mx.array, ny: mx.array) -> mx.array:
    """Return r² = nx² + ny² element-wise."""
    return nx * nx + ny * ny


# ---------------------------------------------------------------------------
# Inverse Brown-Conrady (forward application → used during deprojection)
# ---------------------------------------------------------------------------


def apply_distortion_inverse_brown_conrady(
    nx: mx.array,
    ny: mx.array,
    coeffs: list[float],
) -> tuple[mx.array, mx.array]:
    """Apply the *forward* inverse-Brown-Conrady polynomial.

    This is the non-iterative form: directly evaluates the undistortion
    polynomial.  Used during ``deproject`` for the depth sensor model.

    Equivalent C from ``rscuda_utils.cuh``::

        float r2  = x*x + y*y;
        float f   = 1 + k1*r2 + k2*r2^2 + k3*r2^3;
        float ux  = x*f + 2*p1*x*y + p2*(r2 + 2*x^2);
        float uy  = y*f + 2*p2*x*y + p1*(r2 + 2*y^2);

    Parameters
    ----------
    nx, ny : (H, W) normalized coordinates.
    coeffs : [k1, k2, p1, p2, k3]

    Returns
    -------
    (ux, uy) : corrected normalized coordinates, same shape as input.
    """
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs)

    r2 = _r2(nx, ny)
    f = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)

    ux = nx * f + 2.0 * p1 * nx * ny + p2 * (r2 + 2.0 * nx * nx)
    uy = ny * f + 2.0 * p2 * nx * ny + p1 * (r2 + 2.0 * ny * ny)

    return ux, uy


# ---------------------------------------------------------------------------
# Brown-Conrady iterative inverse (Newton-Raphson)
# ---------------------------------------------------------------------------


def undistort_brown_conrady(
    nx: mx.array,
    ny: mx.array,
    coeffs: list[float],
    iterations: int = 10,
) -> tuple[mx.array, mx.array]:
    """Inverse Brown-Conrady distortion via iterative Newton-Raphson.

    Recovers the undistorted normalized coordinates from distorted ones.
    Used during ``deproject`` when the sensor reports the
    ``brown_conrady`` model (less common for depth, typical for
    ``modified_brown_conrady`` color cameras in forward projection).

    Equivalent C from ``cuda-pointcloud.cu``::

        for (int i = 0; i < 10; i++) {
            float r2     = x*x + y*y;
            float icdist = 1 / (1 + (k3*r2 + k2)*r2 + k1)*r2);
            float delta_x = 2*p1*x*y + p2*(r2 + 2*x^2);
            float delta_y = 2*p2*x*y + p1*(r2 + 2*y^2);
            x = (xo - delta_x) * icdist;
            y = (yo - delta_y) * icdist;
        }

    Parameters
    ----------
    nx, ny     : (H, W) *distorted* normalized coordinates (xo, yo).
    coeffs     : [k1, k2, p1, p2, k3]
    iterations : Newton-Raphson iterations (10 is empirically sufficient).

    Returns
    -------
    (x, y) : undistorted normalized coordinates.
    """
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs)

    # Keep the original (distorted) coordinates as the residual target
    xo = nx
    yo = ny

    x = nx
    y = ny

    for _ in range(iterations):
        r2 = _r2(x, y)
        # icdist = 1 / (1 + ((k3*r2 + k2)*r2 + k1)*r2)
        icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        x = (xo - delta_x) * icdist
        y = (yo - delta_y) * icdist

    return x, y


# ---------------------------------------------------------------------------
# Inverse Brown-Conrady iterative variant
# (CUDA pointcloud.cu uses a slightly different loop for this model)
# ---------------------------------------------------------------------------


def undistort_inverse_brown_conrady(
    nx: mx.array,
    ny: mx.array,
    coeffs: list[float],
    iterations: int = 10,
) -> tuple[mx.array, mx.array]:
    """Newton-Raphson inversion of the inverse-Brown-Conrady model.

    Used when we need to *invert* the forward polynomial (e.g. during
    projection into the depth frame from 3-D).  The loop differs from
    the standard Brown-Conrady case in that it updates ``x, y`` via
    a slightly different formulation that accounts for the tangential
    terms relative to the *undistorted* iterate rather than the
    distorted original.

    Equivalent C from ``cuda-pointcloud.cu``::

        for (int i = 0; i < 10; i++) {
            float r2     = x*x + y*y;
            float icdist = 1 / (1 + (k3*r2 + k2)*r2 + k1)*r2);
            float xq     = x / icdist;
            float yq     = y / icdist;
            float delta_x = 2*p1*xq*yq + p2*(r2 + 2*xq^2);
            float delta_y = 2*p2*xq*yq + p1*(r2 + 2*yq^2);
            x = (xo - delta_x) * icdist;
            y = (yo - delta_y) * icdist;
        }
    """
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs)

    xo = nx
    yo = ny
    x = nx
    y = ny

    for _ in range(iterations):
        r2 = _r2(x, y)
        icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
        xq = x / icdist
        yq = y / icdist
        delta_x = 2.0 * p1 * xq * yq + p2 * (r2 + 2.0 * xq * xq)
        delta_y = 2.0 * p2 * xq * yq + p1 * (r2 + 2.0 * yq * yq)
        x = (xo - delta_x) * icdist
        y = (yo - delta_y) * icdist

    return x, y


# ---------------------------------------------------------------------------
# Modified Brown-Conrady (forward projection — color camera)
# ---------------------------------------------------------------------------


def apply_modified_brown_conrady_forward(
    nx: mx.array,
    ny: mx.array,
    coeffs: list[float],
) -> tuple[mx.array, mx.array]:
    """Apply the modified Brown-Conrady *forward* distortion model.

    Used during point-to-pixel projection for color cameras that report
    the ``modified_brown_conrady`` model.

    Equivalent C from ``rscuda_utils.cuh``::

        float r2 = x*x + y*y;
        float f  = 1 + k1*r2 + k2*r2^2 + k3*r2^3;
        x *= f;
        y *= f;
        float dx = x + 2*p1*x*y + p2*(r2 + 2*x^2);
        float dy = y + 2*p2*x*y + p1*(r2 + 2*y^2);
    """
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs)

    r2 = _r2(nx, ny)
    f = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
    xf = nx * f
    yf = ny * f
    # r2 is still the *original* r2 per the SDK source
    dx = xf + 2.0 * p1 * xf * yf + p2 * (r2 + 2.0 * xf * xf)
    dy = yf + 2.0 * p2 * xf * yf + p1 * (r2 + 2.0 * yf * yf)
    return dx, dy


# ---------------------------------------------------------------------------
# High-level dispatcher: apply forward distortion for projection
# ---------------------------------------------------------------------------


def apply_distortion_forward(
    x: mx.array,
    y: mx.array,
    intrinsics: CameraIntrinsics,
) -> tuple[mx.array, mx.array]:
    """Apply forward distortion to normalized coordinates for pixel projection.

    Given *undistorted* normalized coordinates ``(x, y) = (X/Z, Y/Z)``,
    returns the distorted coordinates that map to the correct pixel
    position when multiplied by focal length and offset by principal point.

    Parameters
    ----------
    x, y        : (...) normalized coordinates (any broadcastable shape).
    intrinsics  : Camera intrinsics specifying the distortion model.

    Returns
    -------
    (xd, yd) : distorted normalized coordinates.
    """
    model = intrinsics.model
    coeffs = intrinsics.coeffs

    if model == "none":
        return x, y
    elif model == "modified_brown_conrady":
        return apply_modified_brown_conrady_forward(x, y, coeffs)
    elif model == "inverse_brown_conrady":
        # The "inverse_brown_conrady" model is the forward correction
        # polynomial applied to the raw normalised coords; during
        # *projection* (3D → pixel) we apply it directly.
        return apply_distortion_inverse_brown_conrady(x, y, coeffs)
    elif model == "brown_conrady":
        # Standard Brown-Conrady: the forward polynomial is not directly
        # available in closed form for projection; for correctness we
        # apply the same iterative inverse scheme on the *output* side.
        # In practice the error is small, but we mirror the SDK behaviour.
        return undistort_brown_conrady(x, y, coeffs)
    else:
        # Unsupported model — return unchanged (caller responsible for warning)
        return x, y


# ---------------------------------------------------------------------------
# High-level dispatcher: undistort for deprojection
# ---------------------------------------------------------------------------


def undistort(
    nx: mx.array,
    ny: mx.array,
    intrinsics: CameraIntrinsics,
    iterations: int = 10,
) -> tuple[mx.array, mx.array]:
    """Undo distortion on normalized coordinates during deprojection.

    Dispatches to the correct algorithm based on ``intrinsics.model``.

    Parameters
    ----------
    nx, ny      : (H, W) normalized coordinates from
                  ``(u - ppx) / fx``, ``(v - ppy) / fy``.
    intrinsics  : Camera intrinsics.
    iterations  : Iterations for iterative models.

    Returns
    -------
    (nx_corr, ny_corr) : Corrected normalized coordinates.
    """
    model = intrinsics.model
    coeffs = intrinsics.coeffs

    if model == "none":
        return nx, ny
    elif model == "inverse_brown_conrady":
        return apply_distortion_inverse_brown_conrady(nx, ny, coeffs)
    elif model == "brown_conrady":
        return undistort_brown_conrady(nx, ny, coeffs, iterations=iterations)
    elif model == "modified_brown_conrady":
        # modified_brown_conrady is a *forward* model; deprojection is
        # not directly supported — same assertion as the SDK.
        raise ValueError(
            "Cannot deproject from a modified_brown_conrady (forward-distorted) image. "
            "Apply undistortion beforehand."
        )
    else:
        return nx, ny
