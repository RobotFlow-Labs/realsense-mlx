"""Depth frame statistics for quality assessment and filtering evaluation.

Design notes
------------
* All heavy arithmetic is done on NumPy arrays (converted from MLX via
  ``np.array(mx_array, copy=False)``) so that we can use ``np.median`` and
  similar functions not yet present in MLX core.
* The comparison metrics follow standard image-quality conventions:

  - RMSE  : root-mean-square error over *all* valid pixels
  - MAE   : mean absolute error over all valid pixels
  - PSNR  : 10 * log10(MAX² / MSE), MAX = maximum of the ``before`` frame
  - SSIM  : approximated by the luminance × structure terms only (no
            covariance window); acceptable for depth QA without SciPy.
  - ``holes_filled``    : pixels that were 0 in ``before`` but non-zero in ``after``
  - ``holes_created``   : pixels that were non-zero in ``before`` but 0 in ``after``
  - ``smoothness_improvement`` : ratio (std_before - std_after) / std_before,
    clamped to [-1, 1]; positive = smoother after filtering.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = ["DepthStats"]


def _to_float32(arr: mx.array | np.ndarray) -> np.ndarray:
    """Materialise an MLX or NumPy array as float32 NumPy."""
    if isinstance(arr, mx.array):
        mx.eval(arr)
        return np.array(arr, copy=False).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


class DepthStats:
    """Compute statistics on depth frames for quality assessment.

    All methods are ``@staticmethod`` — no instance state needed.

    Examples
    --------
    >>> import mlx.core as mx
    >>> depth = mx.full((480, 640), 1000, dtype=mx.uint16)
    >>> stats = DepthStats.compute(depth, depth_scale=0.001)
    >>> stats["valid_ratio"]
    1.0
    >>> stats["mean_m"]  # doctest: +ELLIPSIS
    1.0...
    """

    @staticmethod
    def compute(
        depth: mx.array | np.ndarray,
        depth_scale: float = 0.001,
    ) -> dict:
        """Compute descriptive statistics for a single depth frame.

        Parameters
        ----------
        depth       : (H, W) uint16 depth frame (raw sensor counts).
        depth_scale : Metres per raw count.  Default ``0.001`` (1 mm/count).

        Returns
        -------
        dict with keys:

        ``valid_ratio``
            Fraction of pixels with depth > 0.
        ``min_m``, ``max_m``, ``mean_m``, ``std_m``, ``median_m``
            Depth statistics in metres, computed over valid pixels only.
            ``None`` if no valid pixels exist.
        ``hole_count``
            Number of pixels with zero depth.
        ``hole_ratio``
            Fraction of pixels with zero depth.
        ``edge_pixel_count``
            Approximate count of depth-edge pixels detected by a simple
            Sobel-like horizontal + vertical difference threshold
            (threshold = 5× ``depth_scale`` in metres, i.e. ~5 mm).
        """
        raw = _to_float32(depth)   # float32, keeps zero-depth as 0.0
        H, W = raw.shape
        total = float(H * W)

        # Hole analysis
        is_valid = raw > 0.0
        hole_count = int(np.sum(~is_valid))
        valid_count = int(np.sum(is_valid))

        # Depth in metres — valid pixels only
        depth_m = raw * depth_scale
        valid_depths = depth_m[is_valid]

        if valid_count > 0:
            min_m = float(valid_depths.min())
            max_m = float(valid_depths.max())
            mean_m = float(valid_depths.mean())
            std_m = float(valid_depths.std())
            median_m = float(np.median(valid_depths))
        else:
            min_m = max_m = mean_m = std_m = median_m = None  # type: ignore[assignment]

        # Edge pixel count — simple finite-difference magnitude threshold
        # Use the full (H, W) depth_m array (zeros for invalid pixels).
        # Threshold: 5 × depth_scale to avoid counting quantisation noise.
        edge_thresh = 5.0 * depth_scale
        diff_h = np.abs(np.diff(depth_m, axis=1))   # (H, W-1)
        diff_v = np.abs(np.diff(depth_m, axis=0))   # (H-1, W)

        # Pad back to H×W so we can OR them
        edge_h = np.zeros_like(depth_m, dtype=bool)
        edge_v = np.zeros_like(depth_m, dtype=bool)
        edge_h[:, :-1] = diff_h > edge_thresh
        edge_v[:-1, :] = diff_v > edge_thresh
        edge_pixel_count = int(np.sum(edge_h | edge_v))

        return {
            "valid_ratio": valid_count / total if total > 0 else 0.0,
            "min_m": min_m,
            "max_m": max_m,
            "mean_m": mean_m,
            "std_m": std_m,
            "median_m": median_m,
            "hole_count": hole_count,
            "hole_ratio": hole_count / total if total > 0 else 0.0,
            "edge_pixel_count": edge_pixel_count,
        }

    @staticmethod
    def compare(
        before: mx.array | np.ndarray,
        after: mx.array | np.ndarray,
        depth_scale: float = 0.001,
    ) -> dict:
        """Compare two depth frames (e.g. before and after filtering).

        Both arrays must have the same shape ``(H, W)``.

        Parameters
        ----------
        before      : (H, W) depth frame before processing.
        after       : (H, W) depth frame after processing.
        depth_scale : Metres per raw count.

        Returns
        -------
        dict with keys:

        ``rmse``
            Root-mean-square error in metres over pixels that are valid
            in *both* frames.
        ``mae``
            Mean absolute error in metres over jointly-valid pixels.
        ``psnr``
            Peak signal-to-noise ratio in dB.  Computed as
            ``10 * log10(max_val² / mse)`` where ``max_val`` is the maximum
            depth in metres in ``before``.  ``inf`` when RMSE is 0.
        ``ssim_approx``
            Simplified SSIM approximation: mean_product / (mean_before × mean_after),
            clipped to [0, 1].  Returns ``1.0`` when inputs are identical.
        ``holes_filled``
            Pixel count: zero in ``before``, non-zero in ``after``.
        ``holes_created``
            Pixel count: non-zero in ``before``, zero in ``after``.
        ``smoothness_improvement``
            (std_before - std_after) / std_before over jointly-valid pixels.
            Positive = smoother after, negative = noisier after.
            Returns ``0.0`` when std_before is zero.
        """
        b_raw = _to_float32(before)
        a_raw = _to_float32(after)

        if b_raw.shape != a_raw.shape:
            raise ValueError(
                f"before shape {b_raw.shape} != after shape {a_raw.shape}"
            )

        b_m = b_raw * depth_scale
        a_m = a_raw * depth_scale

        b_valid = b_raw > 0.0
        a_valid = a_raw > 0.0
        both_valid = b_valid & a_valid

        # Hole changes
        holes_filled = int(np.sum(~b_valid & a_valid))
        holes_created = int(np.sum(b_valid & ~a_valid))

        # Error metrics — over jointly-valid pixels
        if both_valid.sum() > 0:
            b_vals = b_m[both_valid]
            a_vals = a_m[both_valid]
            diff = b_vals - a_vals
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
            rmse = float(math.sqrt(mse))

            max_val = float(b_m[b_valid].max()) if b_valid.any() else 1.0
            if mse > 0.0 and max_val > 0.0:
                psnr = float(10.0 * math.log10(max_val ** 2 / mse))
            else:
                psnr = float("inf")

            # Simplified SSIM approximation
            mean_b = float(b_vals.mean())
            mean_a = float(a_vals.mean())
            mean_prod = float(np.mean(b_vals * a_vals))
            denom = mean_b * mean_a
            if denom > 0.0:
                ssim_approx = float(np.clip(mean_prod / denom, 0.0, 1.0))
            else:
                ssim_approx = 1.0 if mse == 0.0 else 0.0

            # Smoothness
            std_b = float(b_vals.std())
            std_a = float(a_vals.std())
            if std_b > 0.0:
                smoothness_improvement = float(
                    np.clip((std_b - std_a) / std_b, -1.0, 1.0)
                )
            else:
                smoothness_improvement = 0.0
        else:
            rmse = mae = 0.0
            psnr = float("inf")
            ssim_approx = 1.0
            smoothness_improvement = 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "psnr": psnr,
            "ssim_approx": ssim_approx,
            "holes_filled": holes_filled,
            "holes_created": holes_created,
            "smoothness_improvement": smoothness_improvement,
        }
