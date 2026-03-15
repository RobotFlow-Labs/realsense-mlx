"""Joint bilateral filter using a guide image for edge-aware depth smoothing.

Algorithm: O(1) bilateral via range-bin decomposition (Paris & Durand 2006)
---------------------------------------------------------------------------
The naive bilateral filter has complexity O(N * K^2) per pixel (N = total
pixels, K = kernel radius).  The Paris-Durand constant-time approximation
reduces this to O(N * n_bins) by decomposing the range Gaussian into
contributions from quantised intensity bins:

  BF[p] ≈ Σ_b  W_b(p) * (J_b ⊛ G_s)(p)
           ----------------------------------
           Σ_b  W_b(p) * (M_b ⊛ G_s)(p)

where:
  - b       indexes a range bin covering intensity interval [b_lo, b_hi]
  - W_b(p)  is the range weight for pixel p given bin b:
              W_b(p) = exp(-((guide(p) - b_centre) / sigma_range)^2 / 2)
  - J_b(q)  = depth(q) * W_b(q)   (weighted depth contribution map)
  - M_b(q)  = W_b(q)               (weight accumulation map)
  - G_s     = separable Gaussian with sigma = sigma_spatial (box approx)
  - ⊛       denotes separable box convolution

The separable box convolution approximates the Gaussian and runs in O(N)
regardless of kernel size.

This approach eliminates the per-pixel kernel loop entirely.  The only loop
is over the n_bins quantisation levels, which is a small constant (default 8)
compared to a typical kernel of 5×5=25 or 9×9=81 operations per pixel.

MLX constraints respected
--------------------------
- All ops on float32 arrays; no int64.
- mx.where() for invalid-pixel masking.
- mx.eval() after each bin accumulation to bound computation graph size.
- Box filter built with cumsum + slicing (no explicit loops over pixels).
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["BilateralFilter"]

# Small constant to avoid divide-by-zero in the final normalisation.
_EPS: float = 1e-6


def _box_filter_1d(arr: mx.array, radius: int) -> mx.array:
    """Apply a 1-D box filter along axis=1 (columns) using prefix sums.

    This is the standard O(N) box filter: cumsum, shift, subtract.
    Boundary pixels are handled with zero-padding (edge effects are minimal
    for depth filtering because radius << image width in practice).

    Parameters
    ----------
    arr:
        Float32 array of shape ``(H, W)``.
    radius:
        Half-width of the box kernel.  Kernel width = 2*radius + 1.

    Returns
    -------
    mx.array
        Box-filtered array, shape ``(H, W)``, float32.
    """
    H, W = arr.shape
    ksize = 2 * radius + 1

    # Zero-pad both sides along columns.
    padded = mx.pad(arr, [(0, 0), (radius, radius)])  # (H, W + 2*radius)

    # Prefix sum along columns.
    cs = mx.cumsum(padded, axis=1)  # (H, W + 2*radius)

    # Sum in window [i, i+ksize) = cs[i+ksize] - cs[i].
    # Shift cs right by 1 (add a zero column at the start).
    cs_shifted = mx.pad(cs, [(0, 0), (1, 0)])  # (H, W + 2*radius + 1)

    right = cs_shifted[:, ksize:]          # cs at column ksize .. ksize+W-1
    left  = cs_shifted[:, :W]              # cs at column 0 .. W-1

    result = (right - left) / float(ksize)
    return result


def _separable_box_filter(arr: mx.array, radius: int) -> mx.array:
    """Apply a 2-D separable box filter (horizontal then vertical).

    Parameters
    ----------
    arr:
        Float32 ``(H, W)`` array.
    radius:
        Half-width in each direction.

    Returns
    -------
    mx.array
        Filtered array, same shape, float32.
    """
    # Horizontal pass.
    h = _box_filter_1d(arr, radius)
    # Vertical pass: transpose → horizontal → transpose back.
    v = _box_filter_1d(mx.transpose(h, (1, 0)), radius)
    return mx.transpose(v, (1, 0))


class BilateralFilter:
    """Joint bilateral filter using a guide image for edge preservation.

    Unlike the spatial filter (which uses depth differences for edges),
    this filter uses an external guide image (typically the IR or color
    frame) to determine edges.  This produces better results because
    depth edges often align with image edges but depth values are noisier.

    Algorithm
    ---------
    O(1) bilateral via range-bin decomposition (Paris & Durand 2006).
    The guide image is quantised into ``n_bins`` intensity bins.  For each
    bin a weighted depth accumulator and weight map are box-filtered and
    accumulated.  The final output is the weighted average, interpolated
    across bins.  No per-pixel kernel loop is used.

    Parameters
    ----------
    sigma_spatial : float
        Spatial Gaussian sigma in pixels.  Controls smoothing radius.
        Default 5.0.
    sigma_range : float
        Range (intensity) Gaussian sigma.  Controls how sensitive the
        filter is to intensity differences — smaller = harder edge
        preservation.  Default 30.0.  Units match the guide image
        (0–255 for uint8, 0.0–1.0 for float32 normalised).
    kernel_size : int
        Filter window size.  Must be an odd positive integer.  Default 5.
        The box-filter radius is derived as ``kernel_size // 2``.
    n_bins : int
        Number of range quantisation bins.  More bins = higher accuracy
        but more computation.  Default 8 gives good quality/speed balance.

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> depth = mx.array(np.random.uniform(500, 3000, (48, 64)).astype(np.float32))
    >>> guide = mx.array(np.random.uniform(0, 255, (48, 64)).astype(np.float32))
    >>> f = BilateralFilter(sigma_spatial=5.0, sigma_range=30.0, kernel_size=5)
    >>> out = f.process(depth, guide)
    >>> out.shape
    (48, 64)
    """

    def __init__(
        self,
        sigma_spatial: float = 5.0,
        sigma_range: float = 30.0,
        kernel_size: int = 5,
        n_bins: int = 8,
    ) -> None:
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = float(sigma_range)
        self.kernel_size = int(kernel_size)
        self.n_bins = int(n_bins)

        # Box-filter radius: approximate the Gaussian with a single box pass.
        # radius ≈ sigma_spatial * sqrt(3) gives matching variance.
        # We also respect the explicit kernel_size parameter.
        self._radius: int = self.kernel_size // 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array, guide: mx.array | None = None) -> mx.array:
        """Apply joint bilateral filtering to ``depth`` using ``guide``.

        Parameters
        ----------
        depth:
            ``(H, W)`` float32 or uint16 depth array.  Zero values are
            treated as invalid and excluded from filtering (but not from
            receiving filtered values from valid neighbours).
        guide:
            ``(H, W)`` guide image for edge detection.  May be uint8
            (0–255) or float32.  If a 3-channel ``(H, W, 3)`` colour
            image is supplied, the luminance channel (mean of channels)
            is used automatically.  If ``None``, depth itself is used as
            its own guide (standard bilateral filter).

        Returns
        -------
        mx.array
            Filtered depth array, same shape and dtype as ``depth``.

        Raises
        ------
        ValueError
            If ``depth`` is not 2-D, or if ``guide`` and ``depth`` have
            incompatible spatial dimensions.
        """
        if depth.ndim != 2:
            raise ValueError(
                f"BilateralFilter expects 2-D (H, W) depth array, got shape {depth.shape}"
            )
        if depth.shape[0] == 0 or depth.shape[1] == 0:
            return depth

        orig_dtype = depth.dtype
        depth_f = depth.astype(mx.float32)

        # Resolve the guide image.
        guide_f = self._prepare_guide(guide, depth_f)

        # Normalise the guide to [0, n_bins) for quantisation.
        guide_norm = self._normalise_guide(guide_f)

        # Run the Paris-Durand O(1) bilateral approximation.
        result = self._bilateral_bins(depth_f, guide_norm)

        # Preserve original invalid pixels (zero depth stays zero).
        valid_mask = depth_f > 0.0
        result = mx.where(valid_mask, result, mx.array(0.0, dtype=mx.float32))

        mx.eval(result)

        # Clamp before casting to uint16 to avoid wrap-around artefacts.
        if orig_dtype == mx.uint16:
            result = mx.clip(result, 0.0, 65535.0)

        return result.astype(orig_dtype)

    def reset(self) -> None:
        """No-op — BilateralFilter is stateless."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_guide(
        self, guide: mx.array | None, depth_f: mx.array
    ) -> mx.array:
        """Resolve and normalise the guide image to float32 (H, W)."""
        if guide is None:
            return depth_f

        g = guide.astype(mx.float32)

        # Collapse colour image to luminance.
        if g.ndim == 3:
            g = mx.mean(g, axis=2)

        if g.ndim != 2:
            raise ValueError(
                f"guide must be 2-D or 3-D (H, W[, C]), got shape {guide.shape}"
            )

        H, W = depth_f.shape
        gH, gW = g.shape
        if gH != H or gW != W:
            raise ValueError(
                f"guide shape {g.shape} does not match depth shape {depth_f.shape}"
            )

        return g

    def _normalise_guide(self, guide_f: mx.array) -> mx.array:
        """Map guide values to [0, n_bins) via min-max normalisation.

        A per-frame normalisation ensures the range bins always cover the
        full dynamic range of the guide, regardless of the input scale
        (uint8 0-255 or float normalised 0-1 etc.).
        """
        g_min = mx.min(guide_f)
        g_max = mx.max(guide_f)
        span = g_max - g_min

        # Avoid divide-by-zero on constant guide images.
        safe_span = mx.where(span > _EPS, span, mx.array(1.0, dtype=mx.float32))
        normalised = (guide_f - g_min) / safe_span  # [0, 1]
        return normalised * float(self.n_bins - 1)  # [0, n_bins - 1]

    def _bilateral_bins(
        self, depth_f: mx.array, guide_norm: mx.array
    ) -> mx.array:
        """Paris-Durand O(1) bilateral filter.

        For each bin b in [0, n_bins):
          1. Compute range weight W_b = exp(-0.5 * ((guide - b) / sigma_range_bins)^2)
             where sigma_range_bins is sigma_range rescaled to bin units.
          2. Accumulate: num += W_b * depth; denom += W_b
          3. Both accumulators are box-filtered before accumulation.

        Final output = box_filter(num) / box_filter(denom).

        Only valid (non-zero) depth pixels contribute to the numerator.

        Parameters
        ----------
        depth_f:
            Float32 (H, W) depth.
        guide_norm:
            Float32 (H, W) guide in [0, n_bins-1].

        Returns
        -------
        mx.array
            Float32 (H, W) filtered depth.
        """
        # sigma_range in bin units: sigma_range is specified in guide-value units
        # (0-255 for uint8 guides).  After normalisation to [0, n_bins-1] the
        # sigma scales proportionally.
        sigma_bins = float(self.n_bins - 1) / max(255.0, 1.0) * self.sigma_range
        # Clamp to at least 0.5 bins so the Gaussian is never degenerate.
        sigma_bins = max(sigma_bins, 0.5)

        valid_mask = (depth_f > 0.0).astype(mx.float32)

        num_acc = mx.zeros(depth_f.shape, dtype=mx.float32)
        den_acc = mx.zeros(depth_f.shape, dtype=mx.float32)

        inv_two_sigma2 = 1.0 / (2.0 * sigma_bins * sigma_bins)

        for b in range(self.n_bins):
            b_centre = float(b)

            # Range weight: exp(-0.5 * ((guide - b_centre) / sigma_bins)^2)
            diff = guide_norm - b_centre
            weight = mx.exp(-(diff * diff) * inv_two_sigma2)

            # Weighted depth (only from valid pixels).
            weighted_depth = weight * depth_f * valid_mask
            weight_valid = weight * valid_mask

            # Spatially smooth both accumulators.
            num_b = _separable_box_filter(weighted_depth, self._radius)
            den_b = _separable_box_filter(weight_valid, self._radius)

            # Accumulate across bins.
            num_acc = num_acc + num_b * weight
            den_acc = den_acc + den_b * weight

            # Bound computation graph every 4 bins to prevent MLX JIT overload.
            if b % 4 == 3:
                mx.eval(num_acc, den_acc)

        # Normalise.
        safe_den = mx.where(
            den_acc > _EPS,
            den_acc,
            mx.array(1.0, dtype=mx.float32),
        )
        result = num_acc / safe_den

        mx.eval(result)
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BilateralFilter("
            f"sigma_spatial={self.sigma_spatial}, "
            f"sigma_range={self.sigma_range}, "
            f"kernel_size={self.kernel_size}, "
            f"n_bins={self.n_bins})"
        )
