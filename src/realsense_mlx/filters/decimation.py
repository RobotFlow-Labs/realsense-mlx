"""Decimation (spatial downsampling) filter for depth frames.

Reduces frame resolution by an integer ``scale`` factor.  The output frame
is ``(H // scale, W // scale)``.  Scale is clamped to [1, 8].

Algorithm
---------
- Scale 1    : no-op, frame returned unchanged.
- Scale 2–3  : median of each ``scale × scale`` tile.
- Scale 4–8  : mean of *valid* (non-zero) pixels in each tile.  If a tile
               has no valid pixels, the output is 0.

Tile-based approach with MLX
------------------------------
Reshape from ``(H, W)`` → ``(H//s, s, W//s, s)`` → transpose axes so
tiles are contiguous → reshape to ``(H//s * W//s, s*s)`` → reduce along
axis 1 with median or masked mean.

MLX constraints observed
------------------------
- No int64: all index math in int32 / float32.
- mx.median(arr, axis) for median.
- mx.where() for invalid-pixel masking.
- mx.eval(arr) to materialise.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["DecimationFilter"]


class DecimationFilter:
    """Downsample a depth frame by an integer scale factor.

    Parameters
    ----------
    scale:
        Downsampling factor in [1, 8].  Values outside this range are
        silently clamped.

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> depth = mx.array(np.random.randint(500, 2000, (480, 640), dtype=np.uint16))
    >>> f = DecimationFilter(scale=2)
    >>> out = f.process(depth)
    >>> out.shape
    (240, 320)
    """

    def __init__(self, scale: int = 2) -> None:
        self.scale: int = max(1, min(8, int(scale)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array) -> mx.array:
        """Decimate *depth* by ``self.scale``.

        Parameters
        ----------
        depth:
            ``(H, W)`` depth array (uint16 or float32).

        Returns
        -------
        mx.array
            ``(H // scale, W // scale)`` with the same dtype as input.
            The frame is *cropped* to a multiple of ``scale`` before
            processing (RS2 SDK behaviour).
        """
        # FIX 8: input validation — must be a 2-D array.
        if depth.ndim != 2:
            raise ValueError(
                f"DecimationFilter expects 2-D (H, W) array, got shape {depth.shape}"
            )

        if self.scale == 1:
            return depth

        s = self.scale
        H, W = depth.shape

        # Crop to multiples of scale.
        H_crop = (H // s) * s
        W_crop = (W // s) * s
        cropped = depth[:H_crop, :W_crop]

        Ho, Wo = H_crop // s, W_crop // s

        # Reshape into tiles: (Ho, s, Wo, s) → (Ho, Wo, s, s) → (Ho*Wo, s*s)
        tiles = cropped.reshape(Ho, s, Wo, s)
        tiles = mx.transpose(tiles, (0, 2, 1, 3))          # (Ho, Wo, s, s)
        tiles = tiles.reshape(Ho * Wo, s * s)               # (Ho*Wo, s²)

        if s <= 3:
            result = self._median_reduce(tiles, Ho, Wo)
        else:
            result = self._valid_mean_reduce(tiles, Ho, Wo, depth.dtype)

        mx.eval(result)
        return result.astype(depth.dtype)

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------

    def _median_reduce(self, tiles: mx.array, Ho: int, Wo: int) -> mx.array:
        """Compute per-tile median of *valid* (non-zero) pixels (scale 2–3).

        FIX 7: The naive ``mx.median`` over the raw tile (including zeros)
        produces the wrong result whenever a tile contains invalid pixels —
        the zeros drag the median toward 0.

        Strategy:
        - For tiles with no zeros (all valid), compute the standard median.
        - For tiles that contain at least one zero, compute the mean of the
          valid (non-zero) pixels instead.  A tile with *all* zeros yields 0.

        This matches the RS2 SDK behaviour of ignoring invalid pixels when
        computing the representative depth for a decimated pixel.
        """
        tiles_f = tiles.astype(mx.float32)          # (N, k)  N = Ho*Wo
        valid_mask = tiles_f > 0.0                  # (N, k)  bool-like float32

        # --- standard median of the full tile (correct when no zeros) ---
        medians = mx.median(tiles_f, axis=1)        # (N,)

        # --- valid-mean fallback for tiles that contain at least one zero ---
        valid_f = valid_mask.astype(mx.float32)
        counts  = mx.sum(valid_f, axis=1)           # (N,)
        sums    = mx.sum(tiles_f * valid_f, axis=1) # (N,)
        safe_counts = mx.where(
            counts > 0.0, counts, mx.array(1.0, dtype=mx.float32)
        )
        valid_means = sums / safe_counts            # (N,)
        valid_means = mx.where(
            counts > 0.0, valid_means, mx.array(0.0, dtype=mx.float32)
        )

        # Determine which tiles have any zero pixels.
        k = tiles_f.shape[1]
        has_invalid = counts < float(k)             # (N,) — True when any pixel is zero

        # Select: median for fully-valid tiles, valid-mean for tiles with holes.
        result = mx.where(has_invalid, valid_means, medians)
        return result.reshape(Ho, Wo)

    # Integer dtypes that require rounding bias when converting from float mean.
    _INTEGER_DTYPES: frozenset = frozenset({mx.uint16, mx.uint8, mx.int32})

    def _valid_mean_reduce(
        self, tiles: mx.array, Ho: int, Wo: int, orig_dtype: mx.Dtype
    ) -> mx.array:
        """Compute mean of valid (non-zero) pixels per tile (scale 4–8).

        Invalid pixels (zero) are excluded from both the sum and the count.
        Tiles with no valid pixels yield 0 in the output.

        FIX 6: The +0.5 rounding bias is only applied when the original dtype
        is an integer type (uint16, uint8, int32).  For float32 inputs it
        introduces a systematic upward bias that corrupts disparity values.
        """
        tiles_f = tiles.astype(mx.float32)
        valid = tiles_f > 0.0                              # (Ho*Wo, s²) bool

        # Sum of valid values per tile.
        valid_f = valid.astype(mx.float32)
        sums   = mx.sum(tiles_f * valid_f, axis=1)         # (Ho*Wo,)
        counts = mx.sum(valid_f, axis=1)                   # (Ho*Wo,)

        # Avoid division by zero; if count==0 result stays 0.
        safe_counts = mx.where(counts > 0.0, counts, mx.array(1.0, dtype=mx.float32))
        means = sums / safe_counts
        means = mx.where(counts > 0.0, means, mx.array(0.0, dtype=mx.float32))
        # FIX 6: only add rounding bias for integer dtypes to get nearest-integer
        # rounding when the result is cast back (e.g. float→uint16).
        # For float32 inputs the +0.5 would introduce a systematic upward shift.
        if orig_dtype in self._INTEGER_DTYPES:
            means = means + 0.5
        return means.reshape(Ho, Wo)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DecimationFilter(scale={self.scale})"
