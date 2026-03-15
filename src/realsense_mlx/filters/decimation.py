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
        """Compute per-tile median (scale 2–3).

        ``mx.median`` operates on float32; we cast, compute, and let the
        caller re-cast to the original dtype.
        """
        tiles_f = tiles.astype(mx.float32)
        medians = mx.median(tiles_f, axis=1)   # (Ho*Wo,)
        return medians.reshape(Ho, Wo)

    def _valid_mean_reduce(
        self, tiles: mx.array, Ho: int, Wo: int, orig_dtype: mx.Dtype
    ) -> mx.array:
        """Compute mean of valid (non-zero) pixels per tile (scale 4–8).

        Invalid pixels (zero) are excluded from both the sum and the count.
        Tiles with no valid pixels yield 0 in the output.
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
        # Round to nearest for integer types.
        means = means + 0.5
        return means.reshape(Ho, Wo)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DecimationFilter(scale={self.scale})"
