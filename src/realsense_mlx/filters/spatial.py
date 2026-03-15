"""Spatial (edge-preserving) filter based on the Domain Transform Filter.

The filter performs ``iterations`` passes, each consisting of a horizontal
bilateral-style recursive pass (left→right then right→left) and a vertical
pass (implemented by transposing and re-applying the horizontal pass).

Algorithm (one horizontal pass, left→right)
--------------------------------------------
For each column i from 1 to W-1:

    diff      = |in[row, i] - in[row, i-1]|
    weight    = exp(-diff / (delta * alpha + eps))
    out[row, i] = in[row, i] + weight * (out[row, i-1] - in[row, i])

This is the Domain Transform recursive formulation:
- When weight → 0 (strong edge): out[i] = in[i]   (current pixel preserved)
- When weight → 1 (flat region): out[i] = out[i-1] (smoothed toward neighbour)

Previous-pixel validity check: update only fires when out[i-1] > 0.

Vectorisation strategy
-----------------------
All ROWS are processed in parallel for each column step — this gives us
full MLX vectorisation across the height dimension.  The column scan
itself is sequential (each step depends on the previous output), so we
loop over W columns in Python.

WARNING: For 640×480 this loop runs 639 iterations per direction × 2
directions × ``iterations`` × 2 axes = up to ~10,000 Python→MLX
dispatch calls per frame.  This is acceptable for v1 correctness but
will be slow at real-time rates.  A Metal kernel is the path to
real-time performance for this filter.

MLX constraints observed
------------------------
- No int64.
- mx.where() for masked updates.
- arr.at[:, col] has no .set(); subtract-then-add used instead.
- mx.eval() to materialise after each full pass.
"""

from __future__ import annotations

import mlx.core as mx

# Small epsilon to prevent division by zero in weight computation.
_EPS: float = 1e-6


class SpatialFilter:
    """Edge-preserving domain transform spatial filter.

    Parameters
    ----------
    alpha:
        Smoothing strength [0, 1].  Higher = more smoothing.
    delta:
        Depth difference sensitivity in the same units as the input
        (depth counts if applied to raw depth, or disparity units).
        Larger = less edge preservation.
    iterations:
        Number of filter passes (1–5).  Each pass uses bilateral weights.
    hole_fill:
        If > 0, apply ``HoleFillingFilter(mode=hole_fill)`` after spatial
        smoothing.  Set to 0 to disable.

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> depth = mx.array(np.random.randint(500, 3000, (48, 64)).astype(np.float32))
    >>> f = SpatialFilter(alpha=0.5, delta=20.0, iterations=2)
    >>> out = f.process(depth)
    >>> out.shape
    (48, 64)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        delta: float = 20.0,
        iterations: int = 2,
        hole_fill: int = 0,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.iterations = max(1, min(5, int(iterations)))
        self.hole_fill = int(hole_fill)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array) -> mx.array:
        """Apply spatial filtering to *depth*.

        Parameters
        ----------
        depth:
            ``(H, W)`` array.  Float32 disparity or uint16/float32 depth.
            Zero values are treated as invalid and are not used as filter
            sources (but may be filled by the filter).

        Returns
        -------
        mx.array
            Smoothed frame with the same shape and dtype as input.
        """
        orig_dtype = depth.dtype
        result = depth.astype(mx.float32)

        for _ in range(self.iterations):
            result = self._horizontal_pass(result)
            result = self._vertical_pass(result)

        mx.eval(result)

        if self.hole_fill > 0:
            from realsense_mlx.filters.hole_filling import HoleFillingFilter
            filler = HoleFillingFilter(mode=self.hole_fill)
            result = filler.process(result.astype(orig_dtype)).astype(mx.float32)

        return result.astype(orig_dtype)

    # ------------------------------------------------------------------
    # Pass implementations
    # ------------------------------------------------------------------

    def _horizontal_pass(self, frame: mx.array) -> mx.array:
        """Bilateral recursive scan: left→right then right→left.

        All rows are processed in parallel per column step.

        NOTE: This loop is intentionally sequential over W columns.
        For real-time performance a Metal kernel is required.
        """
        frame = self._scan_left_to_right(frame)
        frame = self._scan_right_to_left(frame)
        return frame

    def _vertical_pass(self, frame: mx.array) -> mx.array:
        """Vertical bilateral scan via transpose trick."""
        # Transpose so rows become columns, apply horizontal scan, transpose back.
        frame_t = mx.transpose(frame, (1, 0))   # (W, H)
        frame_t = self._horizontal_pass(frame_t)
        return mx.transpose(frame_t, (1, 0))    # (H, W)

    def _scan_left_to_right(self, frame: mx.array) -> mx.array:
        """Single left-to-right recursive pass over all rows."""
        _H, W = frame.shape
        denom = self.delta * self.alpha + _EPS

        for col in range(1, W):
            cur  = frame[:, col]       # (H,) — current INPUT pixel at this column
            prev = frame[:, col - 1]   # (H,) — already-smoothed output from previous col

            # Compute edge weight from the ORIGINAL input difference.
            # We use the current stored frame values (which are the smoothed values
            # from the previous column step) to compute the difference.
            diff   = mx.abs(cur - prev)
            weight = mx.exp(-diff / denom)

            # Domain Transform recursive formula:
            #   out[col] = in[col] + weight * (out[col-1] - in[col])
            # At strong edges (weight→0): out[col] = in[col]  (preserve current value)
            # In flat regions (weight→1): out[col] = out[col-1] (smooth toward neighbour)
            #
            # Only blend where the previous pixel is valid; otherwise keep current.
            prev_valid = prev > 0.0
            updated = cur + weight * (prev - cur)
            new_col = mx.where(prev_valid, updated, cur)

            # MLX ArrayAt has no .set(); use subtract-then-add to replace column.
            frame = frame.at[:, col].subtract(cur).at[:, col].add(new_col)

        return frame

    def _scan_right_to_left(self, frame: mx.array) -> mx.array:
        """Single right-to-left recursive pass over all rows."""
        _H, W = frame.shape
        denom = self.delta * self.alpha + _EPS

        for col in range(W - 2, -1, -1):
            cur   = frame[:, col]      # (H,)
            next_ = frame[:, col + 1]  # (H,) — already-smoothed from previous (right-to-left) step

            diff   = mx.abs(cur - next_)
            weight = mx.exp(-diff / denom)

            # Domain Transform recursive formula (right-to-left direction):
            #   out[col] = in[col] + weight * (out[col+1] - in[col])
            next_valid = next_ > 0.0
            updated = cur + weight * (next_ - cur)
            new_col = mx.where(next_valid, updated, cur)

            # MLX ArrayAt has no .set(); use subtract-then-add to replace column.
            frame = frame.at[:, col].subtract(cur).at[:, col].add(new_col)

        return frame

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpatialFilter("
            f"alpha={self.alpha}, delta={self.delta}, "
            f"iterations={self.iterations}, hole_fill={self.hole_fill})"
        )
