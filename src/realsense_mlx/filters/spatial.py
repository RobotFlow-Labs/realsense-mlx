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

Metal kernel strategy (use_metal=True, default)
------------------------------------------------
Each row is independent.  The column scan within each row is sequential
(each step depends on the previous output).  We therefore launch one Metal
thread per row so that all H rows execute in parallel on the GPU while each
thread performs the W-column scan sequentially in Metal shading language.

This replaces the old Python loop over W columns and eliminates the
~10 000 Python→MLX dispatch calls per frame, giving a large speedup on
640×480 input.

Python fallback strategy (use_metal=False)
------------------------------------------
All ROWS are processed in parallel for each column step — this gives us
full MLX vectorisation across the height dimension.  The column scan
itself is sequential (each step depends on the previous output), so we
loop over W columns in Python.

WARNING: For 640×480 this loop runs 639 iterations per direction × 2
directions × ``iterations`` × 2 axes = up to ~10,000 Python→MLX
dispatch calls per frame.  This is acceptable for correctness testing but
will be slow at real-time rates.

MLX constraints observed
------------------------
- No int64.
- mx.where() for masked updates.
- arr.at[:, col] has no .set(); subtract-then-add used instead.
- mx.eval() to materialise after each full pass.
"""

from __future__ import annotations

import functools

import mlx.core as mx

__all__ = ["SpatialFilter"]

# Small epsilon to prevent division by zero in weight computation.
_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Metal kernel source for the horizontal bilateral recursive scan.
#
# One Metal thread per row (tid = row index).  Each thread:
#   1. Copies its row from `inp` to `out`.
#   2. Left-to-right scan: out[i] = out[i] + w * (out[i-1] - out[i])
#      where w = exp(-|out[i] - out[i-1]| / denom), only when out[i-1] > 0.
#   3. Right-to-left scan: same formula in the reverse direction.
# ---------------------------------------------------------------------------
_HORIZONTAL_METAL_SOURCE = r"""
    int tid = (int)thread_position_in_grid.x;
    int H   = height[0];
    int W   = width[0];
    float denom = delta[0] * alpha[0] + 1e-6f;

    if (tid >= H) return;

    int row_offset = tid * W;

    // Step 1: copy row to output buffer.
    for (int i = 0; i < W; i++) {
        out[row_offset + i] = inp[row_offset + i];
    }

    // Step 2: left-to-right bilateral recursive scan.
    for (int i = 1; i < W; i++) {
        float cur  = out[row_offset + i];
        float prev = out[row_offset + i - 1];
        if (prev > 0.0f) {
            float diff = metal::abs(cur - prev);
            float w    = metal::exp(-diff / denom);
            out[row_offset + i] = cur + w * (prev - cur);
        }
    }

    // Step 3: right-to-left bilateral recursive scan.
    for (int i = W - 2; i >= 0; i--) {
        float cur  = out[row_offset + i];
        float next = out[row_offset + i + 1];
        if (next > 0.0f) {
            float diff = metal::abs(cur - next);
            float w    = metal::exp(-diff / denom);
            out[row_offset + i] = cur + w * (next - cur);
        }
    }
"""


@functools.lru_cache(maxsize=1)
def _get_horizontal_kernel() -> object:
    """JIT-compile and return the Metal horizontal-scan kernel.

    Compiled once per process; subsequent calls return the cached object.
    Thread-safe via ``functools.lru_cache`` — no mutable module-level state.
    """
    return mx.fast.metal_kernel(
        name="spatial_horizontal",
        input_names=["inp", "height", "width", "alpha", "delta"],
        output_names=["out"],
        source=_HORIZONTAL_METAL_SOURCE,
    )


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
    use_metal:
        If ``True`` (default), use the Metal GPU kernel for the horizontal
        bilateral scan.  This eliminates the Python column loop and gives
        a significant speedup on Apple Silicon.  Set to ``False`` to fall
        back to the pure-MLX Python loop (useful for debugging or on
        non-Metal platforms).

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
        use_metal: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.iterations = max(1, min(5, int(iterations)))
        self.hole_fill = int(hole_fill)
        self.use_metal = bool(use_metal)

        # FIX 4: pre-instantiate HoleFillingFilter to avoid repeated construction
        # inside process().  We store None when hole_fill==0 (disabled).
        if self.hole_fill > 0:
            from realsense_mlx.filters.hole_filling import HoleFillingFilter
            self._hole_filling_filter: object | None = HoleFillingFilter(mode=self.hole_fill)
        else:
            self._hole_filling_filter = None

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
        # FIX 2: input validation — must be a 2-D array with non-zero dimensions.
        if depth.ndim != 2:
            raise ValueError(
                f"SpatialFilter expects 2-D (H, W) array, got shape {depth.shape}"
            )
        if depth.shape[0] == 0 or depth.shape[1] == 0:
            return depth

        orig_dtype = depth.dtype
        result = depth.astype(mx.float32)

        for _ in range(self.iterations):
            result = self._horizontal_pass(result)
            result = self._vertical_pass(result)
            # FIX 3: bound computation graph size by materialising after each pass.
            mx.eval(result)

        if self._hole_filling_filter is not None:
            # FIX 4: use pre-constructed HoleFillingFilter instance.
            result = self._hole_filling_filter.process(
                result.astype(orig_dtype)
            ).astype(mx.float32)

        return result.astype(orig_dtype)

    def reset(self) -> None:
        """No-op — SpatialFilter is stateless."""
        pass

    # ------------------------------------------------------------------
    # Pass implementations
    # ------------------------------------------------------------------

    def _horizontal_pass(self, frame: mx.array) -> mx.array:
        """Bilateral recursive scan: left→right then right→left.

        Dispatches to the Metal kernel when ``use_metal=True``, otherwise
        falls back to the pure-Python column loop.
        """
        if self.use_metal:
            return self._horizontal_pass_metal(frame)
        return self._horizontal_pass_python(frame)

    def _horizontal_pass_metal(self, frame: mx.array) -> mx.array:
        """Metal GPU kernel: parallel across rows, sequential within each row.

        Launches H threads (one per row).  Each thread performs the full
        left→right + right→left bilateral scan for its assigned row.

        Parameters
        ----------
        frame:
            Float32 array of shape ``(H, W)``.  Must be C-contiguous (row-major),
            which ``ensure_row_contiguous=True`` (the default) guarantees.

        Returns
        -------
        mx.array
            Filtered frame of shape ``(H, W)``, float32.
        """
        H, W = frame.shape
        kernel = _get_horizontal_kernel()

        outputs = kernel(
            inputs=[
                frame.flatten(),
                mx.array([H], dtype=mx.int32),
                mx.array([W], dtype=mx.int32),
                mx.array([self.alpha], dtype=mx.float32),
                mx.array([self.delta], dtype=mx.float32),
            ],
            output_shapes=[(H * W,)],
            output_dtypes=[mx.float32],
            grid=(H, 1, 1),
            threadgroup=(min(H, 256), 1, 1),
        )
        return outputs[0].reshape(H, W)

    def _horizontal_pass_python(self, frame: mx.array) -> mx.array:
        """Pure-MLX Python loop: left→right then right→left.

        All rows are processed in parallel for each column step — this gives
        full MLX vectorisation across the height dimension.  The column scan
        itself is sequential (each step depends on the previous output), so
        we loop over W columns in Python.

        Kept as reference and fallback.  NOTE: for 640×480, this loop
        emits ~10 000 Python→MLX dispatch calls per frame, which is
        the performance bottleneck at real-time frame rates.
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
        """Single left-to-right recursive pass over all rows (Python loop)."""
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
        """Single right-to-left recursive pass over all rows (Python loop)."""
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
            f"iterations={self.iterations}, hole_fill={self.hole_fill}, "
            f"use_metal={self.use_metal})"
        )
