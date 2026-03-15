"""Hole-filling filter for depth frames.

Three modes mirror the RS2 SDK ``rs2_hole_filling_option``:

Mode 0 – FILL_FROM_LEFT
    Each invalid pixel is replaced by the nearest valid pixel to its left
    on the same row.  This is a scan-line propagation pass.

Mode 1 – FARTHEST (default)
    Each invalid pixel is replaced by the farthest (largest depth value)
    valid pixel among its 4-connected neighbours.  Preserves object
    boundaries by preferring background depth.

Mode 2 – NEAREST
    Each invalid pixel is replaced by the nearest (smallest non-zero depth)
    valid pixel among its 4-connected neighbours.  Useful when foreground
    objects matter more.

MLX constraints observed
------------------------
- No boolean indexing: ``mx.where(mask, a, b)`` throughout.
- Neighbour access via ``mx.pad`` + slicing, not scatter/gather with int64.
- ``arr.at[...].set(...)`` returns a NEW array.
- ``mx.eval(arr)`` to materialise before Python inspection.

Metal kernel strategy for FILL_FROM_LEFT (use_metal=True, default)
-------------------------------------------------------------------
Each row is independent.  The scan within a row is sequential (each pixel
depends on the previous result), so we launch one Metal thread per row.
All H rows execute in parallel on the GPU while each thread performs the
W-column scan sequentially in Metal shading language.

This replaces the old Python loop over W columns (~1280 Python→MLX dispatches
per frame at 1280×720) and eliminates the bottleneck entirely.

Python fallback strategy for FILL_FROM_LEFT (use_metal=False)
--------------------------------------------------------------
All rows are processed in parallel for each column step — full MLX
vectorisation across the height dimension.  The column scan is sequential,
so we loop over W columns in Python.  For large frames this Python loop is
the bottleneck (~50 FPS at 1280×720).
"""

from __future__ import annotations

import functools

import mlx.core as mx

__all__ = ["HoleFillingFilter"]

# ---------------------------------------------------------------------------
# Metal kernel source for the FILL_FROM_LEFT scan.
#
# One Metal thread per row (tid = row index).  Each thread:
#   1. Scans left-to-right along its row.
#   2. Maintains `last_valid`: the most recent non-zero value seen.
#   3. Writes the current value if valid, or `last_valid` if the current
#      pixel is zero (invalid).
# ---------------------------------------------------------------------------
_FILL_LEFT_METAL_SOURCE = r"""
    int tid = (int)thread_position_in_grid.x;
    int H   = height[0];
    int W   = width[0];

    if (tid >= H) return;

    int offset = tid * W;

    float last_valid = 0.0f;
    for (int i = 0; i < W; i++) {
        float val = inp[offset + i];
        if (val > 0.0f) {
            last_valid = val;
        }
        out[offset + i] = (val > 0.0f) ? val : last_valid;
    }
"""


@functools.lru_cache(maxsize=1)
def _get_fill_left_kernel() -> object:
    """JIT-compile and return the Metal FILL_FROM_LEFT kernel.

    Compiled once per process; subsequent calls return the cached object.
    Thread-safe via ``functools.lru_cache`` — no mutable module-level state.
    """
    return mx.fast.metal_kernel(
        name="hole_fill_left",
        input_names=["inp", "height", "width"],
        output_names=["out"],
        source=_FILL_LEFT_METAL_SOURCE,
    )


class HoleFillingFilter:
    """Fill zero-valued (invalid) pixels in a depth frame.

    Parameters
    ----------
    mode:
        Filling strategy:
        - ``HoleFillingFilter.FILL_FROM_LEFT`` (0) – left propagation
        - ``HoleFillingFilter.FARTHEST`` (1) – farthest neighbour (default)
        - ``HoleFillingFilter.NEAREST`` (2) – nearest non-zero neighbour
    use_metal:
        If ``True`` (default), use the Metal GPU kernel for FILL_FROM_LEFT.
        This eliminates the Python column loop and gives a significant speedup
        on Apple Silicon.  Set to ``False`` to fall back to the pure-MLX
        Python loop (useful for debugging or on non-Metal platforms).
        Has no effect for FARTHEST or NEAREST modes (they are always fully
        vectorised with no Python loop).

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> depth = mx.array(np.array([[1000, 0, 0, 2000]], dtype=np.uint16))
    >>> f = HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT)
    >>> out = f.process(depth)
    """

    FILL_FROM_LEFT: int = 0
    FARTHEST: int = 1
    NEAREST: int = 2

    def __init__(self, mode: int = 1, use_metal: bool = True) -> None:
        if mode not in (self.FILL_FROM_LEFT, self.FARTHEST, self.NEAREST):
            raise ValueError(
                f"Unknown hole-fill mode {mode!r}. "
                f"Valid: FILL_FROM_LEFT=0, FARTHEST=1, NEAREST=2."
            )
        self.mode = mode
        self.use_metal = bool(use_metal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array) -> mx.array:
        """Fill holes in *depth*.

        Parameters
        ----------
        depth:
            ``(H, W)`` array.  Zero values are treated as invalid.
            Can be uint16 (raw depth) or float32 (disparity space).

        Returns
        -------
        mx.array
            Same shape and dtype as input with holes filled where possible.
        """
        if self.mode == self.FILL_FROM_LEFT:
            if self.use_metal:
                return self._fill_from_left_metal(depth)
            return self._fill_from_left_python(depth)
        elif self.mode == self.FARTHEST:
            return self._fill_neighbors(depth, take_max=True)
        else:
            return self._fill_neighbors(depth, take_max=False)

    # ------------------------------------------------------------------
    # FILL_FROM_LEFT implementations
    # ------------------------------------------------------------------

    def _fill_from_left_metal(self, depth: mx.array) -> mx.array:
        """Metal GPU kernel: one thread per row, sequential scan within row.

        Launches H threads (one per row).  Each thread performs the full
        left-to-right prefix-fill scan for its assigned row.  All H rows
        execute in parallel on the GPU.

        Parameters
        ----------
        depth:
            ``(H, W)`` array, any dtype.  Converted to float32 internally.

        Returns
        -------
        mx.array
            Same shape and dtype as *depth* with holes filled.
        """
        depth_f = depth.astype(mx.float32)
        H, W = depth_f.shape

        kernel = _get_fill_left_kernel()

        outputs = kernel(
            inputs=[
                depth_f.flatten(),
                mx.array([H], dtype=mx.int32),
                mx.array([W], dtype=mx.int32),
            ],
            output_shapes=[(H * W,)],
            output_dtypes=[mx.float32],
            grid=(H, 1, 1),
            threadgroup=(min(H, 256), 1, 1),
        )
        result = outputs[0].reshape(H, W)
        mx.eval(result)
        return result.astype(depth.dtype)

    def _fill_from_left_python(self, depth: mx.array) -> mx.array:
        """Propagate last valid pixel rightward along each row (Python loop).

        The algorithm is inherently sequential within a row (each pixel
        depends on the previous result).  We iterate over columns but
        process ALL rows in parallel for each column step.

        NOTE: For large frames (e.g. 1280x720) this Python loop over
        W columns is the bottleneck (~50 FPS).  Prefer ``use_metal=True``.
        """
        depth_f = depth.astype(mx.float32)
        H, W = depth_f.shape
        result = depth_f  # functional; each step returns a new array

        for col in range(1, W):
            current = result[:, col]      # (H,)
            prev = result[:, col - 1]     # (H,)
            # Replace invalid (zero) pixel with previous column value.
            filled = mx.where(current > 0.0, current, prev)
            # Functional column update: subtract current value then add new value.
            # (MLX ArrayAt has no .set(); only .add()/.subtract() etc.)
            result = result.at[:, col].subtract(current).at[:, col].add(filled)
            # Prevent unbounded computation graph growth by materialising
            # the array every 64 columns.  Without this, MLX lazily chains all
            # column-update ops into a single graph that becomes very large (and
            # slow to compile) for wide frames (e.g. 1280 cols → 1279 ops).
            if col % 64 == 0:
                mx.eval(result)

        mx.eval(result)
        return result.astype(depth.dtype)

    # ------------------------------------------------------------------
    # Neighbour-based fill (FARTHEST / NEAREST)
    # ------------------------------------------------------------------

    def _fill_neighbors(self, depth: mx.array, *, take_max: bool) -> mx.array:
        """Replace invalid pixels with the farthest or nearest valid neighbour.

        Uses 4-connectivity (up, down, left, right).  We build four shifted
        versions via ``mx.pad`` + slicing, then reduce with mx.max / mx.min.

        Invalid neighbours (zero) are excluded from the reduction by
        substituting a sentinel value:
        - For max (farthest): replace zero neighbours with 0 (excluded by max).
        - For min (nearest): replace zero neighbours with +∞ (excluded by min).
        """
        depth_f = depth.astype(mx.float32)
        H, W = depth_f.shape

        # --- build 4 neighbours via padding ---
        # Pad with zeros on each side; invalid padding pixels are masked out.
        padded = mx.pad(depth_f, [(1, 1), (1, 1)])  # (H+2, W+2)

        up    = padded[0:H,   1:W+1]  # shifted up
        down  = padded[2:H+2, 1:W+1]  # shifted down
        left  = padded[1:H+1, 0:W  ]  # shifted left
        right = padded[1:H+1, 2:W+2]  # shifted right

        if take_max:
            # Farthest: use mx.maximum reduction; zero sentinel keeps invalids out.
            best = mx.maximum(mx.maximum(up, down), mx.maximum(left, right))
        else:
            # Use float('inf') as the sentinel instead of a magic large literal.
            # Any valid depth value is necessarily finite, so an inf sentinel is
            # never confused with a real measurement regardless of depth range.
            inf_sentinel = mx.array(float("inf"), dtype=mx.float32)
            up_    = mx.where(up    > 0.0, up,    inf_sentinel)
            down_  = mx.where(down  > 0.0, down,  inf_sentinel)
            left_  = mx.where(left  > 0.0, left,  inf_sentinel)
            right_ = mx.where(right > 0.0, right, inf_sentinel)
            best_candidate = mx.minimum(
                mx.minimum(up_, down_), mx.minimum(left_, right_)
            )
            # If all neighbours are invalid, sentinel (inf) remains → map back to 0.
            best = mx.where(
                mx.isinf(best_candidate),
                mx.array(0.0, dtype=mx.float32),
                best_candidate,
            )

        # Only fill pixels that are currently invalid (zero).
        is_invalid = depth_f <= 0.0
        result = mx.where(is_invalid, best, depth_f)

        mx.eval(result)
        return result.astype(depth.dtype)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode_names = {0: "FILL_FROM_LEFT", 1: "FARTHEST", 2: "NEAREST"}
        return (
            f"HoleFillingFilter(mode={mode_names[self.mode]}, "
            f"use_metal={self.use_metal})"
        )
