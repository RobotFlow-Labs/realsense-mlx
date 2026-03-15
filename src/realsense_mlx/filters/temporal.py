"""Temporal filter — exponential moving average with persistence masking.

The filter maintains state across calls:

1. Exponential Moving Average (EMA)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   For each valid pixel:
     smoothed[i,j] = alpha * current[i,j] + (1-alpha) * prev[i,j]

   Only pixels that are valid in *both* the current frame and the stored
   history are blended; a pixel valid now but absent previously simply
   inherits the current value.

2. Persistence bitmask
   ~~~~~~~~~~~~~~~~~~~~~~
   An 8-bit uint8 bitmask tracks which of the last 8 frames had a valid
   (non-zero) measurement for each pixel.  On each new frame:
     history = (history << 1) | (current_valid)

   Persistence mode N means "keep a pixel in the output only if it was
   valid in at least N of the last 8 frames".  The required count is
   computed as popcount(history) >= persistence.

   persistence=0 → no persistence gating (pass everything through).
   persistence=1 → must appear at least once in the last 8 frames.
   persistence=8 → must have appeared in every one of the last 8 frames.

MLX constraints observed
------------------------
- No int64: bitmask uses uint8; shifts via mx.left_shift / bitwise_and.
- mx.where() for masking.
- State arrays stored as mx.array; re-created on reset().
- mx.eval() to materialise before returning.

Note on popcount
----------------
MLX has no native popcount.  We compute it by repeatedly checking the LSB
and shifting:  sum of 8 individual bit extractions.  This costs 8 × 3 ops
(shift, and, add) per frame but is fully vectorised.
"""

from __future__ import annotations

import mlx.core as mx


def _popcount8(bitmask: mx.array) -> mx.array:
    """Count set bits in each uint8 element of *bitmask*.

    Returns an int32 array of the same shape with values in [0, 8].
    """
    # We cannot do bitwise ops on uint8 directly in MLX without potential
    # type promotion issues, so cast to int32 first.
    x = bitmask.astype(mx.int32)
    count = mx.zeros(x.shape, dtype=mx.int32)
    for _ in range(8):
        count = count + (x & 1)
        x = mx.right_shift(x, 1)
    return count


class TemporalFilter:
    """Temporal depth filter with EMA smoothing and persistence gating.

    Parameters
    ----------
    alpha:
        EMA weight for the current frame in [0, 1].  Higher = less
        smoothing, faster response to changes.
    delta:
        Depth-change threshold (in the same units as the input).  If the
        absolute difference between the current and previous value exceeds
        ``delta``, the pixel is treated as a changed pixel and the history
        is partially reset (RS2 SDK behaviour — prevents ghost edges from
        persisting after an object moves).
    persistence:
        Minimum number of valid appearances in the last 8 frames required
        to keep a pixel in the output.  0 = disabled.

    State
    -----
    _prev_frame : mx.array | None
        Float32 EMA-smoothed frame from the previous call.
    _history : mx.array | None
        uint8 bitmask tracking validity over the last 8 frames.

    Examples
    --------
    >>> import mlx.core as mx, numpy as np
    >>> f = TemporalFilter(alpha=0.4, delta=20.0, persistence=3)
    >>> for i in range(10):
    ...     frame = mx.array(np.random.randint(500, 2000, (48, 64)).astype(np.uint16))
    ...     out = f.process(frame)
    >>> f.reset()
    """

    def __init__(
        self,
        alpha: float = 0.4,
        delta: float = 20.0,
        persistence: int = 3,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.persistence = max(0, min(8, int(persistence)))

        self._prev_frame: mx.array | None = None
        self._history: mx.array | None = None  # uint8 bitmask per pixel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth: mx.array) -> mx.array:
        """Apply temporal smoothing to *depth*.

        The first call initialises internal state; subsequent calls blend
        with history.

        Parameters
        ----------
        depth:
            ``(H, W)`` array.  Zero values are invalid.  Can be uint16
            (raw depth) or float32 (disparity space).

        Returns
        -------
        mx.array
            Temporally smoothed frame, same shape and dtype as input.
        """
        orig_dtype = depth.dtype
        curr = depth.astype(mx.float32)
        curr_valid = (curr > 0.0).astype(mx.uint8)  # 1 where valid

        if self._prev_frame is None:
            # First frame: initialise state, return as-is.
            self._prev_frame = curr
            self._history = curr_valid
            mx.eval(self._prev_frame, self._history)
            return depth

        prev = self._prev_frame
        history = self._history

        # ------------------------------------------------------------------
        # 1. Update bitmask history: shift left 1 bit, OR in current validity.
        # ------------------------------------------------------------------
        new_history = mx.left_shift(history, 1) & mx.array(0xFF, dtype=mx.uint8)
        new_history = new_history | curr_valid

        # ------------------------------------------------------------------
        # 2. Detect large depth changes (possible occlusion / moving object).
        #    If |curr - prev| > delta, do NOT blend; take current value and
        #    partially reset history to avoid ghosting.
        # ------------------------------------------------------------------
        both_valid = (curr > 0.0) & (prev > 0.0)
        change = mx.abs(curr - prev)
        is_large_change = (change > self.delta) & both_valid

        # Reset history bits for changed pixels to force re-acquisition.
        # We keep only the newest bit (current frame's validity).
        new_history = mx.where(
            is_large_change, curr_valid, new_history
        )

        # ------------------------------------------------------------------
        # 3. EMA blend.
        # ------------------------------------------------------------------
        # Blend only where both current and previous are valid AND no large change.
        can_blend = both_valid & ~is_large_change
        blended = self.alpha * curr + (1.0 - self.alpha) * prev
        # Where current is valid but no previous: keep current.
        # Where neither is valid: keep 0.
        smoothed = mx.where(can_blend, blended, curr)

        # ------------------------------------------------------------------
        # 4. Persistence gating.
        # ------------------------------------------------------------------
        if self.persistence > 0:
            bit_count = _popcount8(new_history)  # int32, values 0–8
            enough_history = bit_count >= self.persistence
            smoothed = mx.where(enough_history, smoothed,
                                mx.array(0.0, dtype=mx.float32))

        # ------------------------------------------------------------------
        # 5. Update state.
        # ------------------------------------------------------------------
        # Only update prev_frame for pixels that were valid in current frame
        # (don't let invalid pixels overwrite good history).
        updated_prev = mx.where(curr > 0.0, smoothed, prev)
        self._prev_frame = updated_prev
        self._history = new_history

        mx.eval(self._prev_frame, self._history)

        # Clip to uint16 range only when the caller will cast to uint16.
        # Float32 inputs are disparity values that legitimately exceed 65535,
        # so clipping them here would corrupt the data.
        if orig_dtype == mx.uint16:
            smoothed = mx.clip(smoothed, 0.0, 65535.0)

        mx.eval(smoothed)
        return smoothed.astype(orig_dtype)

    def reset(self) -> None:
        """Clear all filter state.

        Call this when switching depth streams, changing resolution, or
        after a significant scene change.
        """
        self._prev_frame = None
        self._history = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "initialised" if self._prev_frame is not None else "empty"
        return (
            f"TemporalFilter("
            f"alpha={self.alpha}, delta={self.delta}, "
            f"persistence={self.persistence}, state={state!r})"
        )
