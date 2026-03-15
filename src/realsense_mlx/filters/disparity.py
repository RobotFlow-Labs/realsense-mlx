"""Depth ↔ Disparity conversion filter.

The RealSense SDK stores depth as uint16 values scaled by ``depth_units``
(metres per count, typically 0.001 for millimetre resolution).  The spatial
and temporal filters work more accurately in *disparity* space (proportional
to 1/depth), which makes the error distribution more uniform.

Conversion formula
------------------
  disparity = baseline_mm * focal_px * 32 / (depth_counts * depth_units_m_per_count)

The factor of 32 matches the RS2 SDK fixed-point disparity representation
(5-bit fractional).

MLX constraints observed
------------------------
- No int64: index arithmetic stays in int32 or float32.
- mx.where() instead of boolean indexing.
- arr.at[idx].set/add() returns a NEW array (functional style).
"""

from __future__ import annotations

import mlx.core as mx


class DisparityTransform:
    """Convert depth frames between depth (uint16) and disparity (float32) space.

    Parameters
    ----------
    baseline_mm:
        Stereo baseline in millimetres (camera-specific, e.g. 50.0 mm for D435).
    focal_px:
        Horizontal focal length in pixels (from camera intrinsics).
    depth_units:
        Metres per depth count (``rs2_option.DEPTH_UNITS``, typically 0.001).
    to_disparity:
        Direction flag.  ``True`` = depth→disparity, ``False`` = disparity→depth.

    Examples
    --------
    >>> import mlx.core as mx
    >>> import numpy as np
    >>> depth_np = np.array([[1000, 2000, 0, 500]], dtype=np.uint16)
    >>> depth = mx.array(depth_np)
    >>> t = DisparityTransform(baseline_mm=50.0, focal_px=383.7, depth_units=0.001)
    >>> disp = t.process(depth)          # → float32 disparity
    >>> back = DisparityTransform(50.0, 383.7, 0.001, to_disparity=False)
    >>> reconstructed = back.process(disp)   # → uint16 depth
    """

    def __init__(
        self,
        baseline_mm: float,
        focal_px: float,
        depth_units: float,
        to_disparity: bool = True,
    ) -> None:
        # Pre-compute the disparity-depth factor once.
        # disparity = d2d_factor / depth_counts
        # depth_counts = d2d_factor / disparity
        self.d2d_factor: float = (
            baseline_mm * focal_px * 32.0 / depth_units if depth_units > 0.0 else 0.0
        )
        self.to_disparity = to_disparity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: mx.array) -> mx.array:
        """Apply the configured conversion to *frame*.

        Parameters
        ----------
        frame:
            - When ``to_disparity=True``: ``(H, W)`` uint16 depth counts.
            - When ``to_disparity=False``: ``(H, W)`` float32 disparity values.

        Returns
        -------
        mx.array
            - Disparity frame (float32) when ``to_disparity=True``.
            - Depth frame (uint16) when ``to_disparity=False``.
        """
        if self.to_disparity:
            return self._depth_to_disparity(frame)
        return self._disparity_to_depth(frame)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _depth_to_disparity(self, depth: mx.array) -> mx.array:
        """(H, W) uint16 → (H, W) float32 disparity.

        Zero-depth pixels (invalid measurements) map to disparity 0.0.
        """
        depth_f = depth.astype(mx.float32)
        valid = depth_f > 0.0
        # Safe division: replace zeros with 1.0 before dividing, then mask out.
        safe_depth = mx.where(valid, depth_f, mx.array(1.0, dtype=mx.float32))
        disparity = self.d2d_factor / safe_depth
        return mx.where(valid, disparity, mx.array(0.0, dtype=mx.float32))

    def _disparity_to_depth(self, disparity: mx.array) -> mx.array:
        """(H, W) float32 disparity → (H, W) uint16 depth counts.

        Zero-disparity pixels (invalid) map to depth 0.
        """
        valid = disparity > 0.0
        safe_disp = mx.where(valid, disparity, mx.array(1.0, dtype=mx.float32))
        depth_f = self.d2d_factor / safe_disp + 0.5  # +0.5 for round-to-nearest
        depth_f = mx.where(valid, depth_f, mx.array(0.0, dtype=mx.float32))
        # Clamp to uint16 range before casting.
        depth_f = mx.clip(depth_f, 0.0, 65535.0)
        return depth_f.astype(mx.uint16)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        direction = "depth→disparity" if self.to_disparity else "disparity→depth"
        return (
            f"DisparityTransform("
            f"d2d_factor={self.d2d_factor:.2f}, direction={direction!r})"
        )
