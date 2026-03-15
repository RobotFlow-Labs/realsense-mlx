"""Depth colorization filter using MLX-accelerated LUT lookups.

Converts uint16 depth frames to RGB visualizations via a precomputed
256-entry color lookup table.  Supports 10 named color maps and two
normalization modes: direct linear scaling and histogram equalization.

MLX constraints observed
------------------------
- No int64: all index arithmetic uses int32.
- Histogram computation stays on CPU (numpy) — histogram has sequential
  dependency on sorted cumulative counts that MLX cannot express in a
  single parallel kernel.
- LUT is precomputed once as an mx.array in __init__ to avoid repeated
  host→device copies per frame.
- mx.clip() for clamping, mx.where() for masked selection.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


class DepthColorizer:
    """Convert depth frames to RGB color visualizations.

    Parameters
    ----------
    colormap:
        Name of the color map to use.  One of the 10 keys in
        :attr:`COLORMAPS`.  Default ``"jet"``.
    min_depth:
        Near clip plane in metres.  Pixels closer than this map to the
        first LUT entry.  Default ``0.1``.
    max_depth:
        Far clip plane in metres.  Pixels farther than this map to the
        last LUT entry.  Default ``10.0``.
    equalize:
        When ``True`` (default), apply histogram equalization before LUT
        lookup for better contrast on scenes where depth values cluster
        in a narrow range.  When ``False``, use linear normalization.
    depth_units:
        Metres per depth count (``rs2_option.DEPTH_UNITS``).  Typical
        RealSense value is ``0.001`` (millimetre resolution).

    Examples
    --------
    >>> import numpy as np
    >>> import mlx.core as mx
    >>> from realsense_mlx.filters.colorizer import DepthColorizer
    >>> colorizer = DepthColorizer(colormap="jet", equalize=False)
    >>> depth_np = np.tile(np.arange(640, dtype=np.uint16) * 15, (480, 1))
    >>> depth_mx = mx.array(depth_np)
    >>> rgb = colorizer.colorize(depth_mx)
    >>> mx.eval(rgb)
    >>> rgb.shape
    (480, 640, 3)
    """

    # Each entry: (t, R, G, B)  where t in [0.0, 1.0]
    COLORMAPS: dict[str, list[tuple[float, int, int, int]]] = {
        # Blue → Cyan → Green → Yellow → Red
        "jet": [
            (0.000, 0,   0,   128),
            (0.100, 0,   0,   255),
            (0.250, 0,   128, 255),
            (0.375, 0,   255, 255),
            (0.500, 0,   255, 0),
            (0.625, 255, 255, 0),
            (0.750, 255, 128, 0),
            (0.875, 255, 0,   0),
            (1.000, 128, 0,   0),
        ],
        # RealSense SDK default color map
        "classic": [
            (0.00, 30,  77,  203),
            (0.25, 90,  180, 60),
            (0.50, 255, 255, 0),
            (0.75, 255, 100, 0),
            (1.00, 200, 0,   0),
        ],
        "grayscale": [
            (0.0, 0,   0,   0),
            (1.0, 255, 255, 255),
        ],
        "inv_grayscale": [
            (0.0, 255, 255, 255),
            (1.0, 0,   0,   0),
        ],
        "warm": [
            (0.00, 80,  0,   0),
            (0.33, 255, 100, 0),
            (0.66, 255, 255, 0),
            (1.00, 255, 255, 255),
        ],
        "cold": [
            (0.00, 0,   0,   80),
            (0.33, 0,   100, 255),
            (0.66, 0,   255, 255),
            (1.00, 255, 255, 255),
        ],
        # Terrain-like: ocean → coast → vegetation → desert → rock → snow
        "biomes": [
            (0.00, 0,   0,   204),
            (0.20, 0,   128, 255),
            (0.35, 0,   200, 200),
            (0.50, 34,  139, 34),
            (0.65, 210, 180, 140),
            (0.80, 139, 90,  43),
            (1.00, 255, 250, 250),
        ],
        # Six discrete color bands (hard step transitions)
        "quantized": [
            (0.000, 0,   0,   255),
            (0.167, 0,   0,   255),
            (0.168, 0,   255, 0),
            (0.333, 0,   255, 0),
            (0.334, 255, 255, 0),
            (0.500, 255, 255, 0),
            (0.501, 255, 128, 0),
            (0.667, 255, 128, 0),
            (0.668, 255, 0,   0),
            (0.833, 255, 0,   0),
            (0.834, 128, 0,   128),
            (1.000, 128, 0,   128),
        ],
        # Binary black/white split — useful for threshold testing
        "pattern": [
            (0.000, 0,   0,   0),
            (0.499, 0,   0,   0),
            (0.500, 255, 255, 255),
            (1.000, 255, 255, 255),
        ],
        # Full HSV hue spectrum (rainbow cycle)
        "hue": [
            (0.000, 255, 0,   0),
            (0.167, 255, 255, 0),
            (0.333, 0,   255, 0),
            (0.500, 0,   255, 255),
            (0.667, 0,   0,   255),
            (0.833, 255, 0,   255),
            (1.000, 255, 0,   0),
        ],
    }

    def __init__(
        self,
        colormap: str = "jet",
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        equalize: bool = True,
        depth_units: float = 0.001,
    ) -> None:
        if colormap not in self.COLORMAPS:
            raise ValueError(
                f"Unknown colormap {colormap!r}. "
                f"Available: {sorted(self.COLORMAPS)}"
            )
        if min_depth >= max_depth:
            raise ValueError(
                f"min_depth ({min_depth}) must be less than max_depth ({max_depth})"
            )
        if depth_units <= 0.0:
            raise ValueError(f"depth_units must be positive, got {depth_units}")

        self.colormap = colormap
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.equalize = equalize
        self.depth_units = float(depth_units)

        # Precompute 256-entry LUT once; reused per frame.
        self._lut: mx.array = self._build_lut()

    # ------------------------------------------------------------------
    # LUT construction
    # ------------------------------------------------------------------

    def _build_lut(self) -> mx.array:
        """Interpolate colormap keypoints to a 256-entry (256, 3) uint8 LUT.

        Linear interpolation between adjacent keypoint pairs.  If a keypoint
        pair shares the same ``t`` value (step transition), the upper keypoint
        wins (i.e. the sample at that exact index takes the upper color).

        Returns
        -------
        mx.array
            Shape ``(256, 3)``, dtype ``uint8``.
        """
        keypoints = self.COLORMAPS[self.colormap]
        lut = np.zeros((256, 3), dtype=np.uint8)

        for i in range(256):
            t = i / 255.0

            # Find the two bracketing keypoints.
            # Walk from the end to honour step transitions: when two keypoints
            # share the same t, the one with the smaller index is the lower
            # bound and the one with the larger index is the upper bound.
            lo_idx = 0
            hi_idx = len(keypoints) - 1

            for k in range(len(keypoints) - 1):
                if keypoints[k][0] <= t <= keypoints[k + 1][0]:
                    lo_idx = k
                    hi_idx = k + 1
                    break

            t0, r0, g0, b0 = keypoints[lo_idx]
            t1, r1, g1, b1 = keypoints[hi_idx]

            span = t1 - t0
            if span < 1e-9:
                # Step transition: exact t matches upper keypoint.
                alpha = 1.0
            else:
                alpha = (t - t0) / span

            alpha = max(0.0, min(1.0, alpha))

            lut[i, 0] = int(round(r0 + alpha * (r1 - r0)))
            lut[i, 1] = int(round(g0 + alpha * (g1 - g0)))
            lut[i, 2] = int(round(b0 + alpha * (b1 - b0)))

        return mx.array(lut)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def colorize(self, depth: mx.array) -> mx.array:
        """Colorize a depth frame.

        Parameters
        ----------
        depth:
            Input depth frame, shape ``(H, W)``, dtype ``uint16``.

        Returns
        -------
        mx.array
            RGB output, shape ``(H, W, 3)``, dtype ``uint8``.

        Notes
        -----
        When ``equalize=True`` the histogram is computed on CPU (numpy) because
        cumulative-sum has a sequential dependency that prevents efficient GPU
        parallelism.  The LUT lookup itself runs on MLX/GPU.
        """
        if depth.ndim != 2:
            raise ValueError(
                f"Expected 2-D depth frame (H, W), got shape {depth.shape}"
            )

        if self.equalize:
            return self._colorize_equalized(depth)
        return self._colorize_direct(depth)

    # ------------------------------------------------------------------
    # Normalization modes
    # ------------------------------------------------------------------

    def _colorize_direct(self, depth: mx.array) -> mx.array:
        """Linear [min_depth, max_depth] normalization then LUT lookup.

        Parameters
        ----------
        depth:
            ``(H, W)`` uint16 depth counts.

        Returns
        -------
        mx.array
            ``(H, W, 3)`` uint8 RGB.
        """
        h, w = depth.shape

        depth_f = depth.astype(mx.float32)
        # Convert counts → metres using depth_units scale factor.
        depth_m = depth_f * self.depth_units

        # Normalise to [0, 1] within [min_depth, max_depth].
        depth_range = self.max_depth - self.min_depth
        norm = (depth_m - self.min_depth) / depth_range

        # Map to LUT index in [0, 255].  mx.clip avoids out-of-bounds access.
        indices = mx.clip(norm * 255.0, 0.0, 255.0).astype(mx.int32)

        # Flatten, gather from LUT, reshape back.
        flat_indices = indices.reshape(-1)
        colors = self._lut[flat_indices]  # (H*W, 3)
        return colors.reshape(h, w, 3)

    def _colorize_equalized(self, depth: mx.array) -> mx.array:
        """Histogram equalization normalization then LUT lookup.

        The 16-bit histogram and cumulative distribution function are computed
        on CPU via numpy because the prefix-sum has a data dependency chain.
        The resulting 65536-entry remapping table and the final LUT lookup are
        applied on the MLX device.

        Parameters
        ----------
        depth:
            ``(H, W)`` uint16 depth counts.

        Returns
        -------
        mx.array
            ``(H, W, 3)`` uint8 RGB.
        """
        h, w = depth.shape

        # --- CPU phase: build the 16-bit → 8-bit remapping table -----------
        # Materialize depth on CPU for histogram computation.
        mx.eval(depth)
        depth_np = np.array(depth, copy=False).astype(np.int32)

        hist = np.bincount(depth_np.ravel(), minlength=65536)
        # Exclude invalid (zero-depth) pixels from the equalization so they
        # do not bias the cumulative distribution.
        hist[0] = 0

        cumhist = np.cumsum(hist)
        total = int(cumhist[-1])
        if total > 0:
            # Map each of the 65536 depth levels to an 8-bit LUT index.
            mapping = (cumhist * 255 // total).astype(np.uint8)
        else:
            # All pixels are invalid — produce a black frame.
            mapping = np.zeros(65536, dtype=np.uint8)

        # --- MLX phase: gather and LUT lookup --------------------------------
        mapping_mx = mx.array(mapping)  # (65536,) uint8

        # Index into the remapping table using the raw depth counts.
        # depth is uint16 → cast to int32 for indexing (no int64).
        raw_indices = depth.reshape(-1).astype(mx.int32)  # (H*W,)
        norm_indices = mapping_mx[raw_indices].astype(mx.int32)  # (H*W,)

        colors = self._lut[norm_indices]  # (H*W, 3)
        return colors.reshape(h, w, 3)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_colormap(self, colormap: str) -> None:
        """Switch the active color map and rebuild the LUT.

        Parameters
        ----------
        colormap:
            New colormap name.  Must be a key in :attr:`COLORMAPS`.
        """
        if colormap not in self.COLORMAPS:
            raise ValueError(
                f"Unknown colormap {colormap!r}. "
                f"Available: {sorted(self.COLORMAPS)}"
            )
        self.colormap = colormap
        self._lut = self._build_lut()

    def lut_numpy(self) -> np.ndarray:
        """Return the current LUT as a ``(256, 3)`` numpy uint8 array.

        Useful for inspection and unit tests without a full colorize call.
        """
        mx.eval(self._lut)
        return np.array(self._lut, copy=True)

    def __repr__(self) -> str:
        return (
            f"DepthColorizer("
            f"colormap={self.colormap!r}, "
            f"min_depth={self.min_depth}, "
            f"max_depth={self.max_depth}, "
            f"equalize={self.equalize}, "
            f"depth_units={self.depth_units})"
        )
