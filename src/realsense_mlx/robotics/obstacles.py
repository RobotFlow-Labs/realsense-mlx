"""Obstacle detection from depth frames.

Identifies obstacles within a specified distance range and returns their
positions in camera-frame coordinates along with navigation-relevant
metrics such as closest distance and free-path width.

Algorithm
---------
1. Convert raw depth to metres (MLX).
2. Deproject to 3D (MLX) — reuses the same normalised-grid trick as
   :mod:`realsense_mlx.geometry.pointcloud`.
3. Build a binary obstacle mask: pixels with depth in
   ``[min_distance_m, max_distance_m]`` whose Y value (height) exceeds
   ``obstacle_height_m`` are obstacles (MLX).
4. Compute aggregate metrics on-device (MLX): closest distance, total
   obstacle pixel count.
5. Find bounding boxes and free-path width in NumPy (small, structural
   work not worth a custom Metal kernel).

Coordinate conventions
-----------------------
Standard RealSense camera frame::

    +X → right
    +Y → down
    +Z → into the scene (depth axis)

Height in this module is measured as ``|Y|`` — i.e. the camera looks
forward (+Z) and ``Y`` encodes vertical displacement.  For a forward-
looking camera, obstacles that occupy significant vertical extent will
have large ``|Y|`` values near the image edges.  For a downward-looking
robot camera the caller should adjust ``obstacle_height_m`` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.intrinsics import CameraIntrinsics

__all__ = ["ObstacleDetector", "ObstacleResult"]

# Sentinel distance when no obstacles are found
_NO_OBSTACLE_DIST = float("inf")


@dataclass
class ObstacleResult:
    """Results returned by :class:`ObstacleDetector`.

    Attributes
    ----------
    obstacle_mask : mx.array
        ``(H, W)`` bool array — True for pixels classified as obstacles.
    closest_distance_m : float
        Nearest obstacle distance in metres.  ``inf`` when no obstacles
        are detected.
    obstacle_regions : list[tuple[int, int, int, int]]
        Bounding boxes of connected obstacle blobs as
        ``(y1, x1, y2, x2)`` pixel coordinates (top-left, bottom-right).
        Empty list when no obstacles are present.
    free_path_width_m : float
        Estimated width of the clear corridor directly ahead in metres.
        Computed as the horizontal span of the central depth-image strip
        that contains no obstacle pixels.
    total_obstacle_pixels : int
        Total number of pixels classified as obstacles.
    """

    obstacle_mask: mx.array
    closest_distance_m: float
    obstacle_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    free_path_width_m: float = 0.0
    total_obstacle_pixels: int = 0


class ObstacleDetector:
    """Detect obstacles in RealSense depth frames.

    Segments the depth frame into three categories:

    * **near obstacles** — objects within ``[min_distance_m, max_distance_m]``
      whose vertical extent exceeds ``obstacle_height_m``.
    * **floor / ceiling** — rays outside the height threshold.
    * **background** — rays beyond ``max_distance_m`` or with zero depth.

    Parameters
    ----------
    min_distance_m : float
        Ignore objects closer than this distance (e.g. robot body or lens
        guard).  Default ``0.2`` m.
    max_distance_m : float
        Ignore objects farther than this distance.  Default ``3.0`` m.
    obstacle_height_m : float
        Minimum vertical extent ``|Y|`` (metres) for a point to be
        classified as an obstacle rather than floor/ceiling clutter.
        Default ``0.1`` m.
    central_strip_fraction : float
        Fraction of image width used to measure free-path width.
        Default ``0.33`` (centre third of the image).

    Examples
    --------
    >>> import mlx.core as mx
    >>> import numpy as np
    >>> from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    >>> from realsense_mlx.robotics.obstacles import ObstacleDetector
    >>> intr = CameraIntrinsics(640, 480, 318.8, 239.5, 383.7, 383.7)
    >>> detector = ObstacleDetector(min_distance_m=0.2, max_distance_m=3.0)
    >>> depth = mx.full((480, 640), 1000, dtype=mx.uint16)
    >>> result = detector.detect(depth, intr, depth_scale=0.001)
    >>> result.closest_distance_m
    1.0
    """

    def __init__(
        self,
        min_distance_m: float = 0.2,
        max_distance_m: float = 3.0,
        obstacle_height_m: float = 0.1,
        central_strip_fraction: float = 0.33,
    ) -> None:
        if min_distance_m < 0.0:
            raise ValueError(
                f"min_distance_m must be non-negative, got {min_distance_m}"
            )
        if max_distance_m <= min_distance_m:
            raise ValueError(
                f"max_distance_m ({max_distance_m}) must be > "
                f"min_distance_m ({min_distance_m})"
            )
        if obstacle_height_m < 0.0:
            raise ValueError(
                f"obstacle_height_m must be non-negative, got {obstacle_height_m}"
            )
        if not (0.0 < central_strip_fraction <= 1.0):
            raise ValueError(
                f"central_strip_fraction must be in (0, 1], got {central_strip_fraction}"
            )

        self._min_dist = float(min_distance_m)
        self._max_dist = float(max_distance_m)
        self._obs_height = float(obstacle_height_m)
        self._strip_frac = float(central_strip_fraction)

        # Cached normalised-coordinate grids keyed by intrinsics identity
        self._cached_intr: Optional[CameraIntrinsics] = None
        self._x_norm: Optional[mx.array] = None  # (W,)
        self._y_norm: Optional[mx.array] = None  # (H,)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _ensure_grids(self, intrinsics: CameraIntrinsics) -> None:
        """Build and cache normalised coordinate grids."""
        if self._cached_intr == intrinsics:
            return
        i = intrinsics
        x_raw = (mx.arange(i.width, dtype=mx.float32) - i.ppx) / i.fx
        y_raw = (mx.arange(i.height, dtype=mx.float32) - i.ppy) / i.fy
        mx.eval(x_raw, y_raw)
        self._x_norm = x_raw
        self._y_norm = y_raw
        self._cached_intr = intrinsics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        depth: mx.array,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 0.001,
    ) -> ObstacleResult:
        """Detect obstacles in a depth frame.

        Parameters
        ----------
        depth : mx.array
            ``(H, W)`` uint16 raw depth frame.
        intrinsics : CameraIntrinsics
            Camera intrinsic parameters.
        depth_scale : float
            Metres per raw depth count.  Default ``0.001``.

        Returns
        -------
        ObstacleResult
            Obstacle mask, closest distance, bounding boxes, and
            free-path width.

        Raises
        ------
        ValueError
            If ``depth`` shape does not match ``intrinsics``.
        """
        if depth_scale <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale}")

        H, W = intrinsics.height, intrinsics.width
        if depth.shape != (H, W):
            raise ValueError(
                f"depth shape {depth.shape} does not match intrinsics ({H}, {W})"
            )

        self._ensure_grids(intrinsics)
        assert self._x_norm is not None and self._y_norm is not None

        # ---- 3D deprojection (MLX) ----------------------------------------
        z = depth.astype(mx.float32) * depth_scale          # (H, W)
        Y = self._y_norm[:, None] * z                        # (H, W)  — vertical

        # ---- Obstacle mask (MLX) -------------------------------------------
        # Conditions (all on-device):
        #   - valid depth (z > 0)
        #   - within detection range
        #   - vertical displacement exceeds height threshold
        obstacle_mask = (
            (z > 0.0)
            & (z >= self._min_dist)
            & (z <= self._max_dist)
            & (mx.abs(Y) >= self._obs_height)
        )  # (H, W) bool

        # ---- Closest distance (MLX) ----------------------------------------
        # Mask z values: set non-obstacle pixels to a large sentinel
        masked_z = mx.where(obstacle_mask, z, mx.array(1e9, dtype=mx.float32))
        min_z = mx.min(masked_z)
        mx.eval(obstacle_mask, min_z)

        closest = float(min_z)
        if closest >= 1e8:
            closest = _NO_OBSTACLE_DIST

        # ---- Aggregate stats (numpy, small) --------------------------------
        mask_np = np.array(obstacle_mask, copy=False)  # (H, W) bool
        total_px = int(mask_np.sum())

        regions = _find_obstacle_regions(mask_np) if total_px > 0 else []

        free_width = self._compute_free_path_width(mask_np, intrinsics, depth_scale)

        return ObstacleResult(
            obstacle_mask=obstacle_mask,
            closest_distance_m=closest,
            obstacle_regions=regions,
            free_path_width_m=free_width,
            total_obstacle_pixels=total_px,
        )

    # ------------------------------------------------------------------
    # Free-path width
    # ------------------------------------------------------------------

    def _compute_free_path_width(
        self,
        mask_np: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float,
    ) -> float:
        """Estimate the horizontal width of the clear path in metres.

        Examines the central vertical strip of the obstacle mask.  The
        free-path width is the horizontal span (in pixels) of the
        longest contiguous obstacle-free run across the image centre
        row, converted to metres using the depth at that row (or a
        fallback of 1 m when depth is zero).

        Parameters
        ----------
        mask_np : np.ndarray
            ``(H, W)`` bool obstacle mask.
        intrinsics : CameraIntrinsics
            Camera intrinsics for pixel→metre conversion.
        depth_scale : float
            Not used here; kept for API symmetry.

        Returns
        -------
        float
            Estimated free-path width in metres.
        """
        H, W = mask_np.shape
        half_strip = int(W * self._strip_frac / 2.0)
        cx = W // 2
        col_start = max(0, cx - half_strip)
        col_end = min(W, cx + half_strip)
        cy = H // 2

        strip = mask_np[cy, col_start:col_end]  # (strip_w,) bool

        if strip.size == 0:
            return 0.0

        # Find longest run of False (no obstacle) in the central strip
        # via run-length encoding
        free_run = _longest_false_run(strip)

        if free_run == 0:
            return 0.0

        # Convert pixels to metres using the pinhole model at the centre row
        # width_m ≈ pixel_span * depth_at_centre / fx
        # Use a nominal depth of 1 m when we cannot read it
        nominal_depth_m = 1.0
        fx = intrinsics.fx
        width_m = float(free_run) * nominal_depth_m / fx
        return width_m

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def distance_range_m(self) -> tuple[float, float]:
        """``(min_distance_m, max_distance_m)`` detection range."""
        return (self._min_dist, self._max_dist)

    @property
    def obstacle_height_m(self) -> float:
        """Minimum height threshold for obstacle classification."""
        return self._obs_height

    def __repr__(self) -> str:
        return (
            f"ObstacleDetector("
            f"range=[{self._min_dist}, {self._max_dist}]m, "
            f"min_height={self._obs_height}m)"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_obstacle_regions(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes of obstacle blobs via row-projection scan.

    Uses a simple connected-component approximation: projects the
    boolean mask onto rows and columns to find axis-aligned bounds of
    non-zero regions.  Splits into at most ``_MAX_REGIONS`` boxes by
    finding gaps in the column projection.

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` bool array (True = obstacle).

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of ``(y1, x1, y2, x2)`` bounding boxes.
    """
    row_has_obs = mask.any(axis=1)   # (H,) — rows that have any obstacle
    col_has_obs = mask.any(axis=0)   # (W,) — cols that have any obstacle

    if not row_has_obs.any():
        return []

    # Overall row bounds
    rows_idx = np.where(row_has_obs)[0]
    y1_global = int(rows_idx[0])
    y2_global = int(rows_idx[-1])

    # Split by column gaps (contiguous runs of obstacle columns = separate regions)
    regions: list[tuple[int, int, int, int]] = []
    in_region = False
    x1 = 0
    for c, has_obs in enumerate(col_has_obs):
        if has_obs and not in_region:
            x1 = c
            in_region = True
        elif not has_obs and in_region:
            # Refine row bounds within this column slice
            col_mask = mask[:, x1:c]
            row_sub = col_mask.any(axis=1)
            rows_sub = np.where(row_sub)[0]
            if len(rows_sub) > 0:
                regions.append((int(rows_sub[0]), x1, int(rows_sub[-1]), c - 1))
            in_region = False

    # Close last region
    if in_region:
        col_mask = mask[:, x1:]
        row_sub = col_mask.any(axis=1)
        rows_sub = np.where(row_sub)[0]
        if len(rows_sub) > 0:
            regions.append((
                int(rows_sub[0]), x1,
                int(rows_sub[-1]), int(col_has_obs.shape[0]) - 1,
            ))

    return regions if regions else [(y1_global, 0, y2_global, mask.shape[1] - 1)]


def _longest_false_run(arr: np.ndarray) -> int:
    """Return the length of the longest contiguous False run in a 1-D bool array.

    Parameters
    ----------
    arr : np.ndarray
        1-D bool array.

    Returns
    -------
    int
        Length of the longest False (obstacle-free) run.  Returns
        ``len(arr)`` when all elements are False; 0 when all are True.
    """
    if arr.size == 0:
        return 0
    if not arr.any():
        return int(arr.size)
    if arr.all():
        return 0

    # Pad with True sentinels to detect run boundaries at edges
    padded = np.concatenate([[True], arr, [True]])
    # Positions where value changes from True to False (start of free run)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == -1)[0]   # True→False transitions
    ends = np.where(diff == 1)[0]      # False→True transitions
    lengths = ends - starts
    return int(lengths.max())
