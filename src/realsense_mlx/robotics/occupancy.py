"""2D occupancy grid from depth frames for robot navigation.

Converts depth images into 2D bird's-eye-view occupancy grids suitable
for path planning (A*, RRT, etc.) and obstacle avoidance.

The grid represents a top-down view of the scene where each cell is:
  0 = unknown, 1 = free space, 2 = occupied

Algorithm
---------
1. Deproject depth to 3D points (MLX, vectorised over all pixels).
2. Mask out zero-depth (invalid) pixels.
3. Filter by height band — keep points whose Y coordinate falls in
   ``[min_height_m, max_height_m]``.  Points outside the band are used
   to infer free space (they come from the floor or ceiling, not obstacles).
4. Project surviving points onto the XZ plane (bird's-eye view).
5. Convert XZ coordinates to integer grid-cell indices.
6. Count points per cell with ``np.bincount`` (O(N), cache-friendly).
7. Threshold: cells with ``count >= min_points_per_cell`` are marked
   **occupied** (2); cells that received *any* valid depth ray but no
   obstacle point are marked **free** (1); remaining cells stay
   **unknown** (0).

Coordinate conventions
-----------------------
Follows the RealSense camera frame::

    +X → right
    +Y → down (positive Y means below the camera)
    +Z → into the scene (forward)

The bird's-eye grid maps::

    grid row  ↔  Z axis (forward, row 0 = closest to camera)
    grid col  ↔  X axis (lateral,  col 0 = leftmost)

Grid origin is placed so that the camera sits at
``(col = grid_cols/2, row = 0)`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.intrinsics import CameraIntrinsics

__all__ = ["OccupancyGridGenerator", "OccupancyGrid"]

# Cell value constants
UNKNOWN: int = 0
FREE: int = 1
OCCUPIED: int = 2


@dataclass(frozen=True)
class OccupancyGrid:
    """Container for a generated occupancy grid and its metadata.

    Attributes
    ----------
    grid : mx.array
        ``(rows, cols)`` uint8 array. Values: 0=unknown, 1=free, 2=occupied.
    cell_size_m : float
        Physical size of each cell in metres.
    origin_x_m : float
        World X coordinate (metres) corresponding to grid column 0.
    origin_z_m : float
        World Z coordinate (metres) corresponding to grid row 0.
    n_occupied : int
        Total number of occupied cells.
    n_free : int
        Total number of free cells.
    """

    grid: mx.array
    cell_size_m: float
    origin_x_m: float
    origin_z_m: float
    n_occupied: int
    n_free: int

    @property
    def rows(self) -> int:
        """Number of grid rows (Z axis)."""
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        """Number of grid columns (X axis)."""
        return int(self.grid.shape[1])


class OccupancyGridGenerator:
    """Generate 2D occupancy grids from RealSense depth frames.

    All heavy arithmetic (deprojection, masking, grid-coordinate
    computation, final thresholding) runs on the MLX device.  The only
    step that falls back to NumPy is the scatter-count (``np.bincount``),
    which is O(N) and avoids a prohibitive ``(N, G)`` MLX matrix when
    ``N`` (valid depth pixels) and ``G`` (grid cells) are both large.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Grid dimensions ``(rows, cols)``.  Default ``(200, 200)``.
    cell_size_m : float
        Physical size of each grid cell in metres.  Default ``0.05`` (5 cm).
    min_height_m : float
        Minimum Y value (metres) to classify a point as an obstacle.
        Points with ``Y < min_height_m`` are considered floor/ground.
        Default ``0.1``.
    max_height_m : float
        Maximum Y value (metres) for obstacle classification.
        Points above this are considered ceiling/out-of-range.
        Default ``1.5``.
    min_points_per_cell : int
        Minimum number of 3D points that must project into a cell before
        it is marked occupied.  Acts as a noise filter.  Default ``3``.

    Examples
    --------
    >>> import mlx.core as mx
    >>> import numpy as np
    >>> from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    >>> from realsense_mlx.robotics.occupancy import OccupancyGridGenerator
    >>> intr = CameraIntrinsics(640, 480, 318.8, 239.5, 383.7, 383.7)
    >>> gen = OccupancyGridGenerator(grid_size=(200, 200), cell_size_m=0.05)
    >>> depth = mx.full((480, 640), 1000, dtype=mx.uint16)
    >>> result = gen.generate(depth, intr, depth_scale=0.001)
    >>> result.grid.shape
    (200, 200)
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (200, 200),
        cell_size_m: float = 0.05,
        min_height_m: float = 0.1,
        max_height_m: float = 1.5,
        min_points_per_cell: int = 3,
    ) -> None:
        if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(
                f"grid_size must be a pair of positive ints, got {grid_size}"
            )
        if cell_size_m <= 0.0:
            raise ValueError(
                f"cell_size_m must be positive, got {cell_size_m}"
            )
        if min_height_m >= max_height_m:
            raise ValueError(
                f"min_height_m ({min_height_m}) must be < max_height_m ({max_height_m})"
            )
        if min_points_per_cell < 1:
            raise ValueError(
                f"min_points_per_cell must be >= 1, got {min_points_per_cell}"
            )

        self._grid_rows, self._grid_cols = int(grid_size[0]), int(grid_size[1])
        self._cell_size_m = float(cell_size_m)
        self._min_height_m = float(min_height_m)
        self._max_height_m = float(max_height_m)
        self._min_points = int(min_points_per_cell)

        # Cached coordinate grids — recomputed when intrinsics change
        self._cached_intr: Optional[CameraIntrinsics] = None
        self._x_norm: Optional[mx.array] = None  # (W,) normalised x
        self._y_norm: Optional[mx.array] = None  # (H,) normalised y

        # Precompute grid geometry constants
        # Camera sits at (col = cols/2, row = 0) — origin is left edge of row 0
        self._origin_x_m: float = -(self._grid_cols / 2.0) * self._cell_size_m
        self._origin_z_m: float = 0.0

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _ensure_grids(self, intrinsics: CameraIntrinsics) -> None:
        """Rebuild normalised-coordinate grids when intrinsics change."""
        if self._cached_intr == intrinsics:
            return
        i = intrinsics
        x_raw = (mx.arange(i.width, dtype=mx.float32) - i.ppx) / i.fx   # (W,)
        y_raw = (mx.arange(i.height, dtype=mx.float32) - i.ppy) / i.fy  # (H,)
        mx.eval(x_raw, y_raw)
        self._x_norm = x_raw
        self._y_norm = y_raw
        self._cached_intr = intrinsics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        depth: mx.array,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 0.001,
    ) -> OccupancyGrid:
        """Convert a depth frame to a 2D occupancy grid.

        Parameters
        ----------
        depth : mx.array
            ``(H, W)`` uint16 raw depth values.
        intrinsics : CameraIntrinsics
            Camera intrinsic parameters (focal length, principal point).
        depth_scale : float
            Metres per raw depth count.  Typical D-series value: ``0.001``.

        Returns
        -------
        OccupancyGrid
            Container with the ``(rows, cols)`` uint8 grid and metadata.

        Raises
        ------
        ValueError
            If ``depth`` dimensions do not match ``intrinsics``.
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

        # ---- Step 1: Deproject to 3D (MLX) --------------------------------
        # z in metres: (H, W) float32
        z = depth.astype(mx.float32) * depth_scale

        # X = x_norm * Z,  Y = y_norm * Z
        # x_norm: (W,), broadcast over rows → (H, W)
        # y_norm: (H,), broadcast over cols → (H, W)
        X = self._x_norm[None, :] * z   # (H, W)
        Y = self._y_norm[:, None] * z   # (H, W)
        # Z is already computed above

        # ---- Step 2: Valid-depth mask (MLX) --------------------------------
        valid_depth = z > 0.0  # (H, W) bool

        # ---- Step 3: Height-band filter (MLX) ------------------------------
        # In RealSense coords +Y is down; depth cameras looking forward will
        # see the floor at positive Y, ceiling at negative Y.  We accept
        # user-defined [min, max] so the caller can orient as needed.
        obstacle_mask = (
            valid_depth
            & (Y >= self._min_height_m)
            & (Y <= self._max_height_m)
        )  # (H, W) bool

        # ---- Step 4: Grid coordinate computation (MLX) ---------------------
        # XZ → (grid_col, grid_row)
        gc = mx.floor(
            (X - self._origin_x_m) / self._cell_size_m
        ).astype(mx.int32)  # (H, W)
        gr = mx.floor(
            (z - self._origin_z_m) / self._cell_size_m
        ).astype(mx.int32)  # (H, W)

        # Bounds check
        in_bounds = (
            (gc >= 0) & (gc < self._grid_cols)
            & (gr >= 0) & (gr < self._grid_rows)
        )  # (H, W) bool

        # Masks used for counting
        obs_valid = obstacle_mask & in_bounds        # obstacle pixels in grid
        free_valid = valid_depth & in_bounds & ~obstacle_mask  # floor/ceiling rays

        # ---- Step 5: Scatter-count (numpy bincount — O(N), cache-friendly) ----
        # MLX materialise before numpy conversion
        mx.eval(gc, gr, obs_valid, free_valid)

        G = self._grid_rows * self._grid_cols

        gc_np = np.array(gc, copy=False).ravel()   # (H*W,)
        gr_np = np.array(gr, copy=False).ravel()   # (H*W,)
        obs_np = np.array(obs_valid, copy=False).ravel().astype(bool)
        free_np = np.array(free_valid, copy=False).ravel().astype(bool)

        flat_obs = gr_np[obs_np] * self._grid_cols + gc_np[obs_np]
        flat_free = gr_np[free_np] * self._grid_cols + gc_np[free_np]

        obs_counts = np.bincount(flat_obs, minlength=G).reshape(
            self._grid_rows, self._grid_cols
        )
        free_counts = np.bincount(flat_free, minlength=G).reshape(
            self._grid_rows, self._grid_cols
        )

        # ---- Step 6: Threshold → occupancy values (MLX) -------------------
        obs_counts_mx = mx.array(obs_counts.astype(np.int32))
        free_counts_mx = mx.array(free_counts.astype(np.int32))

        # Build grid: start unknown, mark free, then overwrite with occupied
        grid = mx.zeros((self._grid_rows, self._grid_cols), dtype=mx.uint8)
        grid = mx.where(free_counts_mx > 0, mx.array(FREE, dtype=mx.uint8), grid)
        grid = mx.where(
            obs_counts_mx >= self._min_points,
            mx.array(OCCUPIED, dtype=mx.uint8),
            grid,
        )
        mx.eval(grid)

        n_occupied = int(np.sum(obs_counts >= self._min_points))
        n_free = int(np.sum((free_counts > 0) & (obs_counts < self._min_points)))

        return OccupancyGrid(
            grid=grid,
            cell_size_m=self._cell_size_m,
            origin_x_m=self._origin_x_m,
            origin_z_m=self._origin_z_m,
            n_occupied=n_occupied,
            n_free=n_free,
        )

    def generate_with_visualization(
        self,
        depth: mx.array,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 0.001,
    ) -> tuple[OccupancyGrid, np.ndarray]:
        """Generate occupancy grid and an RGB visualization image.

        Parameters
        ----------
        depth : mx.array
            ``(H, W)`` uint16 depth frame.
        intrinsics : CameraIntrinsics
            Camera intrinsics.
        depth_scale : float
            Metres per raw depth count.

        Returns
        -------
        result : OccupancyGrid
            The computed occupancy grid.
        vis : np.ndarray
            ``(rows, cols, 3)`` uint8 RGB image.
            Black = unknown, white = free, red = occupied.
        """
        result = self.generate(depth, intrinsics, depth_scale)
        mx.eval(result.grid)
        grid_np = np.array(result.grid, copy=False)

        vis = np.zeros((self._grid_rows, self._grid_cols, 3), dtype=np.uint8)
        vis[grid_np == FREE] = [255, 255, 255]    # white — free
        vis[grid_np == OCCUPIED] = [220, 50, 50]  # red — occupied
        # unknown stays black (0, 0, 0)

        return result, vis

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid_size(self) -> tuple[int, int]:
        """Grid dimensions ``(rows, cols)``."""
        return (self._grid_rows, self._grid_cols)

    @property
    def cell_size_m(self) -> float:
        """Physical size of each cell in metres."""
        return self._cell_size_m

    @property
    def height_band(self) -> tuple[float, float]:
        """``(min_height_m, max_height_m)`` obstacle height filter."""
        return (self._min_height_m, self._max_height_m)

    def __repr__(self) -> str:
        return (
            f"OccupancyGridGenerator("
            f"grid={self._grid_rows}x{self._grid_cols}, "
            f"cell={self._cell_size_m*100:.0f}cm, "
            f"height=[{self._min_height_m}, {self._max_height_m}]m, "
            f"min_pts={self._min_points})"
        )
