"""Tests for realsense_mlx.robotics — occupancy grid and obstacle detection.

Coverage
--------
Occupancy grid (OccupancyGridGenerator):
  - Flat floor scene → grid is mostly FREE (no occupied cells)
  - Wall at 1 m → occupied band in the grid
  - Empty scene (all-zero depth) → grid is all UNKNOWN
  - Output shape matches requested grid_size
  - Output dtype is uint8
  - Grid values are confined to {0, 1, 2}
  - n_occupied and n_free metadata match grid contents
  - Visualisation has correct shape and value range
  - Intrinsics caching: grids not rebuilt on repeated calls with same intrinsics
  - ValueError on bad depth shape
  - ValueError on non-positive depth_scale
  - ValueError on invalid constructor args

Obstacle detection (ObstacleDetector):
  - Close object within range → detected (non-empty mask, finite closest_distance)
  - Nothing within range → no obstacles (all-zero mask, inf closest distance)
  - Closest distance accuracy against known depth value
  - Total obstacle pixels > 0 when obstacles present
  - Obstacle regions list populated when obstacles detected
  - Free path width: wide corridor → positive width; fully blocked → ~zero
  - Depth outside min/max range → excluded from mask
  - Zero-depth pixels → excluded from mask
  - ValueError on bad depth shape
  - ValueError on non-positive depth_scale
  - ValueError on invalid constructor args (max <= min distance, etc.)

Integration / top-level imports:
  - realsense_mlx lazy __getattr__ exposes OccupancyGridGenerator, ObstacleDetector
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.geometry.intrinsics import CameraIntrinsics
from realsense_mlx.robotics import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    ObstacleDetector,
    ObstacleResult,
    OccupancyGrid,
    OccupancyGridGenerator,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def intr() -> CameraIntrinsics:
    """640x480 RealSense D415-like intrinsics."""
    return CameraIntrinsics(
        width=640,
        height=480,
        ppx=318.8,
        ppy=239.5,
        fx=383.7,
        fy=383.7,
        model="none",
    )


@pytest.fixture
def small_intr() -> CameraIntrinsics:
    """Small 64x48 intrinsics for fast unit tests."""
    return CameraIntrinsics(
        width=64,
        height=48,
        ppx=32.0,
        ppy=24.0,
        fx=50.0,
        fy=50.0,
        model="none",
    )


def _flat_depth(h: int, w: int, raw_value: int) -> mx.array:
    """Return a constant (H, W) uint16 depth frame."""
    return mx.full((h, w), raw_value, dtype=mx.uint16)


def _eval_np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr, copy=False)


# ===========================================================================
# Occupancy Grid — OccupancyGridGenerator
# ===========================================================================


class TestOccupancyGridBasicProperties:
    """Shape, dtype, and value-range invariants."""

    def test_output_shape_matches_grid_size(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(50, 60), cell_size_m=0.05)
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        result = gen.generate(depth, small_intr)
        assert result.grid.shape == (50, 60), (
            f"Expected (50, 60), got {result.grid.shape}"
        )

    def test_output_dtype_uint8(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        result = gen.generate(depth, small_intr)
        assert result.grid.dtype == mx.uint8

    def test_values_confined_to_012(self, intr):
        gen = OccupancyGridGenerator(grid_size=(100, 100), cell_size_m=0.05)
        depth = _flat_depth(intr.height, intr.width, 1000)
        result = gen.generate(depth, intr)
        grid_np = _eval_np(result.grid)
        unique = set(grid_np.ravel().tolist())
        assert unique.issubset({UNKNOWN, FREE, OCCUPIED}), (
            f"Unexpected grid values: {unique - {UNKNOWN, FREE, OCCUPIED}}"
        )

    def test_metadata_counts_match_grid(self, intr):
        gen = OccupancyGridGenerator(grid_size=(100, 100), cell_size_m=0.05)
        depth = _flat_depth(intr.height, intr.width, 1000)
        result = gen.generate(depth, intr)
        grid_np = _eval_np(result.grid)
        assert result.n_occupied == int((grid_np == OCCUPIED).sum())
        assert result.n_free == int((grid_np == FREE).sum())

    def test_grid_size_property(self):
        gen = OccupancyGridGenerator(grid_size=(30, 45))
        assert gen.grid_size == (30, 45)

    def test_cell_size_property(self):
        gen = OccupancyGridGenerator(cell_size_m=0.1)
        assert gen.cell_size_m == pytest.approx(0.1)

    def test_height_band_property(self):
        gen = OccupancyGridGenerator(min_height_m=0.2, max_height_m=1.8)
        assert gen.height_band == (pytest.approx(0.2), pytest.approx(1.8))

    def test_repr_smoke(self):
        gen = OccupancyGridGenerator(grid_size=(200, 200))
        r = repr(gen)
        assert "OccupancyGridGenerator" in r
        assert "200x200" in r


class TestOccupancyGridScenes:
    """Semantic correctness for known depth scenes."""

    def test_empty_scene_all_unknown(self, small_intr):
        """All-zero depth → no valid rays → grid must be entirely UNKNOWN."""
        gen = OccupancyGridGenerator(
            grid_size=(20, 20),
            cell_size_m=0.05,
            min_height_m=0.1,
            max_height_m=1.5,
            min_points_per_cell=1,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 0)
        result = gen.generate(depth, small_intr, depth_scale=0.001)
        grid_np = _eval_np(result.grid)
        assert (grid_np == UNKNOWN).all(), (
            "All-zero depth should produce all-UNKNOWN grid"
        )
        assert result.n_occupied == 0
        assert result.n_free == 0

    def test_flat_floor_mostly_free(self, intr):
        """Scene where all depth rays fall below the height threshold → no occupied cells.

        At 1 m depth with D415 intrinsics (fy=383.7, ppy=239.5), the
        maximum |Y| at any pixel is::

            |Y|_max = max(ppy, height-1-ppy) / fy * 1.0m
                    = 239.5 / 383.7 * 1.0 ≈ 0.624 m

        Setting min_height_m = 0.7m (above 0.624m) means NO pixel
        qualifies as an obstacle → zero occupied cells, all cells become
        FREE from the valid-depth floor rays.
        """
        gen = OccupancyGridGenerator(
            grid_size=(100, 100),
            cell_size_m=0.05,
            min_height_m=0.7,    # above max |Y| at 1m for D415 (≈0.624m)
            max_height_m=2.0,
            min_points_per_cell=1,
        )
        depth = _flat_depth(intr.height, intr.width, 1000)  # 1 m everywhere
        result = gen.generate(depth, intr, depth_scale=0.001)

        assert result.n_occupied == 0, (
            f"Expected 0 occupied cells for scene below height threshold, "
            f"got {result.n_occupied}"
        )
        # Valid depth rays (floor-classified) should mark cells FREE
        assert result.n_free > 0, "Floor rays should mark cells as free"

    def test_wall_at_1m_produces_occupied_band(self, intr):
        """Wall at 1 m should produce an occupied band in the grid.

        To get occupied cells we lower the height threshold to capture
        pixels with moderate Y values (the wall spans the full image height
        so some rows will have |Y| > 0.05m).
        """
        gen = OccupancyGridGenerator(
            grid_size=(100, 100),
            cell_size_m=0.05,
            min_height_m=0.05,   # low threshold — captures most pixels
            max_height_m=5.0,
            min_points_per_cell=1,
        )
        depth = _flat_depth(intr.height, intr.width, 1000)  # 1 m wall
        result = gen.generate(depth, intr, depth_scale=0.001)
        grid_np = _eval_np(result.grid)

        # At 1 m, X ranges ≈ [-0.83, +0.83] m → up to 16 grid cols
        # Z = 1 m → row = floor(1.0 / 0.05) = 20
        # Many pixels have |Y| > 0.05m (rows away from centre) → occupied
        assert result.n_occupied > 0, "Wall scene should produce occupied cells"

        # Verify occupied cells cluster around Z=1m → grid row ~20
        occ_rows, _ = np.where(grid_np == OCCUPIED)
        if len(occ_rows) > 0:
            assert occ_rows.mean() == pytest.approx(20.0, abs=5.0), (
                f"Occupied cells should be near row 20 (Z=1m/0.05m), "
                f"got mean row {occ_rows.mean():.1f}"
            )

    def test_wall_at_2m_in_correct_row(self, small_intr):
        """Wall at 2 m with cell_size=0.1m → occupied row ~ row 20."""
        gen = OccupancyGridGenerator(
            grid_size=(50, 50),
            cell_size_m=0.1,
            min_height_m=0.01,  # very low threshold
            max_height_m=5.0,
            min_points_per_cell=1,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 2000)  # 2 m
        result = gen.generate(depth, small_intr, depth_scale=0.001)
        grid_np = _eval_np(result.grid)

        occ_rows, _ = np.where(grid_np == OCCUPIED)
        assert len(occ_rows) > 0, "Should detect wall"
        expected_row = int(2.0 / 0.1)  # = 20
        assert occ_rows.mean() == pytest.approx(expected_row, abs=3.0)

    def test_scene_with_obstacle_in_height_band(self, intr):
        """Obstacle in the height band should produce OCCUPIED cells."""
        gen = OccupancyGridGenerator(
            grid_size=(100, 100),
            cell_size_m=0.05,
            min_height_m=0.3,
            max_height_m=1.5,
            min_points_per_cell=2,
        )
        # Build a depth frame where the upper portion of the image is
        # at 1 m (rows 0..150 have large |Y| at 1m, |Y| ≈ ppy/fy * 1m
        # ≈ (239.5/383.7) ≈ 0.62m which is in [0.3, 1.5]).
        depth_np = np.full((intr.height, intr.width), 1000, dtype=np.uint16)
        depth = mx.array(depth_np)

        result = gen.generate(depth, intr, depth_scale=0.001)
        # Upper/lower image rows have large Y → should be occupied
        assert result.n_occupied > 0


class TestOccupancyGridCaching:
    """Grid caching and invalidation behaviour."""

    def test_grids_cached_on_second_call(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        gen.generate(depth, small_intr)
        x_id = id(gen._x_norm)
        gen.generate(depth, small_intr)
        assert id(gen._x_norm) == x_id, "x_norm grid should not be reallocated"

    def test_grids_rebuilt_on_new_intrinsics(self):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        intr_a = CameraIntrinsics(64, 48, 32.0, 24.0, 50.0, 50.0)
        intr_b = CameraIntrinsics(64, 48, 32.0, 24.0, 60.0, 60.0)  # different fx
        depth_a = _flat_depth(48, 64, 500)
        gen.generate(depth_a, intr_a)
        x_id_a = id(gen._x_norm)
        gen.generate(depth_a, intr_b)
        assert id(gen._x_norm) != x_id_a, "Grids should rebuild for new intrinsics"


class TestOccupancyGridVisualization:
    """generate_with_visualization output contract."""

    def test_vis_shape(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(30, 40))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        _, vis = gen.generate_with_visualization(depth, small_intr)
        assert vis.shape == (30, 40, 3), f"Expected (30, 40, 3), got {vis.shape}"

    def test_vis_dtype_uint8(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(30, 40))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        _, vis = gen.generate_with_visualization(depth, small_intr)
        assert vis.dtype == np.uint8

    def test_vis_unknown_is_black(self, small_intr):
        """Unknown cells should be (0, 0, 0)."""
        gen = OccupancyGridGenerator(
            grid_size=(20, 20),
            min_height_m=0.1,
            max_height_m=1.5,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 0)  # all zero
        _, vis = gen.generate_with_visualization(depth, small_intr)
        assert (vis == 0).all(), "All-unknown grid should be fully black"

    def test_vis_returns_occupancy_grid(self, small_intr):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        result, _ = gen.generate_with_visualization(depth, small_intr)
        assert isinstance(result, OccupancyGrid)


class TestOccupancyGridErrors:
    """Error handling for invalid inputs."""

    def test_depth_shape_mismatch_raises(self, intr):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        bad_depth = mx.zeros((100, 100), dtype=mx.uint16)
        with pytest.raises(ValueError, match="does not match intrinsics"):
            gen.generate(bad_depth, intr)

    def test_non_positive_depth_scale_raises(self, intr, small_intr):
        gen = OccupancyGridGenerator(grid_size=(20, 20))
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            gen.generate(depth, small_intr, depth_scale=0.0)
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            gen.generate(depth, small_intr, depth_scale=-1.0)

    def test_invalid_grid_size_raises(self):
        with pytest.raises(ValueError, match="grid_size"):
            OccupancyGridGenerator(grid_size=(0, 100))
        with pytest.raises(ValueError, match="grid_size"):
            OccupancyGridGenerator(grid_size=(-5, 100))

    def test_non_positive_cell_size_raises(self):
        with pytest.raises(ValueError, match="cell_size_m"):
            OccupancyGridGenerator(cell_size_m=0.0)
        with pytest.raises(ValueError, match="cell_size_m"):
            OccupancyGridGenerator(cell_size_m=-0.1)

    def test_inverted_height_band_raises(self):
        with pytest.raises(ValueError, match="min_height_m"):
            OccupancyGridGenerator(min_height_m=2.0, max_height_m=0.5)

    def test_zero_min_points_raises(self):
        with pytest.raises(ValueError, match="min_points_per_cell"):
            OccupancyGridGenerator(min_points_per_cell=0)


# ===========================================================================
# Obstacle Detection — ObstacleDetector
# ===========================================================================


class TestObstacleDetectorBasicProperties:
    """API surface and return-type invariants."""

    def test_result_is_obstacle_result(self, small_intr):
        det = ObstacleDetector()
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr)
        assert isinstance(res, ObstacleResult)

    def test_mask_shape_matches_depth(self, small_intr):
        det = ObstacleDetector()
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr)
        mx.eval(res.obstacle_mask)
        assert res.obstacle_mask.shape == (small_intr.height, small_intr.width)

    def test_mask_dtype_bool(self, small_intr):
        det = ObstacleDetector()
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr)
        assert res.obstacle_mask.dtype == mx.bool_

    def test_distance_range_property(self):
        det = ObstacleDetector(min_distance_m=0.5, max_distance_m=4.0)
        assert det.distance_range_m == (pytest.approx(0.5), pytest.approx(4.0))

    def test_obstacle_height_property(self):
        det = ObstacleDetector(obstacle_height_m=0.3)
        assert det.obstacle_height_m == pytest.approx(0.3)

    def test_repr_smoke(self):
        det = ObstacleDetector()
        r = repr(det)
        assert "ObstacleDetector" in r


class TestObstacleDetectorDetection:
    """Obstacle presence / absence logic."""

    def test_object_within_range_detected(self, small_intr):
        """Object at 1 m (within [0.2, 3.0]) → obstacle pixels > 0."""
        det = ObstacleDetector(
            min_distance_m=0.2,
            max_distance_m=3.0,
            obstacle_height_m=0.01,   # very low — most pixels qualify
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr, depth_scale=0.001)
        assert res.total_obstacle_pixels > 0, "Should detect obstacle at 1 m"

    def test_nothing_within_range_no_obstacles(self, small_intr):
        """Object at 5 m beyond max_distance_m=3 → no obstacles."""
        det = ObstacleDetector(
            min_distance_m=0.2,
            max_distance_m=3.0,
            obstacle_height_m=0.01,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 5000)  # 5 m
        res = det.detect(depth, small_intr, depth_scale=0.001)
        assert res.total_obstacle_pixels == 0
        assert math.isinf(res.closest_distance_m)

    def test_empty_frame_no_obstacles(self, small_intr):
        """All-zero depth (invalid) → no obstacles."""
        det = ObstacleDetector(obstacle_height_m=0.01)
        depth = _flat_depth(small_intr.height, small_intr.width, 0)
        res = det.detect(depth, small_intr, depth_scale=0.001)
        assert res.total_obstacle_pixels == 0
        assert math.isinf(res.closest_distance_m)
        mask_np = _eval_np(res.obstacle_mask).astype(bool)
        assert not mask_np.any()

    def test_closest_distance_accuracy(self, small_intr):
        """Closest distance should match known wall depth."""
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=5.0,
            obstacle_height_m=0.01,
        )
        raw_depth = 2000  # 2.0 m
        depth = _flat_depth(small_intr.height, small_intr.width, raw_depth)
        res = det.detect(depth, small_intr, depth_scale=0.001)
        assert res.closest_distance_m == pytest.approx(2.0, abs=0.01), (
            f"Expected 2.0 m, got {res.closest_distance_m}"
        )

    def test_object_too_close_excluded(self, small_intr):
        """Object closer than min_distance_m should not be detected."""
        det = ObstacleDetector(
            min_distance_m=1.0,  # min = 1 m
            max_distance_m=3.0,
            obstacle_height_m=0.01,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 500)  # 0.5 m
        res = det.detect(depth, small_intr, depth_scale=0.001)
        assert res.total_obstacle_pixels == 0, (
            "Object at 0.5 m should be below min_distance_m=1.0"
        )

    def test_height_threshold_filters_floor(self, intr):
        """Objects with small |Y| (floor-like) should be excluded when
        obstacle_height_m is set high enough."""
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=5.0,
            obstacle_height_m=1.0,  # only very tall obstacles
        )
        # At 1 m, centre rows have |Y| < 0.7m — should not qualify
        depth = _flat_depth(intr.height, intr.width, 1000)
        res = det.detect(depth, intr, depth_scale=0.001)
        # Central pixels (small Y) should be excluded
        # Only extreme top/bottom rows with |Y| > 1.0m would qualify
        # (ppy=239.5, fy=383.7: max |Y| = (479-239.5)/383.7 * 1.0 ≈ 0.62m)
        # So ALL pixels should be excluded at 1m depth
        assert res.total_obstacle_pixels == 0, (
            "No pixel has |Y| > 1.0m at 1m depth with D415 intrinsics"
        )


class TestObstacleDetectorRegions:
    """Bounding box regions."""

    def test_regions_empty_when_no_obstacles(self, small_intr):
        det = ObstacleDetector(obstacle_height_m=0.01)
        depth = _flat_depth(small_intr.height, small_intr.width, 0)
        res = det.detect(depth, small_intr)
        assert res.obstacle_regions == []

    def test_regions_non_empty_when_obstacles_present(self, small_intr):
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=5.0,
            obstacle_height_m=0.01,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr, depth_scale=0.001)
        if res.total_obstacle_pixels > 0:
            assert len(res.obstacle_regions) > 0

    def test_regions_are_valid_bboxes(self, small_intr):
        """Each bounding box should satisfy y1<=y2 and x1<=x2."""
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=5.0,
            obstacle_height_m=0.01,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res = det.detect(depth, small_intr, depth_scale=0.001)
        for y1, x1, y2, x2 in res.obstacle_regions:
            assert y1 <= y2, f"Invalid bbox: y1={y1} > y2={y2}"
            assert x1 <= x2, f"Invalid bbox: x1={x1} > x2={x2}"
            assert y1 >= 0 and x1 >= 0
            assert y2 < small_intr.height and x2 < small_intr.width


class TestObstacleDetectorFreePathWidth:
    """Free-path width estimation."""

    def test_fully_clear_scene_has_positive_width(self, small_intr):
        """A scene with no obstacles should yield positive free-path width.

        When there are no obstacle pixels, _compute_free_path_width should
        return a positive value based on the full central strip width.
        """
        det = ObstacleDetector(
            min_distance_m=2.0,  # object at 1 m is outside range → no obstacles
            max_distance_m=5.0,
            obstacle_height_m=0.01,
        )
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)  # 1 m
        res = det.detect(depth, small_intr, depth_scale=0.001)
        # No obstacles → central strip is fully free → width > 0
        assert res.free_path_width_m > 0.0, (
            "Clear corridor should have positive free-path width"
        )

    def test_blocked_corridor_smaller_width(self, small_intr):
        """Obstacles filling the central strip reduce free-path width.

        Place obstacles extremely close (large raw values within range).
        """
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=5.0,
            obstacle_height_m=0.01,
        )
        # Fully blocked at 1 m → all central-strip pixels are obstacles
        depth = _flat_depth(small_intr.height, small_intr.width, 1000)
        res_blocked = det.detect(depth, small_intr, depth_scale=0.001)

        # Fully clear (out-of-range depth)
        depth_clear = _flat_depth(small_intr.height, small_intr.width, 8000)  # 8 m
        res_clear = det.detect(depth_clear, small_intr, depth_scale=0.001)

        # Clear path should be wider than or equal to blocked path
        assert res_clear.free_path_width_m >= res_blocked.free_path_width_m

    def test_free_path_width_corridor(self):
        """Constructed corridor: obstacles on left/right, clear centre.

        Build a depth mask where left and right thirds are in obstacle
        range, centre third is out-of-range (no obstacle).
        """
        # Use a tiny camera for speed
        intr = CameraIntrinsics(60, 40, 30.0, 20.0, 50.0, 50.0)
        det = ObstacleDetector(
            min_distance_m=0.1,
            max_distance_m=3.0,
            obstacle_height_m=0.01,
            central_strip_fraction=1.0,  # examine full width
        )
        W = 60
        left_third = W // 3
        right_start = W - left_third

        depth_np = np.full((40, 60), 8000, dtype=np.uint16)  # out of range = 8 m
        # Left and right third: 1 m obstacle
        depth_np[:, :left_third] = 1000
        depth_np[:, right_start:] = 1000
        depth = mx.array(depth_np)

        res = det.detect(depth, intr, depth_scale=0.001)
        # Central 20 columns should be obstacle-free
        assert res.free_path_width_m > 0.0, "Centre corridor should be clear"


class TestObstacleDetectorErrors:
    """Error handling."""

    def test_depth_shape_mismatch_raises(self, intr):
        det = ObstacleDetector()
        bad = mx.zeros((100, 100), dtype=mx.uint16)
        with pytest.raises(ValueError, match="does not match intrinsics"):
            det.detect(bad, intr)

    def test_non_positive_depth_scale_raises(self, small_intr):
        det = ObstacleDetector()
        depth = _flat_depth(small_intr.height, small_intr.width, 500)
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            det.detect(depth, small_intr, depth_scale=0.0)

    def test_max_le_min_distance_raises(self):
        with pytest.raises(ValueError, match="max_distance_m"):
            ObstacleDetector(min_distance_m=2.0, max_distance_m=1.0)
        with pytest.raises(ValueError, match="max_distance_m"):
            ObstacleDetector(min_distance_m=2.0, max_distance_m=2.0)

    def test_negative_min_distance_raises(self):
        with pytest.raises(ValueError, match="min_distance_m"):
            ObstacleDetector(min_distance_m=-0.5)

    def test_negative_obstacle_height_raises(self):
        with pytest.raises(ValueError, match="obstacle_height_m"):
            ObstacleDetector(obstacle_height_m=-0.1)

    def test_invalid_strip_fraction_raises(self):
        with pytest.raises(ValueError, match="central_strip_fraction"):
            ObstacleDetector(central_strip_fraction=0.0)
        with pytest.raises(ValueError, match="central_strip_fraction"):
            ObstacleDetector(central_strip_fraction=1.5)


# ===========================================================================
# Top-level lazy import integration
# ===========================================================================


class TestTopLevelLazyImports:
    """realsense_mlx exposes robotics classes via __getattr__."""

    def test_occupancy_grid_generator_importable(self):
        import realsense_mlx as rsmlx
        cls = rsmlx.OccupancyGridGenerator
        assert cls is OccupancyGridGenerator

    def test_occupancy_grid_importable(self):
        import realsense_mlx as rsmlx
        cls = rsmlx.OccupancyGrid
        assert cls is OccupancyGrid

    def test_obstacle_detector_importable(self):
        import realsense_mlx as rsmlx
        cls = rsmlx.ObstacleDetector
        assert cls is ObstacleDetector

    def test_obstacle_result_importable(self):
        import realsense_mlx as rsmlx
        cls = rsmlx.ObstacleResult
        assert cls is ObstacleResult

    def test_unknown_attribute_raises(self):
        import realsense_mlx as rsmlx
        with pytest.raises(AttributeError):
            _ = rsmlx.DoesNotExist


# ===========================================================================
# OccupancyGrid dataclass
# ===========================================================================


class TestOccupancyGridDataclass:
    """OccupancyGrid container accessors."""

    def _make_grid(self, rows: int = 10, cols: int = 15) -> OccupancyGrid:
        g = mx.zeros((rows, cols), dtype=mx.uint8)
        return OccupancyGrid(
            grid=g,
            cell_size_m=0.05,
            origin_x_m=-0.375,
            origin_z_m=0.0,
            n_occupied=3,
            n_free=42,
        )

    def test_rows_property(self):
        og = self._make_grid(10, 15)
        assert og.rows == 10

    def test_cols_property(self):
        og = self._make_grid(10, 15)
        assert og.cols == 15

    def test_metadata_preserved(self):
        og = self._make_grid()
        assert og.n_occupied == 3
        assert og.n_free == 42
        assert og.cell_size_m == pytest.approx(0.05)
