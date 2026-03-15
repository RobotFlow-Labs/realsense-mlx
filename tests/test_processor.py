"""Tests for RealsenseProcessor and ProcessingResult.

Covers:
- Filter-only path (no pointcloud, no mesh)
- Point cloud generation with correct decimation-adjusted intrinsics
- Mesh generation (vertices, faces, normals)
- Colour alignment (depth + colour frames)
- Depth statistics attachment
- PLY export (point cloud and mesh)
- OBJ export
- Decimated intrinsics correctness
- Processing time tracking
- Error handling (bad input shapes)
- Top-level package re-exports
- RealsenseProcessor repr
- reset() propagates to temporal filter
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.pipeline import PipelineConfig
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.processor import ProcessingResult, RealsenseProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr, copy=False)


def _make_intrinsics(
    width: int = 640,
    height: int = 480,
    fx: float = 600.0,
    fy: float = 600.0,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        ppx=width / 2.0,
        ppy=height / 2.0,
        fx=fx,
        fy=fy,
    )


def _make_depth(height: int = 480, width: int = 640, value: int = 1000) -> mx.array:
    """Uniform depth frame at *value* counts (~1 m with depth_scale=0.001)."""
    return mx.array(np.full((height, width), value, dtype=np.uint16))


def _make_color(height: int = 480, width: int = 640) -> mx.array:
    """Simple colour ramp as a (H, W, 3) uint8 array."""
    r = np.tile(
        np.arange(height, dtype=np.uint8).reshape(-1, 1), (1, width)
    )  # (H, W)
    g = np.zeros((height, width), dtype=np.uint8)
    b = np.tile(
        np.arange(width, dtype=np.uint8).reshape(1, -1), (height, 1)
    )  # (H, W)
    return mx.array(np.stack([r, g, b], axis=-1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def depth_intr() -> CameraIntrinsics:
    return _make_intrinsics(640, 480)


@pytest.fixture
def color_intr() -> CameraIntrinsics:
    return _make_intrinsics(640, 480, fx=615.0, fy=615.0)


@pytest.fixture
def depth_frame() -> mx.array:
    return _make_depth()


@pytest.fixture
def color_frame() -> mx.array:
    return _make_color()


@pytest.fixture
def cfg_dec2() -> PipelineConfig:
    """Pipeline with decimation=2, other filters minimal."""
    return PipelineConfig(
        decimation_scale=2,
        spatial_iterations=1,
        enable_spatial=False,
        enable_hole_fill=False,
    )


@pytest.fixture
def cfg_dec1() -> PipelineConfig:
    """Pipeline with no decimation."""
    return PipelineConfig(
        decimation_scale=1,
        spatial_iterations=1,
        enable_spatial=False,
        enable_hole_fill=False,
    )


# ---------------------------------------------------------------------------
# Task 1: DecimationFilter.adjust_intrinsics
# ---------------------------------------------------------------------------


class TestAdjustIntrinsics:
    """Validate that DecimationFilter.adjust_intrinsics produces correct values."""

    def test_scale2_halves_dimensions(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = _make_intrinsics(640, 480, fx=600.0, fy=600.0)
        f = DecimationFilter(scale=2)
        dec = f.adjust_intrinsics(intr)

        assert dec.width == 320
        assert dec.height == 240

    def test_scale2_halves_focal_lengths(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = _make_intrinsics(640, 480, fx=600.0, fy=600.0)
        f = DecimationFilter(scale=2)
        dec = f.adjust_intrinsics(intr)

        assert dec.fx == pytest.approx(300.0)
        assert dec.fy == pytest.approx(300.0)

    def test_scale2_halves_principal_point(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = CameraIntrinsics(640, 480, ppx=320.0, ppy=240.0, fx=600.0, fy=600.0)
        f = DecimationFilter(scale=2)
        dec = f.adjust_intrinsics(intr)

        assert dec.ppx == pytest.approx(160.0)
        assert dec.ppy == pytest.approx(120.0)

    def test_scale4_quarters_dimensions(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = _make_intrinsics(640, 480, fx=600.0, fy=600.0)
        f = DecimationFilter(scale=4)
        dec = f.adjust_intrinsics(intr)

        assert dec.width == 160
        assert dec.height == 120
        assert dec.fx == pytest.approx(150.0)

    def test_scale1_returns_identical_values(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = CameraIntrinsics(640, 480, ppx=320.0, ppy=240.0, fx=600.0, fy=600.0)
        f = DecimationFilter(scale=1)
        dec = f.adjust_intrinsics(intr)

        assert dec.width == 640
        assert dec.height == 480
        assert dec.fx == pytest.approx(600.0)
        assert dec.ppx == pytest.approx(320.0)

    def test_model_and_coeffs_preserved(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = CameraIntrinsics(
            640, 480, 320.0, 240.0, 600.0, 600.0,
            model="none",
            coeffs=[0.1, -0.2, 0.01, 0.0, 0.05],
        )
        f = DecimationFilter(scale=2)
        dec = f.adjust_intrinsics(intr)

        assert dec.model == "none"
        assert dec.coeffs == pytest.approx([0.1, -0.2, 0.01, 0.0, 0.05])

    def test_adjust_intrinsics_does_not_mutate_original(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
        f = DecimationFilter(scale=2)
        _ = f.adjust_intrinsics(intr)

        # Original unchanged
        assert intr.width == 640
        assert intr.height == 480
        assert intr.fx == pytest.approx(600.0)

    def test_last_output_shape_set_after_process(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        f = DecimationFilter(scale=2)
        depth = _make_depth(480, 640)
        out = f.process(depth)
        mx.eval(out)

        assert f._last_output_shape is not None
        assert f._last_output_shape == (240, 320)

    def test_last_output_shape_scale1(self):
        from realsense_mlx.filters.decimation import DecimationFilter

        f = DecimationFilter(scale=1)
        depth = _make_depth(480, 640)
        _ = f.process(depth)

        assert f._last_output_shape == (480, 640)


# ---------------------------------------------------------------------------
# Task 2: RealsenseProcessor — basic construction
# ---------------------------------------------------------------------------


class TestProcessorConstruction:
    def test_default_construction(self, depth_intr):
        proc = RealsenseProcessor(depth_intr)
        assert proc.depth_scale == 0.001
        assert proc.enable_pointcloud is True
        assert proc.enable_colorize is True
        assert proc.enable_mesh is False

    def test_mesh_enables_pointcloud_automatically(self, depth_intr):
        proc = RealsenseProcessor(depth_intr, enable_mesh=True, enable_pointcloud=False)
        assert proc.enable_pointcloud is True
        assert proc.enable_mesh is True

    def test_invalid_depth_scale_raises(self, depth_intr):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            RealsenseProcessor(depth_intr, depth_scale=0.0)

    def test_negative_depth_scale_raises(self, depth_intr):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            RealsenseProcessor(depth_intr, depth_scale=-1.0)

    def test_repr_contains_dims(self, depth_intr):
        proc = RealsenseProcessor(depth_intr)
        r = repr(proc)
        assert "640x480" in r
        assert "pc=True" in r


# ---------------------------------------------------------------------------
# Task 2: process() — filter only (no pointcloud/mesh)
# ---------------------------------------------------------------------------


class TestProcessFilterOnly:
    def test_filtered_depth_shape_decimation_2(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec2,
            enable_pointcloud=False,
            enable_colorize=False,
        )
        result = proc.process(depth_frame)

        assert result.filtered_depth.shape == (240, 320)

    def test_filtered_depth_shape_no_decimation(self, depth_intr, cfg_dec1, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec1,
            enable_pointcloud=False,
            enable_colorize=False,
        )
        result = proc.process(depth_frame)

        assert result.filtered_depth.shape == (480, 640)

    def test_no_pointcloud_when_disabled(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec2,
            enable_pointcloud=False,
            enable_colorize=False,
        )
        result = proc.process(depth_frame)
        assert result.points is None

    def test_no_colored_depth_when_disabled(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec2,
            enable_pointcloud=False,
            enable_colorize=False,
        )
        result = proc.process(depth_frame)
        assert result.colored_depth is None

    def test_no_mesh_when_disabled(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec2,
            enable_pointcloud=True,
            enable_colorize=False,
            enable_mesh=False,
        )
        result = proc.process(depth_frame)
        assert result.vertices is None
        assert result.faces is None

    def test_3d_input_raises(self, depth_intr, cfg_dec1):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec1)
        bad = mx.zeros((480, 640, 1), dtype=mx.uint16)
        with pytest.raises(ValueError, match="2-D"):
            proc.process(bad)


# ---------------------------------------------------------------------------
# Task 2: process() — with point cloud enabled
# ---------------------------------------------------------------------------


class TestProcessPointCloud:
    def test_pointcloud_shape_after_decimation(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.points is not None
        assert result.points.shape == (240, 320, 3)

    def test_pointcloud_is_float32(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.points is not None
        assert result.points.dtype == mx.float32

    def test_intrinsics_attached_to_result(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.intrinsics is not None
        assert result.intrinsics.width == 320
        assert result.intrinsics.height == 240

    def test_decimated_intrinsics_focal_halved(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.intrinsics is not None
        assert result.intrinsics.fx == pytest.approx(depth_intr.fx / 2)
        assert result.intrinsics.fy == pytest.approx(depth_intr.fy / 2)

    def test_zero_depth_produces_zero_xyz(self, depth_intr, cfg_dec1):
        """All-zero input depth should yield a point cloud with all zeros."""
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec1)
        depth = mx.zeros((480, 640), dtype=mx.uint16)
        result = proc.process(depth)

        assert result.points is not None
        pts = _np(result.points)
        assert np.all(pts == 0.0)

    def test_valid_depth_produces_positive_z(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.points is not None
        pts = _np(result.points)
        z_vals = pts[:, :, 2]
        # Centre pixels should have positive Z (1 m depth, no holes)
        h, w = z_vals.shape
        assert z_vals[h // 2, w // 2] > 0.0


# ---------------------------------------------------------------------------
# Task 2: process() — with mesh enabled
# ---------------------------------------------------------------------------


class TestProcessMesh:
    def test_mesh_vertices_shape(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        assert result.vertices is not None
        assert result.vertices.ndim == 2
        assert result.vertices.shape[1] == 3

    def test_mesh_faces_shape(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        assert result.faces is not None
        assert result.faces.ndim == 2
        assert result.faces.shape[1] == 3

    def test_mesh_normals_present(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        assert result.normals is not None
        assert result.normals.ndim == 2
        assert result.normals.shape[1] == 3

    def test_mesh_vertices_match_vertex_count(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        assert result.vertices is not None
        assert result.normals is not None
        assert result.vertices.shape[0] == result.normals.shape[0]

    def test_no_mesh_on_zero_depth(self, depth_intr, cfg_dec2):
        """All-zero depth → all points at Z=0 → DepthMeshGenerator suppresses faces."""
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        depth = mx.zeros((480, 640), dtype=mx.uint16)
        result = proc.process(depth)

        assert result.faces is not None
        assert result.faces.shape[0] == 0


# ---------------------------------------------------------------------------
# Task 2: process() — with colour alignment
# ---------------------------------------------------------------------------


class TestProcessAlignment:
    def test_aligned_color_shape_matches_depth(
        self, depth_intr, color_intr, cfg_dec2, depth_frame, color_frame
    ):
        proc = RealsenseProcessor(
            depth_intr,
            color_intrinsics=color_intr,
            pipeline_config=cfg_dec2,
        )
        result = proc.process(depth_frame, color_frame)

        assert result.aligned_color is not None
        H, W = result.filtered_depth.shape
        assert result.aligned_color.shape == (H, W, 3)

    def test_aligned_color_is_uint8(
        self, depth_intr, color_intr, cfg_dec2, depth_frame, color_frame
    ):
        proc = RealsenseProcessor(
            depth_intr,
            color_intrinsics=color_intr,
            pipeline_config=cfg_dec2,
        )
        result = proc.process(depth_frame, color_frame)

        assert result.aligned_color is not None
        assert result.aligned_color.dtype == mx.uint8

    def test_no_aligned_color_when_no_color_frame(
        self, depth_intr, color_intr, cfg_dec2, depth_frame
    ):
        proc = RealsenseProcessor(
            depth_intr,
            color_intrinsics=color_intr,
            pipeline_config=cfg_dec2,
        )
        result = proc.process(depth_frame, color=None)
        assert result.aligned_color is None

    def test_no_aligned_color_when_no_color_intrinsics(
        self, depth_intr, cfg_dec2, depth_frame, color_frame
    ):
        proc = RealsenseProcessor(
            depth_intr,
            color_intrinsics=None,
            pipeline_config=cfg_dec2,
        )
        result = proc.process(depth_frame, color=color_frame)
        assert result.aligned_color is None


# ---------------------------------------------------------------------------
# Task 2: process() — statistics
# ---------------------------------------------------------------------------


class TestProcessStats:
    def test_stats_dict_attached_when_enabled(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_stats=True
        )
        result = proc.process(depth_frame)

        assert result.stats is not None
        assert "valid_ratio" in result.stats
        assert "mean_m" in result.stats

    def test_stats_none_when_disabled(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_stats=False
        )
        result = proc.process(depth_frame)
        assert result.stats is None

    def test_stats_valid_ratio_is_float(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_stats=True
        )
        result = proc.process(depth_frame)

        assert result.stats is not None
        assert isinstance(result.stats["valid_ratio"], float)
        assert 0.0 <= result.stats["valid_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Task 2: process() — processing time
# ---------------------------------------------------------------------------


class TestProcessingTime:
    def test_processing_time_is_positive(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert result.processing_time_ms > 0.0

    def test_processing_time_is_float(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        assert isinstance(result.processing_time_ms, float)

    def test_processing_time_reasonable_range(self, depth_intr, cfg_dec2, depth_frame):
        """Processing should complete in under 10 seconds even on slow hardware."""
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        # Generous upper bound for test environments
        assert result.processing_time_ms < 10_000.0


# ---------------------------------------------------------------------------
# Task 2: export_ply — point cloud and mesh
# ---------------------------------------------------------------------------


class TestExportPLY:
    def test_export_ply_point_cloud(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        ply_path = str(tmp_path / "test.ply")
        n_written = proc.export_ply(result, ply_path)

        assert n_written > 0
        assert Path(ply_path).exists()
        assert Path(ply_path).stat().st_size > 0

    def test_export_ply_mesh(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        ply_path = str(tmp_path / "mesh.ply")
        n_written = proc.export_ply(result, ply_path)

        # Mesh export returns face count
        assert n_written >= 0
        assert Path(ply_path).exists()

    def test_export_ply_no_data_raises(self, depth_intr, cfg_dec1):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec1,
            enable_pointcloud=False,
            enable_mesh=False,
        )
        # Build a result manually without points
        fake_result = ProcessingResult(
            filtered_depth=mx.zeros((480, 640), dtype=mx.uint16)
        )
        with pytest.raises(ValueError):
            proc.export_ply(fake_result, "/tmp/nope.ply")

    def test_export_ply_creates_parent_dirs(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        nested = str(tmp_path / "sub" / "dir" / "out.ply")
        proc.export_ply(result, nested)

        assert Path(nested).exists()

    def test_export_ply_binary_header(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        """Check PLY file starts with the expected ASCII header."""
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        ply_path = tmp_path / "check.ply"
        proc.export_ply(result, str(ply_path))

        content = ply_path.read_bytes()
        assert content.startswith(b"ply\n")
        assert b"binary_little_endian" in content
        assert b"end_header" in content


# ---------------------------------------------------------------------------
# Task 2: export_obj
# ---------------------------------------------------------------------------


class TestExportOBJ:
    def test_export_obj_point_cloud(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        obj_path = str(tmp_path / "test.obj")
        n_written = proc.export_obj(result, obj_path)

        assert n_written > 0
        assert Path(obj_path).exists()

    def test_export_obj_mesh(self, depth_intr, cfg_dec2, depth_frame, tmp_path):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_mesh=True
        )
        result = proc.process(depth_frame)

        obj_path = str(tmp_path / "mesh.obj")
        n_written = proc.export_obj(result, obj_path)

        assert n_written > 0
        assert Path(obj_path).exists()

    def test_export_obj_contains_vertex_lines(
        self, depth_intr, cfg_dec2, depth_frame, tmp_path
    ):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        result = proc.process(depth_frame)

        obj_path = tmp_path / "verts.obj"
        proc.export_obj(result, str(obj_path))

        content = obj_path.read_text(encoding="ascii")
        assert "# Wavefront OBJ" in content
        lines = content.splitlines()
        vertex_lines = [l for l in lines if l.startswith("v ")]
        assert len(vertex_lines) > 0

    def test_export_obj_no_data_raises(self, depth_intr, cfg_dec1):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec1,
            enable_pointcloud=False,
            enable_mesh=False,
        )
        # Fake result with no geometry
        fake_result = ProcessingResult(
            filtered_depth=mx.zeros((480, 640), dtype=mx.uint16)
        )
        with pytest.raises(ValueError):
            proc.export_obj(fake_result, "/tmp/nope.obj")


# ---------------------------------------------------------------------------
# Consecutive calls — intrinsics consistency
# ---------------------------------------------------------------------------


class TestConsecutiveCalls:
    def test_consecutive_calls_have_consistent_intrinsics(
        self, depth_intr, cfg_dec2, depth_frame
    ):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        r1 = proc.process(depth_frame)
        r2 = proc.process(depth_frame)

        assert r1.intrinsics is not None
        assert r2.intrinsics is not None
        assert r1.intrinsics.width == r2.intrinsics.width
        assert r1.intrinsics.fx == pytest.approx(r2.intrinsics.fx)

    def test_consecutive_pointcloud_shapes_consistent(
        self, depth_intr, cfg_dec2, depth_frame
    ):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        r1 = proc.process(depth_frame)
        r2 = proc.process(depth_frame)

        assert r1.points is not None
        assert r2.points is not None
        assert r1.points.shape == r2.points.shape


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_does_not_crash(self, depth_intr, cfg_dec2):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        proc.reset()  # should not raise

    def test_reset_allows_continued_processing(
        self, depth_intr, cfg_dec2, depth_frame
    ):
        proc = RealsenseProcessor(depth_intr, pipeline_config=cfg_dec2)
        _ = proc.process(depth_frame)
        proc.reset()
        result = proc.process(depth_frame)

        assert result.filtered_depth.shape == (240, 320)


# ---------------------------------------------------------------------------
# Top-level package re-exports
# ---------------------------------------------------------------------------


class TestPackageReexports:
    def test_realsense_processor_importable_from_package(self):
        import realsense_mlx as rsmlx

        assert rsmlx.RealsenseProcessor is RealsenseProcessor

    def test_processing_result_importable_from_package(self):
        import realsense_mlx as rsmlx

        assert rsmlx.ProcessingResult is ProcessingResult

    def test_realsense_processor_via_package_works(self):
        import realsense_mlx as rsmlx

        intr = _make_intrinsics()
        cfg = PipelineConfig(
            decimation_scale=2, enable_spatial=False, enable_hole_fill=False
        )
        proc = rsmlx.RealsenseProcessor(intr, pipeline_config=cfg)
        depth = _make_depth()
        result = proc.process(depth)

        assert isinstance(result, rsmlx.ProcessingResult)
        assert result.filtered_depth.shape == (240, 320)


# ---------------------------------------------------------------------------
# ProcessingResult dataclass
# ---------------------------------------------------------------------------


class TestProcessingResult:
    def test_default_optional_fields_are_none(self):
        depth = mx.zeros((240, 320), dtype=mx.uint16)
        r = ProcessingResult(filtered_depth=depth)

        assert r.points is None
        assert r.colored_depth is None
        assert r.aligned_color is None
        assert r.vertices is None
        assert r.faces is None
        assert r.normals is None
        assert r.stats is None
        assert r.intrinsics is None
        assert r.processing_time_ms == 0.0

    def test_processing_time_default_zero(self):
        depth = mx.zeros((240, 320), dtype=mx.uint16)
        r = ProcessingResult(filtered_depth=depth)
        assert r.processing_time_ms == 0.0


# ---------------------------------------------------------------------------
# Colourize path
# ---------------------------------------------------------------------------


class TestColorize:
    def test_colored_depth_shape(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_colorize=True
        )
        result = proc.process(depth_frame)

        assert result.colored_depth is not None
        H, W = result.filtered_depth.shape
        assert result.colored_depth.shape == (H, W, 3)

    def test_colored_depth_is_uint8(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr, pipeline_config=cfg_dec2, enable_colorize=True
        )
        result = proc.process(depth_frame)

        assert result.colored_depth is not None
        assert result.colored_depth.dtype == mx.uint8

    def test_custom_colormap_accepted(self, depth_intr, cfg_dec2, depth_frame):
        proc = RealsenseProcessor(
            depth_intr,
            pipeline_config=cfg_dec2,
            colormap="classic",
        )
        result = proc.process(depth_frame)

        assert result.colored_depth is not None
