"""Package import safety tests.

Verifies that every public symbol documented in the package is importable
without errors and exposes the expected interface.  These tests act as a
first-pass smoke test; they do not test functional correctness.

Tests are designed to be fast (no large array allocations) and to catch
circular import issues, missing __init__ re-exports, or broken module
paths introduced by refactoring.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Top-level package
# ---------------------------------------------------------------------------

class TestTopLevelPackage:
    def test_import_realsense_mlx(self):
        import realsense_mlx  # noqa: F401 — import side-effect is what matters

    def test_version_attribute_exists(self):
        import realsense_mlx
        assert hasattr(realsense_mlx, "__version__")

    def test_version_is_string(self):
        import realsense_mlx
        assert isinstance(realsense_mlx.__version__, str)

    def test_version_is_semver_like(self):
        import realsense_mlx
        parts = realsense_mlx.__version__.split(".")
        assert len(parts) >= 2, "Version should follow major.minor[.patch] format"
        for part in parts:
            assert part.isdigit(), f"Version part {part!r} is not numeric"

    def test_depth_colorizer_lazy_import(self):
        import realsense_mlx
        cls = realsense_mlx.DepthColorizer
        assert cls.__name__ == "DepthColorizer"

    def test_depth_pipeline_lazy_import(self):
        import realsense_mlx
        cls = realsense_mlx.DepthPipeline
        assert cls.__name__ == "DepthPipeline"

    def test_point_cloud_generator_lazy_import(self):
        import realsense_mlx
        cls = realsense_mlx.PointCloudGenerator
        assert cls.__name__ == "PointCloudGenerator"

    def test_camera_intrinsics_lazy_import(self):
        import realsense_mlx
        cls = realsense_mlx.CameraIntrinsics
        assert cls.__name__ == "CameraIntrinsics"

    def test_unknown_attribute_raises(self):
        import realsense_mlx
        import pytest
        with pytest.raises(AttributeError):
            _ = realsense_mlx.DoesNotExist


# ---------------------------------------------------------------------------
# Filters sub-package
# ---------------------------------------------------------------------------

class TestFiltersImport:
    def test_import_filters_package(self):
        import realsense_mlx.filters  # noqa: F401

    def test_import_colorizer_module(self):
        from realsense_mlx.filters import colorizer  # noqa: F401

    def test_import_depth_colorizer_class(self):
        from realsense_mlx.filters.colorizer import DepthColorizer
        assert callable(DepthColorizer)

    def test_depth_colorizer_instantiates(self):
        from realsense_mlx.filters.colorizer import DepthColorizer
        c = DepthColorizer()
        assert c.colormap == "jet"

    def test_import_disparity_transform(self):
        from realsense_mlx.filters.disparity import DisparityTransform
        assert callable(DisparityTransform)

    def test_disparity_transform_instantiates(self):
        from realsense_mlx.filters.disparity import DisparityTransform
        t = DisparityTransform(baseline_mm=50.0, focal_px=383.7, depth_units=0.001)
        assert t.to_disparity is True

    def test_import_depth_pipeline(self):
        from realsense_mlx.filters import DepthPipeline
        assert callable(DepthPipeline)

    def test_import_spatial_filter(self):
        from realsense_mlx.filters import SpatialFilter
        assert callable(SpatialFilter)

    def test_import_temporal_filter(self):
        from realsense_mlx.filters import TemporalFilter
        assert callable(TemporalFilter)

    def test_colorizer_colormaps_are_ten(self):
        """Exactly 10 named color maps must be registered."""
        from realsense_mlx.filters.colorizer import DepthColorizer
        assert len(DepthColorizer.COLORMAPS) == 10

    def test_all_colormap_names_are_strings(self):
        from realsense_mlx.filters.colorizer import DepthColorizer
        for name in DepthColorizer.COLORMAPS:
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Geometry sub-package
# ---------------------------------------------------------------------------

class TestGeometryImport:
    def test_import_geometry_package(self):
        import realsense_mlx.geometry  # noqa: F401

    def test_import_camera_intrinsics(self):
        from realsense_mlx.geometry import CameraIntrinsics
        assert callable(CameraIntrinsics)

    def test_import_camera_extrinsics(self):
        from realsense_mlx.geometry.intrinsics import CameraExtrinsics
        assert callable(CameraExtrinsics)

    def test_camera_intrinsics_instantiates(self):
        from realsense_mlx.geometry import CameraIntrinsics
        ci = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
        )
        assert ci.width == 640

    def test_camera_extrinsics_identity(self):
        from realsense_mlx.geometry.intrinsics import CameraExtrinsics
        ex = CameraExtrinsics.identity()
        assert ex.is_identity

    def test_import_point_cloud_generator(self):
        from realsense_mlx.geometry import PointCloudGenerator
        assert callable(PointCloudGenerator)


# ---------------------------------------------------------------------------
# Utils sub-package
# ---------------------------------------------------------------------------

class TestUtilsImport:
    def test_import_utils_package(self):
        import realsense_mlx.utils  # noqa: F401

    def test_import_timer(self):
        from realsense_mlx.utils import Timer
        assert callable(Timer)

    def test_import_benchmark_component(self):
        from realsense_mlx.utils import benchmark_component
        assert callable(benchmark_component)

    def test_import_benchmark_module(self):
        from realsense_mlx.utils import benchmark  # noqa: F401

    def test_timer_instantiates(self):
        from realsense_mlx.utils import Timer
        t = Timer("test")
        assert t.name == "test"
        assert t.elapsed_ms == 0.0

    def test_timer_context_manager(self):
        import mlx.core as mx
        from realsense_mlx.utils import Timer
        with Timer("noop") as t:
            _ = mx.zeros((10, 10))
        assert t.elapsed_ms >= 0.0

    def test_benchmark_component_callable(self):
        import mlx.core as mx
        import numpy as np
        from realsense_mlx.utils import benchmark_component

        def noop(x: mx.array) -> mx.array:
            return x + mx.zeros_like(x)

        arr = mx.array(np.zeros((4, 4), dtype=np.float32))
        stats = benchmark_component(noop, (arr,), warmup=1, iterations=3)
        assert set(stats.keys()) == {"mean_ms", "std_ms", "min_ms", "max_ms", "fps"}
        assert stats["mean_ms"] >= 0.0
        assert stats["fps"] > 0.0

    def test_benchmark_component_invalid_iterations(self):
        import mlx.core as mx
        import pytest
        from realsense_mlx.utils import benchmark_component

        arr = mx.zeros((2, 2))
        with pytest.raises(ValueError, match="iterations"):
            benchmark_component(lambda x: x, (arr,), iterations=0)
