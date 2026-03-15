"""pytest-benchmark integration for realsense-mlx components.

Run with::

    # Requires:  pip install pytest-benchmark
    pytest benchmarks/bench_pytest.py -v
    pytest benchmarks/bench_pytest.py -v --benchmark-sort=mean
    pytest benchmarks/bench_pytest.py -v --benchmark-json=bench_results.json

Each test uses the ``benchmark`` fixture from pytest-benchmark, which
automatically warms up the callable and handles statistical collection.
MLX device synchronisation is achieved by wrapping the callable in a
thin function that calls ``mx.eval`` on the output before returning.

Parameterisation
----------------
All tests are parameterised over "480p" and "720p" resolutions.  The
``res`` fixture id appears in the test name, e.g.::

    bench_yuy2_to_rgb[480p]
    bench_yuy2_to_rgb[720p]

Design notes
------------
- Each test constructs its input arrays once in a pytest fixture so that
  array allocation cost is not included in the benchmark measurement.
- Stateful filters (TemporalFilter, SpatialFilter) are reset between
  benchmark rounds where the state would accumulate across iterations.
  For TemporalFilter the fixture pre-seeds 8 frames of history to
  represent steady-state operation.
- The full-pipeline test pre-seeds temporal state similarly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import mlx.core as mx

from realsense_mlx.converters.format_converter import (
    extract_ir_y8,
    split_y8i,
    uyvy_to_yuyv,
    yuy2_to_bgr,
    yuy2_to_rgb,
)
from realsense_mlx.filters.colorizer import DepthColorizer
from realsense_mlx.filters.decimation import DecimationFilter
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
from realsense_mlx.filters.spatial import SpatialFilter
from realsense_mlx.filters.temporal import TemporalFilter
from realsense_mlx.geometry.align import Aligner
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 640),
    "720p": (720, 1280),
}

_BASELINE_MM: float = 50.0
_FOCAL_PX: float = 383.7
_DEPTH_UNITS: float = 0.001


# ---------------------------------------------------------------------------
# Session-scoped fixtures for input data
# ---------------------------------------------------------------------------

@pytest.fixture(params=["480p", "720p"])
def res(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[return-value]


@pytest.fixture()
def hw(res: str) -> tuple[int, int]:
    return RESOLUTIONS[res]


@pytest.fixture()
def depth_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(42)
    arr = rng.integers(500, 8000, size=(h, w), dtype=np.uint16)
    arr[h // 4: h // 2, w // 4: w // 2] = 0
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def disp_mx(depth_mx: mx.array) -> mx.array:
    d2d = _BASELINE_MM * _FOCAL_PX * 32.0 / _DEPTH_UNITS
    d_np = np.array(depth_mx, copy=False)
    safe = np.where(d_np > 0, d_np.astype(np.float32), 1.0)
    disp_np = np.where(d_np > 0, d2d / safe, 0.0).astype(np.float32)
    out = mx.array(disp_np)
    mx.eval(out)
    return out


@pytest.fixture()
def yuy2_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(2 * h * w,), dtype=np.uint8)
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def uyvy_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 65536, size=(h * w,), dtype=np.uint16)
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def y8i_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(17)
    arr = rng.integers(0, 256, size=(2 * h * w,), dtype=np.uint8)
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def ir16_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(23)
    arr = rng.integers(0, 1024, size=(h, w), dtype=np.uint16)
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def color_mx(hw: tuple[int, int]) -> mx.array:
    h, w = hw
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    out = mx.array(arr)
    mx.eval(out)
    return out


@pytest.fixture()
def depth_f32(depth_mx: mx.array) -> mx.array:
    out = depth_mx.astype(mx.float32)
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# Helper: wrap fn so mx.eval is called inside the benchmark loop
# ---------------------------------------------------------------------------

def _mlx(fn, *args):
    """Call fn(*args) and synchronise before returning."""
    result = fn(*args)
    mx.eval(result)
    return result


# ---------------------------------------------------------------------------
# Format converter benchmarks
# ---------------------------------------------------------------------------

def bench_yuy2_to_rgb(benchmark, yuy2_mx, hw):
    h, w = hw
    benchmark(_mlx, yuy2_to_rgb, yuy2_mx, w, h)


def bench_yuy2_to_bgr(benchmark, yuy2_mx, hw):
    h, w = hw
    benchmark(_mlx, yuy2_to_bgr, yuy2_mx, w, h)


def bench_uyvy_to_yuyv(benchmark, uyvy_mx):
    benchmark(_mlx, uyvy_to_yuyv, uyvy_mx)


def bench_split_y8i(benchmark, y8i_mx, hw):
    h, w = hw

    def _run(src):
        left, right = split_y8i(src, w, h)
        mx.eval(left, right)
        return left

    benchmark(_run, y8i_mx)


def bench_extract_ir_y8(benchmark, ir16_mx):
    benchmark(_mlx, extract_ir_y8, ir16_mx)


# ---------------------------------------------------------------------------
# Point cloud benchmarks
# ---------------------------------------------------------------------------

def bench_pointcloud_nodist(benchmark, depth_mx, hw):
    h, w = hw
    intr = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX, "none")
    gen = PointCloudGenerator(intr, depth_scale=_DEPTH_UNITS)
    # Force grid precomputation outside benchmark loop
    _ = gen.generate(depth_mx)
    mx.eval(_)
    benchmark(_mlx, gen.generate, depth_mx)


def bench_pointcloud_brown_conrady(benchmark, depth_mx, hw):
    h, w = hw
    intr = CameraIntrinsics(
        w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX,
        model="brown_conrady",
        coeffs=[-0.055, 0.065, 0.001, -0.0005, -0.021],
    )
    gen = PointCloudGenerator(intr, depth_scale=_DEPTH_UNITS)
    _ = gen.generate(depth_mx)
    mx.eval(_)
    benchmark(_mlx, gen.generate, depth_mx)


# ---------------------------------------------------------------------------
# Alignment benchmark
# ---------------------------------------------------------------------------

def bench_align_color_to_depth(benchmark, depth_mx, color_mx, hw):
    h, w = hw
    d_intr = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX)
    c_intr = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX)
    ext = CameraExtrinsics.identity()
    aligner = Aligner(d_intr, c_intr, ext, depth_scale=_DEPTH_UNITS)
    benchmark(_mlx, aligner.align_color_to_depth, depth_mx, color_mx)


# ---------------------------------------------------------------------------
# Spatial filter benchmark
# ---------------------------------------------------------------------------

def bench_spatial_filter(benchmark, depth_f32):
    filt = SpatialFilter(alpha=0.5, delta=20.0, iterations=2)
    benchmark(_mlx, filt.process, depth_f32)


# ---------------------------------------------------------------------------
# Temporal filter benchmark (steady-state)
# ---------------------------------------------------------------------------

def bench_temporal_filter(benchmark, depth_mx, hw):
    h, w = hw
    rng = np.random.default_rng(42)
    filt = TemporalFilter(alpha=0.4, delta=20.0, persistence=3)
    for _ in range(8):
        seed = mx.array(rng.integers(500, 5000, size=(h, w), dtype=np.uint16))
        filt.process(seed)

    live = mx.array(rng.integers(500, 5000, size=(h, w), dtype=np.uint16))
    mx.eval(live)

    benchmark(_mlx, filt.process, live)


# ---------------------------------------------------------------------------
# Decimation filter benchmark
# ---------------------------------------------------------------------------

def bench_decimation_scale2(benchmark, depth_mx):
    filt = DecimationFilter(scale=2)
    benchmark(_mlx, filt.process, depth_mx)


def bench_decimation_scale4(benchmark, depth_mx):
    filt = DecimationFilter(scale=4)
    benchmark(_mlx, filt.process, depth_mx)


# ---------------------------------------------------------------------------
# Hole filling benchmark
# ---------------------------------------------------------------------------

def bench_hole_fill_farthest(benchmark, depth_mx):
    filt = HoleFillingFilter(mode=HoleFillingFilter.FARTHEST)
    benchmark(_mlx, filt.process, depth_mx)


def bench_hole_fill_nearest(benchmark, depth_mx):
    filt = HoleFillingFilter(mode=HoleFillingFilter.NEAREST)
    benchmark(_mlx, filt.process, depth_mx)


def bench_hole_fill_left(benchmark, depth_mx):
    filt = HoleFillingFilter(mode=HoleFillingFilter.FILL_FROM_LEFT)
    benchmark(_mlx, filt.process, depth_mx)


# ---------------------------------------------------------------------------
# Disparity transform benchmark
# ---------------------------------------------------------------------------

def bench_depth_to_disparity(benchmark, depth_mx):
    t = DisparityTransform(
        baseline_mm=_BASELINE_MM, focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS, to_disparity=True,
    )
    benchmark(_mlx, t.process, depth_mx)


def bench_disparity_to_depth(benchmark, disp_mx):
    t = DisparityTransform(
        baseline_mm=_BASELINE_MM, focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS, to_disparity=False,
    )
    benchmark(_mlx, t.process, disp_mx)


# ---------------------------------------------------------------------------
# Colorizer benchmark
# ---------------------------------------------------------------------------

def bench_colorizer_jet_direct(benchmark, depth_mx):
    c = DepthColorizer(colormap="jet", equalize=False)
    benchmark(_mlx, c.colorize, depth_mx)


def bench_colorizer_jet_equalized(benchmark, depth_mx):
    c = DepthColorizer(colormap="jet", equalize=True)
    benchmark(_mlx, c.colorize, depth_mx)


def bench_colorizer_hsv_direct(benchmark, depth_mx):
    c = DepthColorizer(colormap="hsv", equalize=False)
    benchmark(_mlx, c.colorize, depth_mx)


# ---------------------------------------------------------------------------
# Full pipeline benchmark
# ---------------------------------------------------------------------------

def bench_full_pipeline(benchmark, depth_mx, hw):
    h, w = hw
    rng = np.random.default_rng(77)
    cfg = PipelineConfig(
        decimation_scale=2,
        spatial_iterations=2,
        temporal_alpha=0.4,
        temporal_persistence=3,
        hole_fill_mode=1,
        baseline_mm=_BASELINE_MM,
        focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS,
    )
    pipeline = DepthPipeline(cfg)
    for _ in range(8):
        seed = mx.array(rng.integers(500, 5000, size=(h, w), dtype=np.uint16))
        pipeline.process(seed)

    benchmark(_mlx, pipeline.process, depth_mx)
