"""Comprehensive benchmark suite for all realsense-mlx components.

Run from the project root::

    python benchmarks/bench_all.py
    python benchmarks/bench_all.py --resolutions 480p 720p --iters 50
    python benchmarks/bench_all.py --components converters pointcloud --iters 30

Covers
------
- Format converters  : YUY2→RGB/BGR/RGBA/BGRA/Y16, UYVY→YUYV, split_y8i,
                       split_y12i, extract_ir_y8, extract_ir_y16
- Point cloud        : no-distortion AND Brown-Conrady at 480p / 720p
- Alignment          : color-to-depth (identity extrinsics)
- Spatial filter     : alpha=0.5, delta=20, iterations=2
- Temporal filter    : warmed-up (8 seed frames) so EMA state is live
- Decimation         : scale 2 (median) and scale 4 (valid-mean)
- Hole filling       : modes 0 (left), 1 (farthest), 2 (nearest)
- Disparity          : depth→disparity and disparity→depth
- Colorizer          : all 10 colormaps × direct + equalized
- Full pipeline      : DepthPipeline end-to-end (decimation=2)

MLX vs NumPy CPU comparison is produced for:
- YUY2→RGB
- PointCloud (no-distortion)
- Spatial filter
- Hole filling (farthest)
- Colorizer (jet, equalize=False)

Results are saved as JSON in ``benchmarks/results/<timestamp>.json``.
Memory snapshots (peak / active) are taken before and after each suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Ensure package is importable when run without an editable install
# ---------------------------------------------------------------------------
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import mlx.core as mx

from realsense_mlx.converters.format_converter import (
    yuy2_to_rgb, yuy2_to_bgr, yuy2_to_rgba, yuy2_to_bgra, yuy2_to_y16,
    uyvy_to_yuyv, split_y8i, split_y12i, extract_ir_y8, extract_ir_y16,
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
from realsense_mlx.utils.benchmark import Timer, benchmark_component


# ---------------------------------------------------------------------------
# Resolution table
# ---------------------------------------------------------------------------

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}

# Stereo rig constants used consistently across all geometry benchmarks
_BASELINE_MM: float = 50.0
_FOCAL_PX: float = 383.7
_DEPTH_UNITS: float = 0.001


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_COL_W: int = 48
_LINE_W: int = _COL_W + 64


def _sep(char: str = "=") -> None:
    print(char * _LINE_W)


def _header(title: str) -> None:
    print()
    _sep()
    print(f"  {title}")
    _sep()
    print(
        f"  {'Component':<{_COL_W}s}"
        f"  {'mean':>9}"
        f"  {'std':>7}"
        f"  {'min':>7}"
        f"  {'max':>7}"
        f"  {'fps':>9}"
    )
    print("  " + "-" * (_LINE_W - 2))


def _row(label: str, stats: dict[str, float], extra: str = "") -> None:
    suffix = f"  {extra}" if extra else ""
    print(
        f"  {label:<{_COL_W}s}"
        f"  {stats['mean_ms']:8.2f}ms"
        f"  ±{stats['std_ms']:5.2f}"
        f"  {stats['min_ms']:6.2f}"
        f"  {stats['max_ms']:6.2f}"
        f"  {stats['fps']:8.1f} fps"
        f"{suffix}"
    )


def _memory_snapshot(tag: str) -> dict[str, int]:
    mx.eval(mx.zeros(1))  # flush before sampling
    peak = mx.get_peak_memory()
    active = mx.get_active_memory()
    cache = mx.get_cache_memory()
    print(f"  [mem:{tag}] peak={peak/1e6:.1f} MB  active={active/1e6:.1f} MB  cache={cache/1e6:.1f} MB")
    return {"peak_bytes": peak, "active_bytes": active, "cache_bytes": cache}


# ---------------------------------------------------------------------------
# NumPy reference implementations for MLX vs CPU comparisons
# ---------------------------------------------------------------------------

def _np_yuy2_to_rgb(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """Reference NumPy port of the YUY2→RGB BT.601 kernel."""
    flat = src.reshape(-1).astype(np.int32)
    n = flat.shape[0] // 4
    quad = flat.reshape(n, 4)
    y0, u, y1, v = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]

    def _yuv_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray):
        c = y - 16
        d = u - 128
        e = v - 128
        r = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255).astype(np.uint8)
        g = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255).astype(np.uint8)
        b = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255).astype(np.uint8)
        return r, g, b

    r0, g0, b0 = _yuv_rgb(y0, u, v)
    r1, g1, b1 = _yuv_rgb(y1, u, v)
    rgb = np.stack([
        np.stack([r0, g0, b0], axis=1),
        np.stack([r1, g1, b1], axis=1),
    ], axis=1).reshape(2 * n, 3)
    return rgb.reshape(height, width, 3)


def _np_pointcloud(depth_np: np.ndarray, fx: float, fy: float,
                   ppx: float, ppy: float, scale: float) -> np.ndarray:
    """Reference NumPy point cloud (no distortion)."""
    H, W = depth_np.shape
    z = depth_np.astype(np.float32) * scale
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    X = ((u[None, :] - ppx) / fx) * z
    Y = ((v[:, None] - ppy) / fy) * z
    return np.stack([X, Y, z], axis=-1)


def _benchmark_numpy(fn: Callable, args: tuple, warmup: int, iterations: int) -> dict[str, float]:
    """Benchmark a pure-NumPy function (no MLX eval synchronisation needed)."""
    for _ in range(warmup):
        fn(*args)
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


def _speedup_tag(mlx_stats: dict, np_stats: dict) -> str:
    if np_stats["mean_ms"] > 0 and mlx_stats["mean_ms"] > 0:
        ratio = np_stats["mean_ms"] / mlx_stats["mean_ms"]
        direction = "faster" if ratio >= 1.0 else "slower"
        return f"[MLX {abs(ratio):.1f}x {direction} vs NumPy]"
    return ""


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_depth(h: int, w: int, seed: int = 42) -> tuple[np.ndarray, mx.array]:
    rng = np.random.default_rng(seed)
    np_arr = rng.integers(500, 8000, size=(h, w), dtype=np.uint16)
    np_arr[h // 4: h // 2, w // 4: w // 2] = 0  # rectangular hole region
    mx_arr = mx.array(np_arr)
    mx.eval(mx_arr)
    return np_arr, mx_arr


def _make_disp(depth_np: np.ndarray) -> tuple[np.ndarray, mx.array]:
    d2d = _BASELINE_MM * _FOCAL_PX * 32.0 / _DEPTH_UNITS
    safe = np.where(depth_np > 0, depth_np.astype(np.float32), 1.0)
    disp_np = np.where(depth_np > 0, d2d / safe, 0.0).astype(np.float32)
    disp_mx = mx.array(disp_np)
    mx.eval(disp_mx)
    return disp_np, disp_mx


def _make_yuy2(h: int, w: int, seed: int = 7) -> tuple[np.ndarray, mx.array]:
    rng = np.random.default_rng(seed)
    np_arr = rng.integers(0, 256, size=(2 * h * w,), dtype=np.uint8)
    mx_arr = mx.array(np_arr)
    mx.eval(mx_arr)
    return np_arr, mx_arr


def _make_color_frame(h: int, w: int, seed: int = 99) -> tuple[np.ndarray, mx.array]:
    rng = np.random.default_rng(seed)
    np_arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mx_arr = mx.array(np_arr)
    mx.eval(mx_arr)
    return np_arr, mx_arr


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------

def bench_timer_overhead(warmup: int, iterations: int) -> dict[str, Any]:
    """Measure bare Timer synchronisation overhead (two mx.eval barriers)."""
    _header("Timer synchronisation overhead (baseline)")
    times: list[float] = []
    for _ in range(warmup + iterations):
        with Timer() as t:
            pass
    # Reset and measure properly
    times = []
    for _ in range(iterations):
        with Timer() as t:
            pass
        times.append(t.elapsed_ms)
    arr = np.array(times)
    stats = {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": 1000.0 / float(np.mean(arr)),
    }
    _row("Timer barrier overhead", stats)
    return {"overhead": stats}


def bench_converters(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark all format converters at *resolution*."""
    h, w = RESOLUTIONS[resolution]
    rng = np.random.default_rng(42)
    _header(f"Format Converters  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    # ---- YUY2 source -------------------------------------------------------
    yuy2_np, yuy2_mx = _make_yuy2(h, w)

    for label, fn, args_mlx in [
        ("yuy2_to_rgb",  yuy2_to_rgb,  (yuy2_mx, w, h)),
        ("yuy2_to_bgr",  yuy2_to_bgr,  (yuy2_mx, w, h)),
        ("yuy2_to_rgba", yuy2_to_rgba, (yuy2_mx, w, h)),
        ("yuy2_to_bgra", yuy2_to_bgra, (yuy2_mx, w, h)),
        ("yuy2_to_y16",  yuy2_to_y16,  (yuy2_mx, w, h)),
    ]:
        stats = benchmark_component(fn, args_mlx, warmup=warmup, iterations=iterations)
        if label == "yuy2_to_rgb":
            np_stats = _benchmark_numpy(_np_yuy2_to_rgb, (yuy2_np, w, h), warmup, iterations)
            tag = _speedup_tag(stats, np_stats)
            _row(label, stats, tag)
            results[label] = {"mlx": stats, "numpy": np_stats}
        else:
            _row(label, stats)
            results[label] = stats

    # ---- UYVY source (packed as uint16) ------------------------------------
    uyvy_np = rng.integers(0, 65536, size=(h * w,), dtype=np.uint16)
    uyvy_mx = mx.array(uyvy_np)
    mx.eval(uyvy_mx)
    stats = benchmark_component(uyvy_to_yuyv, (uyvy_mx,), warmup=warmup, iterations=iterations)
    _row("uyvy_to_yuyv", stats)
    results["uyvy_to_yuyv"] = stats

    # ---- Y8I stereo split --------------------------------------------------
    y8i_np = rng.integers(0, 256, size=(2 * h * w,), dtype=np.uint8)
    y8i_mx = mx.array(y8i_np)
    mx.eval(y8i_mx)
    stats = benchmark_component(
        lambda src: split_y8i(src, w, h), (y8i_mx,),
        warmup=warmup, iterations=iterations,
    )
    _row("split_y8i", stats)
    results["split_y8i"] = stats

    # ---- Y12I stereo split -------------------------------------------------
    y12i_np = rng.integers(0, 256, size=(3 * h * w,), dtype=np.uint8)
    y12i_mx = mx.array(y12i_np)
    mx.eval(y12i_mx)
    stats = benchmark_component(
        lambda src: split_y12i(src, w, h), (y12i_mx,),
        warmup=warmup, iterations=iterations,
    )
    _row("split_y12i", stats)
    results["split_y12i"] = stats

    # ---- IR extraction -----------------------------------------------------
    ir16_np = rng.integers(0, 1024, size=(h, w), dtype=np.uint16)
    ir16_mx = mx.array(ir16_np)
    mx.eval(ir16_mx)
    for label, fn in [("extract_ir_y8", extract_ir_y8), ("extract_ir_y16", extract_ir_y16)]:
        stats = benchmark_component(fn, (ir16_mx,), warmup=warmup, iterations=iterations)
        _row(label, stats)
        results[label] = stats

    return results


def bench_pointcloud(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark PointCloudGenerator with and without distortion."""
    h, w = RESOLUTIONS[resolution]
    depth_np, depth_mx = _make_depth(h, w)
    _header(f"PointCloudGenerator  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    # No-distortion (pinhole only)
    intr_nodist = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX, "none")
    gen_nodist = PointCloudGenerator(intr_nodist, depth_scale=_DEPTH_UNITS)
    stats_mlx = benchmark_component(gen_nodist.generate, (depth_mx,), warmup=warmup, iterations=iterations)
    np_stats = _benchmark_numpy(
        _np_pointcloud,
        (depth_np, _FOCAL_PX, _FOCAL_PX, w / 2.0, h / 2.0, _DEPTH_UNITS),
        warmup, iterations,
    )
    tag = _speedup_tag(stats_mlx, np_stats)
    _row("pointcloud no-distortion (MLX)", stats_mlx, tag)
    _row("pointcloud no-distortion (NumPy)", np_stats)
    results["no_distortion"] = {"mlx": stats_mlx, "numpy": np_stats}

    # Brown-Conrady distortion (5 non-zero coefficients)
    intr_bc = CameraIntrinsics(
        w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX,
        model="brown_conrady",
        coeffs=[-0.055, 0.065, 0.001, -0.0005, -0.021],
    )
    gen_bc = PointCloudGenerator(intr_bc, depth_scale=_DEPTH_UNITS)
    stats_bc = benchmark_component(gen_bc.generate, (depth_mx,), warmup=warmup, iterations=iterations)
    _row("pointcloud brown-conrady (MLX)", stats_bc)
    results["brown_conrady"] = stats_bc

    return results


def bench_alignment(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark Aligner.align_color_to_depth at *resolution*."""
    h, w = RESOLUTIONS[resolution]
    depth_np, depth_mx = _make_depth(h, w)
    _header(f"Alignment (color→depth)  @  {resolution}  ({h}x{w})")

    # Color frame at the same resolution (identity extrinsics = same sensor)
    _, color_mx = _make_color_frame(h, w)

    d_intr = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX)
    c_intr = CameraIntrinsics(w, h, w / 2.0, h / 2.0, _FOCAL_PX, _FOCAL_PX)
    ext = CameraExtrinsics.identity()
    aligner = Aligner(d_intr, c_intr, ext, depth_scale=_DEPTH_UNITS)

    stats = benchmark_component(
        aligner.align_color_to_depth,
        (depth_mx, color_mx),
        warmup=warmup,
        iterations=iterations,
    )
    _row("align_color_to_depth (identity ext)", stats)
    return {"align_color_to_depth": stats}


def bench_spatial(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark SpatialFilter at *resolution*.

    Note: SpatialFilter has an O(W) sequential Python loop — it is the
    slowest component and the primary Metal-kernel candidate.
    """
    h, w = RESOLUTIONS[resolution]
    _, depth_mx = _make_depth(h, w)
    depth_f32 = depth_mx.astype(mx.float32)
    mx.eval(depth_f32)
    _header(f"SpatialFilter  @  {resolution}  ({h}x{w})  [sequential W-loop]")

    results: dict[str, Any] = {}

    for iters_val in (1, 2):
        filt = SpatialFilter(alpha=0.5, delta=20.0, iterations=iters_val)
        label = f"spatial alpha=0.5 delta=20 iters={iters_val}"
        stats = benchmark_component(filt.process, (depth_f32,), warmup=warmup, iterations=iterations)
        _row(label, stats)
        results[f"iters_{iters_val}"] = stats

    return results


def bench_temporal(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark TemporalFilter with warmed-up state (8 seed frames)."""
    h, w = RESOLUTIONS[resolution]
    rng = np.random.default_rng(42)
    _header(f"TemporalFilter (warmed-up)  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    for persistence in (0, 3):
        filt = TemporalFilter(alpha=0.4, delta=20.0, persistence=persistence)
        # Seed with 8 frames to fill the 8-bit history bitmask
        for seed_idx in range(8):
            seed_np = rng.integers(500, 5000, size=(h, w), dtype=np.uint16)
            seed_mx = mx.array(seed_np)
            filt.process(seed_mx)

        # Now benchmark with a fresh "live" frame
        live_np = rng.integers(500, 5000, size=(h, w), dtype=np.uint16)
        live_mx = mx.array(live_np)
        mx.eval(live_mx)

        # Re-seed after each timed call to keep the filter in a consistent state
        def _reset_and_run(frame: mx.array, _filt: TemporalFilter = filt) -> mx.array:
            return _filt.process(frame)

        stats = benchmark_component(_reset_and_run, (live_mx,), warmup=warmup, iterations=iterations)
        label = f"temporal alpha=0.4 persistence={persistence}"
        _row(label, stats)
        results[f"persistence_{persistence}"] = stats

    return results


def bench_decimation(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark DecimationFilter at scale 2 and scale 4."""
    h, w = RESOLUTIONS[resolution]
    _, depth_mx = _make_depth(h, w)
    _header(f"DecimationFilter  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    for scale in (2, 4):
        filt = DecimationFilter(scale=scale)
        out_h, out_w = h // scale, w // scale
        label = f"decimation scale={scale} → {out_h}x{out_w} ({'median' if scale <= 3 else 'valid-mean'})"
        stats = benchmark_component(filt.process, (depth_mx,), warmup=warmup, iterations=iterations)
        _row(label, stats)
        results[f"scale_{scale}"] = stats

    return results


def bench_hole_filling(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark HoleFillingFilter for all 3 modes."""
    h, w = RESOLUTIONS[resolution]
    _, depth_mx = _make_depth(h, w)
    _header(f"HoleFillingFilter  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}
    mode_names = {0: "FILL_FROM_LEFT", 1: "FARTHEST", 2: "NEAREST"}

    for mode in (0, 1, 2):
        filt = HoleFillingFilter(mode=mode)
        label = f"hole_fill mode={mode} ({mode_names[mode]})"
        stats = benchmark_component(filt.process, (depth_mx,), warmup=warmup, iterations=iterations)
        _row(label, stats)
        results[f"mode_{mode}"] = stats

    return results


def bench_disparity(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark DisparityTransform in both directions."""
    h, w = RESOLUTIONS[resolution]
    depth_np, depth_mx = _make_depth(h, w)
    _, disp_mx = _make_disp(depth_np)
    _header(f"DisparityTransform  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    for label, to_disp, frame in [
        ("depth → disparity", True, depth_mx),
        ("disparity → depth", False, disp_mx),
    ]:
        transform = DisparityTransform(
            baseline_mm=_BASELINE_MM,
            focal_px=_FOCAL_PX,
            depth_units=_DEPTH_UNITS,
            to_disparity=to_disp,
        )
        stats = benchmark_component(transform.process, (frame,), warmup=warmup, iterations=iterations)
        _row(label, stats)
        results[label.replace(" ", "_").replace("→", "to")] = stats

    return results


def bench_colorizer(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark DepthColorizer across all 10 colormaps × direct + equalized."""
    h, w = RESOLUTIONS[resolution]
    depth_np, depth_mx = _make_depth(h, w)
    _header(f"DepthColorizer  @  {resolution}  ({h}x{w})")

    results: dict[str, Any] = {}

    for cmap in sorted(DepthColorizer.COLORMAPS):
        for equalize in (False, True):
            label = f"{cmap:<18s}  equalize={equalize}"
            colorizer = DepthColorizer(colormap=cmap, equalize=equalize)
            stats = benchmark_component(
                colorizer.colorize, (depth_mx,),
                warmup=warmup, iterations=iterations,
            )
            if cmap == "jet" and not equalize:
                # MLX vs NumPy comparison for one representative cmap
                def _np_colorize(d: np.ndarray) -> np.ndarray:
                    # Minimal reference: normalize + map to uint8 (no colormap LUT)
                    valid = d > 0
                    mn, mx_val = int(d[valid].min()) if valid.any() else 0, int(d[valid].max()) if valid.any() else 1
                    normalized = np.clip((d.astype(np.float32) - mn) / max(mx_val - mn, 1), 0.0, 1.0)
                    grey = (normalized * 255).astype(np.uint8)
                    return np.stack([grey, grey, grey], axis=-1)

                np_stats = _benchmark_numpy(_np_colorize, (depth_np,), warmup, iterations)
                tag = _speedup_tag(stats, np_stats)
                _row(label, stats, tag)
                results[f"{cmap}_direct"] = {"mlx": stats, "numpy": np_stats}
            else:
                _row(label, stats)
                results[f"{cmap}_{'eq' if equalize else 'direct'}"] = stats

    return results


def bench_full_pipeline(
    resolution: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark DepthPipeline end-to-end (all 6 stages)."""
    h, w = RESOLUTIONS[resolution]
    _, depth_mx = _make_depth(h, w)
    _header(f"DepthPipeline (end-to-end)  @  {resolution}  ({h}x{w})")

    cfg = PipelineConfig(
        decimation_scale=2,
        spatial_alpha=0.5,
        spatial_delta=20.0,
        spatial_iterations=2,
        temporal_alpha=0.4,
        temporal_delta=20.0,
        temporal_persistence=3,
        hole_fill_mode=1,
        baseline_mm=_BASELINE_MM,
        focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS,
    )
    pipeline = DepthPipeline(cfg)

    # Seed temporal state with 8 frames before timing
    rng = np.random.default_rng(77)
    for _ in range(8):
        seed = mx.array(rng.integers(500, 5000, size=(h, w), dtype=np.uint16))
        pipeline.process(seed)

    # Benchmark: pipeline resets temporal state between runs would distort results,
    # so we measure the steady-state performance (temporal filter has live state).
    stats = benchmark_component(pipeline.process, (depth_mx,), warmup=warmup, iterations=iterations)
    out_h, out_w = h // 2, w // 2
    label = f"full pipeline (decimate×2 → {out_h}x{out_w}, spatial×2, temporal, hole-fill)"
    _row(label, stats)

    return {"full_pipeline": stats}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_results: dict[str, Any]) -> None:
    """Print a compact cross-resolution summary for key components."""
    print()
    _sep("=")
    print("  SUMMARY TABLE  —  mean latency (ms)  and  fps")
    _sep("=")
    resolutions = [r for r in ("480p", "720p", "1080p") if r in all_results]
    header_cols = "".join(f"  {r:>22}" for r in resolutions)
    print(f"  {'Component':<40s}{header_cols}")
    print("  " + "-" * (40 + 24 * len(resolutions)))

    def _fmt(stats: dict | None) -> str:
        if stats is None:
            return f"{'n/a':>22}"
        m = stats.get("mean_ms", stats.get("mlx", {}).get("mean_ms", None))
        if m is None:
            return f"{'n/a':>22}"
        fps = 1000.0 / m if m > 0 else 0
        return f"  {m:7.2f}ms  {fps:7.1f}fps"

    key_components: list[tuple[str, list[str]]] = [
        ("yuy2_to_rgb",          ["converters", "yuy2_to_rgb"]),
        ("split_y8i",            ["converters", "split_y8i"]),
        ("pointcloud (no-dist)", ["pointcloud", "no_distortion"]),
        ("pointcloud (BC)",      ["pointcloud", "brown_conrady"]),
        ("align color→depth",   ["alignment",  "align_color_to_depth"]),
        ("spatial iters=2",      ["spatial",    "iters_2"]),
        ("temporal persist=3",   ["temporal",   "persistence_3"]),
        ("decimation scale=2",   ["decimation", "scale_2"]),
        ("decimation scale=4",   ["decimation", "scale_4"]),
        ("hole-fill FARTHEST",   ["hole_filling", "mode_1"]),
        ("hole-fill NEAREST",    ["hole_filling", "mode_2"]),
        ("disparity→depth",     ["disparity",  "disparity_to_depth"]),
        ("colorizer jet direct", ["colorizer",  "jet_direct"]),
        ("full pipeline",        ["pipeline",   "full_pipeline"]),
    ]

    for display_name, path in key_components:
        row_str = f"  {display_name:<40s}"
        for res in resolutions:
            res_data = all_results.get(res, {})
            node: Any = res_data
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            row_str += _fmt(node if isinstance(node, dict) and "mean_ms" in node else
                            node.get("mlx") if isinstance(node, dict) and "mlx" in node else None)
        print(row_str)

    print()


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _save_results(all_results: dict[str, Any], warmup: int, iterations: int) -> Path:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_{ts}.json"

    payload = {
        "timestamp": ts,
        "warmup": warmup,
        "iterations": iterations,
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

_ALL_COMPONENTS = [
    "overhead", "converters", "pointcloud", "alignment",
    "spatial", "temporal", "decimation", "hole_filling",
    "disparity", "colorizer", "pipeline",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_all.py",
        description="Comprehensive benchmark for all realsense-mlx components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=sorted(RESOLUTIONS),
        default=["480p", "720p"],
        metavar="RES",
        help="Resolutions to benchmark (default: 480p 720p).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=_ALL_COMPONENTS + ["all"],
        default=["all"],
        help=f"Components to benchmark (default: all). Choices: {_ALL_COMPONENTS}",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up iterations before timing (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per configuration (default: 50).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON results to benchmarks/results/.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    components: set[str] = set(args.components)
    if "all" in components:
        components = set(_ALL_COMPONENTS)

    print()
    _sep()
    print(
        f"  realsense-mlx  |  comprehensive benchmark\n"
        f"  warmup={args.warmup}  iterations={args.iters}  "
        f"resolutions={args.resolutions}\n"
        f"  components={sorted(components)}"
    )
    _sep()

    mx.reset_peak_memory()
    mem_start = _memory_snapshot("start")

    all_results: dict[str, Any] = {}

    if "overhead" in components:
        bench_timer_overhead(warmup=args.warmup, iterations=args.iters)

    for res in args.resolutions:
        all_results.setdefault(res, {})

        if "converters" in components:
            all_results[res]["converters"] = bench_converters(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-converters-{res}")

        if "pointcloud" in components:
            all_results[res]["pointcloud"] = bench_pointcloud(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-pointcloud-{res}")

        if "alignment" in components:
            all_results[res]["alignment"] = bench_alignment(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-alignment-{res}")

        if "spatial" in components:
            all_results[res]["spatial"] = bench_spatial(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-spatial-{res}")

        if "temporal" in components:
            all_results[res]["temporal"] = bench_temporal(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-temporal-{res}")

        if "decimation" in components:
            all_results[res]["decimation"] = bench_decimation(
                res, warmup=args.warmup, iterations=args.iters
            )

        if "hole_filling" in components:
            all_results[res]["hole_filling"] = bench_hole_filling(
                res, warmup=args.warmup, iterations=args.iters
            )

        if "disparity" in components:
            all_results[res]["disparity"] = bench_disparity(
                res, warmup=args.warmup, iterations=args.iters
            )

        if "colorizer" in components:
            all_results[res]["colorizer"] = bench_colorizer(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-colorizer-{res}")

        if "pipeline" in components:
            all_results[res]["pipeline"] = bench_full_pipeline(
                res, warmup=args.warmup, iterations=args.iters
            )
            _memory_snapshot(f"post-pipeline-{res}")

    mem_end = _memory_snapshot("end")

    _print_summary(all_results)

    if not args.no_save:
        all_results["__memory__"] = {"start": mem_start, "end": mem_end}
        out_path = _save_results(all_results, warmup=args.warmup, iterations=args.iters)
        print(f"  Results saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
