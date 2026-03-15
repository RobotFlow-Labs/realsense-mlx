"""Quick benchmark script for all realsense-mlx components.

Run from the project root::

    python benchmarks/bench_all.py
    python benchmarks/bench_all.py --resolutions 480p 720p --iters 100

This script exercises every processing component at the two standard
RealSense streaming resolutions (480p = 640x480, 720p = 1280x720) and
prints a formatted timing table to stdout.

Components benchmarked
----------------------
- DepthColorizer (all 10 colormaps × direct + equalized)
- DisparityTransform (depth→disparity and disparity→depth)

Usage as a standalone script is intentional — it avoids pytest overhead
and gives deterministic iteration counts without framework interference.
For automated CI timing use ``rs-mlx-bench`` (the console-script entry
point installed by pyproject.toml) instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run from the project root without
# having performed an editable install.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import mlx.core as mx

from realsense_mlx.filters.colorizer import DepthColorizer
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.utils.benchmark import Timer, benchmark_component


# ---------------------------------------------------------------------------
# Resolution table
# ---------------------------------------------------------------------------

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_COL_W = 44

def _header(title: str) -> None:
    width = _COL_W + 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print(
        f"  {'Component':<{_COL_W}s}  "
        f"{'mean':>9}  {'std':>7}  {'min':>7}  {'max':>7}  {'fps':>9}"
    )
    print("  " + "-" * (width - 2))


def _row(label: str, stats: dict[str, float]) -> None:
    print(
        f"  {label:<{_COL_W}s}  "
        f"{stats['mean_ms']:8.2f}ms  "
        f"±{stats['std_ms']:5.2f}  "
        f"{stats['min_ms']:6.2f}  "
        f"{stats['max_ms']:6.2f}  "
        f"{stats['fps']:8.1f} fps"
    )


# ---------------------------------------------------------------------------
# Per-component benchmarks
# ---------------------------------------------------------------------------

def bench_colorizer(
    resolution: str,
    warmup: int,
    iterations: int,
) -> None:
    """Benchmark DepthColorizer across all 10 colormaps at *resolution*."""
    h, w = RESOLUTIONS[resolution]
    rng = np.random.default_rng(42)
    depth_np = rng.integers(500, 5000, size=(h, w), dtype=np.uint16)
    # Inject a rectangular invalid region to exercise the equalization path.
    depth_np[h // 4 : h // 2, w // 4 : w // 2] = 0
    depth_mx = mx.array(depth_np)
    mx.eval(depth_mx)

    _header(f"DepthColorizer  @  {resolution}  ({h} x {w})")

    for cmap in sorted(DepthColorizer.COLORMAPS):
        for equalize in (False, True):
            label = f"{cmap:<18s}  equalize={equalize}"
            colorizer = DepthColorizer(colormap=cmap, equalize=equalize)
            stats = benchmark_component(
                colorizer.colorize,
                (depth_mx,),
                warmup=warmup,
                iterations=iterations,
            )
            _row(label, stats)


def bench_disparity(
    resolution: str,
    warmup: int,
    iterations: int,
) -> None:
    """Benchmark DisparityTransform in both directions at *resolution*."""
    h, w = RESOLUTIONS[resolution]
    rng = np.random.default_rng(42)

    depth_np = rng.integers(500, 8000, size=(h, w), dtype=np.uint16)
    depth_mx = mx.array(depth_np)
    mx.eval(depth_mx)

    # Pre-compute a float32 disparity frame for the reverse direction.
    d2d = 50.0 * 383.7 * 32.0 / 0.001
    safe = np.where(depth_np > 0, depth_np.astype(np.float32), 1.0)
    disp_np = np.where(depth_np > 0, d2d / safe, 0.0).astype(np.float32)
    disp_mx = mx.array(disp_np)
    mx.eval(disp_mx)

    _header(f"DisparityTransform  @  {resolution}  ({h} x {w})")

    for label, to_disp, frame in (
        ("depth → disparity", True, depth_mx),
        ("disparity → depth", False, disp_mx),
    ):
        transform = DisparityTransform(
            baseline_mm=50.0,
            focal_px=383.7,
            depth_units=0.001,
            to_disparity=to_disp,
        )
        stats = benchmark_component(
            transform.process,
            (frame,),
            warmup=warmup,
            iterations=iterations,
        )
        _row(label, stats)


def bench_timer_overhead() -> None:
    """Measure Timer synchronization overhead (two mx.eval barriers)."""
    _header("Timer synchronization overhead (baseline)")
    times = []
    for _ in range(20):
        with Timer() as t:
            pass  # Measure bare overhead of two mx.eval calls.
        times.append(t.elapsed_ms)
    arr = np.array(times)
    _row(
        "Timer barrier overhead",
        {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "fps": 1000.0 / float(np.mean(arr)),
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_all.py",
        description="Benchmark all realsense-mlx components.",
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
        choices=["colorizer", "disparity", "overhead", "all"],
        default=["all"],
        help="Components to include (default: all).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    components: set[str] = set(args.components)
    if "all" in components:
        components = {"colorizer", "disparity", "overhead"}

    print(
        f"\nrealsense-mlx benchmarks  |  "
        f"warmup={args.warmup}  iterations={args.iters}  "
        f"resolutions={args.resolutions}"
    )

    if "overhead" in components:
        bench_timer_overhead()

    for res in args.resolutions:
        if "colorizer" in components:
            bench_colorizer(res, warmup=args.warmup, iterations=args.iters)
        if "disparity" in components:
            bench_disparity(res, warmup=args.warmup, iterations=args.iters)

    print()


if __name__ == "__main__":
    main()
