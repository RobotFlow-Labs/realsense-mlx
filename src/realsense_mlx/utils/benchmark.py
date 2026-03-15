"""Benchmarking utilities and CLI entry point for realsense-mlx.

Provides :class:`Timer`, :func:`benchmark_component`, and a ``main``
function registered as the ``rs-mlx-bench`` console script entry point.

Design notes
------------
- :class:`Timer` calls ``mx.eval(mx.zeros(1))`` before *and* after the
  timed block to flush any pending lazy computations and ensure the
  measured interval covers actual device execution, not just graph
  construction.
- :func:`benchmark_component` runs a configurable warm-up phase so that
  JIT compilation and memory allocation costs are excluded from the
  reported statistics.
- All timing is done with ``time.perf_counter`` for sub-microsecond
  resolution on macOS.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing MLX operations with device synchronization.

    Inserts an ``mx.eval`` barrier before *and* after the timed block so
    that the measured interval reflects actual device execution time rather
    than lazy-graph construction overhead.

    Attributes
    ----------
    name:
        Optional label used in ``__repr__`` and log output.
    elapsed_ms:
        Wall-clock duration of the last timed block in milliseconds.
        Populated after ``__exit__`` is called.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from realsense_mlx.utils.benchmark import Timer
    >>> a = mx.ones((1024, 1024))
    >>> with Timer("matmul") as t:
    ...     result = a @ a
    ...     mx.eval(result)
    >>> print(f"{t.name}: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        # Flush any pending MLX work before starting the clock.
        mx.eval(mx.zeros(1))
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        # Flush MLX work queued inside the block, then stop the clock.
        mx.eval(mx.zeros(1))
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"Timer{label}(elapsed_ms={self.elapsed_ms:.3f})"


# ---------------------------------------------------------------------------
# benchmark_component
# ---------------------------------------------------------------------------

def benchmark_component(
    fn: Callable[..., mx.array],
    args: tuple[Any, ...],
    warmup: int = 5,
    iterations: int = 50,
) -> dict[str, float]:
    """Benchmark a processing function and return timing statistics.

    Parameters
    ----------
    fn:
        Callable to benchmark.  Should accept ``*args`` and return an
        ``mx.array`` (or any object that can be passed to ``mx.eval``).
    args:
        Positional arguments forwarded to ``fn`` on every call.
    warmup:
        Number of warm-up iterations excluded from statistics.  These
        allow JIT compilation and memory allocation to complete before
        measurement begins.
    iterations:
        Number of timed iterations used to compute statistics.

    Returns
    -------
    dict[str, float]
        Keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``, ``fps``.

    Examples
    --------
    >>> import mlx.core as mx
    >>> import numpy as np
    >>> from realsense_mlx.utils.benchmark import benchmark_component
    >>> from realsense_mlx.filters.colorizer import DepthColorizer
    >>> colorizer = DepthColorizer(equalize=False)
    >>> depth = mx.array(np.random.randint(500, 5000, (480, 640), dtype=np.uint16))
    >>> stats = benchmark_component(colorizer.colorize, (depth,), warmup=3, iterations=10)
    >>> assert "mean_ms" in stats and stats["fps"] > 0
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    # Warm-up: let JIT, kernel compilation and memory allocation settle.
    for _ in range(warmup):
        result = fn(*args)
        mx.eval(result)

    times: list[float] = []
    for _ in range(iterations):
        with Timer() as t:
            result = fn(*args)
            mx.eval(result)
        times.append(t.elapsed_ms)

    times_arr = np.array(times, dtype=np.float64)
    mean_ms = float(np.mean(times_arr))

    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "fps": 1000.0 / mean_ms if mean_ms > 0.0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Helpers for the CLI
# ---------------------------------------------------------------------------

def _print_stats(label: str, stats: dict[str, float]) -> None:
    """Pretty-print benchmark statistics to stdout."""
    print(
        f"  {label:<40s}  "
        f"{stats['mean_ms']:7.2f} ms  "
        f"± {stats['std_ms']:5.2f}  "
        f"[{stats['min_ms']:.2f}, {stats['max_ms']:.2f}]  "
        f"{stats['fps']:7.1f} fps"
    )


def _bench_colorizer(resolution: str, warmup: int, iterations: int) -> None:
    """Benchmark DepthColorizer across all colormaps at a given resolution."""
    from realsense_mlx.filters.colorizer import DepthColorizer

    if resolution == "480p":
        h, w = 480, 640
    elif resolution == "720p":
        h, w = 720, 1280
    elif resolution == "1080p":
        h, w = 1080, 1920
    else:
        raise ValueError(f"Unknown resolution {resolution!r}")

    rng = np.random.default_rng(42)
    depth_np = rng.integers(500, 5000, size=(h, w), dtype=np.uint16)
    depth_np[h // 4 : h // 2, w // 4 : w // 2] = 0  # inject a hole region
    depth_mx = mx.array(depth_np)
    mx.eval(depth_mx)

    print(f"\n  DepthColorizer @ {resolution} ({h}x{w})")
    print(f"  {'Component':<40s}  {'mean':>10}  {'std':>7}  {'range':>14}  {'fps':>10}")
    print("  " + "-" * 90)

    for cmap in sorted(DepthColorizer.COLORMAPS):
        for equalize in (False, True):
            label = f"{cmap}  equalize={equalize}"
            colorizer = DepthColorizer(colormap=cmap, equalize=equalize)
            stats = benchmark_component(
                colorizer.colorize, (depth_mx,),
                warmup=warmup, iterations=iterations,
            )
            _print_stats(label, stats)


def _bench_disparity(resolution: str, warmup: int, iterations: int) -> None:
    """Benchmark DisparityTransform at a given resolution."""
    from realsense_mlx.filters.disparity import DisparityTransform

    if resolution == "480p":
        h, w = 480, 640
    elif resolution == "720p":
        h, w = 720, 1280
    elif resolution == "1080p":
        h, w = 1080, 1920
    else:
        raise ValueError(f"Unknown resolution {resolution!r}")

    rng = np.random.default_rng(42)
    depth_np = rng.integers(500, 8000, size=(h, w), dtype=np.uint16)
    depth_mx = mx.array(depth_np)
    mx.eval(depth_mx)

    print(f"\n  DisparityTransform @ {resolution} ({h}x{w})")
    print(f"  {'Component':<40s}  {'mean':>10}  {'std':>7}  {'range':>14}  {'fps':>10}")
    print("  " + "-" * 90)

    for direction, to_disp in (("depth→disparity", True), ("disparity→depth", False)):
        transform = DisparityTransform(
            baseline_mm=50.0, focal_px=383.7, depth_units=0.001,
            to_disparity=to_disp,
        )
        if to_disp:
            frame = depth_mx
        else:
            # Build a float32 disparity frame for the reverse direction.
            disp_np = (50.0 * 383.7 * 32.0 / depth_np.astype(np.float32)).astype(np.float32)
            disp_np[depth_np == 0] = 0.0
            frame = mx.array(disp_np)
            mx.eval(frame)

        stats = benchmark_component(
            transform.process, (frame,),
            warmup=warmup, iterations=iterations,
        )
        _print_stats(direction, stats)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the ``rs-mlx-bench`` console script.

    Usage
    -----
    .. code-block:: bash

        rs-mlx-bench                          # all components, 480p + 720p
        rs-mlx-bench --components colorizer   # colorizer only
        rs-mlx-bench --resolutions 720p       # 720p only
        rs-mlx-bench --warmup 10 --iters 100  # custom timing parameters
    """
    parser = argparse.ArgumentParser(
        prog="rs-mlx-bench",
        description="Benchmark realsense-mlx processing components on Apple Silicon.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["colorizer", "disparity", "all"],
        default=["all"],
        help="Components to benchmark (default: all).",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=["480p", "720p", "1080p"],
        default=["480p", "720p"],
        help="Resolutions to test (default: 480p 720p).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations before timing (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed iterations (default: 50).",
    )

    args = parser.parse_args()

    components: set[str] = set(args.components)
    if "all" in components:
        components = {"colorizer", "disparity"}

    resolutions: list[str] = args.resolutions

    print("=" * 96)
    print(f"  realsense-mlx benchmark  |  warmup={args.warmup}  iterations={args.iters}")
    print("=" * 96)

    for res in resolutions:
        if "colorizer" in components:
            _bench_colorizer(res, warmup=args.warmup, iterations=args.iters)
        if "disparity" in components:
            _bench_disparity(res, warmup=args.warmup, iterations=args.iters)

    print("\n" + "=" * 96)


if __name__ == "__main__":
    main()
