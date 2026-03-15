#!/usr/bin/env python3
"""Pre-push self-test — runs in ~5 seconds, covers all major subsystems.

Gives an instant green/red signal before you push.

Usage
-----
    python scripts/quick_check.py
    python scripts/quick_check.py --no-tests   # skip pytest (faster)
    python scripts/quick_check.py --verbose    # show all sub-step detail

Output::

    Tests:     938 passing (7.2s)
    Filters:   ALL passing (10 scenes x 8 filters)
    Metal:     3 kernels compiled OK
    Benchmark: Pipeline 271 FPS, PC 3196 FPS, Align 4320 FPS
    Memory:    45 MB active, 92 MB peak, no leaks
    Status:    READY TO SHIP
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable when run directly
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_DIM = "\033[2m"

_CHECKMARK = "OK"
_CROSS = "FAIL"
_WARN = "WARN"


def _ok(msg: str) -> str:
    return f"{_GREEN}{_BOLD}{_CHECKMARK}{_RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}{_BOLD}{_CROSS}{_RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}{_BOLD}{_WARN}{_RESET}  {msg}"


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def _sync() -> None:
    mx.eval(mx.zeros(1))


def _bench_fn(fn, warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    """Return (mean_ms, fps).  Fully materialises MLX results."""
    for _ in range(warmup):
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            arrays = [x for x in r if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)

    times: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            arrays = [x for x in r if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = float(np.mean(times))
    return mean, (1000.0 / mean if mean > 0.0 else float("inf"))


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _depth(H: int, W: int, seed: int = 42) -> mx.array:
    rng = np.random.default_rng(seed)
    arr = rng.integers(500, 5000, size=(H, W), dtype=np.uint16)
    arr[H // 4 : H // 2, W // 4 : W // 2] = 0
    d = mx.array(arr)
    mx.eval(d)
    return d


def _color(H: int, W: int, seed: int = 7) -> mx.array:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    c = mx.array(arr)
    mx.eval(c)
    return c


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def check_pytest(verbose: bool) -> CheckResult:
    """Run pytest and report pass/fail count."""
    t0 = time.perf_counter()
    venv_pytest = _root / ".venv" / "bin" / "pytest"
    pytest_cmd = str(venv_pytest) if venv_pytest.exists() else "pytest"

    cmd = [pytest_cmd, "tests/", "--tb=short", "-q"]
    if verbose:
        cmd.append("-v")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_root),
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        output = proc.stdout + proc.stderr

        # Extract summary line, e.g. "938 passed in 7.2s" or "1 failed in 4.1s"
        summary_line = ""
        for line in reversed(output.splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                summary_line = line.strip()
                break

        passed = proc.returncode == 0
        detail = summary_line if summary_line else f"exit code {proc.returncode}"

        if verbose and not passed:
            print(_DIM + output + _RESET)

        return CheckResult("Tests", passed, detail, elapsed)
    except subprocess.TimeoutExpired:
        return CheckResult("Tests", False, "timed out after 120s", time.perf_counter() - t0)
    except FileNotFoundError:
        return CheckResult("Tests", False, "pytest not found — run: uv pip install -e '.[dev]'")


def check_filters(verbose: bool) -> CheckResult:
    """Validate all depth filters across 10 synthetic scenes."""
    from realsense_mlx.filters import (
        DecimationFilter,
        HoleFillingFilter,
        SpatialFilter,
        TemporalFilter,
    )
    from realsense_mlx.filters.colorizer import DepthColorizer
    from realsense_mlx.filters.disparity import DisparityTransform

    H, W = 480, 640
    rng = np.random.default_rng(0)

    def _scene(name: str) -> np.ndarray:
        if name == "flat":
            return np.full((H, W), 1500, dtype=np.uint16)
        if name == "ramp":
            return np.tile(np.linspace(500, 5000, W, dtype=np.float32).astype(np.uint16), (H, 1))
        if name == "sphere":
            yy, xx = np.mgrid[:H, :W]
            r = np.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)
            d = np.full((H, W), 2500, dtype=np.float32)
            m = r < min(H, W) // 3
            d[m] = 2000 - 500 * np.cos(np.pi * r[m] / (min(H, W) // 3))
            return d.clip(1, 65535).astype(np.uint16)
        if name == "noisy":
            return rng.normal(2000, 50, (H, W)).clip(1, 65535).astype(np.uint16)
        if name == "holes_15":
            arr = rng.integers(500, 5000, (H, W), dtype=np.uint16)
            arr[rng.random((H, W)) < 0.15] = 0
            return arr
        if name == "holes_50":
            arr = rng.integers(500, 5000, (H, W), dtype=np.uint16)
            arr[rng.random((H, W)) < 0.50] = 0
            return arr
        if name == "extreme":
            arr = np.zeros((H, W), dtype=np.uint16)
            arr[: H // 3] = 1
            arr[H // 3 : 2 * H // 3] = 65535
            return arr
        if name == "all_zero":
            return np.zeros((H, W), dtype=np.uint16)
        if name == "all_max":
            return np.full((H, W), 65535, dtype=np.uint16)
        # gradient
        return rng.integers(500, 5000, (H, W), dtype=np.uint16)

    scene_names = [
        "flat", "ramp", "sphere", "noisy",
        "holes_15", "holes_50", "extreme",
        "all_zero", "all_max", "gradient",
    ]

    failures: list[str] = []
    total = 0

    for scene_name in scene_names:
        arr = _scene(scene_name)
        d = mx.array(arr)
        df = d.astype(mx.float32)

        checks: list[tuple[str, bool]] = []
        try:
            out = DecimationFilter(scale=2).process(d)
            mx.eval(out)
            checks.append(("DecimationFilter", out.shape[0] <= H))

            sf = SpatialFilter(iterations=2)
            out = sf.process(df)
            mx.eval(out)
            checks.append(("SpatialFilter", out.shape == df.shape))

            tf = TemporalFilter(alpha=0.4)
            for _ in range(3):
                out = tf.process(d)
            mx.eval(out)
            checks.append(("TemporalFilter", out.shape == d.shape))

            for mode in [0, 1, 2]:
                out = HoleFillingFilter(mode=mode).process(d)
                mx.eval(out)
                checks.append((f"HoleFilling/m{mode}", out.shape == d.shape))

            dt = DisparityTransform(baseline_mm=50, focal_px=383.7, depth_units=0.001)
            disp = dt.process(d)
            mx.eval(disp)
            checks.append(("Disparity", disp.shape == d.shape))

            col = DepthColorizer(colormap="jet").colorize(d)
            mx.eval(col)
            checks.append(("Colorizer", col.shape == (*d.shape, 3)))

        except Exception as exc:
            failures.append(f"{scene_name}: {exc}")
            continue

        for name, ok in checks:
            total += 1
            if not ok:
                failures.append(f"{scene_name}/{name}: shape mismatch")
            elif verbose:
                pass  # too verbose

    passed = len(failures) == 0
    detail = (
        f"ALL passing ({len(scene_names)} scenes x 8 filters)"
        if passed
        else f"{len(failures)} failures: {', '.join(failures[:3])}"
    )
    return CheckResult("Filters", passed, detail)


def check_metal_kernels(verbose: bool) -> CheckResult:
    """Confirm the three Metal-backed kernels compile and produce valid output."""
    from realsense_mlx.geometry.align import Aligner
    from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
    from realsense_mlx.filters.spatial import SpatialFilter
    from realsense_mlx.filters.hole_filling import HoleFillingFilter

    H, W = 480, 640
    d = _depth(H, W)
    c = _color(H, W)
    df = d.astype(mx.float32)
    mx.eval(df)

    intr = CameraIntrinsics(W, H, W / 2.0, H / 2.0, W * 0.6, W * 0.6)
    extr = CameraExtrinsics.identity()

    failures: list[str] = []
    compiled: list[str] = []

    # Kernel 1: Metal spatial filter
    try:
        sf = SpatialFilter(use_metal=True, iterations=2)
        out = sf.process(df)
        mx.eval(out)
        if out.shape == df.shape:
            compiled.append("SpatialFilter[Metal]")
        else:
            failures.append("SpatialFilter[Metal]: wrong shape")
    except Exception as exc:
        failures.append(f"SpatialFilter[Metal]: {exc}")

    # Kernel 2: Metal alignment
    try:
        al = Aligner(intr, intr, extr, 0.001, use_metal=True)
        out = al.align_color_to_depth(d, c)
        mx.eval(out)
        if out.shape[:2] == d.shape:
            compiled.append("Aligner[Metal]")
        else:
            failures.append("Aligner[Metal]: wrong shape")
    except Exception as exc:
        failures.append(f"Aligner[Metal]: {exc}")

    # Kernel 3: Metal hole filling
    try:
        hf = HoleFillingFilter(mode=0, use_metal=True)
        out = hf.process(d)
        mx.eval(out)
        if out.shape == d.shape:
            compiled.append("HoleFilling[Metal]")
        else:
            failures.append("HoleFilling[Metal]: wrong shape")
    except Exception as exc:
        failures.append(f"HoleFilling[Metal]: {exc}")

    passed = len(failures) == 0
    n = len(compiled)
    detail = (
        f"{n} kernel{'s' if n != 1 else ''} compiled OK"
        if passed
        else f"{len(failures)} failed: {', '.join(failures)}"
    )
    return CheckResult("Metal", passed, detail)


def check_benchmark(verbose: bool) -> CheckResult:
    """Run 10-iteration micro-benchmarks and report headline FPS numbers."""
    from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    from realsense_mlx.geometry.pointcloud import PointCloudGenerator
    from realsense_mlx.geometry.align import Aligner
    from realsense_mlx.geometry.intrinsics import CameraExtrinsics

    H, W = 480, 640
    d = _depth(H, W)
    c = _color(H, W)
    intr = CameraIntrinsics(W, H, W / 2.0, H / 2.0, W * 0.6, W * 0.6)
    extr = CameraExtrinsics.identity()

    pipe = DepthPipeline(PipelineConfig(decimation_scale=2))
    pc = PointCloudGenerator(intr, 0.001)
    al = Aligner(intr, intr, extr, 0.001)

    # Warmup
    for _ in range(5):
        mx.eval(pipe.process(d))
        mx.eval(pc.generate(d))
        mx.eval(al.align_color_to_depth(d, c))

    _, pipe_fps = _bench_fn(lambda: pipe.process(d))
    _, pc_fps = _bench_fn(lambda: pc.generate(d))
    _, al_fps = _bench_fn(lambda: al.align_color_to_depth(d, c))

    warnings: list[str] = []
    if pipe_fps < 30:
        warnings.append(f"Pipeline below 30 FPS ({pipe_fps:.0f})")
    if pc_fps < 30:
        warnings.append(f"PointCloud below 30 FPS ({pc_fps:.0f})")
    if al_fps < 30:
        warnings.append(f"Alignment below 30 FPS ({al_fps:.0f})")

    detail = (
        f"Pipeline {pipe_fps:.0f} FPS, "
        f"PC {pc_fps:.0f} FPS, "
        f"Align {al_fps:.0f} FPS"
    )
    passed = len(warnings) == 0
    result = CheckResult("Benchmark", passed, detail)
    result.warnings = warnings
    return result


def check_memory(verbose: bool) -> CheckResult:
    """Process 200 frames and verify memory growth stays under 100 MB."""
    from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig

    pipe = DepthPipeline(PipelineConfig())
    rng = np.random.default_rng(99)

    mx.reset_peak_memory()
    mem_before = mx.get_active_memory() / 1e6

    for i in range(200):
        arr = rng.integers(500, 5000, (480, 640), dtype=np.uint16)
        d = mx.array(arr)
        out = pipe.process(d)
        mx.eval(out)
        if i % 50 == 49:
            _sync()

    _sync()
    mem_after = mx.get_active_memory() / 1e6
    peak_mb = mx.get_peak_memory() / 1e6
    growth = mem_after - mem_before

    passed = growth < 100.0
    leak_tag = "no leaks" if passed else f"LEAK +{growth:.0f} MB"
    detail = f"{mem_after:.0f} MB active, {peak_mb:.0f} MB peak, {leak_tag}"

    warnings: list[str] = []
    if growth >= 50:
        warnings.append(f"Memory grew by {growth:.0f} MB (threshold: 100 MB)")

    result = CheckResult("Memory", passed, detail)
    result.warnings = warnings
    return result


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

_LABEL_W = 12


def _print_result(r: CheckResult) -> None:
    icon = _ok if r.passed else _fail
    label_padded = f"{r.label}:".ljust(_LABEL_W)
    time_tag = f"{_DIM}({r.elapsed_s:.1f}s){_RESET}" if r.elapsed_s > 0.1 else ""
    print(f"{icon(label_padded)}  {r.detail}  {time_tag}")
    for w in r.warnings:
        print(f"           {_YELLOW}{w}{_RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-push self-test for realsense-mlx.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip the pytest step (faster, but less thorough).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show extra detail for each check.",
    )
    args = parser.parse_args()

    print()
    print(f"{_BOLD}realsense-mlx quick_check{_RESET}")
    print("─" * 60)

    results: list[CheckResult] = []

    if not args.no_tests:
        r = check_pytest(args.verbose)
        _print_result(r)
        results.append(r)

    checks = [
        ("Filters", check_filters),
        ("Metal", check_metal_kernels),
        ("Benchmark", check_benchmark),
        ("Memory", check_memory),
    ]
    for _label, fn in checks:
        r = fn(args.verbose)
        _print_result(r)
        results.append(r)

    all_passed = all(r.passed for r in results)
    has_warnings = any(r.warnings for r in results)

    print("─" * 60)
    if all_passed:
        status_colour = _YELLOW if has_warnings else _GREEN
        status_text = "READY TO SHIP (with warnings)" if has_warnings else "READY TO SHIP"
        print(f"Status:      {status_colour}{_BOLD}{status_text}{_RESET}")
        return 0
    else:
        failed_names = [r.label for r in results if not r.passed]
        print(
            f"Status:      {_RED}{_BOLD}NOT READY{_RESET}  "
            f"({', '.join(failed_names)} failed)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
