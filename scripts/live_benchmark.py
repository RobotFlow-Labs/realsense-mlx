#!/usr/bin/env python3
"""Live performance monitor — runs continuously, shows FPS for all components.

Processes synthetic depth frames in a tight loop and refreshes a terminal
dashboard every second.  Useful during development to catch regressions
the instant they happen — no test run needed.

Usage
-----
    python scripts/live_benchmark.py              # 480p (default)
    python scripts/live_benchmark.py --720p       # test at 720p
    python scripts/live_benchmark.py --1080p      # test at 1080p
    python scripts/live_benchmark.py --camera     # use webcam frames

Output (updates in-place every second)::

    ┌─────────────────────────────────────────────────┐
    │  realsense-mlx Live Performance Monitor         │
    │  Resolution: 480x640 | Memory: 45 MB            │
    ├─────────────────────────────────────────────────┤
    │  Pipeline:    273 FPS  (3.7ms)   ████████████  │
    │  PointCloud: 3196 FPS  (0.3ms)   █████████████ │
    │  Alignment:  4320 FPS  (0.2ms)   █████████████ │
    │  MeshGen:     209 FPS  (4.8ms)   ████████      │
    │  Colorizer:  1100 FPS  (0.9ms)   █████████████ │
    │  Bilateral:    72 FPS (13.9ms)   ████          │
    │  E2E Full:    136 FPS  (7.4ms)   ██████        │
    ├─────────────────────────────────────────────────┤
    │  Frames: 1,234 | Uptime: 12.3s | Peak: 92 MB  │
    └─────────────────────────────────────────────────┘

Press Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable when run directly from the project root
# ---------------------------------------------------------------------------
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from realsense_mlx.filters.bilateral import BilateralFilter
from realsense_mlx.filters.colorizer import DepthColorizer
from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
from realsense_mlx.geometry.align import Aligner
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.mesh import DepthMeshGenerator
from realsense_mlx.geometry.pointcloud import PointCloudGenerator

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

_BOX_W: int = 57           # total inner width of the box (excluding │ borders)
_BAR_MAX_W: int = 13       # maximum bar length in characters
_REFRESH_HZ: float = 1.0   # seconds between dashboard refreshes

# ANSI escape sequences
_ESC = "\033["
_CLEAR_LINE = _ESC + "2K"
_CURSOR_UP = _ESC + "{n}A"
_HIDE_CURSOR = _ESC + "?25l"
_SHOW_CURSOR = _ESC + "?25h"
_RESET = _ESC + "0m"
_BOLD = _ESC + "1m"
_GREEN = _ESC + "32m"
_YELLOW = _ESC + "33m"
_RED = _ESC + "31m"
_CYAN = _ESC + "36m"
_DIM = _ESC + "2m"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Number of rows rendered by the dashboard so the cursor can be rewound.
_DASHBOARD_ROWS: int = 0


@dataclass
class ComponentStat:
    """Rolling statistics for a single pipeline component."""

    name: str
    fps: float = 0.0
    ms: float = 0.0
    # Ring buffer of the last N per-second FPS readings (for sparkline).
    history: list[float] = field(default_factory=list)
    # Maximum FPS ever observed — used to normalise bar width.
    peak_fps: float = 1.0

    def update(self, fps: float, ms: float) -> None:
        self.fps = fps
        self.ms = ms
        self.history.append(fps)
        if len(self.history) > 60:
            self.history.pop(0)
        if fps > self.peak_fps:
            self.peak_fps = fps


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    """Force MLX to complete all pending operations."""
    mx.eval(mx.zeros(1))


def _bench(fn: Callable, iters: int = 10) -> tuple[float, float]:
    """Return (mean_ms, fps) for *fn* across *iters* calls.

    Each call is individually timed with MLX device synchronisation so that
    lazy evaluation does not bleed across measurements.
    """
    _sync()
    times: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        result = fn()
        # Materialise whatever the function returned.
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, (tuple, list)):
            arrays = [r for r in result if isinstance(r, mx.array)]
            if arrays:
                mx.eval(*arrays)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    mean_ms = float(np.mean(times))
    fps = 1000.0 / mean_ms if mean_ms > 0.0 else float("inf")
    return mean_ms, fps


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _make_depth(H: int, W: int, rng: np.random.Generator) -> mx.array:
    arr = rng.integers(500, 5000, size=(H, W), dtype=np.uint16)
    arr[H // 4 : H // 2, W // 4 : W // 2] = 0  # rectangular hole
    d = mx.array(arr)
    mx.eval(d)
    return d


def _make_color(H: int, W: int, rng: np.random.Generator) -> mx.array:
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    c = mx.array(arr)
    mx.eval(c)
    return c


# ---------------------------------------------------------------------------
# Component factories (built once, reused every second)
# ---------------------------------------------------------------------------


class LiveComponents:
    """Holds all processing components for a given resolution."""

    def __init__(self, H: int, W: int) -> None:
        self.H = H
        self.W = W
        self._rng = np.random.default_rng(42)

        intr = CameraIntrinsics(W, H, W / 2.0, H / 2.0, W * 0.6, W * 0.6)
        extr = CameraExtrinsics.identity()

        cfg = PipelineConfig(decimation_scale=2)
        self.pipeline = DepthPipeline(cfg)
        self.colorizer = DepthColorizer(colormap="jet")
        self.pc_gen = PointCloudGenerator(intr, depth_scale=0.001)
        self.aligner = Aligner(intr, intr, extr, depth_scale=0.001)
        self.bilateral = BilateralFilter(sigma_spatial=5.0, sigma_range=30.0, kernel_size=5)

        # Decimated intrinsics (after scale=2)
        dec_H, dec_W = H // 2, W // 2
        dec_intr = CameraIntrinsics(dec_W, dec_H, dec_W / 2.0, dec_H / 2.0, dec_W * 0.6, dec_W * 0.6)
        self.mesh_gen = DepthMeshGenerator(max_edge_length=0.05)
        self.pc_gen_dec = PointCloudGenerator(dec_intr, depth_scale=0.001)

        # Pre-compute frame fixtures (refreshed every 10 seconds to vary data)
        self._depth: mx.array = _make_depth(H, W, self._rng)
        self._color: mx.array = _make_color(H, W, self._rng)
        self._depth_f32 = self._depth.astype(mx.float32)
        mx.eval(self._depth_f32)
        # Guide image for bilateral (single channel)
        self._guide = self._depth_f32

        # Warm up every component so JIT compilation is excluded from the
        # first live reading.
        self._warmup()
        self._fixture_tick = time.monotonic()

    def _warmup(self) -> None:
        """Run a few forward passes through each component to trigger JIT."""
        for _ in range(3):
            filt = self.pipeline.process(self._depth)
            mx.eval(filt)
            pts = self.pc_gen.generate(self._depth)
            mx.eval(pts)
            aln = self.aligner.align_color_to_depth(self._depth, self._color)
            mx.eval(aln)
            col = self.colorizer.colorize(self._depth)
            mx.eval(col)
            bf = self.bilateral.process(self._depth_f32, self._guide)
            mx.eval(bf)
        # Mesh warmup on decimated depth
        dec = self.pipeline.process(self._depth)
        mx.eval(dec)
        pts_dec = self.pc_gen_dec.generate(dec)
        mx.eval(pts_dec)
        verts, faces = self.mesh_gen.generate(pts_dec)
        mx.eval(verts, faces)

    def refresh_fixtures(self) -> None:
        """Rotate to fresh synthetic frames every 10 seconds."""
        now = time.monotonic()
        if now - self._fixture_tick > 10.0:
            self._depth = _make_depth(self.H, self.W, self._rng)
            self._color = _make_color(self.H, self.W, self._rng)
            self._depth_f32 = self._depth.astype(mx.float32)
            mx.eval(self._depth, self._color, self._depth_f32)
            self._guide = self._depth_f32
            self._fixture_tick = now

    def sample_all(self) -> dict[str, tuple[float, float]]:
        """Run 10-iteration micro-benchmarks on every component.

        Returns a mapping of component name → (mean_ms, fps).
        """
        self.refresh_fixtures()
        d = self._depth
        c = self._color
        df = self._depth_f32
        guide = self._guide

        results: dict[str, tuple[float, float]] = {}

        # 1. Full filter pipeline
        results["Pipeline"] = _bench(lambda: self.pipeline.process(d))

        # 2. Point cloud (full-resolution)
        results["PointCloud"] = _bench(lambda: self.pc_gen.generate(d))

        # 3. Alignment
        results["Alignment"] = _bench(lambda: self.aligner.align_color_to_depth(d, c))

        # 4. Mesh generation (operates on decimated depth)
        dec = self.pipeline.process(d)
        mx.eval(dec)
        pts_dec = self.pc_gen_dec.generate(dec)
        mx.eval(pts_dec)
        results["MeshGen"] = _bench(lambda: self.mesh_gen.generate(pts_dec))

        # 5. Colorizer
        results["Colorizer"] = _bench(lambda: self.colorizer.colorize(d))

        # 6. Bilateral filter
        results["Bilateral"] = _bench(lambda: self.bilateral.process(df, guide))

        # 7. End-to-end: pipeline + point cloud + colorize
        def _e2e() -> tuple[mx.array, mx.array, mx.array]:
            filt = self.pipeline.process(d)
            pts = self.pc_gen.generate(d)
            col = self.colorizer.colorize(filt)
            mx.eval(filt, pts, col)
            return filt, pts, col

        results["E2E Full"] = _bench(_e2e)

        return results


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

def _fps_colour(fps: float) -> str:
    if fps >= 60:
        return _GREEN
    if fps >= 30:
        return _YELLOW
    return _RED


def _bar(fps: float, peak_fps: float, max_w: int = _BAR_MAX_W) -> str:
    """Render a proportional ASCII bar chart segment."""
    if peak_fps <= 0:
        return " " * max_w
    filled = min(int(round(fps / peak_fps * max_w)), max_w)
    return "█" * filled + " " * (max_w - filled)


def _border_top() -> str:
    return "┌" + "─" * _BOX_W + "┐"


def _border_mid() -> str:
    return "├" + "─" * _BOX_W + "┤"


def _border_bot() -> str:
    return "└" + "─" * _BOX_W + "┘"


def _row(content: str) -> str:
    """Pad *content* to exactly _BOX_W chars and wrap in box characters."""
    return "│" + content.ljust(_BOX_W) + "│"


def _render_dashboard(
    stats: dict[str, ComponentStat],
    H: int,
    W: int,
    frame_count: int,
    uptime_s: float,
    peak_mb: float,
    active_mb: float,
    using_camera: bool,
) -> list[str]:
    """Build the dashboard as a list of lines (no trailing newlines)."""
    lines: list[str] = []
    source = "camera" if using_camera else "synthetic"

    lines.append(_border_top())
    lines.append(_row(f"  {_BOLD}realsense-mlx Live Performance Monitor{_RESET}"))
    lines.append(_row(f"  Resolution: {H}x{W}  |  Source: {source}  |  Mem: {active_mb:.0f} MB active"))
    lines.append(_border_mid())

    # Find global peak fps for relative bar widths
    global_peak = max((s.peak_fps for s in stats.values()), default=1.0)

    for name, stat in stats.items():
        colour = _fps_colour(stat.fps)
        bar = _bar(stat.fps, global_peak)
        # Layout: "  Name:  NNNN FPS  (NN.Nms)  BBBBBBBBBBBBB"
        label = f"{name}:"
        fps_str = f"{stat.fps:6.0f} FPS"
        ms_str = f"({stat.ms:5.1f}ms)"
        line = f"  {label:<12s} {colour}{fps_str}{_RESET}  {_DIM}{ms_str}{_RESET}  {colour}{bar}{_RESET}"
        lines.append(_row(line))

    lines.append(_border_mid())
    status = (
        f"  Frames: {frame_count:,}  |  Uptime: {uptime_s:.1f}s  |  Peak: {peak_mb:.0f} MB"
    )
    lines.append(_row(status))
    lines.append(_border_bot())
    return lines


def _print_dashboard(
    lines: list[str],
    first_render: bool,
) -> None:
    """Overwrite the previous dashboard in the terminal."""
    global _DASHBOARD_ROWS  # noqa: PLW0603
    if not first_render and _DASHBOARD_ROWS > 0:
        # Move cursor up to overwrite previous output.
        sys.stdout.write(f"\033[{_DASHBOARD_ROWS}A")
    for line in lines:
        sys.stdout.write(_CLEAR_LINE + line + "\n")
    sys.stdout.flush()
    _DASHBOARD_ROWS = len(lines)


# ---------------------------------------------------------------------------
# Camera source (optional)
# ---------------------------------------------------------------------------


def _try_open_camera() -> "Optional[object]":
    """Try to open a webcam via OpenCV.  Returns cap or None."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _grab_camera_depth(cap: "object", H: int, W: int) -> Optional[mx.array]:
    """Grab one frame, convert luminance to synthetic uint16 depth."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return None
    ret, frame = cap.read()  # type: ignore[union-attr]
    if not ret:
        return None
    resized = cv2.resize(frame, (W, H))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    fake = (gray.astype(np.uint16) * 20 + 500)
    d = mx.array(fake)
    mx.eval(d)
    return d


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous live benchmark for realsense-mlx.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    res_group = parser.add_mutually_exclusive_group()
    res_group.add_argument(
        "--480p",
        dest="resolution",
        action="store_const",
        const="480p",
        help="Run at 480×640 (default).",
    )
    res_group.add_argument(
        "--720p",
        dest="resolution",
        action="store_const",
        const="720p",
        help="Run at 720×1280.",
    )
    res_group.add_argument(
        "--1080p",
        dest="resolution",
        action="store_const",
        const="1080p",
        help="Run at 1080×1920.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use a connected webcam/RealSense as the depth source (falls back to synthetic).",
    )
    parser.set_defaults(resolution="480p")
    return parser


_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


def main() -> None:
    args = _build_arg_parser().parse_args()
    H, W = _RESOLUTIONS[args.resolution]

    # Hide cursor for cleaner display; restore on exit.
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    _stop = False

    def _on_sigint(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    cap = None
    if args.camera:
        cap = _try_open_camera()
        if cap is None:
            print("No camera found — falling back to synthetic depth.")
            args.camera = False

    print(f"Initialising components at {args.resolution} ({H}x{W})…")
    components = LiveComponents(H, W)
    mx.reset_peak_memory()

    stats: dict[str, ComponentStat] = {
        name: ComponentStat(name=name)
        for name in ["Pipeline", "PointCloud", "Alignment", "MeshGen",
                     "Colorizer", "Bilateral", "E2E Full"]
    }

    t_start = time.monotonic()
    t_last_refresh = t_start
    frame_count = 0
    first_render = True

    print(f"Running — press Ctrl+C to stop.\n")

    try:
        while not _stop:
            # If camera mode, inject a real frame into the fixture.
            if args.camera and cap is not None:
                live_depth = _grab_camera_depth(cap, H, W)
                if live_depth is not None:
                    components._depth = live_depth
                    components._depth_f32 = live_depth.astype(mx.float32)
                    mx.eval(components._depth_f32)
                    components._guide = components._depth_f32

            # Sample all components (10 iterations each).
            raw = components.sample_all()
            frame_count += 10  # 10 iters per component, Pipeline is representative

            # Update stat objects.
            for name, (ms, fps) in raw.items():
                if name in stats:
                    stats[name].update(fps, ms)

            # Refresh display at 1 Hz.
            now = time.monotonic()
            if now - t_last_refresh >= _REFRESH_HZ:
                peak_mb = mx.get_peak_memory() / 1e6
                active_mb = mx.get_active_memory() / 1e6
                uptime = now - t_start
                lines = _render_dashboard(
                    stats, H, W, frame_count,
                    uptime, peak_mb, active_mb,
                    using_camera=args.camera,
                )
                _print_dashboard(lines, first_render)
                first_render = False
                t_last_refresh = now

    finally:
        sys.stdout.write(_SHOW_CURSOR + "\n")
        sys.stdout.flush()
        if cap is not None:
            try:
                cap.release()  # type: ignore[union-attr]
            except Exception:
                pass
        print("Live benchmark stopped.")


if __name__ == "__main__":
    main()
