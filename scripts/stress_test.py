#!/usr/bin/env python3
"""Comprehensive stress test for realsense-mlx.

Tests ALL features at multiple resolutions, data patterns, and edge cases.
Works without a RealSense camera — uses synthetic data + any available camera
via OpenCV to test the MLX processing pipeline.

Usage:
    python scripts/stress_test.py
    python scripts/stress_test.py --with-camera      # also test live camera
    python scripts/stress_test.py --resolution 1080p  # stress at 1080p
    python scripts/stress_test.py --duration 60       # run for 60 seconds
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ──────────────────────────────────────────────────────────────────────
# Test result tracking
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    time_ms: float = 0.0
    fps: float = 0.0
    error: str = ""
    details: str = ""


@dataclass
class StressReport:
    results: list[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add(self, r: TestResult):
        self.results.append(r)
        icon = "PASS" if r.passed else "FAIL"
        fps_str = f"  {r.fps:.0f} FPS" if r.fps > 0 else ""
        time_str = f"  {r.time_ms:.1f}ms" if r.time_ms > 0 else ""
        print(f"  [{icon}] {r.name}{time_str}{fps_str}")
        if r.error:
            print(f"         {r.error}")

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self):
        elapsed = self.end_time - self.start_time
        print(f"\n{'='*72}")
        print(f"  STRESS TEST COMPLETE: {self.passed} passed, {self.failed} failed "
              f"in {elapsed:.1f}s")
        if self.failed > 0:
            print(f"\n  Failures:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.error}")
        print(f"{'='*72}")


# ──────────────────────────────────────────────────────────────────────
# Synthetic scene generators
# ──────────────────────────────────────────────────────────────────────

def make_flat_wall(H, W, distance_mm=1500):
    """Flat wall at constant distance."""
    return np.full((H, W), distance_mm, dtype=np.uint16)


def make_ramp(H, W, near_mm=500, far_mm=5000):
    """Linear depth ramp left to right."""
    ramp = np.linspace(near_mm, far_mm, W, dtype=np.float32)
    return np.tile(ramp.astype(np.uint16), (H, 1))


def make_sphere(H, W, center_mm=2000, radius_mm=500):
    """Sphere centered in frame."""
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = min(H, W) // 3
    depth = np.full((H, W), center_mm + radius_mm, dtype=np.float32)
    mask = r < max_r
    depth[mask] = center_mm - radius_mm * np.cos(np.pi * r[mask] / max_r)
    return depth.clip(1, 65535).astype(np.uint16)


def make_noisy(H, W, base_mm=2000, noise_std=50):
    """Noisy flat surface."""
    return (np.random.normal(base_mm, noise_std, (H, W)).clip(1, 65535)).astype(np.uint16)


def make_with_holes(H, W, hole_ratio=0.15):
    """Random depth with specified hole ratio."""
    depth = np.random.randint(500, 5000, (H, W), dtype=np.uint16)
    mask = np.random.random((H, W)) < hole_ratio
    depth[mask] = 0
    # Add rectangular hole
    h4, w4 = H // 4, W // 4
    depth[h4:h4*2, w4:w4*2] = 0
    return depth


def make_extreme_values(H, W):
    """Test min/max uint16 values."""
    depth = np.zeros((H, W), dtype=np.uint16)
    depth[:H//3] = 1          # minimum valid
    depth[H//3:2*H//3] = 65535  # maximum
    depth[2*H//3:] = 0          # invalid
    return depth


def make_color(H, W):
    """Random BGR color frame."""
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Stress test groups
# ──────────────────────────────────────────────────────────────────────

RESOLUTIONS = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "4K": (2160, 3840),
}


def bench_fn(fn, warmup=3, iters=20):
    """Benchmark a callable, return (mean_ms, fps)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        mx.eval(mx.zeros(1))  # sync
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, tuple):
            mx.eval(*[r for r in result if isinstance(r, mx.array)])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean = np.mean(times)
    return mean, 1000.0 / mean if mean > 0 else 0


def stress_filters(report: StressReport, H: int, W: int):
    """Stress all depth filters at given resolution."""
    from realsense_mlx.filters import (
        DecimationFilter, SpatialFilter, TemporalFilter,
        HoleFillingFilter, DisparityTransform, DepthPipeline,
    )
    from realsense_mlx.filters.colorizer import DepthColorizer
    from realsense_mlx.filters.bilateral import BilateralFilter
    from realsense_mlx.filters.enhancement import DepthEnhancer

    scenes = {
        "flat_wall": make_flat_wall(H, W),
        "ramp": make_ramp(H, W),
        "sphere": make_sphere(H, W),
        "noisy": make_noisy(H, W),
        "holes_15pct": make_with_holes(H, W, 0.15),
        "holes_50pct": make_with_holes(H, W, 0.50),
        "extreme": make_extreme_values(H, W),
        "all_zero": np.zeros((H, W), dtype=np.uint16),
        "all_max": np.full((H, W), 65535, dtype=np.uint16),
        "single_pixel": np.array([[1000]], dtype=np.uint16),
    }

    # --- Individual filters ---
    for name, arr in scenes.items():
        d = mx.array(arr)
        try:
            # Decimation
            dec = DecimationFilter(scale=2)
            out = dec.process(d)
            mx.eval(out)
            ok = out.shape[0] <= H and out.shape[1] <= W
            report.add(TestResult(f"DecimationFilter({name})", ok))

            # Spatial (Metal + Python)
            for metal in [True, False]:
                tag = "Metal" if metal else "Python"
                sf = SpatialFilter(iterations=2, use_metal=metal)
                out = sf.process(d.astype(mx.float32))
                mx.eval(out)
                report.add(TestResult(f"SpatialFilter/{tag}({name})", out.shape == d.shape))

            # Temporal (multi-frame)
            tf = TemporalFilter(alpha=0.4, persistence=3)
            for _ in range(5):
                out = tf.process(d)
            mx.eval(out)
            report.add(TestResult(f"TemporalFilter({name})", out.shape == d.shape))

            # Hole filling (all modes)
            for mode in [0, 1, 2]:
                hf = HoleFillingFilter(mode=mode, use_metal=(mode == 0))
                out = hf.process(d)
                mx.eval(out)
                report.add(TestResult(f"HoleFill/mode={mode}({name})", out.shape == d.shape))

            # Disparity transform round-trip
            dt = DisparityTransform(baseline_mm=50, focal_px=383.7, depth_units=0.001)
            disp = dt.process(d)
            mx.eval(disp)
            dt2 = DisparityTransform(baseline_mm=50, focal_px=383.7, depth_units=0.001,
                                     to_disparity=False)
            back = dt2.process(disp)
            mx.eval(back)
            report.add(TestResult(f"DisparityRoundTrip({name})", back.shape == d.shape))

            # Colorizer (all 10 maps)
            for cmap in ["jet", "classic", "grayscale", "warm", "cold",
                         "biomes", "quantized", "pattern", "hue", "inv_grayscale"]:
                col = DepthColorizer(colormap=cmap)
                out = col.colorize(d)
                mx.eval(out)
                ok = out.shape == (*d.shape, 3) or (d.size <= 1)
                report.add(TestResult(f"Colorizer/{cmap}({name})", ok))

        except Exception as e:
            report.add(TestResult(f"Filter/{name}", False, error=str(e)))

    # --- Bilateral filter ---
    try:
        depth = mx.array(make_noisy(H, W)).astype(mx.float32)
        guide = mx.array(make_color(H, W)[:, :, 0]).astype(mx.float32)
        bf = BilateralFilter(sigma_spatial=5.0, sigma_range=30.0, kernel_size=5)
        out = bf.process(depth, guide)
        mx.eval(out)
        report.add(TestResult(f"BilateralFilter({H}x{W})", out.shape == depth.shape))
    except Exception as e:
        report.add(TestResult(f"BilateralFilter({H}x{W})", False, error=str(e)))

    # --- Full pipeline ---
    try:
        pipe = DepthPipeline()
        d = mx.array(make_with_holes(H, W))
        for _ in range(5):
            pipe.process(d)
        ms, fps = bench_fn(lambda: pipe.process(d), warmup=3, iters=10)
        report.add(TestResult(f"DepthPipeline({H}x{W})", True, time_ms=ms, fps=fps))
    except Exception as e:
        report.add(TestResult(f"DepthPipeline({H}x{W})", False, error=str(e)))

    # --- Enhancement pipeline ---
    try:
        enh = DepthEnhancer()
        d = mx.array(make_noisy(H, W))
        for _ in range(3):
            enh.process(d)
        ms, fps = bench_fn(lambda: enh.process(d), warmup=2, iters=10)
        report.add(TestResult(f"DepthEnhancer({H}x{W})", True, time_ms=ms, fps=fps))
    except Exception as e:
        report.add(TestResult(f"DepthEnhancer({H}x{W})", False, error=str(e)))


def stress_geometry(report: StressReport, H: int, W: int):
    """Stress geometry modules."""
    from realsense_mlx.geometry import (
        PointCloudGenerator, Aligner, DepthMeshGenerator,
        CameraIntrinsics, CameraExtrinsics,
    )

    intr = CameraIntrinsics(W, H, W / 2, H / 2, W * 0.6, W * 0.6)
    intr_bc = CameraIntrinsics(W, H, W / 2, H / 2, W * 0.6, W * 0.6,
                                model="brown_conrady",
                                coeffs=[0.1, -0.25, 0.001, -0.001, 0.05])
    extr = CameraExtrinsics.identity()

    scenes = {
        "flat": make_flat_wall(H, W),
        "sphere": make_sphere(H, W),
        "noisy": make_noisy(H, W),
        "holes": make_with_holes(H, W),
    }

    for name, arr in scenes.items():
        d = mx.array(arr)
        color = mx.array(make_color(H, W))

        try:
            # Point cloud (no distortion)
            pc = PointCloudGenerator(intr, 0.001)
            pts = pc.generate(d)
            mx.eval(pts)
            ok = pts.shape == (H, W, 3)
            report.add(TestResult(f"PointCloud/none({name})", ok))

            # Point cloud (Brown-Conrady)
            pc_bc = PointCloudGenerator(intr_bc, 0.001)
            pts_bc = pc_bc.generate(d)
            mx.eval(pts_bc)
            report.add(TestResult(f"PointCloud/BC({name})", pts_bc.shape == (H, W, 3)))

            # Alignment (Metal + MLX)
            for metal in [True, False]:
                tag = "Metal" if metal else "MLX"
                al = Aligner(intr, intr, extr, 0.001, use_metal=metal)
                aligned = al.align_color_to_depth(d, color)
                mx.eval(aligned)
                report.add(TestResult(f"Align/{tag}({name})", aligned.shape[:2] == d.shape))

            # Depth-to-color alignment
            al2 = Aligner(intr, intr, extr, 0.001)
            d2c = al2.align_depth_to_color(d)
            mx.eval(d2c)
            report.add(TestResult(f"AlignD2C({name})", d2c.shape == d.shape))

            # Mesh generation
            mg = DepthMeshGenerator(max_edge_length=0.05)
            verts, faces = mg.generate(pts)
            mx.eval(verts)
            report.add(TestResult(f"MeshGen({name})", verts.ndim == 2 and verts.shape[1] == 3))

            # Normals
            normals = mg.compute_normals(verts, faces)
            report.add(TestResult(f"Normals({name})", normals.shape == verts.shape))

        except Exception as e:
            report.add(TestResult(f"Geometry/{name}", False, error=str(e)))

    # --- Benchmark geometry ---
    d = mx.array(make_noisy(H, W))
    color = mx.array(make_color(H, W))

    pc = PointCloudGenerator(intr, 0.001)
    ms, fps = bench_fn(lambda: pc.generate(d), warmup=3, iters=20)
    report.add(TestResult(f"PointCloud bench({H}x{W})", True, time_ms=ms, fps=fps))

    al = Aligner(intr, intr, extr, 0.001, use_metal=True)
    ms, fps = bench_fn(lambda: al.align_color_to_depth(d, color), warmup=3, iters=20)
    report.add(TestResult(f"Align Metal bench({H}x{W})", True, time_ms=ms, fps=fps))

    pts = pc.generate(d)
    mg = DepthMeshGenerator(0.05)
    ms, fps = bench_fn(lambda: mg.generate(pts), warmup=2, iters=10)
    report.add(TestResult(f"MeshGen bench({H}x{W})", True, time_ms=ms, fps=fps))


def stress_export(report: StressReport, H: int, W: int):
    """Stress export formats."""
    from realsense_mlx.geometry import PointCloudGenerator, DepthMeshGenerator, CameraIntrinsics

    intr = CameraIntrinsics(W, H, W / 2, H / 2, W * 0.6, W * 0.6)
    d = mx.array(make_sphere(H, W))
    color = mx.array(make_color(H, W))

    pc = PointCloudGenerator(intr, 0.001)
    pts = pc.generate(d)
    mg = DepthMeshGenerator(0.05)
    verts, faces = mg.generate(pts)
    normals = mg.compute_normals(verts, faces)

    with tempfile.TemporaryDirectory() as tmpdir:
        # PLY point cloud
        try:
            path = os.path.join(tmpdir, "points.ply")
            n = pc.export_ply(pts, path, colors=color)
            ok = os.path.exists(path) and os.path.getsize(path) > 100
            report.add(TestResult(f"ExportPLY/points({H}x{W})", ok, details=f"{n} points"))
        except Exception as e:
            report.add(TestResult(f"ExportPLY/points", False, error=str(e)))

        # PLY mesh
        try:
            path = os.path.join(tmpdir, "mesh.ply")
            n = pc.export_ply_mesh(verts, faces, path, colors=color, normals=normals)
            ok = os.path.exists(path) and os.path.getsize(path) > 100
            report.add(TestResult(f"ExportPLY/mesh({H}x{W})", ok, details=f"{n} verts"))
        except Exception as e:
            report.add(TestResult(f"ExportPLY/mesh", False, error=str(e)))

        # OBJ
        try:
            path = os.path.join(tmpdir, "mesh.obj")
            n = pc.export_obj(pts, path, faces=faces, colors=color)
            ok = os.path.exists(path) and os.path.getsize(path) > 100
            report.add(TestResult(f"ExportOBJ({H}x{W})", ok, details=f"{n} verts"))
        except Exception as e:
            report.add(TestResult(f"ExportOBJ", False, error=str(e)))


def stress_processor(report: StressReport, H: int, W: int):
    """Stress the end-to-end processor."""
    from realsense_mlx.processor import RealsenseProcessor
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics

    intr = CameraIntrinsics(W, H, W / 2, H / 2, W * 0.6, W * 0.6)

    configs = [
        ("filter_only", dict(enable_pointcloud=False, enable_mesh=False)),
        ("with_pc", dict(enable_pointcloud=True, enable_mesh=False)),
        ("with_mesh", dict(enable_pointcloud=True, enable_mesh=True)),
        ("with_stats", dict(enable_pointcloud=True, enable_stats=True)),
        ("full", dict(enable_pointcloud=True, enable_mesh=True,
                       enable_colorize=True, enable_stats=True)),
    ]

    for tag, kwargs in configs:
        try:
            proc = RealsenseProcessor(intr, depth_scale=0.001, **kwargs)
            d = mx.array(make_with_holes(H, W))
            color = mx.array(make_color(H, W))
            for _ in range(3):
                proc.process(d, color)
            ms, fps = bench_fn(lambda: proc.process(d, color), warmup=2, iters=10)
            report.add(TestResult(f"Processor/{tag}({H}x{W})", True, time_ms=ms, fps=fps))
        except Exception as e:
            report.add(TestResult(f"Processor/{tag}", False, error=str(e)))


def stress_transport(report: StressReport):
    """Stress shared memory transport."""
    from realsense_mlx.transport.shm_frame import ShmFrameWriter, ShmFrameReader

    name = f"stress_test_{os.getpid()}"
    try:
        writer = ShmFrameWriter(name, 640, 480, channels=1, dtype=np.uint16)
        reader = ShmFrameReader(name)

        # Write 100 frames rapidly
        t0 = time.perf_counter()
        for i in range(100):
            frame = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
            writer.write(frame)
        write_time = (time.perf_counter() - t0) * 1000

        # Read last frame
        frame, seq = reader.read()
        ok = frame is not None and frame.shape == (480, 640) and seq > 0

        reader.close()
        writer.close(unlink=True)

        fps = 100 / (write_time / 1000)
        report.add(TestResult(f"SHM/100_writes", ok, time_ms=write_time, fps=fps))
    except Exception as e:
        report.add(TestResult(f"SHM/transport", False, error=str(e)))


def stress_recorder(report: StressReport):
    """Stress frame recording/playback."""
    from realsense_mlx.capture.recorder import FrameRecorder, FramePlayer
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics

    intr = CameraIntrinsics(640, 480, 320, 240, 600, 600)

    with tempfile.TemporaryDirectory() as tmpdir:
        rec_dir = os.path.join(tmpdir, "test_rec")
        try:
            # Record 20 frames
            rec = FrameRecorder(rec_dir)
            rec.start(intr, depth_scale=0.001)
            for i in range(20):
                d = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
                c = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                rec.add_frame(mx.array(d), mx.array(c), timestamp=float(i) / 30.0)
            summary = rec.stop()
            report.add(TestResult(f"Recorder/20_frames", summary["frame_count"] == 20))

            # Playback
            player = FramePlayer(rec_dir)
            player.open()
            count = 0
            for depth, color, ts in player:
                count += 1
                assert depth.shape == (480, 640)
            player.close()
            report.add(TestResult(f"Player/20_frames", count == 20))

            # Seek
            player.open()
            player.seek(10)
            d, c, ts = player.next_frame()
            player.close()
            report.add(TestResult(f"Player/seek", d is not None))

        except Exception as e:
            report.add(TestResult(f"Recorder", False, error=str(e)))


def stress_camera_opencv(report: StressReport, duration: float = 5.0):
    """Stress test using any connected camera via OpenCV."""
    try:
        import cv2
    except ImportError:
        report.add(TestResult("Camera/opencv", False, error="opencv not installed"))
        return

    from realsense_mlx.filters import DepthPipeline
    from realsense_mlx.filters.colorizer import DepthColorizer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        report.add(TestResult("Camera/open", False, error="No camera found"))
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    report.add(TestResult(f"Camera/open({w}x{h})", True))

    # Capture frames and run through MLX processing
    colorizer = DepthColorizer(colormap="jet")
    frames_captured = 0
    mlx_times = []
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frames_captured += 1

        # Convert to grayscale uint16 (simulate depth from camera luminance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fake_depth = (gray.astype(np.uint16) * 20 + 500)  # scale to depth-like range

        # Run through MLX
        t0 = time.perf_counter()
        d_mx = mx.array(fake_depth)
        colored = colorizer.colorize(d_mx)
        mx.eval(colored)
        mlx_times.append((time.perf_counter() - t0) * 1000)

    cap.release()

    cam_fps = frames_captured / (time.perf_counter() - t_start)
    mlx_mean = np.mean(mlx_times) if mlx_times else 0
    mlx_fps = 1000 / mlx_mean if mlx_mean > 0 else 0

    report.add(TestResult(f"Camera/capture({frames_captured} frames)",
                          frames_captured > 0, fps=cam_fps))
    report.add(TestResult(f"Camera/MLX_colorize({w}x{h})",
                          len(mlx_times) > 0, time_ms=mlx_mean, fps=mlx_fps))

    # Stress test: capture + full MLX processing
    from realsense_mlx.geometry import CameraIntrinsics, PointCloudGenerator

    intr = CameraIntrinsics(w, h, w / 2, h / 2, w * 0.6, w * 0.6)
    pc = PointCloudGenerator(intr, 0.001)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    e2e_times = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration and len(e2e_times) < 100:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fake_depth = (gray.astype(np.uint16) * 20 + 500)

        t0 = time.perf_counter()
        d_mx = mx.array(fake_depth)
        colored = colorizer.colorize(d_mx)
        points = pc.generate(d_mx)
        mx.eval(colored, points)
        e2e_times.append((time.perf_counter() - t0) * 1000)

    cap.release()

    if e2e_times:
        mean = np.mean(e2e_times)
        report.add(TestResult(f"Camera/E2E_colorize+pointcloud({w}x{h})",
                              True, time_ms=mean, fps=1000 / mean))


def stress_memory(report: StressReport):
    """Memory stress: process many frames, check for leaks."""
    from realsense_mlx.filters import DepthPipeline

    pipe = DepthPipeline()

    # Measure memory before
    mem_before = mx.get_active_memory() / 1e6

    # Process 200 frames
    for i in range(200):
        d = mx.array(np.random.randint(500, 5000, (480, 640), dtype=np.uint16))
        pipe.process(d)
        if i % 50 == 0:
            mx.eval(mx.zeros(1))  # sync

    mem_after = mx.get_active_memory() / 1e6
    peak = mx.get_peak_memory() / 1e6

    # Memory should not grow unbounded (allow 100MB tolerance)
    growth = mem_after - mem_before
    ok = growth < 100  # MB

    report.add(TestResult(
        f"Memory/200_frames",
        ok,
        details=f"before={mem_before:.0f}MB after={mem_after:.0f}MB "
                f"peak={peak:.0f}MB growth={growth:.0f}MB"
    ))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive stress test")
    parser.add_argument("--resolution", default="480p",
                        choices=list(RESOLUTIONS.keys()),
                        help="Primary resolution to test")
    parser.add_argument("--with-camera", action="store_true",
                        help="Include live camera tests")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Camera test duration in seconds")
    parser.add_argument("--all-resolutions", action="store_true",
                        help="Test at all resolutions (slow)")
    args = parser.parse_args()

    report = StressReport()
    report.start_time = time.perf_counter()

    resolutions = list(RESOLUTIONS.items()) if args.all_resolutions else \
                  [(args.resolution, RESOLUTIONS[args.resolution])]

    for res_name, (H, W) in resolutions:
        print(f"\n{'='*72}")
        print(f"  STRESS TEST @ {res_name} ({H}x{W})")
        print(f"{'='*72}")

        print(f"\n--- Filters ---")
        stress_filters(report, H, W)

        print(f"\n--- Geometry ---")
        stress_geometry(report, H, W)

        print(f"\n--- Export ---")
        stress_export(report, H, W)

        print(f"\n--- Processor ---")
        stress_processor(report, H, W)

    print(f"\n--- Transport ---")
    stress_transport(report)

    print(f"\n--- Recorder ---")
    stress_recorder(report)

    print(f"\n--- Memory ---")
    stress_memory(report)

    if args.with_camera:
        print(f"\n--- Live Camera ({args.duration}s) ---")
        stress_camera_opencv(report, duration=args.duration)

    report.end_time = time.perf_counter()
    report.summary()

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
