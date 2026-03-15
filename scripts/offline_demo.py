#!/usr/bin/env python3
"""Offline demo using synthetic depth data (no camera needed).

Generates a synthetic depth scene, runs the full MLX processing pipeline via
:class:`~realsense_mlx.processor.RealsenseProcessor`, and shows before/after
results.  Useful for verifying the pipeline works correctly on Apple Silicon
without a physical RealSense camera.

The demo uses ``RealsenseProcessor`` which handles decimation-adjusted
intrinsics automatically — no manual intrinsics scaling required.

Synthetic scene
---------------
The scene contains:
- A tilted ground plane with Gaussian noise.
- A solid sphere at the image centre.
- A second smaller sphere offset to the right.
- Random holes (simulated invalid pixels, value = 0) at ~8% density.
- A Gaussian noise layer added on top.

The demo then:
1. Runs the full ``RealsenseProcessor`` pipeline (decimation → spatial →
   temporal × frames → hole filling → colourisation → point cloud).
2. Colourizes before/after with the requested colormap.
3. Optionally displays side-by-side with OpenCV.
4. Optionally exports a PLY point cloud file.
5. Prints timing and basic statistics to stdout.

Usage
-----
::

    python scripts/offline_demo.py
    python scripts/offline_demo.py --colormap classic --no-display
    python scripts/offline_demo.py --width 1280 --height 720 --frames 20
    python scripts/offline_demo.py --colormap hue --equalize
    python scripts/offline_demo.py --benchmark 100
    python scripts/offline_demo.py --export-ply /tmp/scene.ply

Press ``q`` or ``Esc`` in the window to quit early.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# Synthetic depth scene generator
# ---------------------------------------------------------------------------


def _make_sphere_mask(
    H: int,
    W: int,
    cx: float,
    cy: float,
    radius: float,
) -> np.ndarray:
    """Return a (H, W) float32 array with a sphere depth profile.

    Pixels outside the sphere have value 0.0.  Inside, value represents
    the distance from the camera to the sphere surface, centred at
    ``sphere_z`` with the given radius.
    """
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    inside = dist2 <= radius ** 2
    # z offset from sphere centre plane (positive = closer to camera)
    dz = np.sqrt(np.maximum(0.0, radius ** 2 - dist2))
    depth = np.where(inside, dz, 0.0)
    return depth


def generate_synthetic_depth(
    width: int = 640,
    height: int = 480,
    depth_units: float = 0.001,
    rng_seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic uint16 depth frame.

    The scene (in metres) contains:
    - A ground plane tilted slightly toward the camera
      (z ≈ 1.5 m at top, 3.0 m at bottom of frame).
    - A large sphere (r ≈ 0.3 m) centred at (W/2, H/2, 2.0 m).
    - A small sphere (r ≈ 0.15 m) right of centre at (0.65W, 0.4H, 1.8 m).
    - Gaussian noise (σ ≈ 2 mm).
    - ~8% random invalid (zero) pixels.

    Parameters
    ----------
    width, height:
        Frame dimensions.
    depth_units:
        Metres per count.  Default 0.001 (RealSense D-series default).
    rng_seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray
        ``(height, width)`` uint16 array.
    """
    rng = np.random.default_rng(rng_seed)

    # ---- Ground plane ---------------------------------------------------
    z_top = 1.5
    z_bottom = 3.0
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    plane_m = z_top + t * (z_bottom - z_top)
    plane_m = np.broadcast_to(plane_m, (height, width)).copy()

    # ---- Large sphere (nearer) ------------------------------------------
    large_cx = width * 0.50
    large_cy = height * 0.50
    large_r_m = 0.30
    large_r_px = large_r_m / (z_bottom * depth_units * 1000.0 / width) * width * 0.5
    large_r_px = max(20.0, large_r_px)
    sphere_large = _make_sphere_mask(height, width, large_cx, large_cy, large_r_px)

    sphere_large_m = sphere_large * (large_r_m / max(large_r_px, 1.0))
    z_sphere_centre_large = 2.0

    depth_sphere_large = np.where(
        sphere_large > 0,
        z_sphere_centre_large - sphere_large_m,
        0.0,
    )

    # ---- Small sphere (even closer) -------------------------------------
    small_cx = width * 0.65
    small_cy = height * 0.38
    small_r_m = 0.14
    small_r_px = max(
        12.0, small_r_m / (z_bottom * depth_units * 1000.0 / width) * width * 0.4
    )
    sphere_small = _make_sphere_mask(height, width, small_cx, small_cy, small_r_px)
    small_r_ratio = small_r_m / max(small_r_px, 1.0)
    depth_sphere_small = np.where(
        sphere_small > 0,
        1.75 - sphere_small * small_r_ratio,
        0.0,
    )

    # ---- Compose scene (take minimum = nearest surface) -----------------
    scene_m = plane_m.copy()
    mask_large = depth_sphere_large > 0
    scene_m[mask_large] = np.minimum(
        scene_m[mask_large], depth_sphere_large[mask_large]
    )
    mask_small = depth_sphere_small > 0
    scene_m[mask_small] = np.minimum(
        scene_m[mask_small], depth_sphere_small[mask_small]
    )

    # ---- Gaussian noise (σ = 2 mm) --------------------------------------
    noise_m = rng.normal(0.0, 0.002, size=(height, width)).astype(np.float32)
    scene_m = np.clip(scene_m + noise_m, 0.0, 10.0)

    # ---- Convert to uint16 counts ---------------------------------------
    scene_counts = (scene_m / depth_units).astype(np.float32)
    scene_counts = np.clip(scene_counts, 0, 65535)

    # ---- Simulate invalid pixels (holes) --------------------------------
    hole_mask = rng.random(size=(height, width)) < 0.08
    scene_counts[hole_mask] = 0.0

    return scene_counts.astype(np.uint16)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Resolution
    res = parser.add_argument_group("scene resolution")
    res.add_argument("--width", type=int, default=640, metavar="W")
    res.add_argument("--height", type=int, default=480, metavar="H")

    # Colorizer
    col = parser.add_argument_group("colorizer")
    col.add_argument(
        "--colormap",
        default="jet",
        choices=[
            "jet", "classic", "grayscale", "inv_grayscale",
            "warm", "cold", "biomes", "quantized", "pattern", "hue",
        ],
        help="Depth colormap (default: %(default)s).",
    )
    col.add_argument(
        "--equalize", action="store_true", default=False,
        help="Enable histogram equalization in the colorizer.",
    )

    # Pipeline
    flt = parser.add_argument_group("pipeline")
    flt.add_argument(
        "--decimation", type=int, default=2,
        choices=range(1, 9),
        metavar="N",
        help="Decimation scale 1–8 (default: 2).",
    )
    flt.add_argument(
        "--no-filter", dest="filter", action="store_false", default=True,
        help="Skip all post-processing (sets decimation=1).",
    )
    flt.add_argument(
        "--enable-mesh", action="store_true", default=False,
        help="Generate triangle mesh from the point cloud.",
    )
    flt.add_argument(
        "--enable-stats", action="store_true", default=False,
        help="Print depth statistics for the filtered frame.",
    )

    # Export
    exp = parser.add_argument_group("export")
    exp.add_argument(
        "--export-ply", metavar="PATH", default=None,
        help="Export the point cloud (or mesh if --enable-mesh) to a PLY file.",
    )
    exp.add_argument(
        "--export-obj", metavar="PATH", default=None,
        help="Export the point cloud (or mesh) to an OBJ file.",
    )

    # Simulation
    sim = parser.add_argument_group("simulation")
    sim.add_argument(
        "--frames", type=int, default=5,
        metavar="N",
        help="Number of synthetic frames to feed into the pipeline "
             "(accumulates temporal filter state).  Default: %(default)s.",
    )
    sim.add_argument(
        "--seed", type=int, default=42,
        help="NumPy random seed for reproducible scenes (default: %(default)s).",
    )
    sim.add_argument(
        "--fov-deg", type=float, default=69.4,
        metavar="DEG",
        help="Horizontal field-of-view in degrees for synthetic intrinsics "
             "(default: %(default)s — approx D435 RGB).",
    )

    # Benchmark
    parser.add_argument(
        "--benchmark", type=int, default=0,
        metavar="N",
        help="Run N iterations and print throughput; implies --no-display.",
    )

    # Display
    parser.add_argument(
        "--no-display", dest="display", action="store_false", default=True,
        help="Skip the OpenCV display window (print stats only).",
    )
    parser.add_argument(
        "--window-width", type=int, default=1280,
        help="Viewer window width (default: %(default)s).",
    )
    parser.add_argument(
        "--window-height", type=int, default=480,
        help="Viewer window height (default: %(default)s).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the offline demo.  Returns an exit code."""
    from realsense_mlx.filters.colorizer import DepthColorizer
    from realsense_mlx.filters.pipeline import PipelineConfig
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    from realsense_mlx.processor import RealsenseProcessor

    depth_units = 0.001  # metres per count (standard RealSense D-series)

    # ---- Synthetic camera intrinsics ------------------------------------
    # Build pinhole intrinsics for the full-resolution frame.
    # RealsenseProcessor will automatically derive decimated intrinsics
    # after the first process() call.
    intr = CameraIntrinsics.make_pinhole(
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
    )

    # ---- Build processor ------------------------------------------------
    dec_scale = args.decimation if args.filter else 1
    pipe_cfg = PipelineConfig(
        decimation_scale=dec_scale,
        depth_units=depth_units,
    )
    proc = RealsenseProcessor(
        depth_intrinsics=intr,
        depth_scale=depth_units,
        pipeline_config=pipe_cfg,
        enable_pointcloud=True,
        enable_mesh=args.enable_mesh,
        enable_colorize=True,
        enable_stats=args.enable_stats,
        colormap=args.colormap,
    )

    # ---- Colorizer for raw depth (before processing) --------------------
    colorizer_raw = DepthColorizer(
        colormap=args.colormap,
        equalize=args.equalize,
        depth_units=depth_units,
    )

    # ---- Generate synthetic scene and warm up temporal state ------------
    print(
        f"Generating {args.frames} synthetic depth frames "
        f"({args.width}x{args.height}, seed={args.seed}) …"
    )
    print(f"  Intrinsics: {intr}")

    depth_raw: mx.array | None = None
    last_result = None

    for i in range(args.frames):
        frame_np = generate_synthetic_depth(
            width=args.width,
            height=args.height,
            depth_units=depth_units,
            rng_seed=args.seed + i,
        )
        depth_mx = mx.array(frame_np)

        last_result = proc.process(depth_mx)
        mx.eval(last_result.filtered_depth)

        if i == 0:
            depth_raw = depth_mx

    if depth_raw is None or last_result is None:
        print("ERROR: no frames generated.", file=sys.stderr)
        return 1

    # ---- Print result summary -------------------------------------------
    print(f"\nResult summary:")
    print(f"  Raw frame:      {depth_raw.shape}  dtype={depth_raw.dtype}")
    print(
        f"  Filtered frame: {last_result.filtered_depth.shape}"
        f"  dtype={last_result.filtered_depth.dtype}"
    )
    if last_result.intrinsics is not None:
        print(f"  Decimated intr: {last_result.intrinsics}")
    if last_result.points is not None:
        print(f"  Point cloud:    {last_result.points.shape}")
    if last_result.vertices is not None:
        print(
            f"  Mesh vertices:  {last_result.vertices.shape}"
            f"  faces: {last_result.faces.shape if last_result.faces is not None else 'N/A'}"
        )
    if last_result.processing_time_ms > 0:
        print(f"  Processing:     {last_result.processing_time_ms:.2f} ms/frame")

    if last_result.stats is not None:
        s = last_result.stats
        valid_pct = 100.0 * s["valid_ratio"]
        print(f"\nDepth statistics (filtered frame):")
        print(f"  Valid pixels:  {valid_pct:.1f}%")
        if s["mean_m"] is not None:
            print(
                f"  Depth range:   {s['min_m']:.3f} m  →  {s['max_m']:.3f} m"
                f"  (mean {s['mean_m']:.3f} m, std {s['std_m']*1000:.2f} mm)"
            )

    # ---- Benchmark mode -------------------------------------------------
    if args.benchmark > 0:
        print(f"\nBenchmark: {args.benchmark} iterations …")
        frame_np = generate_synthetic_depth(
            args.width, args.height, depth_units, args.seed
        )
        depth_bench = mx.array(frame_np)
        proc.reset()

        t0 = time.perf_counter()
        for _ in range(args.benchmark):
            r = proc.process(depth_bench)
            mx.eval(r.filtered_depth)
            if r.colored_depth is not None:
                mx.eval(r.colored_depth)
        elapsed = time.perf_counter() - t0

        fps = args.benchmark / elapsed
        ms_per_frame = elapsed * 1000.0 / args.benchmark
        print(
            f"  {args.benchmark} frames in {elapsed:.3f}s "
            f"=> {fps:.1f} fps  ({ms_per_frame:.2f} ms/frame)"
        )
        return 0

    # ---- Export ---------------------------------------------------------
    if args.export_ply is not None:
        n = proc.export_ply(last_result, args.export_ply)
        kind = "faces" if last_result.faces is not None else "points"
        print(f"\nExported {n} {kind} to: {args.export_ply}")

    if args.export_obj is not None:
        n = proc.export_obj(last_result, args.export_obj)
        print(f"Exported {n} vertices to: {args.export_obj}")

    # ---- Colorize raw frame for display ---------------------------------
    colored_raw = colorizer_raw.colorize(depth_raw)
    colored_proc = last_result.colored_depth
    if colored_proc is None:
        colored_proc = colorizer_raw.colorize(last_result.filtered_depth)
    mx.eval(colored_raw, colored_proc)

    # ---- Display --------------------------------------------------------
    show = args.display
    if show:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print(
                "WARNING: opencv-python not installed — skipping display.\n"
                "Install with:  uv pip install opencv-python",
                file=sys.stderr,
            )
            show = False

    if show:
        from realsense_mlx.display import RealsenseViewer

        label = (
            f"Offline Demo — {args.colormap} | "
            f"left=raw ({depth_raw.shape[1]}x{depth_raw.shape[0]})  "
            f"right=filtered ({last_result.filtered_depth.shape[1]}x"
            f"{last_result.filtered_depth.shape[0]})  "
            f"(q/Esc to quit)"
        )
        viewer = RealsenseViewer(
            title=label,
            width=args.window_width,
            height=args.window_height,
        )
        print(f"\nDisplaying result.")
        print("Press 'q' or Esc in the window to close.")

        try:
            with viewer:
                while viewer.is_open():
                    viewer.show_side_by_side(colored_raw, colored_proc)
        except KeyboardInterrupt:
            pass
    else:
        import os

        out_dir = "/tmp/realsense_mlx_offline_demo"
        os.makedirs(out_dir, exist_ok=True)

        def _save(arr: mx.array, path: str) -> None:
            """Save RGB mx.array as PNG via OpenCV (if available) or raw npy."""
            mx.eval(arr)
            np_arr = np.array(arr, copy=False)
            try:
                import cv2

                bgr = np_arr[:, :, ::-1]  # RGB → BGR
                cv2.imwrite(path, bgr)
                print(f"  Saved: {path}")
            except ImportError:
                npy_path = path.replace(".png", ".npy")
                np.save(npy_path, np_arr)
                print(f"  Saved (numpy): {npy_path}")

        print(f"\nSaving output images to {out_dir}/")
        _save(colored_raw, f"{out_dir}/depth_raw.png")
        _save(colored_proc, f"{out_dir}/depth_filtered.png")

    print("\nDone.")
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
