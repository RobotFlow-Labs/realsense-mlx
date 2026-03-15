#!/usr/bin/env python3
"""Live depth processing demo with MLX acceleration.

Captures depth (and optionally colour) from a connected Intel RealSense
camera, runs the full post-processing pipeline on the Apple Silicon GPU
via MLX, colourizes the result, and displays it in an OpenCV window.

Requirements
------------
- A connected Intel RealSense depth camera (D415, D435, D455, etc.)
- pyrealsense2 built from source (no ARM64 macOS wheels on PyPI)
- opencv-python  (``uv pip install opencv-python``)

Usage
-----
::

    python scripts/live_depth_demo.py
    python scripts/live_depth_demo.py --colormap classic --no-filter
    python scripts/live_depth_demo.py --width 848 --height 480 --fps 30
    python scripts/live_depth_demo.py --colormap jet --no-equalize --side-by-side
    python scripts/live_depth_demo.py --decimation 1 --no-temporal

Press ``q`` or ``Esc`` in the viewer window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time


# ---------------------------------------------------------------------------
# Dependency checks (deferred, with helpful messages)
# ---------------------------------------------------------------------------


def _check_deps() -> bool:
    """Verify optional dependencies are present and print guidance if not."""
    ok = True

    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        print(
            "ERROR: pyrealsense2 is not installed.\n"
            "Build from source:\n"
            "  git clone https://github.com/IntelRealSense/librealsense.git\n"
            "  cd librealsense && mkdir build && cd build\n"
            "  cmake .. -DBUILD_PYTHON_BINDINGS=ON "
            "-DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release\n"
            "  make -j$(sysctl -n hw.ncpu) && sudo make install",
            file=sys.stderr,
        )
        ok = False

    try:
        import cv2  # noqa: F401
    except ImportError:
        print(
            "ERROR: opencv-python is not installed.\n"
            "Install it with:  uv pip install opencv-python",
            file=sys.stderr,
        )
        ok = False

    return ok


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Resolution & frame-rate ------------------------------------------
    res = parser.add_argument_group("resolution / frame-rate")
    res.add_argument(
        "--width", type=int, default=640,
        metavar="W",
        help="Stream width in pixels (default: %(default)s).",
    )
    res.add_argument(
        "--height", type=int, default=480,
        metavar="H",
        help="Stream height in pixels (default: %(default)s).",
    )
    res.add_argument(
        "--fps", type=int, default=30,
        help="Target frame-rate (default: %(default)s).",
    )

    # -- Colorizer --------------------------------------------------------
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
        "--min-depth", type=float, default=0.1,
        metavar="M",
        help="Near clip in metres (default: %(default)s).",
    )
    col.add_argument(
        "--max-depth", type=float, default=6.0,
        metavar="M",
        help="Far clip in metres (default: %(default)s).",
    )
    col.add_argument(
        "--no-equalize", dest="equalize", action="store_false", default=True,
        help="Disable histogram equalization (use linear normalization).",
    )

    # -- Pipeline filters -------------------------------------------------
    flt = parser.add_argument_group("depth pipeline filters")
    flt.add_argument(
        "--no-filter", dest="filter", action="store_false", default=True,
        help="Skip all post-processing (display raw depth).",
    )
    flt.add_argument(
        "--decimation", type=int, default=2,
        metavar="N",
        choices=range(1, 9),
        help="Decimation scale factor 1–8 (default: %(default)s; 1=off).",
    )
    flt.add_argument(
        "--no-temporal", dest="temporal", action="store_false", default=True,
        help="Disable temporal filter.",
    )
    flt.add_argument(
        "--no-spatial", dest="spatial", action="store_false", default=True,
        help="Disable spatial filter.",
    )
    flt.add_argument(
        "--no-hole-fill", dest="hole_fill", action="store_false", default=True,
        help="Disable hole-filling.",
    )

    # -- Display ----------------------------------------------------------
    disp = parser.add_argument_group("display")
    disp.add_argument(
        "--side-by-side", action="store_true", default=False,
        help="Show colorized depth next to the RGB colour frame.",
    )
    disp.add_argument(
        "--window-width", type=int, default=1280,
        help="Viewer window width in pixels (default: %(default)s).",
    )
    disp.add_argument(
        "--window-height", type=int, default=480,
        help="Viewer window height in pixels (default: %(default)s).",
    )

    # -- Misc -------------------------------------------------------------
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print per-frame timing statistics.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        metavar="N",
        help="Stop after N frames (0 = run until window closed).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the live demo loop.  Returns an exit code."""
    if not _check_deps():
        return 1

    # ---- Imports after dep-check ----------------------------------------
    import mlx.core as mx  # noqa: F401 — ensure MLX initialises on the Metal device

    from realsense_mlx.capture import RealsenseCapture, CaptureConfig
    from realsense_mlx.display import RealsenseViewer
    from realsense_mlx.filters import DepthPipeline, PipelineConfig
    from realsense_mlx.filters.colorizer import DepthColorizer

    # ---- Configure capture ----------------------------------------------
    capture_cfg = CaptureConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=True,
        enable_color=args.side_by_side,
    )

    # ---- Configure pipeline ---------------------------------------------
    pipeline: DepthPipeline | None = None
    if args.filter:
        pipe_cfg = PipelineConfig(
            decimation_scale=args.decimation,
            enable_spatial=args.spatial,
            enable_hole_fill=args.hole_fill,
        )
        if not args.temporal:
            pipe_cfg.temporal_alpha = 1.0   # no smoothing (current frame only)
        pipeline = DepthPipeline(pipe_cfg)

    # ---- Configure colorizer --------------------------------------------
    colorizer = DepthColorizer(
        colormap=args.colormap,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        equalize=args.equalize,
    )

    # ---- Create viewer --------------------------------------------------
    viewer = RealsenseViewer(
        title=f"RealSense MLX — {args.colormap}",
        width=args.window_width,
        height=args.window_height,
    )

    # ---- Start camera ---------------------------------------------------
    try:
        capture = RealsenseCapture(capture_cfg)
        capture.start()
    except (ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        depth_intr = capture.get_depth_intrinsics()
        print(f"Camera ready.  depth_scale={capture.depth_scale:.6f} m/count")
        if depth_intr is not None:
            print(f"Depth intrinsics: {depth_intr}")
        print(f"Pipeline: {pipeline!r}")
        print(f"Colorizer: {colorizer!r}\n")

    # ---- Frame loop -----------------------------------------------------
    frame_count = 0
    t_start = time.perf_counter()
    t_last_report = t_start

    try:
        with viewer:
            while viewer.is_open():
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    break

                frames = capture.get_frames()
                if frames.depth is None:
                    continue

                t_frame_start = time.perf_counter()

                # Post-process
                depth = frames.depth
                if pipeline is not None:
                    depth = pipeline.process(depth)

                # Colorize
                colored = colorizer.colorize(depth)
                mx.eval(colored)

                t_proc = (time.perf_counter() - t_frame_start) * 1000.0

                # Display
                if args.side_by_side and frames.color is not None:
                    viewer.show_side_by_side(colored, frames.color)
                else:
                    viewer.show_depth(colored)

                frame_count += 1

                if args.verbose:
                    now = time.perf_counter()
                    if now - t_last_report >= 1.0:
                        elapsed = now - t_start
                        fps = frame_count / elapsed
                        print(
                            f"  frame {frame_count:5d} | "
                            f"fps={fps:5.1f} | "
                            f"proc={t_proc:5.2f}ms"
                        )
                        t_last_report = now

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"\nStopped. {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps avg)")
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
