#!/usr/bin/env python3
"""Live stereo depth demo — ZED 2i (or any USB stereo camera).

Opens a side-by-side stereo camera via OpenCV, runs SGBM + MLX
post-processing on every frame, and displays a colourised depth map
side-by-side with the left eye image.

Keyboard controls
-----------------
q       Quit
s       Save current depth map as PLY point cloud
f       Toggle MLX filter chain on/off
r       Reset temporal filter state
+/-     Increase / decrease num_disparities by 16
h       Show this help overlay

Usage
-----
    python scripts/stereo_depth_demo.py
    python scripts/stereo_depth_demo.py --device 0
    python scripts/stereo_depth_demo.py --device 0 --baseline 120 --focal 700
    python scripts/stereo_depth_demo.py --device 0 --width 2560 --height 720
    python scripts/stereo_depth_demo.py --dual --left 0 --right 2
    python scripts/stereo_depth_demo.py --no-filter     # raw SGBM only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live stereo depth demo — any USB stereo camera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Camera selection
    cam_group = p.add_mutually_exclusive_group()
    cam_group.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        metavar="N",
        help="Camera device ID for side-by-side layout (default: auto-detect = 0)",
    )
    cam_group.add_argument(
        "--dual",
        action="store_true",
        help="Use dual-camera layout (two separate USB devices)",
    )

    p.add_argument("--left",  type=int, default=0, metavar="N",
                   help="Left camera device ID (dual mode only)")
    p.add_argument("--right", type=int, default=1, metavar="N",
                   help="Right camera device ID (dual mode only)")

    # Frame geometry
    p.add_argument("--width",  type=int, default=2560,
                   help="Full frame width (side-by-side) or per-eye width (dual)")
    p.add_argument("--height", type=int, default=720,
                   help="Frame height")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Target camera frame rate")

    # Stereo geometry
    p.add_argument("--baseline", "-b", type=float, default=120.0,
                   help="Stereo baseline in millimetres (ZED 2i = 120)")
    p.add_argument("--focal", "-f", type=float, default=700.0,
                   help="Focal length in pixels (ZED 2i @ 720p ≈ 700)")

    # SGBM tuning
    p.add_argument("--disparities", type=int, default=128,
                   help="Number of disparity levels (must be divisible by 16)")
    p.add_argument("--block-size", type=int, default=5,
                   help="SGBM block size (odd integer, 3–11)")

    # MLX filter toggle
    p.add_argument("--no-filter", action="store_true",
                   help="Disable MLX filter chain (show raw SGBM output)")

    # Output
    p.add_argument("--colormap", default="jet",
                   choices=["jet", "classic", "grayscale", "warm", "cold",
                            "biomes", "quantized", "hue"],
                   help="Depth colormap")
    p.add_argument("--ply-dir", default=".",
                   help="Directory to save PLY files (created if missing)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# PLY export helper
# ---------------------------------------------------------------------------

def save_ply(depth_m: np.ndarray, focal_px: float, ply_path: Path) -> None:
    """Export depth map as a PLY point cloud.

    Parameters
    ----------
    depth_m:
        ``(H, W)`` float32 depth in metres.  Zero = invalid.
    focal_px:
        Focal length in pixels (used for back-projection).
    ply_path:
        Output file path.
    """
    H, W = depth_m.shape
    cx, cy = W / 2.0, H / 2.0

    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth_m
    valid = z > 0.0

    x = (cols[valid] - cx) * z[valid] / focal_px
    y = (rows[valid] - cy) * z[valid] / focal_px
    z_v = z[valid]

    points = np.stack([x, y, z_v], axis=1).astype(np.float32)
    n_pts = points.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(points.tobytes())

    print(f"[stereo_demo] Saved {n_pts:,} points → {ply_path}")


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------

HELP_LINES = [
    "Keys:",
    "  q     Quit",
    "  s     Save PLY",
    "  f     Toggle MLX filters",
    "  r     Reset temporal filter",
    "  +/-   +/- 16 disparities",
    "  h     Toggle help",
]


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    num_disp: int,
    mlx_enabled: bool,
    show_help: bool,
) -> None:
    """Draw FPS, disparity info, and optional help on *frame* in-place."""
    try:
        import cv2
    except ImportError:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    color_white = (255, 255, 255)
    color_yellow = (0, 220, 220)
    color_red = (80, 80, 255)

    def put(text: str, row: int, col: int = 10, color=color_white) -> None:
        pos = (col, row)
        # Black shadow for legibility on any background.
        cv2.putText(frame, text, (col + 1, row + 1), font, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, text, pos, font, scale, color, thick, cv2.LINE_AA)

    status_color = color_yellow if mlx_enabled else color_red
    mlx_status = "MLX ON" if mlx_enabled else "MLX OFF"

    put(f"FPS: {fps:.1f}  disp={num_disp}  {mlx_status}",
        row=24, color=status_color)

    if show_help:
        for i, line in enumerate(HELP_LINES):
            put(line, row=50 + i * 22)


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> int:
    """Run the live stereo depth demo.

    Returns
    -------
    int
        Exit code (0 = clean exit).
    """
    try:
        import cv2
    except ImportError:
        print(
            "ERROR: opencv-python is required.\n"
            "       Install with: pip install opencv-python",
            file=sys.stderr,
        )
        return 1

    import mlx.core as mx
    from realsense_mlx.stereo.camera import StereoCamera, StereoCameraError
    from realsense_mlx.stereo.depth import StereoDepthEstimator, StereoDepthConfig

    # --- Build camera ---
    if args.dual:
        cam = StereoCamera.from_dual(
            left_id=args.left,
            right_id=args.right,
            width=args.width,
            height=args.height,
            target_fps=args.fps,
        )
    else:
        cam = StereoCamera.from_side_by_side(
            device_id=args.device,
            width=args.width,
            height=args.height,
            target_fps=args.fps,
        )

    # --- Build depth estimator ---
    cfg = StereoDepthConfig(
        baseline_mm=args.baseline,
        focal_px=args.focal,
        num_disparities=args.disparities,
        block_size=args.block_size,
        enable_spatial=not args.no_filter,
        enable_temporal=not args.no_filter,
        enable_hole_fill=not args.no_filter,
        colormap=args.colormap,
    )
    estimator = StereoDepthEstimator(config=cfg)

    print(f"[stereo_demo] Camera: {cam!r}")
    print(f"[stereo_demo] Estimator: {estimator!r}")
    print("[stereo_demo] Press 'q' to quit, 'h' for help.")

    try:
        cam.start()
    except StereoCameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mlx_enabled: bool = not args.no_filter
    show_help: bool = False
    frame_count: int = 0
    ply_dir = Path(args.ply_dir)

    # Timing
    t0 = time.perf_counter()
    display_fps: float = 0.0

    try:
        while True:
            # --- Capture ---
            try:
                left_bgr, right_bgr = cam.capture()
            except StereoCameraError as exc:
                print(f"[stereo_demo] Capture error: {exc}", file=sys.stderr)
                break

            # --- Update MLX filter toggle ---
            estimator.config.enable_spatial = mlx_enabled
            estimator.config.enable_temporal = mlx_enabled
            estimator.config.enable_hole_fill = mlx_enabled

            # --- Compute depth ---
            t_start = time.perf_counter()
            depth_mx, depth_bgr = estimator.compute_with_color(left_bgr, right_bgr)
            mx.eval(depth_mx)
            t_elapsed = time.perf_counter() - t_start

            frame_count += 1

            # Update display FPS estimate (exponential moving average).
            elapsed_total = time.perf_counter() - t0
            if elapsed_total > 0:
                instant_fps = 1.0 / max(t_elapsed, 1e-6)
                display_fps = 0.9 * display_fps + 0.1 * instant_fps

            # --- Build display frame ---
            # Resize depth colourisation to match left image height if needed.
            h_left, w_left = left_bgr.shape[:2]
            h_dep, w_dep = depth_bgr.shape[:2]

            if (h_dep, w_dep) != (h_left, w_left):
                depth_bgr = cv2.resize(
                    depth_bgr, (w_left, h_left), interpolation=cv2.INTER_LINEAR
                )

            # Side-by-side display: left eye | depth colourisation.
            display = np.concatenate([left_bgr, depth_bgr], axis=1)

            draw_overlay(
                display,
                fps=cam.fps if cam.fps > 0.0 else display_fps,
                num_disp=estimator.config.num_disparities,
                mlx_enabled=mlx_enabled,
                show_help=show_help,
            )

            cv2.imshow("Stereo Depth Demo  [q=quit | s=save | f=filter | h=help]",
                       display)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                mx.eval(depth_mx)
                depth_np = np.array(depth_mx, copy=False)
                ts = int(time.time())
                ply_path = ply_dir / f"stereo_depth_{ts:010d}.ply"
                save_ply(depth_np, args.focal, ply_path)

            elif key == ord("f"):
                mlx_enabled = not mlx_enabled
                if not mlx_enabled:
                    # Clear stale temporal state when disabling.
                    estimator.reset()
                print(f"[stereo_demo] MLX filters: {'ON' if mlx_enabled else 'OFF'}")

            elif key == ord("r"):
                estimator.reset()
                print("[stereo_demo] Temporal filter reset.")

            elif key == ord("h"):
                show_help = not show_help

            elif key == ord("+") or key == ord("="):
                new_d = estimator.config.num_disparities + 16
                estimator.config.num_disparities = new_d
                estimator._sgbm = estimator._build_sgbm()
                print(f"[stereo_demo] num_disparities = {new_d}")

            elif key == ord("-"):
                new_d = max(16, estimator.config.num_disparities - 16)
                estimator.config.num_disparities = new_d
                estimator._sgbm = estimator._build_sgbm()
                print(f"[stereo_demo] num_disparities = {new_d}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"[stereo_demo] Processed {frame_count} frames. Bye.")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_demo(parse_args()))
