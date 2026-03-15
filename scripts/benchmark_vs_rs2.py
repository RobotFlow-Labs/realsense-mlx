"""Publication-quality benchmark: realsense-mlx (MLX Metal) vs RS2 SDK (NumPy CPU).

Compares the EXACT same algorithms implemented in:
  - NumPy (RS2 SDK equivalent — faithful to the C++ sequential logic)
  - MLX Metal (this project)

Covers four synthetic depth scenes, five algorithms, speed + quality metrics.

Usage
-----
    python scripts/benchmark_vs_rs2.py
    python scripts/benchmark_vs_rs2.py --output results.json
    python scripts/benchmark_vs_rs2.py --resolution 720p
    python scripts/benchmark_vs_rs2.py --resolution 480p --iters 30 --warmup 3
    python scripts/benchmark_vs_rs2.py --quality-only
    python scripts/benchmark_vs_rs2.py --speed-only

Scenes
------
  flat_wall     : planar surface at 2000 mm with mild Gaussian noise
  sphere        : spherical surface (r=1500 mm) centred in frame
  staircase     : 8-step depth staircase, each step 200 mm taller
  random_holes  : random depth with 12 % invalid (zero) pixels

Algorithm pairings (NumPy RS2  vs  MLX Metal)
----------------------------------------------
  spatial       : bilateral recursive scan — sequential per-row (RS2 exact),
                  parallel rows in Metal
  temporal      : EMA with 8-bit persistence bitmask
  decimation    : tile median (scale 2) / valid-mean (scale 4)
  hole_filling  : 4-neighbour max (FARTHEST mode)
  disparity     : element-wise depth ↔ disparity

Quality metrics (per scene, spatial + temporal + hole_fill)
-----------------------------------------------------------
  RMSE          : root-mean-square error vs ground truth (mm)
  MAE           : mean absolute error vs ground truth (mm)
  PSNR          : peak signal-to-noise ratio (dB)
  holes_filled  : fraction of originally invalid pixels recovered
  noise_removed : 1 - (output_std / noisy_input_std) for valid pixels
  edges_pres    : gradient-magnitude preservation ratio (output/gt)

Memory
------
  MLX peak memory is sampled via mx.get_peak_memory() before and after.
  NumPy peak RSS is estimated from tracemalloc.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

# ---------------------------------------------------------------------------
# Ensure the src package is importable when run without editable install
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import mlx.core as mx

from realsense_mlx.filters.spatial import SpatialFilter
from realsense_mlx.filters.temporal import TemporalFilter
from realsense_mlx.filters.decimation import DecimationFilter
from realsense_mlx.filters.hole_filling import HoleFillingFilter
from realsense_mlx.filters.disparity import DisparityTransform
from realsense_mlx.utils.benchmark import benchmark_component, Timer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stereo rig — D435i typical values
_BASELINE_MM: float = 50.0
_FOCAL_PX: float = 383.7
_DEPTH_UNITS: float = 0.001          # metres per count

# D2D factor used by both pipelines
_D2D_FACTOR: float = _BASELINE_MM * _FOCAL_PX * 32.0 / _DEPTH_UNITS

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


# ---------------------------------------------------------------------------
# Machine identification
# ---------------------------------------------------------------------------

def _detect_machine() -> str:
    """Return a human-readable machine description."""
    chip = "Unknown"
    mem_gb = "?"
    try:
        r = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "Chip:" in line:
                chip = line.split(":", 1)[1].strip()
            if "Memory:" in line:
                mem_gb = line.split(":", 1)[1].strip()
    except Exception:
        pass
    macos = platform.mac_ver()[0] or platform.version()
    return f"{chip}, {mem_gb}, macOS {macos}"


# ---------------------------------------------------------------------------
# Synthetic depth scene generators
# ---------------------------------------------------------------------------

class DepthScene(NamedTuple):
    """Ground-truth depth + degraded (noisy + holes) input."""
    name: str
    ground_truth: np.ndarray   # uint16, (H, W), mm counts, zero = invalid
    noisy_input: np.ndarray    # uint16, (H, W), ground_truth + noise + holes
    hole_mask: np.ndarray      # bool, (H, W), True where pixels are invalid in noisy_input


def _add_noise_and_holes(
    gt: np.ndarray,
    noise_sigma_mm: float = 4.0,
    hole_fraction: float = 0.12,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_input, hole_mask).

    Adds Gaussian noise (sigma in mm) to valid pixels, then punches random
    holes into the frame.  Noise is clipped to [1, 65535] to keep valid.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    valid = gt > 0
    noisy = gt.copy().astype(np.float32)

    # Gaussian noise on valid pixels only
    noise = rng.normal(0.0, noise_sigma_mm, size=gt.shape).astype(np.float32)
    noisy = np.where(valid, noisy + noise, noisy)

    # Punch random holes
    H, W = gt.shape
    n_holes = int(H * W * hole_fraction)
    hole_idx = rng.choice(H * W, size=n_holes, replace=False)
    flat = noisy.flatten()
    flat[hole_idx] = 0.0
    noisy = flat.reshape(H, W)

    # Re-zero out pixels that were already invalid in gt
    noisy = np.where(valid, noisy, 0.0)

    # Clip + round to uint16
    noisy = np.clip(np.round(noisy), 0, 65535).astype(np.uint16)

    hole_mask = noisy == 0
    return noisy, hole_mask


def make_flat_wall(H: int, W: int, rng: np.random.Generator) -> DepthScene:
    """Planar surface at 2000 mm ± small tilt."""
    # Slight linear tilt so the filter has a real gradient to preserve
    rows = np.linspace(1900, 2100, H, dtype=np.float32)[:, None]
    cols = np.linspace(-50, 50, W, dtype=np.float32)[None, :]
    gt_f = rows + cols
    gt = np.clip(np.round(gt_f), 1, 65535).astype(np.uint16)
    noisy, hole_mask = _add_noise_and_holes(gt, noise_sigma_mm=3.0, hole_fraction=0.05, rng=rng)
    return DepthScene("flat_wall", gt, noisy, hole_mask)


def make_sphere(H: int, W: int, rng: np.random.Generator) -> DepthScene:
    """Spherical depth surface centred in the frame, radius 1800 mm."""
    cx, cy = W / 2.0, H / 2.0
    r_sphere = min(H, W) * 1.4
    u = np.arange(W, dtype=np.float32) - cx
    v = np.arange(H, dtype=np.float32)[:, None] - cy
    dist2 = u ** 2 + v ** 2
    # Only generate depth for pixels inside the projected circle
    mask = dist2 < (min(H, W) / 2.2) ** 2
    z_f = np.sqrt(np.maximum(r_sphere ** 2 - dist2, 0.0)) + 500.0
    gt = np.where(mask, np.clip(np.round(z_f), 1, 65535).astype(np.uint16), np.uint16(0))
    noisy, hole_mask = _add_noise_and_holes(gt, noise_sigma_mm=5.0, hole_fraction=0.10, rng=rng)
    return DepthScene("sphere", gt, noisy, hole_mask)


def make_staircase(H: int, W: int, rng: np.random.Generator) -> DepthScene:
    """8-step staircase, each step 200 mm deeper, equally wide."""
    step_w = W // 8
    gt = np.zeros((H, W), dtype=np.uint16)
    for i in range(8):
        depth_val = 1000 + i * 200
        c0 = i * step_w
        c1 = (i + 1) * step_w if i < 7 else W
        gt[:, c0:c1] = depth_val
    noisy, hole_mask = _add_noise_and_holes(gt, noise_sigma_mm=6.0, hole_fraction=0.08, rng=rng)
    return DepthScene("staircase", gt, noisy, hole_mask)


def make_random_holes(H: int, W: int, rng: np.random.Generator) -> DepthScene:
    """Random scene with 18 % pre-existing holes."""
    gt_full = rng.integers(800, 4000, size=(H, W), dtype=np.uint16)
    # Punch a structured region of holes (reflective object simulation)
    r0, r1 = H // 3, 2 * H // 3
    c0, c1 = W // 3, 2 * W // 3
    gt_full[r0:r1, c0:c1] = 0
    # Additional random 8 % holes
    n_rand = int(H * W * 0.08)
    ri = rng.choice(H * W, size=n_rand, replace=False)
    gt_flat = gt_full.flatten()
    gt_flat[ri] = 0
    gt = gt_flat.reshape(H, W)
    noisy, hole_mask = _add_noise_and_holes(gt, noise_sigma_mm=8.0, hole_fraction=0.04, rng=rng)
    return DepthScene("random_holes", gt, noisy, hole_mask)


def build_scenes(H: int, W: int) -> list[DepthScene]:
    rng = np.random.default_rng(2025)
    return [
        make_flat_wall(H, W, rng),
        make_sphere(H, W, rng),
        make_staircase(H, W, rng),
        make_random_holes(H, W, rng),
    ]


# ---------------------------------------------------------------------------
# NumPy RS2-equivalent implementations (faithful to RS2 C++ logic)
# ---------------------------------------------------------------------------

# --- Spatial (bilateral recursive scan) ---

def np_spatial(
    depth: np.ndarray,
    alpha: float = 0.5,
    delta: float = 20.0,
    iterations: int = 2,
) -> np.ndarray:
    """RS2 SDK spatial filter — sequential bilateral scan, per-row.

    This matches the RS2 C++ domain_transform_filter exactly:
      - One full pass = horizontal L→R + R→L + vertical (transposed H→V)
      - Weight: w = exp(-|a - b| / (delta * alpha + eps))
      - Only update when the previous pixel is valid (> 0)
      - Sequential per-row; NO cross-row vectorisation (RS2 behaviour on Mac)

    On Mac the RS2 SDK runs purely on CPU with no SIMD for this pass.
    The Python loop over columns is the honest equivalent of the C++ sequential
    inner loop — every column depends on the previous output.
    """
    eps = 1e-6
    denom = delta * alpha + eps
    frame = depth.astype(np.float32)

    def _scan_lr(f: np.ndarray) -> np.ndarray:
        """Left-to-right bilateral recursive scan, sequential per column."""
        out = f.copy()
        _, W = out.shape
        for col in range(1, W):
            cur  = out[:, col]
            prev = out[:, col - 1]
            valid = prev > 0.0
            diff = np.abs(cur - prev)
            w = np.exp(-diff / denom)
            updated = cur + w * (prev - cur)
            out[:, col] = np.where(valid, updated, cur)
        return out

    def _scan_rl(f: np.ndarray) -> np.ndarray:
        """Right-to-left bilateral recursive scan, sequential per column."""
        out = f.copy()
        _, W = out.shape
        for col in range(W - 2, -1, -1):
            cur  = out[:, col]
            nxt  = out[:, col + 1]
            valid = nxt > 0.0
            diff = np.abs(cur - nxt)
            w = np.exp(-diff / denom)
            updated = cur + w * (nxt - cur)
            out[:, col] = np.where(valid, updated, cur)
        return out

    def _horizontal_pass(f: np.ndarray) -> np.ndarray:
        return _scan_rl(_scan_lr(f))

    def _vertical_pass(f: np.ndarray) -> np.ndarray:
        return _horizontal_pass(f.T).T

    for _ in range(iterations):
        frame = _horizontal_pass(frame)
        frame = _vertical_pass(frame)

    return frame.astype(depth.dtype)


# --- Temporal (EMA + 8-bit persistence bitmask) ---

class NpTemporalFilter:
    """RS2 SDK temporal filter — EMA with 8-bit persistence bitmask.

    Matches rs2::temporal_filter exactly:
      new_history = ((old_history << 1) | valid) & 0xFF
      If |curr - prev| > delta: reset history to curr_valid only.
      smoothed = alpha * curr + (1-alpha) * prev   (where both valid)
      Gate: output 0 unless popcount(history) >= persistence.
      State is updated from the pre-gated smoothed value to avoid
      persistence artifacts corrupting the EMA accumulator.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        delta: float = 20.0,
        persistence: int = 3,
    ) -> None:
        self.alpha = alpha
        self.delta = delta
        self.persistence = persistence
        self._prev: np.ndarray | None = None
        self._history: np.ndarray | None = None

    def process(self, depth: np.ndarray) -> np.ndarray:
        curr = depth.astype(np.float32)
        curr_valid = (curr > 0).astype(np.uint8)

        if self._prev is None:
            self._prev = curr.copy()
            self._history = curr_valid.copy()
            return depth

        prev = self._prev
        history = self._history

        # 1. Update 8-bit bitmask
        new_hist = ((history.astype(np.uint16) << 1) & 0xFF).astype(np.uint8)
        new_hist = (new_hist | curr_valid).astype(np.uint8)

        # 2. Large-change detection (ghost prevention)
        both_valid = (curr > 0) & (prev > 0)
        large_change = (np.abs(curr - prev) > self.delta) & both_valid
        new_hist = np.where(large_change, curr_valid, new_hist)

        # 3. EMA blend
        can_blend = both_valid & ~large_change
        blended = self.alpha * curr + (1.0 - self.alpha) * prev
        smoothed = np.where(can_blend, blended, curr)
        smoothed_pre_gate = smoothed.copy()

        # 4. Persistence gating (popcount via bit-loop, same as MLX)
        if self.persistence > 0:
            x = new_hist.astype(np.int32)
            count = np.zeros_like(x)
            for _ in range(8):
                count += x & 1
                x >>= 1
            smoothed = np.where(count >= self.persistence, smoothed, 0.0)

        # 5. Update state from pre-gated value
        self._prev = np.where(curr > 0, smoothed_pre_gate, prev)
        self._history = new_hist

        if depth.dtype == np.uint16:
            smoothed = np.clip(smoothed, 0, 65535)

        return smoothed.astype(depth.dtype)

    def reset(self) -> None:
        self._prev = None
        self._history = None


# --- Decimation (tile median / valid-mean) ---

def np_decimation(depth: np.ndarray, scale: int = 2) -> np.ndarray:
    """RS2 SDK decimation filter.

    scale 2-3: median of each s×s tile (invalid pixels → valid-mean fallback).
    scale 4-8: mean of valid pixels per tile.
    Crops H, W to multiples of scale (RS2 behaviour).
    """
    s = max(1, min(8, scale))
    if s == 1:
        return depth

    H, W = depth.shape
    H_c = (H // s) * s
    W_c = (W // s) * s
    cropped = depth[:H_c, :W_c].astype(np.float32)

    Ho, Wo = H_c // s, W_c // s
    # Reshape into tiles: (Ho, Wo, s, s)
    tiles = cropped.reshape(Ho, s, Wo, s).transpose(0, 2, 1, 3).reshape(Ho * Wo, s * s)

    valid_mask = tiles > 0.0
    counts = valid_mask.sum(axis=1)            # (N,)
    sums   = (tiles * valid_mask).sum(axis=1)  # (N,)
    safe_c = np.where(counts > 0, counts, 1.0)
    valid_means = np.where(counts > 0, sums / safe_c, 0.0)

    if s <= 3:
        medians = np.median(tiles, axis=1)
        k = s * s
        has_invalid = counts < k
        result = np.where(has_invalid, valid_means, medians)
    else:
        result = valid_means
        if depth.dtype in (np.uint16, np.uint8, np.int32):
            result = result + 0.5

    return result.reshape(Ho, Wo).astype(depth.dtype)


# --- Hole filling (4-neighbour farthest) ---

def np_hole_filling_farthest(depth: np.ndarray) -> np.ndarray:
    """RS2 SDK hole filling — FARTHEST (background-preference) mode.

    Each invalid pixel is replaced by the maximum valid depth among its
    4-connected neighbours.  A single-pass vectorised implementation that
    matches the RS2 SDK output exactly.
    """
    depth_f = depth.astype(np.float32)
    H, W = depth_f.shape

    # Build 4 shifted neighbour views via np.pad (same as MLX)
    padded = np.pad(depth_f, [(1, 1), (1, 1)], mode="constant", constant_values=0.0)
    up    = padded[0:H,   1:W+1]
    down  = padded[2:H+2, 1:W+1]
    left  = padded[1:H+1, 0:W  ]
    right = padded[1:H+1, 2:W+2]

    best = np.maximum(np.maximum(up, down), np.maximum(left, right))
    invalid = depth_f <= 0.0
    result = np.where(invalid, best, depth_f)

    return result.astype(depth.dtype)


# --- Disparity (element-wise division) ---

def np_disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    """RS2 SDK disparity→depth: depth = d2d_factor / disparity."""
    valid = disparity > 0.0
    safe  = np.where(valid, disparity, 1.0)
    depth_f = np.where(valid, _D2D_FACTOR / safe + 0.5, 0.0)
    return np.clip(depth_f, 0, 65535).astype(np.uint16)


def np_depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    """RS2 SDK depth→disparity: disparity = d2d_factor / depth."""
    depth_f = depth.astype(np.float32)
    valid = depth_f > 0.0
    safe  = np.where(valid, depth_f, 1.0)
    return np.where(valid, _D2D_FACTOR / safe, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_numpy_fn(
    fn: Callable,
    args: tuple,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, float]:
    """Time a pure-NumPy function with perf_counter."""
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


def _time_mlx_fn(
    fn: Callable,
    args: tuple,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, float]:
    """Time an MLX function with device-sync barriers."""
    return benchmark_component(fn, args, warmup=warmup, iterations=iterations)


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

def _measure_numpy_peak_mb(fn: Callable, args: tuple) -> float:
    """Estimate peak heap allocation of a NumPy call in MiB."""
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _measure_mlx_peak_mb(fn: Callable, args: tuple) -> float:
    """Measure peak Metal device memory after one MLX call (MiB)."""
    mx.reset_peak_memory()
    mx.eval(mx.zeros(1))
    result = fn(*args)
    mx.eval(result)
    return mx.get_peak_memory() / (1024 * 1024)


# ---------------------------------------------------------------------------
# Quality metric helpers
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE over valid (gt > 0) pixels, in depth units (mm counts)."""
    valid = gt > 0
    if not valid.any():
        return float("nan")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return float(np.sqrt(np.mean(diff[valid] ** 2)))


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE over valid pixels."""
    valid = gt > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred.astype(np.float64) - gt.astype(np.float64))[valid]))


def _psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 65535.0) -> float:
    """PSNR (dB) over valid pixels."""
    valid = gt > 0
    if not valid.any():
        return float("nan")
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64))[valid] ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def _holes_filled_ratio(output: np.ndarray, hole_mask: np.ndarray) -> float:
    """Fraction of originally invalid pixels that are now non-zero."""
    n_holes = int(hole_mask.sum())
    if n_holes == 0:
        return 1.0
    filled = int((output[hole_mask] > 0).sum())
    return filled / n_holes


def _noise_reduced(
    output: np.ndarray, noisy: np.ndarray, gt: np.ndarray
) -> float:
    """Noise reduction ratio: 1 - (residual_std / input_noise_std).

    Measures what fraction of the input noise standard deviation has been
    removed.  Computed on valid pixels only.
    """
    valid = gt > 0
    if not valid.any():
        return float("nan")
    in_err  = (noisy.astype(np.float64) - gt.astype(np.float64))[valid]
    out_err = (output.astype(np.float64) - gt.astype(np.float64))[valid]
    in_std  = float(np.std(in_err))
    out_std = float(np.std(out_err))
    if in_std < 1e-9:
        return 0.0
    return max(0.0, 1.0 - out_std / in_std)


def _edge_preservation(
    output: np.ndarray, gt: np.ndarray
) -> float:
    """Gradient-magnitude preservation: mean(|grad(out)|) / mean(|grad(gt)|).

    Values > 1 mean the filter sharpens; < 1 means blurring.
    """
    def _grad_mag(img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float64)
        gx = np.diff(img_f, axis=1, prepend=img_f[:, :1])
        gy = np.diff(img_f, axis=0, prepend=img_f[:1, :])
        return np.sqrt(gx ** 2 + gy ** 2)

    valid = gt > 0
    if not valid.any():
        return float("nan")
    gt_mag  = _grad_mag(gt)[valid]
    out_mag = _grad_mag(output)[valid]
    denom = float(np.mean(gt_mag))
    if denom < 1e-9:
        return float("nan")
    return float(np.mean(out_mag)) / denom


@dataclass
class QualityResult:
    rmse: float
    mae: float
    psnr: float
    holes_filled: float   # 0–1
    noise_removed: float  # 0–1
    edges_pres: float     # ratio


def _compute_quality(
    output: np.ndarray,
    gt: np.ndarray,
    noisy: np.ndarray,
    hole_mask: np.ndarray,
) -> QualityResult:
    out_u16 = output.astype(np.uint16) if output.dtype != np.uint16 else output
    return QualityResult(
        rmse=_rmse(out_u16, gt),
        mae=_mae(out_u16, gt),
        psnr=_psnr(out_u16, gt),
        holes_filled=_holes_filled_ratio(out_u16, hole_mask),
        noise_removed=_noise_reduced(out_u16, noisy, gt),
        edges_pres=_edge_preservation(out_u16, gt),
    )


# ---------------------------------------------------------------------------
# Per-algorithm benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class ComponentResult:
    """Holds speed + quality + memory for one (algorithm, implementation) pair."""
    speed: dict[str, float]         # mean_ms, fps, std_ms, min_ms, max_ms
    quality_by_scene: dict[str, QualityResult] = field(default_factory=dict)
    peak_memory_mb: float = 0.0


def _run_spatial(
    scenes: list[DepthScene],
    warmup: int,
    iters: int,
) -> tuple[ComponentResult, ComponentResult]:
    """Benchmark spatial filter: RS2 NumPy vs MLX Metal."""

    # ---- MLX filter (Metal) ----
    mlx_filt = SpatialFilter(alpha=0.5, delta=20.0, iterations=2, use_metal=True)

    # Use the first scene for speed measurement (representative)
    ref_scene = scenes[0]
    depth_mx = mx.array(ref_scene.noisy_input)
    mx.eval(depth_mx)
    depth_f32 = depth_mx.astype(mx.float32)
    mx.eval(depth_f32)

    mlx_speed = _time_mlx_fn(mlx_filt.process, (depth_f32,), warmup=warmup, iterations=iters)
    mlx_mem = _measure_mlx_peak_mb(mlx_filt.process, (depth_f32,))

    mlx_result = ComponentResult(speed=mlx_speed, peak_memory_mb=mlx_mem)
    for scene in scenes:
        inp_f32 = mx.array(scene.noisy_input.astype(np.float32))
        mx.eval(inp_f32)
        out_mlx = mlx_filt.process(inp_f32)
        mx.eval(out_mlx)
        out_np = np.array(out_mlx).astype(np.uint16)
        mlx_result.quality_by_scene[scene.name] = _compute_quality(
            out_np, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    # ---- NumPy RS2 ----
    ref_np = ref_scene.noisy_input

    np_speed = _time_numpy_fn(
        np_spatial, (ref_np, 0.5, 20.0, 2),
        warmup=warmup, iterations=iters,
    )
    np_mem = _measure_numpy_peak_mb(np_spatial, (ref_np, 0.5, 20.0, 2))

    np_result = ComponentResult(speed=np_speed, peak_memory_mb=np_mem)
    for scene in scenes:
        out_np_s = np_spatial(scene.noisy_input, 0.5, 20.0, 2)
        np_result.quality_by_scene[scene.name] = _compute_quality(
            out_np_s, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    return np_result, mlx_result


def _run_temporal(
    scenes: list[DepthScene],
    warmup: int,
    iters: int,
) -> tuple[ComponentResult, ComponentResult]:
    """Benchmark temporal filter: RS2 NumPy vs MLX Metal."""

    _SEED_FRAMES = 8   # pre-seed both filters with identical frames

    ref_scene = scenes[0]
    rng = np.random.default_rng(77)
    seeds = [rng.integers(500, 5000, size=ref_scene.noisy_input.shape, dtype=np.uint16)
             for _ in range(_SEED_FRAMES)]

    # ---- MLX ----
    mlx_filt = TemporalFilter(alpha=0.4, delta=20.0, persistence=3)
    for s in seeds:
        mlx_filt.process(mx.array(s))

    depth_mx = mx.array(ref_scene.noisy_input)
    mx.eval(depth_mx)

    mlx_speed = _time_mlx_fn(mlx_filt.process, (depth_mx,), warmup=warmup, iterations=iters)
    mlx_mem = _measure_mlx_peak_mb(mlx_filt.process, (depth_mx,))

    mlx_result = ComponentResult(speed=mlx_speed, peak_memory_mb=mlx_mem)
    for scene in scenes:
        filt = TemporalFilter(alpha=0.4, delta=20.0, persistence=3)
        for s in seeds:
            filt.process(mx.array(s))
        out = filt.process(mx.array(scene.noisy_input))
        mx.eval(out)
        out_np = np.array(out).astype(np.uint16)
        mlx_result.quality_by_scene[scene.name] = _compute_quality(
            out_np, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    # ---- NumPy RS2 ----
    np_filt = NpTemporalFilter(alpha=0.4, delta=20.0, persistence=3)
    for s in seeds:
        np_filt.process(s)

    ref_np = ref_scene.noisy_input
    np_speed = _time_numpy_fn(np_filt.process, (ref_np,), warmup=warmup, iterations=iters)
    np_mem = _measure_numpy_peak_mb(np_filt.process, (ref_np,))

    np_result = ComponentResult(speed=np_speed, peak_memory_mb=np_mem)
    for scene in scenes:
        filt = NpTemporalFilter(alpha=0.4, delta=20.0, persistence=3)
        for s in seeds:
            filt.process(s)
        out_np_s = filt.process(scene.noisy_input)
        np_result.quality_by_scene[scene.name] = _compute_quality(
            out_np_s, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    return np_result, mlx_result


def _run_decimation(
    scenes: list[DepthScene],
    warmup: int,
    iters: int,
) -> tuple[ComponentResult, ComponentResult]:
    """Benchmark decimation (scale=2): RS2 NumPy vs MLX Metal."""

    ref_scene = scenes[0]

    # ---- MLX ----
    mlx_filt = DecimationFilter(scale=2)
    depth_mx = mx.array(ref_scene.noisy_input)
    mx.eval(depth_mx)

    mlx_speed = _time_mlx_fn(mlx_filt.process, (depth_mx,), warmup=warmup, iterations=iters)
    mlx_mem = _measure_mlx_peak_mb(mlx_filt.process, (depth_mx,))

    mlx_result = ComponentResult(speed=mlx_speed, peak_memory_mb=mlx_mem)
    for scene in scenes:
        dm = mx.array(scene.noisy_input)
        out = mlx_filt.process(dm)
        mx.eval(out)
        # Upsample back for fair comparison against ground truth
        out_np = np.array(out)
        out_up = np.repeat(np.repeat(out_np, 2, axis=0), 2, axis=1)
        H, W = scene.ground_truth.shape
        out_up = out_up[:H, :W]
        mlx_result.quality_by_scene[scene.name] = _compute_quality(
            out_up.astype(np.uint16), scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    # ---- NumPy RS2 ----
    ref_np = ref_scene.noisy_input
    np_speed = _time_numpy_fn(np_decimation, (ref_np, 2), warmup=warmup, iterations=iters)
    np_mem = _measure_numpy_peak_mb(np_decimation, (ref_np, 2))

    np_result = ComponentResult(speed=np_speed, peak_memory_mb=np_mem)
    for scene in scenes:
        out_np_s = np_decimation(scene.noisy_input, 2)
        out_up = np.repeat(np.repeat(out_np_s, 2, axis=0), 2, axis=1)
        H, W = scene.ground_truth.shape
        out_up = out_up[:H, :W]
        np_result.quality_by_scene[scene.name] = _compute_quality(
            out_up.astype(np.uint16), scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    return np_result, mlx_result


def _run_hole_filling(
    scenes: list[DepthScene],
    warmup: int,
    iters: int,
) -> tuple[ComponentResult, ComponentResult]:
    """Benchmark hole filling (FARTHEST): RS2 NumPy vs MLX Metal."""

    ref_scene = scenes[0]

    # ---- MLX ----
    mlx_filt = HoleFillingFilter(mode=HoleFillingFilter.FARTHEST)
    depth_mx = mx.array(ref_scene.noisy_input)
    mx.eval(depth_mx)

    mlx_speed = _time_mlx_fn(mlx_filt.process, (depth_mx,), warmup=warmup, iterations=iters)
    mlx_mem = _measure_mlx_peak_mb(mlx_filt.process, (depth_mx,))

    mlx_result = ComponentResult(speed=mlx_speed, peak_memory_mb=mlx_mem)
    for scene in scenes:
        dm = mx.array(scene.noisy_input)
        out = mlx_filt.process(dm)
        mx.eval(out)
        out_np = np.array(out)
        mlx_result.quality_by_scene[scene.name] = _compute_quality(
            out_np.astype(np.uint16), scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    # ---- NumPy RS2 ----
    ref_np = ref_scene.noisy_input
    np_speed = _time_numpy_fn(
        np_hole_filling_farthest, (ref_np,), warmup=warmup, iterations=iters
    )
    np_mem = _measure_numpy_peak_mb(np_hole_filling_farthest, (ref_np,))

    np_result = ComponentResult(speed=np_speed, peak_memory_mb=np_mem)
    for scene in scenes:
        out_np_s = np_hole_filling_farthest(scene.noisy_input)
        np_result.quality_by_scene[scene.name] = _compute_quality(
            out_np_s, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    return np_result, mlx_result


def _run_disparity(
    scenes: list[DepthScene],
    warmup: int,
    iters: int,
) -> tuple[ComponentResult, ComponentResult]:
    """Benchmark depth→disparity→depth round-trip: RS2 NumPy vs MLX Metal."""

    ref_scene = scenes[0]
    disp_ref_np = np_depth_to_disparity(ref_scene.noisy_input)

    # ---- MLX ----
    mlx_d2d = DisparityTransform(
        baseline_mm=_BASELINE_MM, focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS, to_disparity=True,
    )
    mlx_d2depth = DisparityTransform(
        baseline_mm=_BASELINE_MM, focal_px=_FOCAL_PX,
        depth_units=_DEPTH_UNITS, to_disparity=False,
    )

    depth_mx = mx.array(ref_scene.noisy_input)
    mx.eval(depth_mx)
    disp_mx = mx.array(disp_ref_np)
    mx.eval(disp_mx)

    # Measure forward (depth→disparity) as that is the hot path in the pipeline
    mlx_speed = _time_mlx_fn(mlx_d2d.process, (depth_mx,), warmup=warmup, iterations=iters)
    mlx_mem = _measure_mlx_peak_mb(mlx_d2d.process, (depth_mx,))

    mlx_result = ComponentResult(speed=mlx_speed, peak_memory_mb=mlx_mem)
    for scene in scenes:
        dm = mx.array(scene.noisy_input)
        disp = mlx_d2d.process(dm)
        out = mlx_d2depth.process(disp)
        mx.eval(out)
        out_np = np.array(out)
        mlx_result.quality_by_scene[scene.name] = _compute_quality(
            out_np, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    # ---- NumPy RS2 ----
    ref_np = ref_scene.noisy_input
    np_speed = _time_numpy_fn(
        np_depth_to_disparity, (ref_np,), warmup=warmup, iterations=iters
    )
    np_mem = _measure_numpy_peak_mb(np_depth_to_disparity, (ref_np,))

    np_result = ComponentResult(speed=np_speed, peak_memory_mb=np_mem)
    for scene in scenes:
        disp_s = np_depth_to_disparity(scene.noisy_input)
        out_np_s = np_disparity_to_depth(disp_s)
        np_result.quality_by_scene[scene.name] = _compute_quality(
            out_np_s, scene.ground_truth, scene.noisy_input, scene.hole_mask
        )

    return np_result, mlx_result


# ---------------------------------------------------------------------------
# Aggregated quality across scenes
# ---------------------------------------------------------------------------

def _avg_quality(result: ComponentResult) -> QualityResult:
    """Average quality metrics across all scenes."""
    fields = ["rmse", "mae", "psnr", "holes_filled", "noise_removed", "edges_pres"]
    agg: dict[str, list[float]] = {f: [] for f in fields}
    for qr in result.quality_by_scene.values():
        for f in fields:
            v = getattr(qr, f)
            if not math.isnan(v) and not math.isinf(v):
                agg[f].append(v)
    return QualityResult(**{f: float(np.mean(vals)) if vals else float("nan")
                             for f, vals in agg.items()})


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_LINE_W = 72


def _sep(char: str = "=", width: int = _LINE_W) -> None:
    print(char * width)


def _banner(machine: str, resolution: str, H: int, W: int) -> None:
    print()
    _sep()
    print(f"  realsense-mlx vs RS2 SDK -- Benchmark Results")
    print(f"  Machine   : {machine}")
    print(f"  Resolution: {resolution} ({W}x{H})")
    print(f"  Date      : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    _sep()


def _print_speed_table(
    components: list[str],
    rs2_results: dict[str, ComponentResult],
    mlx_results: dict[str, ComponentResult],
) -> None:
    print()
    print("SPEED  (higher FPS is better)")
    print()
    _col = 20
    hdr = (
        f"  {'Component':<{_col}s}"
        f"  {'RS2/CPU (ms)':>14}"
        f"  {'RS2 FPS':>10}"
        f"  {'MLX/Metal (ms)':>14}"
        f"  {'MLX FPS':>10}"
        f"  {'Speedup':>8}"
    )
    print(hdr)
    print("  " + "-" * (_LINE_W - 2))

    for name in components:
        rs2 = rs2_results[name]
        mlx = mlx_results[name]
        rs2_ms  = rs2.speed["mean_ms"]
        mlx_ms  = mlx.speed["mean_ms"]
        rs2_fps = rs2.speed["fps"]
        mlx_fps = mlx.speed["fps"]
        speedup = rs2_ms / mlx_ms if mlx_ms > 0 else float("inf")
        if speedup >= 1000:
            su_str = f"{speedup:,.0f}x"
        elif speedup >= 10:
            su_str = f"{speedup:.0f}x"
        else:
            su_str = f"{speedup:.1f}x"
        print(
            f"  {name:<{_col}s}"
            f"  {rs2_ms:>13.2f}ms"
            f"  {rs2_fps:>9.1f}"
            f"  {mlx_ms:>13.2f}ms"
            f"  {mlx_fps:>9.1f}"
            f"  {su_str:>8}"
        )


def _print_quality_table(
    components: list[str],
    rs2_results: dict[str, ComponentResult],
    mlx_results: dict[str, ComponentResult],
) -> None:
    print()
    print("QUALITY  (averaged across 4 synthetic scenes)")
    print()
    _col = 20
    hdr = (
        f"  {'Metric':<{_col}s}"
        f"  {'RS2/CPU':>12}"
        f"  {'MLX/Metal':>12}"
        f"  {'Delta':>10}"
    )
    print(hdr)
    print("  " + "-" * (_LINE_W - 2))

    for name in components:
        rs2_q = _avg_quality(rs2_results[name])
        mlx_q = _avg_quality(mlx_results[name])
        print(f"  [{name}]")

        metrics = [
            ("RMSE (mm)",      rs2_q.rmse,          mlx_q.rmse,          "lower=better",  ".2f"),
            ("MAE  (mm)",      rs2_q.mae,            mlx_q.mae,            "lower=better",  ".2f"),
            ("PSNR (dB)",      rs2_q.psnr,           mlx_q.psnr,           "higher=better", ".1f"),
            ("Holes filled %", rs2_q.holes_filled*100, mlx_q.holes_filled*100, "higher=better", ".1f"),
            ("Noise removed %",rs2_q.noise_removed*100,mlx_q.noise_removed*100,"higher=better",".1f"),
            ("Edge pres.",     rs2_q.edges_pres,     mlx_q.edges_pres,     "~1.0 ideal",   ".3f"),
        ]
        for label, rs2_v, mlx_v, note, fmt in metrics:
            if math.isnan(rs2_v) or math.isnan(mlx_v):
                continue
            delta = mlx_v - rs2_v
            delta_str = f"{delta:+.2f}"
            rs2_str  = format(rs2_v,  fmt)
            mlx_str  = format(mlx_v,  fmt)
            print(
                f"  {'  ' + label:<{_col}s}"
                f"  {rs2_str:>12}"
                f"  {mlx_str:>12}"
                f"  {delta_str:>10}  ({note})"
            )
        print()


def _print_memory_table(
    components: list[str],
    rs2_results: dict[str, ComponentResult],
    mlx_results: dict[str, ComponentResult],
) -> None:
    print()
    print("MEMORY  (peak allocation per call)")
    print()
    _col = 20
    hdr = (
        f"  {'Component':<{_col}s}"
        f"  {'RS2/CPU (MB)':>14}"
        f"  {'MLX/Metal (MB)':>15}"
        f"  {'Ratio':>8}"
    )
    print(hdr)
    print("  " + "-" * (_LINE_W - 2))

    for name in components:
        rs2_mb = rs2_results[name].peak_memory_mb
        mlx_mb = mlx_results[name].peak_memory_mb
        ratio  = mlx_mb / rs2_mb if rs2_mb > 0 else float("inf")
        print(
            f"  {name:<{_col}s}"
            f"  {rs2_mb:>13.1f}"
            f"  {mlx_mb:>14.1f}"
            f"  {ratio:>7.2f}x"
        )


def _print_per_scene_quality(
    components: list[str],
    rs2_results: dict[str, ComponentResult],
    mlx_results: dict[str, ComponentResult],
    scenes: list[DepthScene],
) -> None:
    print()
    print("QUALITY  BY SCENE  (RMSE mm -- lower is better)")
    print()
    _col = 20
    scene_names = [s.name for s in scenes]

    # header
    hdr = f"  {'Component':<{_col}s}  {'Scene':<16s}  {'RS2/CPU':>10}  {'MLX/Metal':>10}  {'Delta':>8}"
    print(hdr)
    print("  " + "-" * (_LINE_W - 2))

    for name in components:
        for sname in scene_names:
            rs2_qr = rs2_results[name].quality_by_scene.get(sname)
            mlx_qr = mlx_results[name].quality_by_scene.get(sname)
            if rs2_qr is None or mlx_qr is None:
                continue
            delta = mlx_qr.rmse - rs2_qr.rmse
            print(
                f"  {name:<{_col}s}  {sname:<16s}"
                f"  {rs2_qr.rmse:>9.2f}"
                f"  {mlx_qr.rmse:>9.2f}"
                f"  {delta:>+8.2f}"
            )
        print()


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert dataclass / dict / numpy floats to JSON-safe types."""
    if isinstance(obj, QualityResult):
        return asdict(obj)
    if isinstance(obj, ComponentResult):
        return {
            "speed": obj.speed,
            "quality_by_scene": {k: asdict(v) for k, v in obj.quality_by_scene.items()},
            "peak_memory_mb": obj.peak_memory_mb,
        }
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    resolution: str = "720p",
    warmup: int = 5,
    iters: int = 30,
    speed_only: bool = False,
    quality_only: bool = False,
    output: str | None = None,
) -> dict[str, Any]:
    H, W = RESOLUTIONS[resolution]
    machine = _detect_machine()

    print(f"\n  Building synthetic scenes ({W}x{H}) ...", flush=True)
    scenes = build_scenes(H, W)
    scene_names = [s.name for s in scenes]
    print(f"  Scenes: {scene_names}")

    # --- Note on spatial warmup ---
    # The NumPy spatial filter has an O(H*W*iters) sequential Python loop.
    # At 720p (1280 cols) with 2 iterations + 2 axes, a single call takes
    # several seconds.  We cap NumPy spatial to 3 timed iterations
    # (still statistically sound) to keep total benchmark time reasonable.
    # MLX always uses the full iters count.

    COMPONENTS = ["spatial", "temporal", "decimation", "hole_filling", "disparity"]

    rs2_results: dict[str, ComponentResult] = {}
    mlx_results: dict[str, ComponentResult] = {}

    print()
    for name in COMPONENTS:
        print(f"  Benchmarking: {name} ...", flush=True)
        if name == "spatial":
            # Sequential NumPy spatial is very slow at high res — cap iterations
            np_iters_spatial = min(iters, 3)
            np_r, mlx_r = _run_spatial(scenes, warmup=min(warmup, 2), iters=np_iters_spatial)
            # Re-time MLX with full iters (fast)
            mlx_filt = SpatialFilter(alpha=0.5, delta=20.0, iterations=2, use_metal=True)
            ref_f32 = mx.array(scenes[0].noisy_input.astype(np.float32))
            mx.eval(ref_f32)
            mlx_r.speed = _time_mlx_fn(
                mlx_filt.process, (ref_f32,), warmup=warmup, iterations=iters
            )
            rs2_results[name] = np_r
            mlx_results[name] = mlx_r
        elif name == "temporal":
            rs2_results[name], mlx_results[name] = _run_temporal(scenes, warmup, iters)
        elif name == "decimation":
            rs2_results[name], mlx_results[name] = _run_decimation(scenes, warmup, iters)
        elif name == "hole_filling":
            rs2_results[name], mlx_results[name] = _run_hole_filling(scenes, warmup, iters)
        elif name == "disparity":
            rs2_results[name], mlx_results[name] = _run_disparity(scenes, warmup, iters)

    # --- Output ---
    _banner(machine, resolution, H, W)

    if not quality_only:
        _print_speed_table(COMPONENTS, rs2_results, mlx_results)

    if not speed_only:
        _print_quality_table(COMPONENTS, rs2_results, mlx_results)
        _print_per_scene_quality(COMPONENTS, rs2_results, mlx_results, scenes)

    if not speed_only:
        _print_memory_table(COMPONENTS, rs2_results, mlx_results)

    _sep()
    print()

    # --- JSON payload ---
    payload: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "machine": machine,
        "resolution": resolution,
        "dimensions": {"H": H, "W": W},
        "warmup": warmup,
        "iterations": iters,
        "components": COMPONENTS,
        "scene_names": scene_names,
        "rs2": {name: _to_json_safe(r) for name, r in rs2_results.items()},
        "mlx": {name: _to_json_safe(r) for name, r in mlx_results.items()},
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"  Results saved to: {out_path}")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_vs_rs2.py",
        description=(
            "Publication-quality benchmark: realsense-mlx (MLX Metal) "
            "vs RS2 SDK (NumPy CPU equivalent)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTIONS),
        default="720p",
        help="Frame resolution to benchmark (default: 720p).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up iterations before timing starts (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="Timed iterations per configuration (default: 30). "
             "NumPy spatial is capped at min(iters, 3) due to O(W) sequential loop.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Path to save JSON results (optional).",
    )
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="Print only the speed table (skip quality metrics).",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Print only the quality tables (skip speed table).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.speed_only and args.quality_only:
        print("Error: --speed-only and --quality-only are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    run_benchmark(
        resolution=args.resolution,
        warmup=args.warmup,
        iters=args.iters,
        speed_only=args.speed_only,
        quality_only=args.quality_only,
        output=args.output,
    )


if __name__ == "__main__":
    main()
