"""Utility modules for realsense-mlx.

Exports
-------
Timer
    Context manager that measures wall-clock time around MLX operations,
    inserting synchronization barriers before and after to give accurate
    device-side latency readings.

benchmark_component
    Benchmark a processing function over multiple iterations and return
    timing statistics (mean, std, min, max, fps).

DepthStats
    Static methods for computing quality metrics on depth frames and
    comparing before/after pairs (RMSE, MAE, PSNR, SSIM approximation,
    hole statistics).
"""

from realsense_mlx.utils.benchmark import Timer, benchmark_component
from realsense_mlx.utils.depth_stats import DepthStats

__all__ = ["Timer", "benchmark_component", "DepthStats"]
