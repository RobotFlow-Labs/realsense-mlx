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
"""

from realsense_mlx.utils.benchmark import Timer, benchmark_component

__all__ = ["Timer", "benchmark_component"]
