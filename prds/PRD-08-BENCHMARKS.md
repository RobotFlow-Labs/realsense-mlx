# PRD-08: Performance Benchmarks & Optimization

## Overview
Establish performance baselines, identify bottlenecks, and optimize the MLX processing pipeline for real-time depth camera usage on Apple Silicon.

## Blocked By
- PRD-00 through PRD-05 (functional correctness first)

## Target Performance

| Component | Resolution | Target FPS | Baseline (CPU numpy) |
|-----------|-----------|------------|---------------------|
| Format converters | 720p | ≥120 FPS | ~30-60 FPS |
| Point cloud | 720p | ≥60 FPS | ~15-20 FPS |
| Spatial filter | 720p | ≥60 FPS | ~20-30 FPS |
| Temporal filter | 720p | ≥120 FPS | ~40-60 FPS |
| Decimation (2×) | 720p | ≥120 FPS | ~30-60 FPS |
| Hole filling | 720p | ≥120 FPS | ~60-90 FPS |
| Disparity transform | 720p | ≥200 FPS | ~50-100 FPS |
| Alignment | 720p | ≥30 FPS | ~10-15 FPS |
| Colorizer | 720p | ≥60 FPS | ~30-40 FPS |
| **Full pipeline** | **720p** | **≥30 FPS** | **~5-8 FPS** |

## Benchmark Framework

```python
# benchmarks/bench_filters.py
import pytest
from pytest_benchmark import fixture as bench_fixture

@pytest.mark.benchmark(group="pointcloud")
def test_pointcloud_720p(benchmark, d415_intrinsics):
    depth = mx.array(np.random.randint(500, 5000, (720, 1280), dtype=np.uint16))
    gen = PointCloudGenerator(d415_intrinsics, 0.001)

    def run():
        result = gen.generate(depth)
        mx.eval(result)  # Force materialization

    benchmark(run)

@pytest.mark.benchmark(group="pipeline")
def test_full_pipeline_720p(benchmark):
    depth = mx.array(np.random.randint(500, 5000, (720, 1280), dtype=np.uint16))
    pipeline = DepthPipeline()

    # Warm up temporal state
    for _ in range(5):
        pipeline.process(depth)

    def run():
        result = pipeline.process(depth)
        mx.eval(result)

    benchmark(run)
```

### Resolution Matrix
Benchmark each component at:
- 480p: 640×480 (D415/D435 default)
- 720p: 1280×720 (D435i max)
- 1080p: 1920×1080 (stress test)

### A/B Comparison
```python
def bench_mlx_vs_cpu(depth, component_name):
    """Compare MLX vs numpy CPU for same operation."""
    mlx_filter = create_mlx_filter(component_name)
    cpu_filter = create_cpu_filter(component_name)

    # MLX
    t0 = time.perf_counter()
    for _ in range(100):
        result = mlx_filter.process(mx.array(depth))
        mx.eval(result)
    mlx_time = (time.perf_counter() - t0) / 100

    # CPU
    t0 = time.perf_counter()
    for _ in range(100):
        result = cpu_filter.process(depth)
    cpu_time = (time.perf_counter() - t0) / 100

    speedup = cpu_time / mlx_time
    print(f"{component_name}: MLX={mlx_time*1000:.1f}ms CPU={cpu_time*1000:.1f}ms Speedup={speedup:.1f}×")
```

## CLI Entry Point

```bash
# Run all benchmarks
rs-mlx-bench

# Specific component
rs-mlx-bench --filter pointcloud

# Specific resolution
rs-mlx-bench --resolution 720p

# Compare with CPU
rs-mlx-bench --compare-cpu

# JSON output for CI tracking
rs-mlx-bench --json > benchmarks/results/$(date +%Y%m%d).json
```

## Optimization Priorities (ranked by pipeline impact)

1. **Spatial filter** — Sequential scan is the bottleneck; consider Metal kernel
2. **Point cloud** — Precompute coordinate grids, cache per intrinsics
3. **Alignment** — Vectorized matmul + gather is key
4. **Decimation** — Reshape + median should be fast natively
5. **Temporal** — Keep state on device, minimize eval() calls
6. **Format conversion** — Already trivially fast
7. **Disparity** — Already trivially fast

## Profiling Strategy

```python
import mlx.core as mx

# Profile individual operations
with mx.stream(mx.cpu):  # vs mx.gpu
    result = filter.process(depth)
    mx.eval(result)

# Memory tracking
print(f"Peak memory: {mx.metal.get_peak_memory() / 1e6:.1f} MB")
print(f"Active memory: {mx.metal.get_active_memory() / 1e6:.1f} MB")
```

## Acceptance Criteria

- [ ] All components benchmarked at 480p, 720p, 1080p
- [ ] Full pipeline achieves ≥30 FPS at 720p
- [ ] MLX vs CPU comparison documented
- [ ] Memory usage profiled and documented
- [ ] JSON results stored for CI regression tracking
- [ ] CLI tool functional

## Estimated Effort
8-12 hours

## Port Difficulty
**LOW** — Benchmarking infrastructure, not algorithmic work
