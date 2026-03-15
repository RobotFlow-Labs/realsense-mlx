# realsense-mlx — PRD Index

## Port Overview
MLX-accelerated processing backend for Intel RealSense cameras on Apple Silicon.
Wraps `pyrealsense2` capture with MLX compute for depth processing, point clouds, and filters.

## PRD Dependency Graph

```
PRD-00 (Project Setup)
  ├── PRD-01 (Format Converters)           [EASY]
  ├── PRD-02 (Point Cloud)                 [MEDIUM]
  │     └── PRD-04 (Alignment)             [HARD]
  ├── PRD-03 (Depth Filters × 5)          [MEDIUM]
  ├── PRD-05 (Colorizer)                   [EASY-MEDIUM]
  │     └── PRD-06 (Display Viewer)        [MEDIUM]
  ├── PRD-07 (Integration Testing)         [parallel with all]
  └── PRD-08 (Benchmarks)                  [after all]
```

## Execution Order (recommended)

| Phase | PRDs | Effort | Can Parallelize |
|-------|------|--------|-----------------|
| 1 | PRD-00 (Setup) | 4-6h | No — blocks everything |
| 2 | PRD-01, PRD-03 (disparity+hole-fill), PRD-05 | 16-20h | YES — independent |
| 3 | PRD-02 (Pointcloud), PRD-03 (decimation+spatial+temporal) | 20-28h | YES — independent |
| 4 | PRD-04 (Alignment) | 16-24h | No — needs PRD-02 |
| 5 | PRD-06 (Viewer) | 8-12h | Can start after PRD-05 |
| 6 | PRD-07, PRD-08 (Testing + Benchmarks) | 20-28h | YES |

**Total estimated: 84-118 hours (10-15 eng-days)**

## CUDA → MLX Port Summary

| CUDA Kernel | PRD | MLX Strategy | Difficulty |
|-------------|-----|-------------|------------|
| 10 format converters | PRD-01 | Vectorized bitwise/arithmetic | EASY |
| pointcloud deproject | PRD-02 | Precomputed grids + broadcast multiply | MEDIUM |
| align map_depth_to_other | PRD-04 | Vectorized matmul + gather | HARD |
| align other_to_depth | PRD-04 | Gather from color frame | MEDIUM |
| align depth_to_other | PRD-04 | Sort-based scatter-min | HARD |
| align replace_to_zero | PRD-04 | mx.where | TRIVIAL |

## Processing Filters → MLX Summary

| Filter | PRD | MLX Strategy | Speedup Target |
|--------|-----|-------------|---------------|
| Disparity transform | PRD-03 | Elementwise division | 8-16× |
| Hole filling | PRD-03 | Neighbor gather + min/max | 6-12× |
| Decimation | PRD-03 | Reshape + median/mean | 8-16× |
| Spatial | PRD-03 | Row-parallel recursive scan | 2-5× |
| Temporal | PRD-03 | Vectorized alpha blend + state | 3-8× |
| Colorizer | PRD-05 | Precomputed LUT + gather | 3-6× |
