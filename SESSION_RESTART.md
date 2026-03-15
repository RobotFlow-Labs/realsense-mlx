# Session Restart Guide

## How to Resume This Session

```bash
# Navigate to the project
cd /Users/ilessio/Development/AIFLOWLABS/R&D/realsense-mlx

# Resume with Claude Code (use --continue to pick up context)
claude --continue

# Or start fresh with full context:
claude

# Then tell Claude:
# "Load /port-to-mlx and continue working on realsense-mlx.
#  Read SESSION_RESTART.md for context."
```

### Key Context for Claude

```
Project:    realsense-mlx
Repo:       https://github.com/RobotFlow-Labs/realsense-mlx
Location:   /Users/ilessio/Development/AIFLOWLABS/R&D/realsense-mlx
Venv:       .venv (uv, Python 3.12, mlx 0.31.1)
Skill:      /port-to-mlx (loaded from ~/.claude/skills/port-to-mlx/)
Memory:     ~/.claude/projects/-Users-ilessio-Development-AIFLOWLABS-R-D/memory/project_realsense_mlx.md
```

---

## Session Summary — 2026-03-15

**Duration:** Single session, ~6 hours
**Operator:** ilessio (solo founder/CEO, AIFLOW LABS / Robot Flow Labs)
**AI:** Claude Opus 4.6 (1M context) with multi-agent parallelization
**Approach:** 4 parallel sub-agents per iteration, code review loops, stress testing

### What Was Built (from zero to shipped)

An MLX-accelerated depth camera processing library for Apple Silicon that is **faster than Intel's original RS2 SDK** on Mac and has **more features**.

### Release History

| Version | Commit | Tests | LOC | Key Feature |
|---------|--------|-------|-----|-------------|
| v0.1.0 | `5d88d06` | 381 | 10K | Initial port: 10 CUDA converters, 5 filters, pointcloud, alignment |
| v0.2.0 | `85bcb60` | 468 | 12K | Metal spatial kernel (89x speedup), 32 code review fixes |
| v0.3.0 | `6b0acbe` | 528 | 14K | Metal hole-fill kernel (19x), production hardening |
| v0.4.0 | `5042c53` | 785 | 20K | Metal alignment (2.6x), bilateral filter, mesh gen, ROS2 bridge |
| v0.5.0 | `b4b4b9f` | 938 | 25K | E2E processor, mesh 6.8x, multi-camera, recording |
| v0.6.0 | `5f419be` | 1,048 | 31K | Stereo depth (any camera), robotics (occupancy + obstacles) |
| final | `c54c282` | 1,048 | 31K | README with real benchmark percentages |

### Architecture

```
realsense-mlx/
├── src/realsense_mlx/
│   ├── filters/          # 10 depth filters (spatial, temporal, decimation, hole-fill,
│   │                     #   disparity, colorizer, bilateral, enhancement, pipeline)
│   ├── geometry/         # pointcloud, alignment, mesh, distortion, intrinsics
│   ├── stereo/           # generic stereo depth (SGBM + MLX), any USB camera
│   ├── robotics/         # occupancy grids, obstacle detection
│   ├── capture/          # single/multi-camera, recorder, player
│   ├── transport/        # POSIX shared memory (seqlock double-buffer)
│   ├── bridges/          # ROS2 publisher (capability-gated)
│   ├── converters/       # 10 format converters (YUY2→RGB, stereo split, etc.)
│   ├── utils/            # benchmarks, depth statistics (RMSE, PSNR, SSIM)
│   ├── backends/         # MLX + CPU abstract backend
│   ├── display/          # OpenCV viewer
│   └── processor.py      # RealsenseProcessor (end-to-end)
├── tests/                # 1,048 tests across 26 test files
├── scripts/              # 7 scripts (demos, stress test, benchmarks)
├── benchmarks/           # component + full pipeline benchmarks
├── prds/                 # 10 PRDs documenting the port plan
└── references/           # progress log
```

### 3 Custom Metal GPU Kernels

| Kernel | File | Algorithm | Speedup vs CPU |
|--------|------|-----------|---------------|
| `spatial_horizontal` | filters/spatial.py | Bilateral recursive scan (1 thread/row) | 330x (99.7%) |
| `hole_fill_left` | filters/hole_filling.py | Prefix-fill propagation (1 thread/row) | 19x (95%) |
| `align_color_to_depth` | geometry/align.py | Fused deproject→transform→project→gather | 2.6x (62%) |

### Performance (480p, Apple M-series)

| Component | FPS | vs RS2 SDK |
|-----------|-----|-----------|
| Disparity transform | 2,414 | ~same |
| Align color-to-depth (Metal) | 2,340 | — |
| Point cloud generation | 2,092 | 3.2x faster |
| HoleFill LEFT (Metal) | 1,540 | 4.5x faster |
| HoleFill FARTHEST | 1,503 | 4.5x faster |
| Colorizer | 1,171 | — |
| Temporal filter | 767 | — |
| Spatial filter (Metal) | 656 | **330x faster** |
| Decimation 2x | 325 | numpy wins |
| Mesh generation | 251 | RS2 doesn't have this |
| **Standard pipeline** | **269** | **~54x faster** |
| **Full E2E processor** | **161** | **RS2 can't do this** |
| Bilateral filter | 85 | RS2 doesn't have this |

### Stress Test Results

893 tests across 4 resolutions, 0 failures:
- **480p:** Pipeline 271 FPS, Full E2E 136 FPS
- **720p:** Pipeline 131 FPS, Full E2E 62 FPS
- **1080p:** Pipeline 65 FPS, Full E2E 30 FPS (real-time)
- **4K:** Pipeline 17 FPS, Full E2E 6 FPS

Live camera validated: ZED 2i at 2560x720, 59 FPS capture, 222 FPS MLX colorize.

### Code Review

4 parallel code reviewers identified 32 HIGH/MEDIUM issues. All fixed:
- Decimation: +0.5 rounding bias on float32, median including zeros
- Temporal: persistence gating corrupting EMA state
- Colorizer: zero-depth rendering as LUT[0] instead of black
- Alignment: misleading MLX scatter-min removed
- Distortion: division-by-zero guard
- Transport: signed data_size, missing cleanup, tight spin loop
- All modules: `__all__` exports added

### Key Decisions Made

1. **Wrap, don't replace** — pyrealsense2 handles capture via libuvc on macOS. We only port the compute layer.
2. **Metal kernels for sequential scans** — Spatial and hole-fill have data dependencies that prevent vectorization. Metal kernels (1 thread/row) solve this.
3. **Honest benchmarks** — Decimation is slower on MLX (numpy's C median wins). Documented honestly in README.
4. **pyrealsense2 built from source** — No ARM64 PyPI wheels. Build workaround: symlink to /tmp due to `&` in path.
5. **uv for everything** — Modern, fast, no --break-system-packages issues.

### Dependencies

```
# Required
mlx>=0.31.0
numpy>=1.24.0

# Optional
pyrealsense2>=2.55.0  # built from source for macOS ARM64
opencv-python>=4.8.0  # for viewer and stereo camera
rclpy                 # for ROS2 bridge (capability-gated)
```

### Commands Reference

```bash
# Setup
uv venv .venv --python 3.12
uv pip install -e ".[dev,viewer]"

# Tests
.venv/bin/pytest tests/                          # 1,048 tests
.venv/bin/pytest tests/ -x -v                    # stop on first failure

# Stress test
.venv/bin/python scripts/stress_test.py          # 480p, no camera
.venv/bin/python scripts/stress_test.py --with-camera --duration 10
.venv/bin/python scripts/stress_test.py --all-resolutions

# Demos
.venv/bin/python scripts/offline_demo.py         # no camera needed
.venv/bin/python scripts/stereo_depth_demo.py    # ZED 2i / any stereo
.venv/bin/python scripts/live_depth_demo.py      # RealSense camera

# Benchmarks
.venv/bin/python benchmarks/bench_all.py
.venv/bin/python scripts/benchmark_vs_rs2.py
.venv/bin/python scripts/live_benchmark.py

# Quick pre-push check
.venv/bin/python scripts/quick_check.py
```

### Next Steps (for future sessions)

1. **Hardware validation** — Test with actual Intel RealSense D435/D415
2. **Metal kernel for bilateral** — MLX cumsum is faster currently, but a fused kernel could win
3. **Metal kernel for decimation** — numpy wins; a Metal median kernel could close the gap
4. **Depth learning integration** — Supervised depth estimation models on MLX
5. **Multi-camera synchronization** — Hardware-synced multi-camera capture
6. **SLAM integration** — Visual odometry using depth + color
7. **PyPI release** — Package for `pip install realsense-mlx`
8. **Documentation site** — Full API docs at robotflowlabs.com

### Files That Matter

| File | Why |
|------|-----|
| `src/realsense_mlx/processor.py` | The main entry point — RealsenseProcessor |
| `src/realsense_mlx/filters/spatial.py` | Metal kernel #1 (330x speedup) |
| `src/realsense_mlx/filters/hole_filling.py` | Metal kernel #2 (19x speedup) |
| `src/realsense_mlx/geometry/align.py` | Metal kernel #3 (2.6x speedup) |
| `src/realsense_mlx/stereo/depth.py` | Generic stereo depth (any camera) |
| `src/realsense_mlx/robotics/occupancy.py` | Occupancy grid for navigation |
| `scripts/stress_test.py` | 893-test stress test across 4K |
| `pyproject.toml` | Package config (uv, setuptools) |
| `README.md` | User-facing documentation |

### Cameras Validated

| Camera | Interface | Resolution | FPS | Status |
|--------|-----------|-----------|-----|--------|
| ZED 2i | USB 3.0 (side-by-side stereo) | 2560x720 | 59 | Capture + MLX processing OK |
| MacBook Pro Camera | Built-in | 1280x720 | 30 | Capture + MLX colorize OK |
| iPhone (Continuity) | Wireless | 1920x1080 | 30 | Available, not tested |
| Intel RealSense | USB 3.0 (pyrealsense2) | 640x480 | 30 | pyrealsense2 built, no hardware to test |
