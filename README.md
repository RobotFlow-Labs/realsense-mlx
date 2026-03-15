# realsense-mlx

**Depth camera processing on Apple Silicon — faster than the original SDK.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.31%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/tests-938%20passing-brightgreen.svg)](#tests)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

---

## Why?

Intel's RealSense SDK uses **CUDA** for GPU acceleration — which doesn't exist on Mac. On Apple Silicon, their filters fall back to single-threaded CPU. The spatial filter runs at **0.3 FPS**.

We rewrote everything in Apple's [MLX](https://github.com/ml-explore/mlx) framework with **custom Metal GPU kernels**. Same algorithms, **1,990x faster**.

| Filter | RS2 SDK on Mac | realsense-mlx | Speedup |
|--------|---------------|---------------|---------|
| Spatial filter | 0.3 FPS | **644 FPS** | **1,990x** |
| Hole filling | 32 FPS | **731 FPS** | **23x** |
| Decimation | 81 FPS | **313 FPS** | **3.8x** |
| Point cloud | 787 FPS | **1,103 FPS** | **1.4x** |
| **Full pipeline** | **~2 FPS** | **200 FPS** | **~100x** |

Plus features the RS2 SDK doesn't have: bilateral filtering, mesh generation, depth statistics, shared memory transport, frame recording, ROS2 bridge, and a single-call end-to-end processor.

---

## Quick Start

```bash
# Install (30 seconds)
git clone https://github.com/RobotFlow-Labs/realsense-mlx.git
cd realsense-mlx
uv venv .venv --python 3.12
uv pip install -e ".[dev,viewer]"

# Run the demo (no camera needed)
.venv/bin/python scripts/offline_demo.py

# Run tests
.venv/bin/pytest tests/
```

---

## 3 Lines to Process Depth

```python
import realsense_mlx as rsmlx

proc = rsmlx.RealsenseProcessor(intrinsics, depth_scale=0.001)
result = proc.process(depth_frame)
# result.filtered_depth, result.points, result.colored_depth — all done
```

---

## Usage Examples

### Filter a depth frame

```python
from realsense_mlx.filters import DepthPipeline
import mlx.core as mx

pipeline = DepthPipeline()
filtered = pipeline.process(mx.array(depth_uint16))
```

### Generate a point cloud

```python
from realsense_mlx.geometry import PointCloudGenerator, CameraIntrinsics

intrinsics = CameraIntrinsics(640, 480, ppx=320, ppy=240, fx=600, fy=600)
gen = PointCloudGenerator(intrinsics, depth_scale=0.001)
points = gen.generate(depth)  # (H, W, 3) float32 XYZ
gen.export_ply(points, "cloud.ply", colors=color_frame)
```

### Generate a triangle mesh

```python
from realsense_mlx.geometry import DepthMeshGenerator

mesh = DepthMeshGenerator(max_edge_length=0.05)
vertices, faces = mesh.generate(points)
gen.export_ply_mesh(vertices, faces, "mesh.ply", colors=color_frame)
```

### Align color to depth

```python
from realsense_mlx.geometry import Aligner, CameraExtrinsics

aligner = Aligner(depth_intr, color_intr, CameraExtrinsics.identity(), 0.001)
aligned_color = aligner.align_color_to_depth(depth, color)  # Metal GPU kernel
```

### End-to-end with one call

```python
from realsense_mlx import RealsenseProcessor

proc = RealsenseProcessor(
    depth_intrinsics=intrinsics,
    depth_scale=0.001,
    enable_pointcloud=True,
    enable_mesh=True,
    enable_colorize=True,
    enable_stats=True,
    colormap="jet",
)

result = proc.process(depth_frame, color_frame)

# Everything you need:
result.filtered_depth     # (H', W') uint16
result.points             # (H', W', 3) float32
result.colored_depth      # (H', W', 3) uint8
result.aligned_color      # (H', W', 3) uint8  (if color provided)
result.vertices           # (N, 3) mesh vertices
result.faces              # (M, 3) mesh faces
result.stats              # {"valid_ratio": 0.95, "mean_m": 1.2, ...}
result.processing_time_ms # 5.0
```

### Record and replay

```python
from realsense_mlx.capture import FrameRecorder, FramePlayer

# Record
rec = FrameRecorder("my_recording")
rec.start(intrinsics)
rec.add_frame(depth, color, timestamp=0.0)
rec.stop()

# Replay
player = FramePlayer("my_recording")
player.open()
for depth, color, ts in player:
    result = proc.process(depth, color)
```

### Live camera (with RealSense connected)

```bash
.venv/bin/python scripts/live_depth_demo.py --colormap jet --side-by-side
```

---

## What's Inside

### 3 Custom Metal GPU Kernels

| Kernel | What it does | Speedup |
|--------|-------------|---------|
| `spatial_horizontal` | Edge-preserving bilateral scan (1 thread/row) | 1,990x |
| `hole_fill_left` | Prefix-fill scan propagation (1 thread/row) | 19x |
| `align_color_to_depth` | Fused deproject→transform→project→gather | 2.6x |

### 10 Depth Processing Filters

| Filter | Description | FPS (480p) |
|--------|-------------|-----------|
| `DecimationFilter` | Downsample by 2-8x with median/mean | 313 |
| `SpatialFilter` | Edge-preserving smoothing (Metal) | 644 |
| `TemporalFilter` | Time-domain EMA with persistence | 500+ |
| `HoleFillingFilter` | Fill invalid pixels (3 modes, Metal) | 731 |
| `DisparityTransform` | Depth ↔ disparity conversion | 1,700 |
| `DepthColorizer` | 10 colormaps, histogram equalization | 1,100 |
| `BilateralFilter` | O(1) guide-image edge preservation | 123 |
| `DepthEnhancer` | Quality pipeline: bilateral→temporal→hole-fill | 78 |
| `DepthPipeline` | Standard RS2 filter chain | 200 |
| `RealsenseProcessor` | E2E: filter→pointcloud→mesh→export | 145 |

### Geometry

| Module | Description | FPS (480p) |
|--------|-------------|-----------|
| `PointCloudGenerator` | Depth → XYZ with distortion correction | 1,103 |
| `Aligner` | Depth ↔ color registration (Metal) | 1,153 |
| `DepthMeshGenerator` | Organized point cloud → triangle mesh | 176 |
| Export | PLY (binary), OBJ, with normals + colors | instant |

### 10 Colormaps

`jet` `classic` `grayscale` `inv_grayscale` `warm` `cold` `biomes` `quantized` `pattern` `hue`

### Also Included

- **10 format converters** — YUY2→RGB/BGR/RGBA, stereo split, IR extraction
- **Shared memory transport** — POSIX shm with seqlock double-buffer
- **Frame recording/playback** — NPZ + metadata.json
- **Multi-camera capture** — discover + sync multiple cameras
- **ROS2 bridge** — publish depth/color/pointcloud/camera_info topics
- **Depth statistics** — RMSE, MAE, PSNR, SSIM, hole counting

---

## Building pyrealsense2 for macOS ARM64

The Intel SDK doesn't publish macOS ARM64 wheels. Build from source:

```bash
# Install deps
brew install cmake libusb

# Build (takes ~5 min)
cd /tmp
ln -sf /path/to/librealsense librealsense-src
mkdir librealsense-build && cd librealsense-build

cmake ../librealsense-src \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DFORCE_RSUSB_BACKEND=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(sysctl -n hw.ncpu)

# Install into your venv
cp Release/pyrealsense2*.so Release/pyrsutils*.so Release/librealsense2.dylib \
   /path/to/realsense-mlx/.venv/lib/python3.12/site-packages/
```

---

## Tests

```bash
.venv/bin/pytest tests/           # 938 tests, ~7 seconds
.venv/bin/pytest tests/ -x -v     # stop on first failure, verbose
.venv/bin/python benchmarks/bench_all.py  # full benchmark suite
```

---

## Project Stats

- **938 tests** passing
- **25,000 LOC** across 37 source files
- **3 Metal GPU kernels** (JIT-compiled, cached per process)
- **0 external deps** beyond MLX + numpy (pyrealsense2 + opencv are optional)
- **Apache 2.0** license

---

## API Reference

```python
import realsense_mlx as rsmlx

# Top-level
rsmlx.RealsenseProcessor    # End-to-end processor
rsmlx.ProcessingResult      # Result container
rsmlx.DepthPipeline         # Standard filter chain
rsmlx.DepthColorizer        # Depth visualization
rsmlx.PointCloudGenerator   # Depth → 3D points
rsmlx.Aligner               # Depth ↔ color alignment
rsmlx.FormatConverter       # YUY2/stereo format conversion
rsmlx.CameraIntrinsics      # Camera parameters

# Filters
from realsense_mlx.filters import (
    DecimationFilter, SpatialFilter, TemporalFilter,
    HoleFillingFilter, DisparityTransform, DepthColorizer,
    BilateralFilter, DepthEnhancer, DepthPipeline,
)

# Geometry
from realsense_mlx.geometry import (
    PointCloudGenerator, Aligner, DepthMeshGenerator,
    CameraIntrinsics, CameraExtrinsics,
)

# Capture
from realsense_mlx.capture import (
    RealsenseCapture, MultiCameraCapture,
    FrameRecorder, FramePlayer,
)

# Utils
from realsense_mlx.utils import Timer, benchmark_component
from realsense_mlx.utils.depth_stats import DepthStats
```

---

Built by [Robot Flow Labs](https://robotflowlabs.com) for robotics on Apple Silicon.
