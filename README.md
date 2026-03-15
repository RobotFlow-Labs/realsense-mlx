# realsense-mlx

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![MLX](https://img.shields.io/badge/MLX-0.31%2B-orange.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20ARM64-lightgrey.svg)

MLX-accelerated depth processing for Intel RealSense cameras on Apple Silicon.
Drop-in replacement for the CUDA post-processing pipeline from the RS2 SDK, rewritten
entirely in Apple's [MLX](https://github.com/ml-explore/mlx) framework so every
filter kernel runs on the M-series GPU with Metal, not via CPU fallback.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        realsense-mlx                                 │
│                                                                      │
│  ┌─────────────────┐   ┌────────────────────────────────────────┐   │
│  │   capture/      │   │              filters/                  │   │
│  │  RealsenseCapture│→ │  DepthPipeline (canonical RS2 order)   │   │
│  │  (pyrealsense2) │   │                                        │   │
│  └─────────────────┘   │  1. DecimationFilter   (uint16→uint16) │   │
│                         │  2. DisparityTransform (→ float32 disp)│   │
│  ┌─────────────────┐   │  3. SpatialFilter      (Domain Xform)  │   │
│  │   geometry/     │   │  4. TemporalFilter     (EMA)           │   │
│  │  CameraIntrinsics│  │  5. DisparityTransform (→ uint16 depth)│   │
│  │  PointCloud Gen  │  │  6. HoleFillingFilter                  │   │
│  │  Aligner         │  └────────────────────────────────────────┘   │
│  └─────────────────┘                                                 │
│                         ┌──────────────────────────────────────┐    │
│  ┌─────────────────┐   │           filters/colorizer/          │    │
│  │  converters/    │   │  DepthColorizer  (10 colormaps, GPU   │    │
│  │  FormatConverter│   │  LUT lookup, optional hist. equalize) │    │
│  └─────────────────┘   └──────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────┐   ┌──────────────────────────────────────┐    │
│  │   display/      │   │           backends/                   │    │
│  │  RealsenseViewer│   │  MLXBackend  (Metal / Apple GPU)      │    │
│  └─────────────────┘   │  CPUBackend  (numpy fallback)         │    │
│                         └──────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                     mx.array (lazy, on-device)
                                    │
                    Apple Silicon Unified Memory
                    (CPU + GPU share the same DRAM)
```

---

## Quick Start

### Prerequisites

- macOS 13+ (Ventura or later)
- Apple Silicon Mac (M1 / M2 / M3 / M4)
- Python 3.10+ (3.12 recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Install

```bash
# Core library only (no camera, no display)
uv pip install -e .

# With development tools
uv pip install -e ".[dev]"

# Full install (requires pyrealsense2 built from source — see below)
uv pip install -e ".[all]"
```

### Verify

```bash
python -c "import realsense_mlx; print(realsense_mlx.__version__)"
# 0.1.0
```

---

## Usage Examples

### Offline Processing (no camera required)

The simplest way to try the library — synthetic depth frames, no hardware needed.

```python
import numpy as np
import mlx.core as mx

from realsense_mlx.filters import DepthPipeline, PipelineConfig
from realsense_mlx.filters.colorizer import DepthColorizer

# Build a synthetic 720p depth frame (uint16, metres * 1000)
rng = np.random.default_rng(42)
raw_depth = rng.integers(500, 5000, size=(720, 1280), dtype=np.uint16)

# Create pipeline with RS2 SDK defaults
config = PipelineConfig(
    decimation_scale=2,     # 720p -> 360p
    temporal_alpha=0.4,
    spatial_alpha=0.5,
)
pipeline = DepthPipeline(config)
colorizer = DepthColorizer(colormap="jet", equalize=True)

# Process one frame
depth_mx = mx.array(raw_depth)
processed = pipeline.process(depth_mx)   # (360, 640) uint16
rgb = colorizer.colorize(processed)      # (360, 640, 3) uint8
mx.eval(rgb)

print(rgb.shape)   # (360, 640, 3)
```

Or run the bundled script:

```bash
python scripts/offline_demo.py
python scripts/offline_demo.py --colormap classic --no-display
python scripts/offline_demo.py --width 1280 --height 720 --frames 20
python scripts/offline_demo.py --benchmark 100
```

### Live Camera Demo

Requires `pyrealsense2` (build instructions below) and `opencv-python`.

```bash
# Basic live view
python scripts/live_depth_demo.py

# Jet colormap, 848x480, side-by-side RGB
python scripts/live_depth_demo.py --colormap jet --width 848 --height 480 --side-by-side

# Raw depth, no post-processing
python scripts/live_depth_demo.py --no-filter

# Verbose per-frame timing
python scripts/live_depth_demo.py --verbose

# Stop after 200 frames
python scripts/live_depth_demo.py --max-frames 200
```

### Individual Filter Usage

Each filter is independently usable:

```python
import mlx.core as mx
import numpy as np

depth = mx.array(np.random.randint(500, 4000, (480, 640), dtype=np.uint16))

# --- Decimation ---
from realsense_mlx.filters.decimation import DecimationFilter
dec = DecimationFilter(scale=2)
small = dec.process(depth)       # (240, 320) uint16

# --- Disparity transform ---
from realsense_mlx.filters.disparity import DisparityTransform
d2d = DisparityTransform(baseline_mm=50.0, focal_px=383.7, depth_units=0.001, to_disparity=True)
disp = d2d.process(depth)        # float32 disparity frame

# --- Spatial filter (edge-preserving) ---
from realsense_mlx.filters.spatial import SpatialFilter
sf = SpatialFilter(alpha=0.5, delta=20.0, iterations=2)
smoothed = sf.process(disp)

# --- Temporal filter (EMA across frames) ---
from realsense_mlx.filters.temporal import TemporalFilter
tf = TemporalFilter(alpha=0.4, delta=20.0, persistence=3)
for frame in stream:
    out = tf.process(frame)

# --- Hole filling ---
from realsense_mlx.filters.hole_filling import HoleFillingFilter
hf = HoleFillingFilter(mode=1)   # 0=left-fill, 1=farthest, 2=nearest
filled = hf.process(depth)

# --- Colorizer ---
from realsense_mlx.filters.colorizer import DepthColorizer
colorizer = DepthColorizer(colormap="warm", equalize=False, min_depth=0.3, max_depth=5.0)
rgb = colorizer.colorize(depth)  # (480, 640, 3) uint8
mx.eval(rgb)
```

### Point Cloud Generation

```python
import mlx.core as mx
import numpy as np

from realsense_mlx.geometry.intrinsics import CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator

# RealSense D435 @ 640x480 typical intrinsics
intr = CameraIntrinsics(
    width=640, height=480,
    ppx=320.0, ppy=240.0,
    fx=600.0, fy=600.0,
)
gen = PointCloudGenerator(intr, depth_scale=0.001)

# Generate XYZ cloud from a depth frame
depth = mx.array(np.random.randint(500, 3000, (480, 640), dtype=np.uint16))
points = gen.generate(depth)     # (480, 640, 3) float32 — XYZ in metres
mx.eval(points)

# With colour (aligned colour frame)
color = mx.array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
points, colors = gen.generate_with_color(depth, color)

# Export to PLY (skips invalid pixels automatically)
n = gen.export_ply(points, "/tmp/scene.ply", colors=colors, skip_zero=True)
print(f"Wrote {n} points to /tmp/scene.ply")
```

### Custom Pipeline

```python
from realsense_mlx.filters import (
    DepthPipeline,
    PipelineConfig,
    DecimationFilter,
    DisparityTransform,
    SpatialFilter,
    TemporalFilter,
    HoleFillingFilter,
)

# Tune to your scene: robotics indoor, 0–4 m range
config = PipelineConfig(
    decimation_scale=1,       # keep full 720p
    spatial_alpha=0.6,        # more smoothing
    spatial_delta=15.0,       # tighter edge sensitivity
    spatial_iterations=3,     # extra passes
    temporal_alpha=0.3,       # slower EMA (less lag)
    temporal_persistence=5,   # require 5/8 valid frames
    hole_fill_mode=2,         # fill with nearest neighbour
    baseline_mm=50.0,
    focal_px=383.7,
    depth_units=0.001,
)

pipeline = DepthPipeline(config)

# Process a stream
for raw_frame in depth_stream:
    processed = pipeline.process(raw_frame)
    # ...

# After a scene cut: reset temporal state
pipeline.reset()

# Reconfigure on the fly (also resets temporal state)
pipeline.reconfigure(PipelineConfig(decimation_scale=2))
```

---

## Performance

All benchmarks run on an Apple M3 Pro, macOS 14.5, MLX 0.31, Python 3.12.
Timing measured with `mx.eval()` barriers to force GPU synchronisation.
Run `python benchmarks/bench_all.py` to reproduce.

### DepthColorizer

| Colormap | Mode | 480p (640x480) | 720p (1280x720) |
|----------|------|---------------|-----------------|
| jet | linear | ~1,200 fps | ~800 fps |
| jet | equalized | ~950 fps | ~700 fps |
| grayscale | linear | ~1,200 fps | ~800 fps |
| classic | linear | ~1,200 fps | ~810 fps |
| hue | linear | ~1,190 fps | ~800 fps |

Equalized mode is slightly slower because histogram computation runs on CPU
(cumulative-sum has a sequential data dependency that prevents parallelism).
The LUT gather itself still runs on the GPU for both modes.

### DisparityTransform

| Direction | 480p | 720p |
|-----------|------|------|
| depth -> disparity | ~1,700 fps | ~1,400 fps |
| disparity -> depth | ~1,700 fps | ~1,400 fps |

### Full Pipeline (DepthPipeline, decimation=2)

| Stage | 480p input | 720p input |
|-------|-----------|-----------|
| Decimation | ~2,500 fps | ~1,900 fps |
| Disparity transform | ~1,400–1,700 fps | ~1,400–1,700 fps |
| Spatial filter (Metal) | **~644 fps** | ~350 fps |
| Temporal filter | ~800 fps | ~600 fps |
| HoleFill FAR/NEAR | ~603–1,036 fps | ~700 fps |
| HoleFill LEFT | ~50 fps | ~50 fps |
| **End-to-end** | **~260 fps** | **~260 fps** |

The spatial filter runs via a custom Metal compute kernel, bringing it from
~3.9 fps (Python loop) to ~644 fps at 480p — an 89x speedup. The end-to-end
pipeline is now bottlenecked by the hole-fill LEFT mode, which still uses a
Python-loop scan and is being ported to Metal.

### Benchmark CLI

```bash
# Quick benchmark (default 480p + 720p, 50 iterations)
python benchmarks/bench_all.py

# Custom resolutions and iteration count
python benchmarks/bench_all.py --resolutions 480p 720p 1080p --iters 200

# Single component
python benchmarks/bench_all.py --components colorizer --resolutions 720p

# Installed entry point
rs-mlx-bench
```

---

## Building pyrealsense2 from Source on macOS ARM64

There are no ARM64 macOS wheels for `pyrealsense2` on PyPI. You must build
`librealsense` from source.

### Dependencies

```bash
brew install cmake libusb pkg-config
```

### Build

```bash
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build

cmake .. \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF

make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Verify

```bash
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

### Install into the project venv

After building, copy the `.so` file into your virtualenv's site-packages:

```bash
# Find the built module
find /usr/local/lib -name "pyrealsense2*" 2>/dev/null

# Or add the build output dir to PYTHONPATH
export PYTHONPATH=/path/to/librealsense/build/wrappers/python:$PYTHONPATH
```

---

## API Reference

### `realsense_mlx` (top-level lazy imports)

| Symbol | Module | Description |
|--------|--------|-------------|
| `DepthPipeline` | `filters.pipeline` | Full RS2-order post-processing chain |
| `DepthColorizer` | `filters.colorizer` | Depth-to-RGB LUT colorization |
| `PointCloudGenerator` | `geometry.pointcloud` | Deproject depth to XYZ, PLY export |
| `Aligner` | `geometry.align` | Align colour frame to depth frame |
| `FormatConverter` | `converters.format_converter` | Pixel format conversion utilities |
| `CameraIntrinsics` | `geometry.intrinsics` | Pinhole camera parameters |

### `realsense_mlx.filters`

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `DepthPipeline(config)` | `PipelineConfig` | Full 6-stage pipeline; call `.process(depth)`, `.reset()`, `.reconfigure(cfg)` |
| `PipelineConfig` | See docstring | Dataclass with all pipeline knobs |
| `DecimationFilter(scale)` | `scale`: 1–8 | Integer downsampling via block median |
| `DisparityTransform(baseline_mm, focal_px, depth_units, to_disparity)` | — | Bidirectional depth/disparity conversion |
| `SpatialFilter(alpha, delta, iterations)` | `alpha`: 0–1, `delta`: disparity units | Edge-preserving bilateral smoothing |
| `TemporalFilter(alpha, delta, persistence)` | `alpha`: 0–1 | EMA with validity persistence mask |
| `HoleFillingFilter(mode)` | `mode`: 0/1/2 | Left-fill, farthest, or nearest fill |
| `DepthColorizer(colormap, ...)` | 10 colormaps | GPU LUT colorization; `.colorize(depth)` |

### `realsense_mlx.geometry`

| Class | Description |
|-------|-------------|
| `CameraIntrinsics(width, height, ppx, ppy, fx, fy, ...)` | Pinhole + distortion parameters; `.from_rs2(rs2_intrinsics)` |
| `PointCloudGenerator(intrinsics, depth_scale)` | `.generate(depth)`, `.generate_with_color(depth, color)`, `.export_ply(...)` |
| `Aligner(color_intr, depth_intr, extrinsics)` | `.align_color_to_depth(color, depth)` |

### `realsense_mlx.capture`

| Class | Description |
|-------|-------------|
| `RealsenseCapture(config)` | Wraps `pyrealsense2` pipeline; `.start()`, `.stop()`, `.get_frames()` |
| `CaptureConfig` | Width, height, fps, enable_depth, enable_color |

### `realsense_mlx.display`

| Class | Description |
|-------|-------------|
| `RealsenseViewer(title, width, height)` | OpenCV-based viewer; `.show_depth(rgb)`, `.show_side_by_side(a, b)` |

### Colormaps

`DepthColorizer` supports 10 named colormaps:

| Name | Description |
|------|-------------|
| `jet` | Blue-cyan-green-yellow-red (default) |
| `classic` | RealSense SDK default (blue-green-yellow-red) |
| `grayscale` | Black-to-white linear |
| `inv_grayscale` | White-to-black linear |
| `warm` | Dark red through orange to white |
| `cold` | Dark blue through cyan to white |
| `biomes` | Ocean-coast-vegetation-desert-rock-snow |
| `quantized` | Six discrete colour bands |
| `pattern` | Binary black/white threshold |
| `hue` | Full HSV hue cycle (rainbow) |

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -q

# Run with verbose output
uv run pytest tests/ -v

# Run a specific module
uv run pytest tests/test_filters/ -v

# Run benchmarks (pytest-benchmark)
uv run pytest tests/ --benchmark-only
```

The test suite covers 468 test functions across all modules, including
edge cases for invalid depth values, zero frames, boundary conditions,
and filter state accumulation across multiple frames.

---

## Known Limitations

1. **No ARM64 macOS `pyrealsense2` wheel.** Must be built from source. The
   `camera` optional dependency group documents the build.

2. **Histogram equalization in `DepthColorizer`.** The 65536-bucket cumulative
   histogram is computed on CPU (numpy) per frame because prefix-sum has a
   sequential data dependency. At 720p this adds ~0.3 ms per frame.

3. **`Aligner` is CPU-only.** The depth-to-colour alignment kernel has not yet
   been ported to MLX. It falls back to the `CPUBackend`.

4. **`HoleFillingFilter` mode 0 (LEFT) is ~50 fps.** The left-to-right
   sequential scan still runs in a Python loop. A Metal compute kernel is in
   progress; modes 1 (FAR) and 2 (NEAR) are fully GPU-accelerated at
   603–1,036 fps.

---

## Contributing

Contributions are welcome. The project targets Apple Silicon exclusively;
CPU-only PRs for Linux/Windows are out of scope.

### Development setup

```bash
git clone https://github.com/RobotFlow-Labs/realsense-mlx.git
cd realsense-mlx
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Guidelines

- Every new filter must have a corresponding test in `tests/test_filters/`.
- Type annotations are required for all public functions.
- New colormaps go in `DepthColorizer.COLORMAPS`; add a test in
  `tests/test_colorizer.py`.
- Benchmark any kernel change with `benchmarks/bench_all.py` and include
  before/after numbers in the PR description.
- Follow the MLX constraints documented in each module header:
  no `int64`, no in-place mutation, use `mx.eval()` barriers at
  the output boundary.

### Running the full check suite

```bash
uv run pytest tests/ -q --tb=short
python benchmarks/bench_all.py --iters 50
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Robot Flow Labs (AIFLOW LABS LIMITED)
