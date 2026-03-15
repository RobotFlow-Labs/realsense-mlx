# PRD-00: Project Setup & Build Infrastructure

## Overview
Bootstrap the `realsense-mlx` Python package — an MLX-accelerated processing backend for Intel RealSense cameras on Apple Silicon. This package wraps `pyrealsense2` capture with MLX compute for depth processing, point cloud generation, and filter pipelines.

## Architecture

```
realsense-mlx/
├── src/realsense_mlx/
│   ├── __init__.py              # Public API
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract ProcessingBackend
│   │   ├── mlx_backend.py       # MLX implementation
│   │   └── cpu_backend.py       # CPU/numpy fallback
│   ├── capture/
│   │   ├── __init__.py
│   │   └── pipeline.py          # Wrapper around pyrealsense2.pipeline
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── decimation.py        # MLX decimation
│   │   ├── spatial.py           # MLX spatial filter
│   │   ├── temporal.py          # MLX temporal filter
│   │   ├── hole_filling.py      # MLX hole filling
│   │   ├── disparity.py         # MLX disparity transform
│   │   └── colorizer.py         # MLX colorizer
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── pointcloud.py        # MLX point cloud generation
│   │   ├── align.py             # MLX depth-color alignment
│   │   ├── intrinsics.py        # Camera intrinsics handling
│   │   └── distortion.py        # Distortion model implementations
│   ├── converters/
│   │   ├── __init__.py
│   │   └── format_converter.py  # MLX format conversions (YUY2→RGB, etc.)
│   ├── display/
│   │   ├── __init__.py
│   │   └── viewer.py            # SDL2/Metal viewer
│   └── utils/
│       ├── __init__.py
│       └── benchmark.py         # Timing utilities
├── tests/
├── benchmarks/
├── scripts/
├── prds/
├── references/
├── pyproject.toml
└── README.md
```

## Key Design Decisions

1. **Wrap, don't replace**: `pyrealsense2` handles all camera capture via libuvc on macOS. We only replace the compute/processing layer with MLX.
2. **Backend abstraction**: `ProcessingBackend` interface allows switching between MLX and CPU/numpy.
3. **Zero-copy bridge**: numpy arrays from `pyrealsense2` → `mx.array` with minimal copying.
4. **Stateful filters**: Temporal filter maintains frame history as MLX arrays on device.
5. **Pipeline composability**: Filters are composable via a `ProcessingPipeline` that chains operations.

## Dependencies

### Base (required)
```
pyrealsense2>=2.55.0
mlx>=0.31.0
numpy>=1.24.0
```

### Optional
```
sdl2>=0.9.16      # For Metal/SDL2 viewer
open3d>=0.18.0    # For point cloud visualization
```

### Dev
```
pytest>=7.0
pytest-benchmark>=4.0
```

## pyproject.toml Spec

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "realsense-mlx"
version = "0.1.0"
description = "MLX-accelerated processing backend for Intel RealSense on Apple Silicon"
requires-python = ">=3.10"
dependencies = [
    "pyrealsense2>=2.55.0",
    "mlx>=0.31.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
viewer = ["pysdl2>=0.9.16"]
viz = ["open3d>=0.18.0"]
dev = ["pytest>=7.0", "pytest-benchmark>=4.0"]
all = ["realsense-mlx[viewer,viz,dev]"]

[project.scripts]
rs-mlx-bench = "realsense_mlx.utils.benchmark:main"
rs-mlx-viewer = "realsense_mlx.display.viewer:main"
```

## Acceptance Criteria

- [ ] `uv pip install -e ".[dev]"` succeeds on macOS ARM64
- [ ] `import realsense_mlx` succeeds without a connected camera
- [ ] `import pyrealsense2` works (libuvc backend on macOS)
- [ ] Backend selection: `ProcessingBackend.create("mlx")` and `ProcessingBackend.create("cpu")`
- [ ] All imports are lazy — no MLX import until backend is actually used
- [ ] pytest discovers and runs test suite

## Blocking

This PRD blocks ALL subsequent PRDs. Must be completed first.

## Estimated Effort
4-6 hours
