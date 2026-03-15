# PRD-07: Integration Testing & End-to-End Validation

## Overview
Comprehensive test suite validating the entire `realsense-mlx` pipeline against `pyrealsense2` CPU reference implementations. Tests must run WITHOUT a connected camera (using recorded BAG files or synthetic data).

## Blocked By
- PRD-00 through PRD-05 (all processing components)

## Test Architecture

```
tests/
├── conftest.py              # Shared fixtures (synthetic data, recorded bags)
├── test_format_converters.py   # PRD-01
├── test_pointcloud.py          # PRD-02
├── test_filters/
│   ├── test_disparity.py       # PRD-03
│   ├── test_hole_filling.py    # PRD-03
│   ├── test_decimation.py      # PRD-03
│   ├── test_spatial.py         # PRD-03
│   └── test_temporal.py        # PRD-03
├── test_align.py               # PRD-04
├── test_colorizer.py           # PRD-05
├── test_pipeline.py            # Full pipeline integration
├── test_import_safety.py       # Import without camera/MLX
└── test_backend_selection.py   # MLX vs CPU backend switching
```

## Test Data Strategy

### Synthetic Data (for unit tests)
```python
# conftest.py

@pytest.fixture
def synthetic_depth_ramp():
    """Linear depth ramp 0→10000 across width."""
    return np.tile(np.arange(640, dtype=np.uint16) * 15, (480, 1))

@pytest.fixture
def synthetic_depth_with_holes():
    """Depth frame with known hole pattern."""
    depth = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
    depth[100:200, 200:400] = 0  # rectangular hole
    return depth

@pytest.fixture
def synthetic_color_bgr():
    """BGR color frame with gradient."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def d415_intrinsics():
    """Real D415 depth camera intrinsics."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=318.8, ppy=239.5,
        fx=383.7, fy=383.7,
        model="brown_conrady",
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
    )
```

### Recorded BAG Files (for integration tests)
Record a short sequence with `rs-record`:
```bash
rs-record -f test_recording.bag -s 30 -d 640x480 -c 640x480
```
Store in `tests/data/` (gitignored, downloaded by CI).

## Validation Framework

### Cross-Framework Comparison
```python
def assert_frames_match(mlx_result: mx.array, rs2_result, atol: float = 0):
    """Compare MLX output against pyrealsense2 reference."""
    mlx_np = np.array(mlx_result)
    rs2_np = np.asanyarray(rs2_result.get_data()) if hasattr(rs2_result, 'get_data') else rs2_result

    if atol == 0:
        assert np.array_equal(mlx_np, rs2_np), f"Exact match failed. Max diff: {np.max(np.abs(mlx_np.astype(int) - rs2_np.astype(int)))}"
    else:
        assert np.allclose(mlx_np.astype(float), rs2_np.astype(float), atol=atol), \
            f"Match failed with atol={atol}. Max diff: {np.max(np.abs(mlx_np.astype(float) - rs2_np.astype(float)))}"
```

### Tolerance Spec by Component

| Component | Tolerance | Reason |
|-----------|-----------|--------|
| Format converters | Exact (atol=0) | Pure integer arithmetic |
| Disparity transform | atol=0 for uint16 | Integer rounding |
| Hole filling | Exact | Discrete comparison |
| Decimation (median) | Exact | Deterministic sort |
| Decimation (mean) | atol=1 | Rounding differences |
| Spatial filter | atol=1 depth unit | Float accumulation order |
| Temporal filter | atol=2 depth units | Float accumulation + state |
| Point cloud (no distortion) | atol=1e-6 | Float precision |
| Point cloud (with distortion) | atol=1e-4 | Iterative convergence |
| Alignment | atol=1 pixel / 1 depth unit | Rounding in reprojection |
| Colorizer | atol=1 per channel | LUT interpolation |

## Key Test Scenarios

### 1. Import Safety
```python
def test_import_without_camera():
    """realsense_mlx imports cleanly even without connected camera."""
    import realsense_mlx
    assert hasattr(realsense_mlx, 'DepthPipeline')

def test_import_without_mlx(monkeypatch):
    """Falls back to CPU backend if MLX unavailable."""
    monkeypatch.setitem(sys.modules, 'mlx', None)
    from realsense_mlx.backends import cpu_backend
    assert cpu_backend is not None
```

### 2. Full Pipeline Equivalence
```python
def test_full_pipeline_matches_rs2(synthetic_depth_ramp, d415_intrinsics):
    """Complete filter chain matches pyrealsense2 output."""
    # MLX pipeline
    mlx_pipeline = DepthPipeline(config=PipelineConfig(
        decimation_scale=2,
        spatial_alpha=0.5, spatial_delta=20, spatial_iterations=2,
        temporal_alpha=0.4, temporal_delta=20, temporal_persistence=3,
        hole_fill_mode="farthest"
    ))

    # Run 20 frames for temporal convergence
    for i in range(20):
        mlx_result = mlx_pipeline.process(mx.array(synthetic_depth_ramp))

    # Compare against rs2
    rs2_pipeline = build_rs2_pipeline(same_config)
    for i in range(20):
        rs2_result = rs2_pipeline_process(synthetic_depth_ramp)

    assert_frames_match(mlx_result, rs2_result, atol=2)
```

### 3. Point Cloud Accuracy
```python
def test_pointcloud_known_geometry(d415_intrinsics):
    """Verify deprojection produces correct 3D coordinates for known depth."""
    # Flat wall at 1 meter
    depth = np.full((480, 640), 1000, dtype=np.uint16)  # 1000mm

    gen = PointCloudGenerator(d415_intrinsics, depth_scale=0.001)
    points = gen.generate(mx.array(depth))

    # Center pixel should be at (0, 0, 1.0)
    center = np.array(points[240, 320])
    assert abs(center[2] - 1.0) < 1e-4
    assert abs(center[0]) < 0.01
    assert abs(center[1]) < 0.01
```

## CI Integration

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: macos-14  # M1/M2 runner
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv venv .venv --python 3.12
      - run: uv pip install -e ".[dev]"
      - run: .venv/bin/pytest tests/ -q --tb=short
```

## Acceptance Criteria

- [ ] ≥90% code coverage on processing modules
- [ ] All tolerance specs met
- [ ] Tests run without connected camera
- [ ] CI passes on macOS ARM64 runner
- [ ] Import safety verified (no crash on missing deps)
- [ ] Performance regression tests (baseline recorded)

## Estimated Effort
12-16 hours

## Priority
Execute alongside PRD development — each PRD should include its tests.
