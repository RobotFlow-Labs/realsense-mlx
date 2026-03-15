# PRD-02: MLX Point Cloud Generation

## Overview
Port the CUDA `kernel_deproject_depth_cuda` to MLX. This is the core depth→3D deprojection used for all point cloud workflows. Generates (H, W, 3) XYZ float32 coordinates from a depth frame + camera intrinsics.

## Blocked By
- PRD-00 (Project Setup)

## Source Reference
- `librealsense/src/cuda/cuda-pointcloud.cu` — kernel + device functions
- `librealsense/src/proc/pointcloud.cpp` — CPU reference
- `librealsense/src/proc/sse/sse-pointcloud.cpp` — SSE optimization (precomputed x/y maps)
- `librealsense/src/cuda/rscuda_utils.cuh` — distortion model implementations

## Algorithm

### Core Deprojection (per pixel)
```
Input: depth[y, x] (uint16), intrinsics (fx, fy, ppx, ppy, distortion_model, coeffs[5])
Output: point[y, x] = (X, Y, Z) float32

1. Normalize pixel coordinates:
   nx = (x - ppx) / fx
   ny = (y - ppy) / fy

2. Apply inverse distortion correction:
   if model == BROWN_CONRADY:
     # Iterative: 10 Newton-Raphson steps
     for i in range(10):
       r2 = nx*nx + ny*ny
       f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2
       ux = nx - 2*coeffs[2]*nx*ny - coeffs[3]*(r2 + 2*nx*nx)
       uy = ny - 2*coeffs[3]*nx*ny - coeffs[2]*(r2 + 2*ny*ny)
       nx = ux / f
       ny = uy / f
   elif model == INVERSE_BROWN_CONRADY:
     r2 = nx*nx + ny*ny
     f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2
     nx = nx*f + 2*coeffs[2]*nx*ny + coeffs[3]*(r2 + 2*nx*nx)
     ny = ny*f + 2*coeffs[3]*nx*ny + coeffs[2]*(r2 + 2*ny*ny)
   elif model == NONE:
     pass  # no correction

3. Scale by depth:
   Z = depth[y, x] * depth_scale
   X = nx * Z
   Y = ny * Z
```

### SSE Optimization Pattern (to replicate)
The SSE implementation precomputes x/y coordinate maps once per intrinsics change:
```
pre_x[x] = (x - ppx) / fx  for all x in [0, width)
pre_y[y] = (y - ppy) / fy  for all y in [0, height)
```
Then deprojection becomes a simple broadcast multiply:
```
X = pre_x * depth * depth_scale
Y = pre_y * depth * depth_scale
Z = depth * depth_scale
```
This is the ideal MLX pattern.

## MLX Implementation Strategy

### Phase 1: No-distortion fast path
```python
def deproject_depth_no_distortion(
    depth: mx.array,         # (H, W) uint16
    fx: float, fy: float,
    ppx: float, ppy: float,
    depth_scale: float
) -> mx.array:               # (H, W, 3) float32
    H, W = depth.shape

    # Precompute normalized coordinate grids (cached per intrinsics)
    x_coords = (mx.arange(W, dtype=mx.float32) - ppx) / fx  # (W,)
    y_coords = (mx.arange(H, dtype=mx.float32) - ppy) / fy  # (H,)

    # Convert depth to meters
    z = depth.astype(mx.float32) * depth_scale  # (H, W)

    # Broadcast multiply
    X = x_coords[None, :] * z  # (H, W)
    Y = y_coords[:, None] * z  # (H, W)

    return mx.stack([X, Y, z], axis=-1)  # (H, W, 3)
```

### Phase 2: Brown-Conrady distortion
```python
def deproject_depth_brown_conrady(
    depth: mx.array,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    iterations: int = 10
) -> mx.array:
    H, W = depth.shape

    # Normalized coords
    nx = (mx.arange(W) - intrinsics.ppx) / intrinsics.fx  # (W,)
    ny = (mx.arange(H) - intrinsics.ppy) / intrinsics.fy  # (H,)

    # Meshgrid
    nx_grid = mx.broadcast_to(nx[None, :], (H, W))
    ny_grid = mx.broadcast_to(ny[:, None], (H, W))

    # Iterative inverse distortion (vectorized over all pixels)
    c = intrinsics.coeffs
    for _ in range(iterations):
        r2 = nx_grid * nx_grid + ny_grid * ny_grid
        f = 1.0 + c[0]*r2 + c[1]*r2*r2 + c[4]*r2*r2*r2
        ux = nx_grid - 2*c[2]*nx_grid*ny_grid - c[3]*(r2 + 2*nx_grid*nx_grid)
        uy = ny_grid - 2*c[3]*nx_grid*ny_grid - c[2]*(r2 + 2*ny_grid*ny_grid)
        nx_grid = ux / f
        ny_grid = uy / f

    z = depth.astype(mx.float32) * depth_scale
    X = nx_grid * z
    Y = ny_grid * z

    return mx.stack([X, Y, z], axis=-1)
```

### Phase 3: Metal kernel (if needed for 60fps)
Only if Phase 2 doesn't meet performance targets.

## Data Structures

```python
@dataclass
class CameraIntrinsics:
    width: int
    height: int
    ppx: float      # principal point x
    ppy: float      # principal point y
    fx: float       # focal length x
    fy: float       # focal length y
    model: str      # "none", "brown_conrady", "inverse_brown_conrady"
    coeffs: list[float]  # distortion coefficients [k1, k2, p1, p2, k3]

    @classmethod
    def from_rs2(cls, rs_intrinsics) -> "CameraIntrinsics":
        """Convert pyrealsense2 intrinsics to our format."""
        ...
```

## Public API

```python
# realsense_mlx/geometry/pointcloud.py

class PointCloudGenerator:
    def __init__(self, intrinsics: CameraIntrinsics, depth_scale: float):
        self._intrinsics = intrinsics
        self._depth_scale = depth_scale
        self._precomputed = None  # Lazy cache for coord grids

    def generate(self, depth: mx.array) -> mx.array:
        """Depth frame → (H, W, 3) XYZ point cloud."""
        ...

    def generate_with_color(
        self, depth: mx.array, color: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Returns (points, colors) both (H, W, 3)."""
        ...

    def export_ply(self, points: mx.array, path: str, colors: mx.array = None):
        """Export to PLY format."""
        ...
```

## Test Plan

1. **No-distortion accuracy**: Generate synthetic depth ramp, verify XYZ against analytical formula
2. **Brown-Conrady accuracy**: Use real RealSense intrinsics, compare against `pyrealsense2.pointcloud().calculate()` — tolerance: atol=1e-4
3. **Inverse Brown-Conrady**: Same as above for cameras using this model
4. **Zero depth handling**: Pixels with depth=0 should produce (0,0,0)
5. **Performance**: Benchmark at 640×480, 1280×720 — target ≥2× over CPU numpy
6. **PLY export**: Round-trip: generate → export → load → compare

## Acceptance Criteria

- [ ] No-distortion path matches CPU reference exactly (atol=1e-6)
- [ ] Brown-Conrady matches CPU reference within atol=1e-4
- [ ] Zero-depth pixels produce (0,0,0) points
- [ ] Coordinate grids cached per intrinsics (not recomputed per frame)
- [ ] Benchmark: ≥2× speedup over numpy at 720p
- [ ] PLY export functional

## Estimated Effort
12-16 hours

## Port Difficulty
**MEDIUM** — Iterative distortion loop is the challenge; vectorized over all pixels compensates
