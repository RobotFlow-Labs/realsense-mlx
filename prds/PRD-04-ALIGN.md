# PRD-04: MLX Depth-Color Alignment

## Overview
Port the CUDA alignment kernels (`cuda-align.cu`) to MLX. Depth-color alignment is critical for any RGB-D application — it registers the depth map to the color camera's coordinate frame (or vice versa).

## Blocked By
- PRD-00 (Project Setup)
- PRD-02 (Point Cloud — shares intrinsics/distortion infrastructure)

## Source Reference
- `librealsense/src/cuda/cuda-align.cu` — 5 kernels
- `librealsense/src/proc/align.cpp` — CPU reference
- `librealsense/src/proc/sse/sse-align.cpp` — SSE optimization
- `librealsense/src/cuda/rscuda_utils.cuh` — 3D geometry device functions

## Algorithm

### Align Color to Depth (most common)
For each depth pixel (x, y):
1. Deproject to 3D: `P_depth = K_depth⁻¹ × [x, y, 1] × depth[y,x]`
2. Transform to color frame: `P_color = R × P_depth + T`
3. Project to color image: `[u, v] = K_color × P_color / P_color.z`
4. Sample color at (u, v) → write to aligned_color[y, x]

### Align Depth to Color (reverse)
For each depth pixel (x, y):
1. Deproject to 3D: `P_depth = K_depth⁻¹ × [x, y, 1] × depth[y,x]`
2. Transform to color frame: `P_color = R × P_depth + T`
3. Project to color pixel: `[u, v] = K_color × P_color / P_color.z`
4. Write depth[y,x] to aligned_depth[v, u] (using min for overlaps)

### Key Geometry Functions
```python
def deproject(pixel, intrinsics, depth):
    """2D pixel + depth → 3D point."""
    x = (pixel[0] - intrinsics.ppx) / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) / intrinsics.fy
    # Apply distortion correction...
    return [x * depth, y * depth, depth]

def transform(point, extrinsics):
    """3D point in frame A → 3D point in frame B."""
    R = extrinsics.rotation  # 3×3
    T = extrinsics.translation  # 3
    return R @ point + T

def project(point, intrinsics):
    """3D point → 2D pixel."""
    x = point[0] / point[2]
    y = point[1] / point[2]
    # Apply forward distortion...
    return [x * intrinsics.fx + intrinsics.ppx,
            y * intrinsics.fy + intrinsics.ppy]
```

## MLX Implementation Strategy

### Data Structures
```python
@dataclass
class CameraExtrinsics:
    rotation: mx.array      # (3, 3) float32
    translation: mx.array   # (3,) float32

    @classmethod
    def from_rs2(cls, rs_extr) -> "CameraExtrinsics":
        return cls(
            rotation=mx.array(rs_extr.rotation).reshape(3, 3),
            translation=mx.array(rs_extr.translation)
        )
```

### Align Color to Depth (vectorized)
```python
def align_color_to_depth(
    depth: mx.array,          # (H_d, W_d) uint16
    color: mx.array,          # (H_c, W_c, C) uint8
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    depth_to_color: CameraExtrinsics,
    depth_scale: float
) -> mx.array:                # (H_d, W_d, C) uint8
    H_d, W_d = depth.shape

    # Step 1: Deproject all depth pixels to 3D (reuse pointcloud)
    points_3d = deproject_depth(depth, depth_intrinsics, depth_scale)  # (H_d, W_d, 3)

    # Step 2: Transform to color frame (vectorized matmul)
    R = depth_to_color.rotation  # (3, 3)
    T = depth_to_color.translation  # (3,)
    points_color = mx.matmul(points_3d.reshape(-1, 3), R.T) + T  # (N, 3)
    points_color = points_color.reshape(H_d, W_d, 3)

    # Step 3: Project to color image coords
    z = points_color[..., 2]
    valid = z > 0
    u = mx.where(valid,
        points_color[..., 0] / z * color_intrinsics.fx + color_intrinsics.ppx, 0)
    v = mx.where(valid,
        points_color[..., 1] / z * color_intrinsics.fy + color_intrinsics.ppy, 0)

    # Step 4: Sample color (nearest neighbor)
    u_int = mx.clip(mx.round(u).astype(mx.int32), 0, color_intrinsics.width - 1)
    v_int = mx.clip(mx.round(v).astype(mx.int32), 0, color_intrinsics.height - 1)

    # Gather from color frame
    flat_idx = v_int * color_intrinsics.width + u_int
    flat_color = color.reshape(-1, color.shape[-1])
    aligned = flat_color[flat_idx.reshape(-1)].reshape(H_d, W_d, -1)

    # Mask invalid pixels
    aligned = mx.where(valid[..., None], aligned, 0)

    return aligned
```

### Align Depth to Color (scatter with min)
```python
def align_depth_to_color(
    depth: mx.array,
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    depth_to_color: CameraExtrinsics,
    depth_scale: float
) -> mx.array:                # (H_c, W_c) uint16
    # Same steps 1-3 as above...

    # Step 4: Scatter depth to color grid (min for overlaps)
    # MLX limitation: no atomic min scatter
    # Workaround: sort by target index, take min per group
    aligned_depth = mx.zeros((color_intrinsics.height, color_intrinsics.width),
                             dtype=mx.uint16)

    # Use blocked scatter pattern from port-to-mlx skill
    flat_idx = v_int * color_intrinsics.width + u_int
    valid_flat = flat_idx[valid].reshape(-1)
    depth_flat = depth[valid].reshape(-1)

    # Sort by target index for min-scatter
    sort_order = mx.argsort(valid_flat)
    sorted_idx = valid_flat[sort_order]
    sorted_depth = depth_flat[sort_order]

    # Segment boundaries → take min per segment
    # Use the repeat_interleave pattern from pointelligence
    aligned_depth = _scatter_min(sorted_idx, sorted_depth,
                                 color_intrinsics.height * color_intrinsics.width)

    return aligned_depth.reshape(color_intrinsics.height, color_intrinsics.width)
```

## Scatter-Min Implementation

The CUDA kernel uses `atomicMin`. MLX has no atomic scatter. Solution:

```python
def _scatter_min(indices: mx.array, values: mx.array, size: int) -> mx.array:
    """Scatter values to output, taking min for duplicate indices."""
    # Sort by index
    order = mx.argsort(indices)
    sorted_idx = indices[order]
    sorted_val = values[order]

    # Find segment boundaries (where index changes)
    boundaries = mx.concatenate([
        mx.array([True]),
        sorted_idx[1:] != sorted_idx[:-1]
    ])

    # Cumulative min within each segment (scan)
    # For each segment, the first value after sort is already a candidate
    # We need running min — can use segment_reduce pattern

    # Simple approach: scatter the sorted values, natural min due to order
    output = mx.full((size,), 65535, dtype=mx.uint16)

    # Walk segments and take min
    # This is the hard part on MLX — may need numpy fallback for now
    output_np = np.full(size, 65535, dtype=np.uint16)
    np.minimum.at(output_np, np.array(sorted_idx), np.array(sorted_val))

    return mx.array(output_np)
```

**Phase 2**: Replace numpy fallback with pure MLX using blocked scatter pattern.

## Public API

```python
# realsense_mlx/geometry/align.py

class Aligner:
    def __init__(self,
                 depth_intrinsics: CameraIntrinsics,
                 color_intrinsics: CameraIntrinsics,
                 depth_to_color: CameraExtrinsics,
                 depth_scale: float):
        ...

    def align_color_to_depth(self, depth: mx.array, color: mx.array) -> mx.array:
        """Align color frame to depth frame coords. Returns (H_d, W_d, C)."""
        ...

    def align_depth_to_color(self, depth: mx.array) -> mx.array:
        """Align depth frame to color frame coords. Returns (H_c, W_c)."""
        ...
```

## Test Plan

1. **Synthetic calibration**: Known intrinsics/extrinsics → verify reprojected pixels match analytically
2. **Real calibration**: Load RealSense calibration, compare against `rs.align()` output
3. **Edge pixels**: Pixels that project outside target frame should be zeroed
4. **Occlusion**: Multiple depth pixels mapping to same color pixel → verify min selection
5. **Performance**: Benchmark at 640×480 depth → 1920×1080 color alignment

## Acceptance Criteria

- [ ] Align-color-to-depth matches pyrealsense2 within ±1 pixel
- [ ] Align-depth-to-color matches pyrealsense2 within ±1 depth unit
- [ ] Out-of-bounds pixels produce zero output
- [ ] Handles cameras with different resolutions
- [ ] ≥2× speedup for color-to-depth, comparable for depth-to-color (numpy fallback)

## Estimated Effort
16-24 hours

## Port Difficulty
**HARD** — Scatter-min for depth-to-color alignment has no native MLX equivalent
