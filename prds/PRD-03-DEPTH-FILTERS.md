# PRD-03: MLX Depth Processing Filters

## Overview
Port the 5 core depth processing filters to MLX: spatial filter, temporal filter, decimation filter, hole-filling filter, and disparity transform. These form the standard RealSense post-processing pipeline.

## Blocked By
- PRD-00 (Project Setup)

## Source References
- `librealsense/src/proc/spatial-filter.cpp` — Edge-preserving spatial smoothing
- `librealsense/src/proc/temporal-filter.cpp` — Time-domain averaging
- `librealsense/src/proc/decimation-filter.cpp` — Downsampling with median/mean
- `librealsense/src/proc/hole-filling-filter.cpp` — Void pixel repair
- `librealsense/src/proc/disparity-transform.cpp` — Depth↔disparity conversion

---

## Filter 1: Disparity Transform (EASIEST — do first)

### Algorithm
```
Depth→Disparity: disparity = baseline_mm * focal_px * 32 / (depth * depth_units)
Disparity→Depth: depth = baseline_mm * focal_px * 32 / (disparity * depth_units)
```
Single elementwise division with zero/invalid handling.

### MLX Implementation
```python
def depth_to_disparity(depth: mx.array, d2d_factor: float) -> mx.array:
    """(H, W) uint16 → (H, W) float32"""
    depth_f = depth.astype(mx.float32)
    valid = depth_f > 0
    result = mx.where(valid, d2d_factor / depth_f, 0.0)
    return result

def disparity_to_depth(disparity: mx.array, d2d_factor: float) -> mx.array:
    """(H, W) float32 → (H, W) uint16"""
    valid = mx.isnormal(disparity) & (disparity > 0)
    depth_f = mx.where(valid, d2d_factor / disparity + 0.5, 0.0)
    return depth_f.astype(mx.uint16)
```

### Acceptance Criteria
- [ ] Matches pyrealsense2 output within atol=0 (exact for uint16)
- [ ] Zero-depth produces zero-disparity and vice versa
- [ ] ≥8× speedup over numpy at 720p

---

## Filter 2: Hole Filling (EASY)

### Algorithm
Three modes:
1. **Fill from left**: Propagate last valid value rightward per row
2. **Farthest from around**: max(up, down, left, right) among valid neighbors
3. **Nearest from around**: min(up, down, left, right) among valid neighbors

### MLX Implementation
```python
def hole_fill_farthest(depth: mx.array) -> mx.array:
    """Fill zeros with max of 4-connected valid neighbors."""
    invalid = (depth == 0)

    # Shifted views for 4 neighbors
    up    = mx.pad(depth[:-1, :], ((1,0),(0,0)))
    down  = mx.pad(depth[1:, :],  ((0,1),(0,0)))
    left  = mx.pad(depth[:, :-1], ((0,0),(1,0)))
    right = mx.pad(depth[:, 1:],  ((0,0),(0,1)))

    # Stack and take max (farthest) among non-zero
    neighbors = mx.stack([up, down, left, right], axis=-1)  # (H, W, 4)
    fill_val = mx.max(neighbors, axis=-1)

    return mx.where(invalid, fill_val, depth)
```

### Acceptance Criteria
- [ ] All 3 modes match pyrealsense2 reference output
- [ ] Only modifies zero-valued pixels
- [ ] ≥4× speedup at 720p

---

## Filter 3: Decimation (MEDIUM)

### Algorithm
- Scale 2-3: Median of kernel pixels (opt_med3 through opt_med9)
- Scale 4-8: Mean of non-zero kernel pixels
- Output dimensions: `(W/scale, H/scale)` padded to 4-byte alignment

### MLX Implementation
```python
def decimate_depth(depth: mx.array, scale: int = 2) -> mx.array:
    """Downsample depth by integer factor with median/mean."""
    H, W = depth.shape
    out_h, out_w = H // scale, W // scale

    # Reshape into tiles
    tiles = depth[:out_h*scale, :out_w*scale].reshape(out_h, scale, out_w, scale)
    flat_tiles = tiles.transpose(0, 2, 1, 3).reshape(out_h, out_w, scale*scale)

    if scale <= 3:
        # Median
        return mx.median(flat_tiles, axis=-1).astype(depth.dtype)
    else:
        # Mean of non-zero
        valid_mask = flat_tiles > 0
        sums = mx.sum(flat_tiles.astype(mx.float32), axis=-1)
        counts = mx.sum(valid_mask.astype(mx.float32), axis=-1)
        counts = mx.maximum(counts, 1.0)  # avoid div by zero
        return (sums / counts + 0.5).astype(depth.dtype)
```

### Acceptance Criteria
- [ ] Scale 2 median matches pyrealsense2 (exact)
- [ ] Scale 4+ mean matches within ±1 depth unit
- [ ] Output dimensions correct with padding
- [ ] Supports Z16 and disparity formats
- [ ] ≥6× speedup at 720p

---

## Filter 4: Spatial Filter (MEDIUM-HARD)

### Algorithm
Domain Transform Filter (Gastal & Oliveira, SIGGRAPH 2011):
- Recursive bidirectional (left→right, right→left) edge-preserving filter
- Applied `iterations` times, alternating horizontal and vertical passes
- Weight: `w = exp(-sqrt(2) / (sigma_s * sigma_r * depth_delta))`
- Output: `out[i] = in[i] + w * (out[i-1] - in[i])`

### MLX Implementation Strategy
The recursive nature (each output depends on previous output) prevents full vectorization.
**Approach: Row-parallel processing**
- Process all rows in parallel for horizontal pass
- Process all columns in parallel for vertical pass
- Within each row/column: sequential scan (unavoidable for recursive filter)

```python
def spatial_filter(depth: mx.array, alpha: float = 0.5, delta: float = 20.0,
                   iterations: int = 2) -> mx.array:
    """Edge-preserving spatial smoothing."""
    result = depth.astype(mx.float32)

    for _ in range(iterations):
        # Horizontal pass (left→right then right→left)
        result = _recursive_filter_horizontal(result, alpha, delta)
        # Vertical pass (top→bottom then bottom→top)
        result = _recursive_filter_vertical(result, alpha, delta)

    return result.astype(depth.dtype)
```

**Note**: May need Metal kernel for competitive performance due to sequential dependency.

### Acceptance Criteria
- [ ] Matches pyrealsense2 within atol=1 depth unit (rounding differences expected)
- [ ] Configurable alpha, delta, iterations
- [ ] Handles zero/invalid pixels (skip in filter chain)
- [ ] ≥2× speedup at 720p (limited by sequential dependency)

---

## Filter 5: Temporal Filter (MEDIUM)

### Algorithm
Exponential moving average with persistence tracking:
```
filtered[y,x] = alpha * current[y,x] + (1-alpha) * previous[y,x]
```
Plus 8-frame history bitmask for persistence-based hole filling.

### MLX Implementation
```python
class TemporalFilter:
    def __init__(self, alpha: float = 0.4, delta: float = 20.0, persistence: int = 3):
        self.alpha = alpha
        self.delta = delta
        self.persistence = persistence
        self._prev_frame: mx.array | None = None
        self._history: mx.array | None = None  # (H, W) uint8 bitmask

    def process(self, depth: mx.array) -> mx.array:
        if self._prev_frame is None:
            self._prev_frame = depth.astype(mx.float32)
            self._history = mx.ones(depth.shape, dtype=mx.uint8)
            return depth

        current = depth.astype(mx.float32)
        prev = self._prev_frame

        # Compute difference
        diff = mx.abs(current - prev)
        valid = (depth > 0) & (diff < self.delta)

        # Alpha blend where valid
        blended = mx.where(valid,
            self.alpha * current + (1 - self.alpha) * prev,
            current)

        # Update history bitmask (shift left, set LSB for valid)
        self._history = (self._history << 1) | valid.astype(mx.uint8)

        # Persistence: use previous value if history says "was valid recently"
        persistent = self._check_persistence(self._history)
        result = mx.where((depth == 0) & persistent, prev, blended)

        self._prev_frame = result
        return result.astype(depth.dtype)
```

### Acceptance Criteria
- [ ] Converges to pyrealsense2 output after 10+ frames (steady-state match)
- [ ] History tracking correct for all 9 persistence modes
- [ ] State properly reset on `reset()` call
- [ ] Frame history stays on MLX device (no CPU round-trip)
- [ ] ≥3× speedup at 720p

---

## Combined Processing Pipeline

```python
class DepthPipeline:
    """Standard RealSense post-processing pipeline on MLX."""

    def __init__(self, config: PipelineConfig = None):
        config = config or PipelineConfig()
        self.decimation = DecimationFilter(scale=config.decimation_scale)
        self.depth_to_disp = DisparityTransform(to_disparity=True)
        self.spatial = SpatialFilter(alpha=config.spatial_alpha, ...)
        self.temporal = TemporalFilter(alpha=config.temporal_alpha, ...)
        self.disp_to_depth = DisparityTransform(to_disparity=False)
        self.hole_fill = HoleFillingFilter(mode=config.hole_fill_mode)

    def process(self, depth: mx.array) -> mx.array:
        """Run full post-processing pipeline."""
        result = self.decimation.process(depth)
        result = self.depth_to_disp.process(result)
        result = self.spatial.process(result)
        result = self.temporal.process(result)
        result = self.disp_to_depth.process(result)
        result = self.hole_fill.process(result)
        return result
```

## Test Plan

For each filter:
1. Synthetic depth frame with known patterns (ramps, edges, holes)
2. Compare against pyrealsense2 CPU filter output (frame-by-frame)
3. Edge cases: all-zero frame, max-value frame, single-pixel frame
4. Pipeline test: chain all filters, compare end-to-end output

## Overall Acceptance Criteria
- [ ] All 5 filters individually validated against pyrealsense2
- [ ] Combined pipeline produces equivalent output
- [ ] Temporal filter state management works across frames
- [ ] Total pipeline at 720p: ≥3× speedup over CPU

## Estimated Effort
20-30 hours total (disparity: 2h, hole-fill: 4h, decimation: 6h, spatial: 8h, temporal: 8h)

## Port Difficulty
**MEDIUM** overall — disparity/hole-fill easy, spatial has sequential dependency challenge
