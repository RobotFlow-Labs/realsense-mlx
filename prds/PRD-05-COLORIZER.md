# PRD-05: MLX Depth Colorizer

## Overview
Port the depth colorizer to MLX. Converts depth/disparity frames to RGB color visualizations using histogram equalization and configurable color maps.

## Blocked By
- PRD-00 (Project Setup)

## Source Reference
- `librealsense/src/proc/colorizer.cpp` — CPU implementation
- `librealsense/src/proc/colorizer.h` — Class definition, color map data

## Algorithm

### Histogram Equalization Path
1. Build cumulative histogram of depth values (sequential)
2. Normalize each pixel: `norm[y,x] = histogram[depth[y,x]] / max_histogram`
3. Map normalized value to color: `rgb = colormap(norm[y,x])`

### Direct Range Path
1. Normalize: `norm = (depth - min_depth) / (max_depth - min_depth)`
2. Map to color: `rgb = colormap(norm)`

### Color Maps (10 built-in)
Each map is a list of (value, R, G, B) keypoints. Interpolate between them.

## MLX Implementation

```python
class DepthColorizer:
    COLORMAPS = {
        "jet": [...],      # Blue→Cyan→Green→Yellow→Red
        "classic": [...],  # RealSense default
        "grayscale": [...],
        "warm": [...],
        "cold": [...],
        # ... 10 total
    }

    def __init__(self, colormap: str = "jet", min_depth: float = 0.1,
                 max_depth: float = 10.0, equalize: bool = True):
        self.colormap = colormap
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.equalize = equalize
        self._lut = self._build_lut()  # (256, 3) uint8 precomputed

    def _build_lut(self) -> mx.array:
        """Precompute 256-entry color LUT from colormap keypoints."""
        # Interpolate keypoints to 256 entries
        ...
        return mx.array(lut, dtype=mx.uint8)  # (256, 3)

    def colorize(self, depth: mx.array) -> mx.array:
        """(H, W) uint16 → (H, W, 3) uint8 RGB."""
        if self.equalize:
            return self._colorize_equalized(depth)
        else:
            return self._colorize_direct(depth)

    def _colorize_direct(self, depth: mx.array) -> mx.array:
        depth_f = depth.astype(mx.float32)
        # Normalize to [0, 255]
        norm = (depth_f - self.min_depth) / (self.max_depth - self.min_depth)
        norm = mx.clip(norm * 255, 0, 255).astype(mx.int32)
        # LUT lookup
        return self._lut[norm.reshape(-1)].reshape(*depth.shape, 3)

    def _colorize_equalized(self, depth: mx.array) -> mx.array:
        # Histogram (CPU — sequential dependency)
        depth_np = np.array(depth)
        hist, _ = np.histogram(depth_np[depth_np > 0], bins=65536, range=(0, 65535))
        cumhist = np.cumsum(hist)
        max_val = cumhist[-1] if cumhist[-1] > 0 else 1

        # Build mapping table: depth_value → [0, 255]
        mapping = (cumhist * 255 / max_val).astype(np.uint8)
        mapping_mx = mx.array(mapping)

        # Apply mapping (MLX gather)
        indices = depth.reshape(-1).astype(mx.int32)
        norm = mapping_mx[indices].reshape(depth.shape)

        # LUT lookup for color
        return self._lut[norm.reshape(-1).astype(mx.int32)].reshape(*depth.shape, 3)
```

## Public API

```python
# realsense_mlx/filters/colorizer.py

class DepthColorizer:
    def __init__(self, colormap: str = "jet", min_depth: float = 0.1,
                 max_depth: float = 10.0, equalize: bool = True): ...

    def colorize(self, depth: mx.array) -> mx.array:
        """Depth frame → RGB visualization. Returns (H, W, 3) uint8."""
        ...

    def set_range(self, min_depth: float, max_depth: float): ...
    def set_colormap(self, name: str): ...

    @staticmethod
    def available_colormaps() -> list[str]: ...
```

## Test Plan

1. **Direct mode**: Ramp input → verify smooth color gradient
2. **Equalized mode**: Clustered values → verify spread across color range
3. **Zero handling**: Zero-depth pixels should be black (0, 0, 0)
4. **All colormaps**: Visual spot-check for each of 10 maps
5. **Benchmark**: 720p — target ≥3× speedup (LUT lookups are fast on MLX)

## Acceptance Criteria

- [ ] All 10 colormaps implemented with correct keypoints
- [ ] Direct and equalized modes both functional
- [ ] Zero-depth → black output
- [ ] LUT precomputed (not recalculated per frame)
- [ ] ≥3× speedup at 720p

## Estimated Effort
6-8 hours

## Port Difficulty
**EASY-MEDIUM** — Histogram is CPU bottleneck, rest is vectorized LUT lookup
