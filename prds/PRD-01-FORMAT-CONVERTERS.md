# PRD-01: MLX Format Converters

## Overview
Port the 10 CUDA format conversion kernels to MLX. These handle YUY2→RGB/BGR/RGBA, stereo frame splitting (Y8I→Y8+Y8, Y12I→Y16+Y16), byte swaps (UYVY→YUYV), and SR300 INZI extraction.

## Blocked By
- PRD-00 (Project Setup)

## Source Reference
- `librealsense/src/cuda/cuda-conversion.cu` — all 10 kernels
- `librealsense/src/cuda/cuda-conversion.cuh` — headers and struct definitions

## Kernels to Port

### Tier 1: Trivial (< 1 hour each)

#### 1.1 `uyvy_to_yuyv` (byte swap)
```python
def uyvy_to_yuyv(src: mx.array) -> mx.array:
    """Swap bytes within uint16 words. src shape: (N,) uint16."""
    return ((src >> 8) & 0x00FF) | ((src << 8) & 0xFF00)
```
- **Input**: `(N,)` uint16
- **Output**: `(N,)` uint16
- **Tolerance**: Exact (bitwise)

#### 1.2 `split_y8i` (stereo Y8 split)
```python
def split_y8i(src: mx.array) -> tuple[mx.array, mx.array]:
    """Split interleaved Y8 stereo to left/right. src shape: (H, W*2) uint8."""
    return src[..., 0::2], src[..., 1::2]
```
- **Input**: `(N*2,)` uint8 (interleaved L, R)
- **Output**: `(N,)` uint8 (left), `(N,)` uint8 (right)
- **Tolerance**: Exact

#### 1.3 `split_y12i` (stereo 12-bit split)
```python
def split_y12i(src: mx.array) -> tuple[mx.array, mx.array]:
    """Unpack 12-bit packed stereo to 16-bit. Bit manipulation."""
```
- **Input**: Packed 12-bit interleaved (3 bytes per stereo pair)
- **Output**: `(N,)` uint16 (left), `(N,)` uint16 (right)
- **Tolerance**: Exact
- **Note**: Bit shifting: `(val << 6) | (val >> 4)` for 12→16 bit expansion

#### 1.4 `z16_y8_from_sr300_inzi` (IR extraction)
```python
def z16_y8_from_inzi(src: mx.array) -> mx.array:
    """Extract IR from INZI format. dest[i] = src[i] >> 2."""
    return (src >> 2).astype(mx.uint8)
```
- **Input**: `(N,)` uint16
- **Output**: `(N,)` uint8
- **Tolerance**: Exact

#### 1.5 `z16_y16_from_sr300_inzi`
```python
def z16_y16_from_inzi(src: mx.array) -> mx.array:
    """Extract 16-bit IR. dest[i] = src[i] << 6."""
    return src << 6
```
- **Input/Output**: `(N,)` uint16
- **Tolerance**: Exact

### Tier 2: Color Conversions (2-3 hours each)

#### 1.6 `yuy2_to_rgb8`
```python
def yuy2_to_rgb8(src: mx.array, width: int, height: int) -> mx.array:
    """YUY2 to RGB8. ITU-R BT.601 color matrix."""
```
- **Input**: `(H, W*2)` uint8 (YUY2 packed: Y0, U, Y1, V per 2 pixels)
- **Output**: `(H, W, 3)` uint8 (RGB)
- **Algorithm**:
  ```
  c = Y - 16
  d = U - 128
  e = V - 128
  R = clamp((298*c + 409*e + 128) >> 8, 0, 255)
  G = clamp((298*c - 100*d - 208*e + 128) >> 8, 0, 255)
  B = clamp((298*c + 516*d + 128) >> 8, 0, 255)
  ```
- **MLX Strategy**: Reshape to extract Y/U/V channels → broadcast multiply → clamp → stack
- **Tolerance**: Exact (integer arithmetic)

#### 1.7 `yuy2_to_bgr8` — same as 1.6, channel swap
#### 1.8 `yuy2_to_rgba8` — same as 1.6, add alpha=255
#### 1.9 `yuy2_to_bgra8` — same as 1.6, channel swap + alpha
#### 1.10 `yuy2_to_y16` — extract Y channel only, expand to 16-bit

## Function Signatures (Public API)

```python
# realsense_mlx/converters/format_converter.py

class FormatConverter:
    """MLX-accelerated format conversion for RealSense frames."""

    @staticmethod
    def yuy2_to_rgb(frame: mx.array, width: int, height: int) -> mx.array: ...

    @staticmethod
    def yuy2_to_bgr(frame: mx.array, width: int, height: int) -> mx.array: ...

    @staticmethod
    def yuy2_to_rgba(frame: mx.array, width: int, height: int) -> mx.array: ...

    @staticmethod
    def yuy2_to_bgra(frame: mx.array, width: int, height: int) -> mx.array: ...

    @staticmethod
    def yuy2_to_y16(frame: mx.array, width: int, height: int) -> mx.array: ...

    @staticmethod
    def uyvy_to_yuyv(frame: mx.array) -> mx.array: ...

    @staticmethod
    def split_y8i(frame: mx.array) -> tuple[mx.array, mx.array]: ...

    @staticmethod
    def split_y12i(frame: mx.array) -> tuple[mx.array, mx.array]: ...

    @staticmethod
    def extract_ir_y8(frame: mx.array) -> mx.array: ...

    @staticmethod
    def extract_ir_y16(frame: mx.array) -> mx.array: ...
```

## Test Plan

For each converter:
1. Generate synthetic input matching the exact format
2. Compare MLX output against `pyrealsense2` CPU output (reference)
3. Test edge cases: zero frames, max values (255/65535), single pixel
4. Benchmark: 640×480, 1280×720, 1920×1080 resolutions

```python
def test_yuy2_to_rgb_matches_reference():
    """Compare MLX YUY2→RGB against pyrealsense2 CPU conversion."""
    yuy2_data = generate_synthetic_yuy2(640, 480)
    mlx_result = FormatConverter.yuy2_to_rgb(mx.array(yuy2_data), 640, 480)
    cpu_result = cpu_yuy2_to_rgb(yuy2_data, 640, 480)  # numpy reference
    assert np.array_equal(np.array(mlx_result), cpu_result)
```

## Acceptance Criteria

- [ ] All 10 converters pass exact-match tests against CPU reference
- [ ] No int64 usage (MLX Metal limitation)
- [ ] Benchmark shows ≥2× speedup over numpy at 720p
- [ ] Zero-copy input from numpy (via `mx.array(np_array)`)

## Estimated Effort
8-12 hours

## Port Difficulty
**EASY** — All elementwise operations, no inter-pixel dependencies
