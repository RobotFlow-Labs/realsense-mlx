"""MLX-accelerated pixel-format converters for Intel RealSense streams.

All 10 kernels are direct ports of the CUDA implementations in
``librealsense/src/cuda/cuda-conversion.cu``.  Every function accepts and
returns ``mx.array`` objects so that the results can be consumed by downstream
MLX pipelines without an intermediate round-trip through NumPy.

YUV→RGB conversion uses the ITU-R BT.601 integer-arithmetic formula identical
to the CUDA source:

    c = Y − 16,  d = U − 128,  e = V − 128
    R = clip((298c + 409e + 128) >> 8, 0, 255)
    G = clip((298c − 100d − 208e + 128) >> 8, 0, 255)
    B = clip((298c + 516d + 128) >> 8, 0, 255)

MLX constraints observed throughout:
- No int64: all intermediate arithmetic is kept in int32.
- Bit-shifts use ``mx.right_shift`` / ``mx.left_shift``.
- Clamping uses ``mx.clip``.
- ``mx.eval`` is called before returning to materialise lazy graphs.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

__all__ = [
    "FormatConverter",
    "uyvy_to_yuyv",
    "split_y8i",
    "split_y12i",
    "extract_ir_y8",
    "extract_ir_y16",
    "yuy2_to_rgb",
    "yuy2_to_bgr",
    "yuy2_to_rgba",
    "yuy2_to_bgra",
    "yuy2_to_y16",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _int32(arr: mx.array) -> mx.array:
    """Cast *arr* to int32 without going through int64."""
    return arr.astype(mx.int32)


def _uint8(arr: mx.array) -> mx.array:
    return arr.astype(mx.uint8)


def _uint16(arr: mx.array) -> mx.array:
    return arr.astype(mx.uint16)


def _clamp_uint8(arr: mx.array) -> mx.array:
    """Clip int32 values to [0, 255] and cast to uint8."""
    return _uint8(mx.clip(arr, 0, 255))


def _yuy2_chromaluma(
    flat: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Extract Y0, U, Y1, V from a flat uint8 YUY2 buffer.

    YUY2 macro-pixel layout (4 bytes per 2 pixels):
        byte 0: Y0
        byte 1: U  (Cb)
        byte 2: Y1
        byte 3: V  (Cr)

    Args:
        flat: 1-D uint8 array of length ``4 * num_super_pixels``.

    Returns:
        Tuple (y0, u, y1, v) — each a 1-D int32 array of length
        ``num_super_pixels``.
    """
    # Reshape to (N, 4) super-pixel view
    n = flat.shape[0] // 4
    quad = flat.reshape((n, 4))

    y0 = _int32(quad[:, 0])
    u  = _int32(quad[:, 1])
    y1 = _int32(quad[:, 2])
    v  = _int32(quad[:, 3])
    return y0, u, y1, v


def _yuv_to_rgb_channels(
    y: mx.array,
    u: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Convert Y/U/V int32 arrays to clamped uint8 R, G, B.

    Uses the ITU-R BT.601 integer-arithmetic formula from the CUDA source.

    Args:
        y: Luma component (int32).
        u: Cb chroma component (int32).
        v: Cr chroma component (int32).

    Returns:
        Tuple (R, G, B) as uint8 arrays.
    """
    c = y - 16
    d = u - 128
    e = v - 128

    r = _clamp_uint8(mx.right_shift(298 * c + 409 * e + 128, 8))
    g = _clamp_uint8(mx.right_shift(298 * c - 100 * d - 208 * e + 128, 8))
    b = _clamp_uint8(mx.right_shift(298 * c + 516 * d + 128, 8))
    return r, g, b


# ---------------------------------------------------------------------------
# Public converter functions
# ---------------------------------------------------------------------------


def uyvy_to_yuyv(src: mx.array) -> mx.array:
    """Swap bytes within each uint16 word to convert UYVY → YUYV.

    Equivalent to the CUDA kernel::

        dst[i] = ((src[i] >> 8) & 0x00FF) | ((src[i] << 8) & 0xFF00)

    Args:
        src: 1-D uint16 array (raw UYVY frame packed as uint16 words).

    Returns:
        1-D uint16 array in YUYV order.
    """
    s = src.astype(mx.uint16)
    lo = mx.right_shift(s, 8) & mx.array(0x00FF, dtype=mx.uint16)
    hi = mx.left_shift(s, 8) & mx.array(0xFF00, dtype=mx.uint16)
    result = lo | hi
    mx.eval(result)
    return result


def split_y8i(
    src: mx.array,
    width: int,
    height: int,
) -> tuple[mx.array, mx.array]:
    """Split an interleaved Y8I stereo frame into left and right uint8 planes.

    Y8I layout: alternating (left, right) byte pairs — exactly
    ``y8i_pixel { uint8_t l; uint8_t r; }`` from the CUDA header.

    Args:
        src: 1-D uint8 array of length ``2 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple (left, right) — each a 2-D uint8 array of shape
        ``(height, width)``.
    """
    count = width * height
    pairs = src.reshape((count, 2))
    left  = pairs[:, 0].reshape((height, width))
    right = pairs[:, 1].reshape((height, width))
    mx.eval(left, right)
    return left, right


def split_y12i(
    src: mx.array,
    width: int,
    height: int,
) -> tuple[mx.array, mx.array]:
    """Unpack 12-bit packed stereo (Y12I) to a pair of uint16 planes.

    Y12I pixel layout (3 bytes per stereo sample — packed bitfield)::

        struct y12i_pixel {
            uint8_t rl:8, rh:4, ll:4, lh:8;
            int l() { return lh << 4 | ll; }
            int r() { return rh << 8 | rl; }
        };

    Byte layout in memory (little-endian bitfield, packed)::

        byte 0 → rl[7:0]               (low byte of right)
        byte 1 → ll[3:0] | rh[3:0]<<4  (low nibble of left, high nibble of right)
        byte 2 → lh[7:0]               (high byte of left)

    The 12-bit value is then expanded to 16-bit by the CUDA kernel::

        a[i] = l() << 6 | l() >> 4;
        b[i] = r() << 6 | r() >> 4;

    Args:
        src: 1-D uint8 array of length ``3 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple (left, right) — each a 2-D uint16 array of shape
        ``(height, width)``.
    """
    count = width * height
    triplets = src.reshape((count, 3))

    b0 = _int32(triplets[:, 0])  # rl[7:0]
    b1 = _int32(triplets[:, 1])  # ll[3:0] | rh[3:0]<<4
    b2 = _int32(triplets[:, 2])  # lh[7:0]

    # left  = lh<<4 | ll   where ll = b1[3:0], lh = b2[7:0]
    ll = b1 & 0x0F
    lh = b2
    l_12 = mx.left_shift(lh, 4) | ll

    # right = rh<<8 | rl   where rl = b0[7:0], rh = b1[7:4]
    rl = b0
    rh = mx.right_shift(b1, 4) & 0x0F
    r_12 = mx.left_shift(rh, 8) | rl

    # Expand 12-bit → 16-bit:  val16 = val12 << 6 | val12 >> 4 (replicating top bits)
    l_16 = _uint16(mx.left_shift(l_12, 6) | mx.right_shift(l_12, 4))
    r_16 = _uint16(mx.left_shift(r_12, 6) | mx.right_shift(r_12, 4))

    left  = l_16.reshape((height, width))
    right = r_16.reshape((height, width))
    mx.eval(left, right)
    return left, right


def extract_ir_y8(src: mx.array) -> mx.array:
    """Extract 8-bit IR from a 16-bit SR300 INZI IR channel.

    Equivalent to the CUDA kernel::

        dest[i] = source[i] >> 2;

    Args:
        src: 1-D or 2-D uint16 array.

    Returns:
        uint8 array of the same shape.
    """
    # The CUDA kernel casts the shifted uint16 directly to uint8, which
    # truncates to the low byte.  This is correct for SR300 INZI data (10-bit
    # range, max shifted value = 255).  We replicate the exact CUDA behavior.
    shifted = mx.right_shift(src.astype(mx.uint16), 2)
    result = (shifted & 0xFF).astype(mx.uint8)
    mx.eval(result)
    return result


def extract_ir_y16(src: mx.array) -> mx.array:
    """Expand a 10-bit IR value to 16-bit by left-shifting 6 places.

    Equivalent to the CUDA kernel::

        dest[i] = source[i] << 6;

    Args:
        src: 1-D or 2-D uint16 array.

    Returns:
        uint16 array of the same shape.
    """
    result = _uint16(mx.left_shift(src.astype(mx.uint16), 6))
    mx.eval(result)
    return result


def yuy2_to_rgb(src: mx.array, width: int, height: int) -> mx.array:
    """Convert YUY2 to packed RGB8.

    Each YUY2 macro-pixel (4 bytes) produces two RGB pixels (6 bytes).

    Args:
        src: 1-D uint8 array of length ``2 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        3-D uint8 array of shape ``(height, width, 3)`` in RGB order.
    """
    y0, u, y1, v = _yuy2_chromaluma(src.reshape(-1))

    r0, g0, b0 = _yuv_to_rgb_channels(y0, u, v)
    r1, g1, b1 = _yuv_to_rgb_channels(y1, u, v)

    n = y0.shape[0]
    # Interleave two pixels per super-pixel → (2*n, 3)
    rgb = mx.stack(
        [
            mx.stack([r0, g0, b0], axis=1),  # (n, 3)
            mx.stack([r1, g1, b1], axis=1),  # (n, 3)
        ],
        axis=1,                               # (n, 2, 3)
    ).reshape((2 * n, 3))                     # (width*height, 3)

    result = rgb.reshape((height, width, 3))
    mx.eval(result)
    return result


def yuy2_to_bgr(src: mx.array, width: int, height: int) -> mx.array:
    """Convert YUY2 to packed BGR8.

    Args:
        src: 1-D uint8 array of length ``2 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        3-D uint8 array of shape ``(height, width, 3)`` in BGR order.
    """
    y0, u, y1, v = _yuy2_chromaluma(src.reshape(-1))

    r0, g0, b0 = _yuv_to_rgb_channels(y0, u, v)
    r1, g1, b1 = _yuv_to_rgb_channels(y1, u, v)

    n = y0.shape[0]
    bgr = mx.stack(
        [
            mx.stack([b0, g0, r0], axis=1),
            mx.stack([b1, g1, r1], axis=1),
        ],
        axis=1,
    ).reshape((2 * n, 3))

    result = bgr.reshape((height, width, 3))
    mx.eval(result)
    return result


def yuy2_to_rgba(src: mx.array, width: int, height: int) -> mx.array:
    """Convert YUY2 to packed RGBA8 (alpha = 255).

    Args:
        src: 1-D uint8 array of length ``2 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        3-D uint8 array of shape ``(height, width, 4)`` in RGBA order.
    """
    y0, u, y1, v = _yuy2_chromaluma(src.reshape(-1))

    r0, g0, b0 = _yuv_to_rgb_channels(y0, u, v)
    r1, g1, b1 = _yuv_to_rgb_channels(y1, u, v)

    n = y0.shape[0]
    alpha = mx.full((n,), 255, dtype=mx.uint8)

    rgba = mx.stack(
        [
            mx.stack([r0, g0, b0, alpha], axis=1),
            mx.stack([r1, g1, b1, alpha], axis=1),
        ],
        axis=1,
    ).reshape((2 * n, 4))

    result = rgba.reshape((height, width, 4))
    mx.eval(result)
    return result


def yuy2_to_bgra(src: mx.array, width: int, height: int) -> mx.array:
    """Convert YUY2 to packed BGRA8 (alpha = 255).

    Args:
        src: 1-D uint8 array of length ``2 * width * height``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        3-D uint8 array of shape ``(height, width, 4)`` in BGRA order.
    """
    y0, u, y1, v = _yuy2_chromaluma(src.reshape(-1))

    r0, g0, b0 = _yuv_to_rgb_channels(y0, u, v)
    r1, g1, b1 = _yuv_to_rgb_channels(y1, u, v)

    n = y0.shape[0]
    alpha = mx.full((n,), 255, dtype=mx.uint8)

    bgra = mx.stack(
        [
            mx.stack([b0, g0, r0, alpha], axis=1),
            mx.stack([b1, g1, r1, alpha], axis=1),
        ],
        axis=1,
    ).reshape((2 * n, 4))

    result = bgra.reshape((height, width, 4))
    mx.eval(result)
    return result


def yuy2_to_y16(src: mx.array, width: int, height: int) -> mx.array:
    """Extract the Y (luma) channel from YUY2 and expand to 16-bit.

    The CUDA kernel stores the 8-bit luma in the *high* byte of each uint16::

        dst[0] = 0;          dst[1] = src[0]   // pixel 0
        dst[2] = 0;          dst[3] = src[2]   // pixel 1

    This is equivalent to ``Y16 = Y8 << 8``.

    Args:
        src: 1-D uint8 array of length ``2 * width * height`` (YUY2).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        2-D uint16 array of shape ``(height, width)``.
    """
    flat = src.reshape(-1)
    n_super = flat.shape[0] // 4
    quad = flat.reshape((n_super, 4))

    y0_u8 = quad[:, 0].astype(mx.uint16)
    y1_u8 = quad[:, 2].astype(mx.uint16)

    # Place luma in the high byte: Y16 = Y8 << 8
    y0_16 = mx.left_shift(y0_u8, 8)
    y1_16 = mx.left_shift(y1_u8, 8)

    # Interleave back to pixel order: (n_super, 2) → (width*height,)
    y16 = mx.stack([y0_16, y1_16], axis=1).reshape((height, width))
    mx.eval(y16)
    return y16


# ---------------------------------------------------------------------------
# High-level convenience class
# ---------------------------------------------------------------------------


class FormatConverter:
    """Stateless collection of MLX pixel-format converters.

    All methods are thin wrappers around the module-level functions above.
    Construct once and reuse; there is no internal state.

    Example::

        converter = FormatConverter()
        rgb = converter.yuy2_to_rgb(raw_frame, width=640, height=480)
        np_rgb = np.array(rgb)
    """

    # ---- UYVY ---------------------------------------------------------------

    @staticmethod
    def uyvy_to_yuyv(src: mx.array) -> mx.array:
        """Byte-swap UYVY uint16 words to YUYV order.

        Args:
            src: 1-D uint16 array.

        Returns:
            1-D uint16 array.
        """
        return uyvy_to_yuyv(src)

    # ---- Stereo splits ------------------------------------------------------

    @staticmethod
    def split_y8i(
        src: mx.array, width: int, height: int
    ) -> tuple[mx.array, mx.array]:
        """Split Y8I interleaved stereo to (left, right) uint8 planes."""
        return split_y8i(src, width, height)

    @staticmethod
    def split_y12i(
        src: mx.array, width: int, height: int
    ) -> tuple[mx.array, mx.array]:
        """Unpack Y12I packed stereo to (left, right) uint16 planes."""
        return split_y12i(src, width, height)

    # ---- IR extraction ------------------------------------------------------

    @staticmethod
    def extract_ir_y8(src: mx.array) -> mx.array:
        """Extract 8-bit IR: ``src >> 2``."""
        return extract_ir_y8(src)

    @staticmethod
    def extract_ir_y16(src: mx.array) -> mx.array:
        """Expand 10-bit IR to 16-bit: ``src << 6``."""
        return extract_ir_y16(src)

    # ---- YUY2 conversions ---------------------------------------------------

    @staticmethod
    def yuy2_to_rgb(src: mx.array, width: int, height: int) -> mx.array:
        """YUY2 → RGB8 ``(height, width, 3)``."""
        return yuy2_to_rgb(src, width, height)

    @staticmethod
    def yuy2_to_bgr(src: mx.array, width: int, height: int) -> mx.array:
        """YUY2 → BGR8 ``(height, width, 3)``."""
        return yuy2_to_bgr(src, width, height)

    @staticmethod
    def yuy2_to_rgba(src: mx.array, width: int, height: int) -> mx.array:
        """YUY2 → RGBA8 ``(height, width, 4)`` with alpha = 255."""
        return yuy2_to_rgba(src, width, height)

    @staticmethod
    def yuy2_to_bgra(src: mx.array, width: int, height: int) -> mx.array:
        """YUY2 → BGRA8 ``(height, width, 4)`` with alpha = 255."""
        return yuy2_to_bgra(src, width, height)

    @staticmethod
    def yuy2_to_y16(src: mx.array, width: int, height: int) -> mx.array:
        """YUY2 → Y16 ``(height, width)`` luma-only."""
        return yuy2_to_y16(src, width, height)
