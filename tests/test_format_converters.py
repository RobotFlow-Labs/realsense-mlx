"""Tests for all 10 format-converter functions in FormatConverter.

Each converter is tested by:
1. Building synthetic input data with known values.
2. Running the NumPy reference (matching the CUDA C logic exactly).
3. Running the MLX implementation via FormatConverter.
4. Asserting element-wise equality (or within 1 LSB for integer rounding).

Fixtures are designed to cover edge-cases: black frames, white frames,
saturated chroma, mixed random data.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from realsense_mlx.converters.format_converter import (
    FormatConverter,
    uyvy_to_yuyv,
    split_y8i,
    split_y12i,
    extract_ir_y8,
    extract_ir_y16,
    yuy2_to_rgb,
    yuy2_to_bgr,
    yuy2_to_rgba,
    yuy2_to_bgra,
    yuy2_to_y16,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WIDTH = 8
HEIGHT = 4
RNG = np.random.default_rng(42)


def to_mx(arr: np.ndarray) -> mx.array:
    return mx.array(arr)


def to_np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# NumPy reference implementations (mirror the CUDA kernels exactly)
# ---------------------------------------------------------------------------


def ref_uyvy_to_yuyv(src: np.ndarray) -> np.ndarray:
    """Byte-swap each uint16 word."""
    return ((src >> 8) & np.uint16(0x00FF)) | ((src << 8) & np.uint16(0xFF00))


def ref_split_y8i(src: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    count = width * height
    pairs = src.reshape(count, 2)
    left  = pairs[:, 0].reshape(height, width)
    right = pairs[:, 1].reshape(height, width)
    return left, right


def ref_split_y12i(
    src: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    count = width * height
    triplets = src.reshape(count, 3).astype(np.int32)
    b0 = triplets[:, 0]
    b1 = triplets[:, 1]
    b2 = triplets[:, 2]

    ll = b1 & 0x0F
    lh = b2
    l_12 = (lh << 4) | ll

    rl = b0
    rh = (b1 >> 4) & 0x0F
    r_12 = (rh << 8) | rl

    l_16 = ((l_12 << 6) | (l_12 >> 4)).astype(np.uint16)
    r_16 = ((r_12 << 6) | (r_12 >> 4)).astype(np.uint16)
    return l_16.reshape(height, width), r_16.reshape(height, width)


def ref_extract_ir_y8(src: np.ndarray) -> np.ndarray:
    return (src >> 2).astype(np.uint8)


def ref_extract_ir_y16(src: np.ndarray) -> np.ndarray:
    return (src.astype(np.uint32) << 6).astype(np.uint16)


def _ref_yuv_pixel(y: int, u: int, v: int) -> tuple[int, int, int]:
    c = int(y) - 16
    d = int(u) - 128
    e = int(v) - 128
    r = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
    g = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
    b = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
    return int(r), int(g), int(b)


def ref_yuy2_to_rgb(src: np.ndarray, width: int, height: int) -> np.ndarray:
    flat = src.ravel()
    n = len(flat) // 4
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n):
        y0, u, y1, v = flat[i*4], flat[i*4+1], flat[i*4+2], flat[i*4+3]
        px0 = _ref_yuv_pixel(y0, u, v)
        px1 = _ref_yuv_pixel(y1, u, v)
        row, col_base = divmod(i * 2, width)
        out[row, col_base] = px0
        out[row, col_base + 1] = px1
    return out


def ref_yuy2_to_bgr(src: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = ref_yuy2_to_rgb(src, width, height)
    return rgb[:, :, ::-1].copy()


def ref_yuy2_to_rgba(src: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = ref_yuy2_to_rgb(src, width, height)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def ref_yuy2_to_bgra(src: np.ndarray, width: int, height: int) -> np.ndarray:
    bgr = ref_yuy2_to_bgr(src, width, height)
    alpha = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, alpha], axis=2)


def ref_yuy2_to_y16(src: np.ndarray, width: int, height: int) -> np.ndarray:
    flat = src.ravel()
    n = len(flat) // 4
    out = np.zeros((height, width), dtype=np.uint16)
    for i in range(n):
        y0, y1 = flat[i*4], flat[i*4+2]
        row, col_base = divmod(i * 2, width)
        out[row, col_base]     = np.uint16(y0) << 8
        out[row, col_base + 1] = np.uint16(y1) << 8
    return out


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def make_uyvy(n: int) -> np.ndarray:
    """Random uint16 UYVY buffer of *n* words."""
    return RNG.integers(0, 0xFFFF, size=n, dtype=np.uint16)


def make_yuy2(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """YUY2 frame: valid Y in [16,235], UV in [16,240]."""
    n_pixels = width * height
    n_super  = n_pixels // 2
    y_vals = RNG.integers(16, 236, size=n_pixels, dtype=np.uint8)
    u_vals = RNG.integers(16, 241, size=n_super,  dtype=np.uint8)
    v_vals = RNG.integers(16, 241, size=n_super,  dtype=np.uint8)

    frame = np.empty(n_pixels * 2, dtype=np.uint8)
    for i in range(n_super):
        frame[i*4 + 0] = y_vals[i*2]
        frame[i*4 + 1] = u_vals[i]
        frame[i*4 + 2] = y_vals[i*2 + 1]
        frame[i*4 + 3] = v_vals[i]
    return frame


def make_yuy2_black(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Black frame: Y=16, U=128, V=128."""
    n_pixels = width * height
    frame = np.empty(n_pixels * 2, dtype=np.uint8)
    frame[0::4] = 16   # Y0
    frame[1::4] = 128  # U
    frame[2::4] = 16   # Y1
    frame[3::4] = 128  # V
    return frame


def make_yuy2_white(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """White frame: Y=235, U=128, V=128."""
    n_pixels = width * height
    frame = np.empty(n_pixels * 2, dtype=np.uint8)
    frame[0::4] = 235
    frame[1::4] = 128
    frame[2::4] = 235
    frame[3::4] = 128
    return frame


def make_y8i(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Interleaved (left, right) uint8 byte pairs."""
    count = width * height
    return RNG.integers(0, 256, size=count * 2, dtype=np.uint8)


def make_y12i(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """3-bytes-per-pixel Y12I packed stereo buffer."""
    count = width * height
    return RNG.integers(0, 256, size=count * 3, dtype=np.uint8)


def make_ir_u16(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return RNG.integers(0, 0x3FFF, size=width * height, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Tests: uyvy_to_yuyv
# ---------------------------------------------------------------------------


class TestUyvyToYuyv:
    def _run(self, np_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected = ref_uyvy_to_yuyv(np_src)
        actual   = to_np(uyvy_to_yuyv(to_mx(np_src)))
        return expected, actual

    def test_random(self):
        src = make_uyvy(128)
        exp, act = self._run(src)
        np.testing.assert_array_equal(act, exp)

    def test_known_value(self):
        # 0xAABB → 0xBBAA
        src = np.array([0xAABB], dtype=np.uint16)
        _, act = self._run(src)
        assert int(act[0]) == 0xBBAA

    def test_identity_symmetric(self):
        # Applying twice should give back the original
        src = make_uyvy(64)
        once = uyvy_to_yuyv(to_mx(src))
        twice = uyvy_to_yuyv(once)
        np.testing.assert_array_equal(to_np(twice), src)

    def test_zero_input(self):
        src = np.zeros(16, dtype=np.uint16)
        exp, act = self._run(src)
        np.testing.assert_array_equal(act, exp)

    def test_all_ones(self):
        src = np.full(16, 0xFFFF, dtype=np.uint16)
        exp, act = self._run(src)
        np.testing.assert_array_equal(act, exp)

    def test_output_dtype(self):
        src = make_uyvy(8)
        result = uyvy_to_yuyv(to_mx(src))
        assert result.dtype == mx.uint16

    def test_via_class(self):
        src = make_uyvy(32)
        expected = ref_uyvy_to_yuyv(src)
        actual = to_np(FormatConverter.uyvy_to_yuyv(to_mx(src)))
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Tests: split_y8i
# ---------------------------------------------------------------------------


class TestSplitY8i:
    def _run(self, src_np: np.ndarray) -> tuple:
        left_ref, right_ref = ref_split_y8i(src_np, WIDTH, HEIGHT)
        left_mx,  right_mx  = split_y8i(to_mx(src_np), WIDTH, HEIGHT)
        return left_ref, right_ref, to_np(left_mx), to_np(right_mx)

    def test_random(self):
        src = make_y8i()
        lr, rr, la, ra = self._run(src)
        np.testing.assert_array_equal(la, lr)
        np.testing.assert_array_equal(ra, rr)

    def test_output_shape(self):
        src = make_y8i()
        left, right = split_y8i(to_mx(src), WIDTH, HEIGHT)
        assert to_np(left).shape  == (HEIGHT, WIDTH)
        assert to_np(right).shape == (HEIGHT, WIDTH)

    def test_output_dtype(self):
        src = make_y8i()
        left, right = split_y8i(to_mx(src), WIDTH, HEIGHT)
        assert left.dtype  == mx.uint8
        assert right.dtype == mx.uint8

    def test_known_pattern(self):
        # Alternating (0xAA, 0xBB) pairs
        count = WIDTH * HEIGHT
        src = np.tile([0xAA, 0xBB], count).astype(np.uint8)
        left, right = split_y8i(to_mx(src), WIDTH, HEIGHT)
        assert np.all(to_np(left)  == 0xAA)
        assert np.all(to_np(right) == 0xBB)

    def test_independence_of_planes(self):
        # Mutating one plane should not affect the other
        src = make_y8i()
        left, right = split_y8i(to_mx(src), WIDTH, HEIGHT)
        l_np = to_np(left).copy()
        r_np = to_np(right).copy()
        assert not np.all(l_np == r_np)  # they differ (random data)

    def test_via_class(self):
        src = make_y8i()
        lr, rr, la, ra = self._run(src)
        lc, rc = FormatConverter.split_y8i(to_mx(src), WIDTH, HEIGHT)
        np.testing.assert_array_equal(to_np(lc), lr)
        np.testing.assert_array_equal(to_np(rc), rr)


# ---------------------------------------------------------------------------
# Tests: split_y12i
# ---------------------------------------------------------------------------


class TestSplitY12i:
    def _run(self, src_np: np.ndarray):
        lr, rr = ref_split_y12i(src_np, WIDTH, HEIGHT)
        la_mx, ra_mx = split_y12i(to_mx(src_np), WIDTH, HEIGHT)
        return lr, rr, to_np(la_mx), to_np(ra_mx)

    def test_random(self):
        src = make_y12i()
        lr, rr, la, ra = self._run(src)
        np.testing.assert_array_equal(la, lr)
        np.testing.assert_array_equal(ra, rr)

    def test_output_shape(self):
        src = make_y12i()
        left, right = split_y12i(to_mx(src), WIDTH, HEIGHT)
        assert to_np(left).shape  == (HEIGHT, WIDTH)
        assert to_np(right).shape == (HEIGHT, WIDTH)

    def test_output_dtype(self):
        src = make_y12i()
        left, right = split_y12i(to_mx(src), WIDTH, HEIGHT)
        assert left.dtype  == mx.uint16
        assert right.dtype == mx.uint16

    def test_zero_input(self):
        count = WIDTH * HEIGHT
        src = np.zeros(count * 3, dtype=np.uint8)
        left, right = split_y12i(to_mx(src), WIDTH, HEIGHT)
        assert np.all(to_np(left)  == 0)
        assert np.all(to_np(right) == 0)

    def test_16bit_range(self):
        """Output values should fit in uint16 (0–65535)."""
        src = make_y12i()
        left, right = split_y12i(to_mx(src), WIDTH, HEIGHT)
        l_np = to_np(left)
        r_np = to_np(right)
        assert l_np.max() <= 65535
        assert r_np.max() <= 65535

    def test_via_class(self):
        src = make_y12i()
        lr, rr, la, ra = self._run(src)
        lc, rc = FormatConverter.split_y12i(to_mx(src), WIDTH, HEIGHT)
        np.testing.assert_array_equal(to_np(lc), lr)
        np.testing.assert_array_equal(to_np(rc), rr)


# ---------------------------------------------------------------------------
# Tests: extract_ir_y8
# ---------------------------------------------------------------------------


class TestExtractIrY8:
    def test_random(self):
        src = make_ir_u16()
        expected = ref_extract_ir_y8(src)
        actual   = to_np(extract_ir_y8(to_mx(src)))
        np.testing.assert_array_equal(actual, expected)

    def test_known_values(self):
        src = np.array([0x0000, 0x0004, 0x03FF, 0xFFFF], dtype=np.uint16)
        exp = ref_extract_ir_y8(src)
        act = to_np(extract_ir_y8(to_mx(src)))
        np.testing.assert_array_equal(act, exp)

    def test_output_dtype(self):
        src = make_ir_u16()
        result = extract_ir_y8(to_mx(src))
        assert result.dtype == mx.uint8

    def test_output_range(self):
        # All outputs must be in [0, 255]
        src = np.arange(0, 256, dtype=np.uint16) * 4
        act = to_np(extract_ir_y8(to_mx(src)))
        assert act.min() >= 0
        assert act.max() <= 255

    def test_via_class(self):
        src = make_ir_u16()
        exp = ref_extract_ir_y8(src)
        act = to_np(FormatConverter.extract_ir_y8(to_mx(src)))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: extract_ir_y16
# ---------------------------------------------------------------------------


class TestExtractIrY16:
    def test_random(self):
        src = make_ir_u16()
        expected = ref_extract_ir_y16(src)
        actual   = to_np(extract_ir_y16(to_mx(src)))
        np.testing.assert_array_equal(actual, expected)

    def test_known_values(self):
        # 1 << 6 = 64
        src = np.array([0x0001, 0x0010, 0x03FF], dtype=np.uint16)
        exp = ref_extract_ir_y16(src)
        act = to_np(extract_ir_y16(to_mx(src)))
        np.testing.assert_array_equal(act, exp)

    def test_output_dtype(self):
        src = make_ir_u16()
        result = extract_ir_y16(to_mx(src))
        assert result.dtype == mx.uint16

    def test_zero_input(self):
        src = np.zeros(16, dtype=np.uint16)
        act = to_np(extract_ir_y16(to_mx(src)))
        assert np.all(act == 0)

    def test_via_class(self):
        src = make_ir_u16()
        exp = ref_extract_ir_y16(src)
        act = to_np(FormatConverter.extract_ir_y16(to_mx(src)))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: yuy2_to_rgb
# ---------------------------------------------------------------------------


class TestYuy2ToRgb:
    def _compare(self, src_np: np.ndarray):
        expected = ref_yuy2_to_rgb(src_np, WIDTH, HEIGHT)
        actual   = to_np(yuy2_to_rgb(to_mx(src_np), WIDTH, HEIGHT))
        np.testing.assert_array_equal(actual, expected)

    def test_random(self):
        self._compare(make_yuy2())

    def test_black_frame(self):
        self._compare(make_yuy2_black())

    def test_white_frame(self):
        self._compare(make_yuy2_white())

    def test_output_shape(self):
        result = to_np(yuy2_to_rgb(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH, 3)

    def test_output_dtype(self):
        result = yuy2_to_rgb(to_mx(make_yuy2()), WIDTH, HEIGHT)
        assert result.dtype == mx.uint8

    def test_output_range(self):
        result = to_np(yuy2_to_rgb(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_saturated_chroma(self):
        # Max V → strong red; Y=128, U=128, V=240
        src = np.empty(WIDTH * HEIGHT * 2, dtype=np.uint8)
        src[0::4] = 128
        src[1::4] = 128
        src[2::4] = 128
        src[3::4] = 240
        self._compare(src)

    def test_via_class(self):
        src = make_yuy2()
        exp = ref_yuy2_to_rgb(src, WIDTH, HEIGHT)
        act = to_np(FormatConverter.yuy2_to_rgb(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: yuy2_to_bgr
# ---------------------------------------------------------------------------


class TestYuy2ToBgr:
    def _compare(self, src_np: np.ndarray):
        expected = ref_yuy2_to_bgr(src_np, WIDTH, HEIGHT)
        actual   = to_np(yuy2_to_bgr(to_mx(src_np), WIDTH, HEIGHT))
        np.testing.assert_array_equal(actual, expected)

    def test_random(self):
        self._compare(make_yuy2())

    def test_black_frame(self):
        self._compare(make_yuy2_black())

    def test_white_frame(self):
        self._compare(make_yuy2_white())

    def test_output_shape(self):
        result = to_np(yuy2_to_bgr(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH, 3)

    def test_output_dtype(self):
        result = yuy2_to_bgr(to_mx(make_yuy2()), WIDTH, HEIGHT)
        assert result.dtype == mx.uint8

    def test_bgr_vs_rgb_channel_order(self):
        src = make_yuy2()
        rgb = to_np(yuy2_to_rgb(to_mx(src), WIDTH, HEIGHT))
        bgr = to_np(yuy2_to_bgr(to_mx(src), WIDTH, HEIGHT))
        # R channel of RGB == B channel of BGR
        np.testing.assert_array_equal(rgb[:, :, 0], bgr[:, :, 2])
        np.testing.assert_array_equal(rgb[:, :, 1], bgr[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 2], bgr[:, :, 0])

    def test_via_class(self):
        src = make_yuy2()
        exp = ref_yuy2_to_bgr(src, WIDTH, HEIGHT)
        act = to_np(FormatConverter.yuy2_to_bgr(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: yuy2_to_rgba
# ---------------------------------------------------------------------------


class TestYuy2ToRgba:
    def _compare(self, src_np: np.ndarray):
        expected = ref_yuy2_to_rgba(src_np, WIDTH, HEIGHT)
        actual   = to_np(yuy2_to_rgba(to_mx(src_np), WIDTH, HEIGHT))
        np.testing.assert_array_equal(actual, expected)

    def test_random(self):
        self._compare(make_yuy2())

    def test_black_frame(self):
        self._compare(make_yuy2_black())

    def test_output_shape(self):
        result = to_np(yuy2_to_rgba(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH, 4)

    def test_output_dtype(self):
        result = yuy2_to_rgba(to_mx(make_yuy2()), WIDTH, HEIGHT)
        assert result.dtype == mx.uint8

    def test_alpha_always_255(self):
        src = make_yuy2()
        result = to_np(yuy2_to_rgba(to_mx(src), WIDTH, HEIGHT))
        assert np.all(result[:, :, 3] == 255)

    def test_rgb_channels_match_rgb_output(self):
        src = make_yuy2()
        rgb  = to_np(yuy2_to_rgb(to_mx(src),  WIDTH, HEIGHT))
        rgba = to_np(yuy2_to_rgba(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(rgba[:, :, :3], rgb)

    def test_via_class(self):
        src = make_yuy2()
        exp = ref_yuy2_to_rgba(src, WIDTH, HEIGHT)
        act = to_np(FormatConverter.yuy2_to_rgba(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: yuy2_to_bgra
# ---------------------------------------------------------------------------


class TestYuy2ToBgra:
    def _compare(self, src_np: np.ndarray):
        expected = ref_yuy2_to_bgra(src_np, WIDTH, HEIGHT)
        actual   = to_np(yuy2_to_bgra(to_mx(src_np), WIDTH, HEIGHT))
        np.testing.assert_array_equal(actual, expected)

    def test_random(self):
        self._compare(make_yuy2())

    def test_black_frame(self):
        self._compare(make_yuy2_black())

    def test_output_shape(self):
        result = to_np(yuy2_to_bgra(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH, 4)

    def test_output_dtype(self):
        result = yuy2_to_bgra(to_mx(make_yuy2()), WIDTH, HEIGHT)
        assert result.dtype == mx.uint8

    def test_alpha_always_255(self):
        src = make_yuy2()
        result = to_np(yuy2_to_bgra(to_mx(src), WIDTH, HEIGHT))
        assert np.all(result[:, :, 3] == 255)

    def test_bgr_channels_match_bgr_output(self):
        src = make_yuy2()
        bgr  = to_np(yuy2_to_bgr(to_mx(src),  WIDTH, HEIGHT))
        bgra = to_np(yuy2_to_bgra(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(bgra[:, :, :3], bgr)

    def test_rgba_vs_bgra_channel_swap(self):
        src = make_yuy2()
        rgba = to_np(yuy2_to_rgba(to_mx(src), WIDTH, HEIGHT))
        bgra = to_np(yuy2_to_bgra(to_mx(src), WIDTH, HEIGHT))
        # R in rgba == B in bgra
        np.testing.assert_array_equal(rgba[:, :, 0], bgra[:, :, 2])
        np.testing.assert_array_equal(rgba[:, :, 1], bgra[:, :, 1])
        np.testing.assert_array_equal(rgba[:, :, 2], bgra[:, :, 0])
        np.testing.assert_array_equal(rgba[:, :, 3], bgra[:, :, 3])

    def test_via_class(self):
        src = make_yuy2()
        exp = ref_yuy2_to_bgra(src, WIDTH, HEIGHT)
        act = to_np(FormatConverter.yuy2_to_bgra(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: yuy2_to_y16
# ---------------------------------------------------------------------------


class TestYuy2ToY16:
    def _compare(self, src_np: np.ndarray):
        expected = ref_yuy2_to_y16(src_np, WIDTH, HEIGHT)
        actual   = to_np(yuy2_to_y16(to_mx(src_np), WIDTH, HEIGHT))
        np.testing.assert_array_equal(actual, expected)

    def test_random(self):
        self._compare(make_yuy2())

    def test_black_frame(self):
        self._compare(make_yuy2_black())

    def test_white_frame(self):
        self._compare(make_yuy2_white())

    def test_output_shape(self):
        result = to_np(yuy2_to_y16(to_mx(make_yuy2()), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH)

    def test_output_dtype(self):
        result = yuy2_to_y16(to_mx(make_yuy2()), WIDTH, HEIGHT)
        assert result.dtype == mx.uint16

    def test_luma_in_high_byte(self):
        """Y16 stores luma in the high byte (low byte is 0)."""
        src = make_yuy2()
        y16 = to_np(yuy2_to_y16(to_mx(src), WIDTH, HEIGHT))
        # Low 8 bits must be zero
        assert np.all((y16 & 0x00FF) == 0)

    def test_zero_luma(self):
        """Y=0 in YUY2 → Y16=0 (high byte = 0)."""
        src = np.zeros(WIDTH * HEIGHT * 2, dtype=np.uint8)
        result = to_np(yuy2_to_y16(to_mx(src), WIDTH, HEIGHT))
        assert np.all(result == 0)

    def test_via_class(self):
        src = make_yuy2()
        exp = ref_yuy2_to_y16(src, WIDTH, HEIGHT)
        act = to_np(FormatConverter.yuy2_to_y16(to_mx(src), WIDTH, HEIGHT))
        np.testing.assert_array_equal(act, exp)


# ---------------------------------------------------------------------------
# Tests: Backend factory
# ---------------------------------------------------------------------------


class TestBackendFactory:
    def test_mlx_backend_create(self):
        from realsense_mlx.backends.base import ProcessingBackend
        backend = ProcessingBackend.create("mlx")
        assert backend.__class__.__name__ == "MLXBackend"

    def test_cpu_backend_create(self):
        from realsense_mlx.backends.base import ProcessingBackend
        backend = ProcessingBackend.create("cpu")
        assert backend.__class__.__name__ == "CPUBackend"

    def test_unknown_backend_raises(self):
        from realsense_mlx.backends.base import ProcessingBackend
        with pytest.raises(ValueError, match="Unknown backend"):
            ProcessingBackend.create("cuda")

    def test_mlx_backend_roundtrip(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        arr = np.array([1, 2, 3], dtype=np.float32)
        device = backend.to_device(arr)
        back = backend.to_numpy(device)
        np.testing.assert_array_equal(back, arr)

    def test_cpu_backend_roundtrip(self):
        from realsense_mlx.backends.cpu_backend import CPUBackend
        backend = CPUBackend()
        arr = np.array([4, 5, 6], dtype=np.float32)
        device = backend.to_device(arr)
        back = backend.to_numpy(device)
        np.testing.assert_array_equal(back, arr)

    def test_mlx_backend_zeros(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        z = backend.to_numpy(backend.zeros((3, 4), dtype=mx.float32))
        assert z.shape == (3, 4)
        assert np.all(z == 0)

    def test_cpu_backend_clip(self):
        from realsense_mlx.backends.cpu_backend import CPUBackend
        backend = CPUBackend()
        arr = np.array([-10, 0, 128, 300], dtype=np.int32)
        out = backend.clip(arr, 0, 255)
        np.testing.assert_array_equal(out, [0, 0, 128, 255])

    def test_mlx_backend_clip(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        arr = mx.array([-10, 0, 128, 300], dtype=mx.int32)
        out = backend.to_numpy(backend.clip(arr, 0, 255))
        np.testing.assert_array_equal(out, [0, 0, 128, 255])

    def test_mlx_backend_where(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        cond = mx.array([True, False, True])
        x    = mx.array([1, 2, 3])
        y    = mx.array([4, 5, 6])
        out  = backend.to_numpy(backend.where(cond, x, y))
        np.testing.assert_array_equal(out, [1, 5, 3])

    def test_mlx_backend_bitwise_ops(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        a = mx.array([0b1010, 0b1100], dtype=mx.uint8)
        b = mx.array([0b1100, 0b1010], dtype=mx.uint8)
        and_out = backend.to_numpy(backend.bitwise_and(a, b))
        or_out  = backend.to_numpy(backend.bitwise_or(a, b))
        np.testing.assert_array_equal(and_out, [0b1000, 0b1000])
        np.testing.assert_array_equal(or_out,  [0b1110, 0b1110])

    def test_mlx_backend_shifts(self):
        from realsense_mlx.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        a = mx.array([16], dtype=mx.uint16)
        rs = backend.to_numpy(backend.right_shift(a, 4))
        ls = backend.to_numpy(backend.left_shift(a, 4))
        assert int(rs[0]) == 1
        assert int(ls[0]) == 256


# ---------------------------------------------------------------------------
# Tests: FormatConverter class interface
# ---------------------------------------------------------------------------


class TestFormatConverterClass:
    """Smoke-tests verifying the class interface delegates correctly."""

    def test_all_methods_exist(self):
        methods = [
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
        for name in methods:
            assert hasattr(FormatConverter, name), f"Missing method: {name}"

    def test_instance_usable(self):
        """FormatConverter can be instantiated (though all methods are static)."""
        conv = FormatConverter()
        src = make_yuy2()
        result = to_np(conv.yuy2_to_rgb(to_mx(src), WIDTH, HEIGHT))
        assert result.shape == (HEIGHT, WIDTH, 3)
