"""Tests for SpatialFilter (Domain Transform edge-preserving filter).

Strategy
--------
We use small frames (e.g. 8x8, 16x16) to keep the sequential column loop
fast in tests while still exercising correctness.

The parametrize fixture ``filter_backend`` runs every test against both
the Metal kernel path (use_metal=True) and the pure-Python MLX loop
(use_metal=False) so that we verify:
  - Both paths produce equivalent numerical results (atol=1e-4).
  - All behavioural invariants hold for each path independently.

Covers:
- Output shape and dtype preservation.
- Uniform frame stays (approximately) unchanged.
- Smoothing of noisy uniform region (variance drops).
- Edge preservation: a hard depth discontinuity should not bleed across.
- All-zero frame stays zero.
- Single-pixel frame.
- iterations=1 vs iterations=3 (more iterations = more smoothing).
- hole_fill integration (valid hole_fill modes).
- Parameter bounds (iterations clamped to [1,5]).
- Metal vs Python numerical equivalence on varied frames.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.spatial import SpatialFilter


def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[True, False], ids=["metal", "python"])
def filter_backend(request) -> bool:
    """Parametrize over Metal (True) and Python (False) backends."""
    return request.param


@pytest.fixture
def default_filter(filter_backend) -> SpatialFilter:
    return SpatialFilter(alpha=0.5, delta=20.0, iterations=2, use_metal=filter_backend)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        f = SpatialFilter()
        assert f.alpha == 0.5
        assert f.delta == 20.0
        assert f.iterations == 2
        assert f.hole_fill == 0
        assert f.use_metal is True

    def test_use_metal_false(self):
        f = SpatialFilter(use_metal=False)
        assert f.use_metal is False

    def test_iterations_clamped_low(self):
        f = SpatialFilter(iterations=0)
        assert f.iterations == 1

    def test_iterations_clamped_high(self):
        f = SpatialFilter(iterations=99)
        assert f.iterations == 5

    def test_repr_contains_use_metal(self):
        r = repr(SpatialFilter(alpha=0.3, delta=15.0, use_metal=True))
        assert "SpatialFilter" in r
        assert "alpha=0.3" in r
        assert "use_metal=True" in r

    def test_repr_python_backend(self):
        r = repr(SpatialFilter(use_metal=False))
        assert "use_metal=False" in r


# ---------------------------------------------------------------------------
# Shape / dtype contracts
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_shape_preserved_uint16(self, default_filter):
        inp = mx.array(np.ones((16, 16), dtype=np.uint16) * 1000)
        out = default_filter.process(inp)
        assert out.shape == (16, 16)
        assert out.dtype == mx.uint16

    def test_shape_preserved_float32(self, default_filter):
        inp = mx.array(np.ones((16, 16), dtype=np.float32) * 1000.0)
        out = default_filter.process(inp)
        assert out.shape == (16, 16)
        assert out.dtype == mx.float32

    def test_single_pixel_valid(self, default_filter):
        inp = mx.array(np.array([[500]], dtype=np.float32))
        out = _np(default_filter.process(inp))
        assert out.shape == (1, 1)
        assert abs(out[0, 0] - 500.0) < 1.0

    def test_single_pixel_zero(self, default_filter):
        inp = mx.array(np.array([[0]], dtype=np.uint16))
        out = _np(default_filter.process(inp))
        assert out[0, 0] == 0


# ---------------------------------------------------------------------------
# All-zero frame
# ---------------------------------------------------------------------------

class TestAllZero:
    def test_all_zero_stays_zero(self, default_filter):
        inp = mx.array(np.zeros((8, 8), dtype=np.uint16))
        out = _np(default_filter.process(inp))
        assert np.all(out == 0)

    def test_all_zero_float32(self, default_filter):
        inp = mx.array(np.zeros((8, 8), dtype=np.float32))
        out = _np(default_filter.process(inp))
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# Uniform frame — should be (approximately) preserved
# ---------------------------------------------------------------------------

class TestUniformFrame:
    def test_uniform_frame_unchanged(self, default_filter):
        """A perfectly uniform frame has zero difference everywhere,
        so weights are 1.0 and the output is the blended value of
        identical neighbours — still the same value."""
        val = 1500.0
        inp = mx.array(np.full((12, 12), val, dtype=np.float32))
        out = _np(default_filter.process(inp))
        # Allow small floating-point drift.
        assert np.allclose(out, val, atol=1.0)

    def test_uniform_uint16_unchanged(self, default_filter):
        inp = mx.array(np.full((12, 12), 800, dtype=np.uint16))
        out = _np(default_filter.process(inp))
        diff = np.abs(out.astype(np.int32) - 800)
        assert np.max(diff) <= 1


# ---------------------------------------------------------------------------
# Noise reduction on flat region
# ---------------------------------------------------------------------------

class TestNoise:
    def test_smoothing_reduces_variance(self, filter_backend):
        f = SpatialFilter(alpha=0.5, delta=500.0, iterations=3, use_metal=filter_backend)
        rng = np.random.default_rng(0)
        # Flat region at ~1000 with +/-50 noise.
        base = np.full((16, 16), 1000.0, dtype=np.float32)
        noise = rng.uniform(-50, 50, (16, 16)).astype(np.float32)
        frame = (base + noise).clip(1.0, 65535.0).astype(np.float32)
        inp = mx.array(frame)
        out = _np(f.process(inp))
        # Valid pixels only.
        var_before = np.var(frame)
        var_after  = np.var(out[out > 0])
        assert var_after < var_before, (
            f"Variance should decrease: before={var_before:.1f} after={var_after:.1f}"
        )


# ---------------------------------------------------------------------------
# Edge preservation
# ---------------------------------------------------------------------------

class TestEdgePreservation:
    def test_hard_edge_not_crossed(self, filter_backend):
        """Create a 2-zone depth image (left=500, right=3000).
        With small delta the filter should not blend across the edge.
        """
        f = SpatialFilter(alpha=0.5, delta=20.0, iterations=2, use_metal=filter_backend)
        frame = np.zeros((8, 8), dtype=np.float32)
        frame[:, :4] = 500.0
        frame[:, 4:] = 3000.0
        inp = mx.array(frame)
        out = _np(f.process(inp))

        # Left zone should remain close to 500, right zone close to 3000.
        left_mean  = np.mean(out[:, :3])   # avoid the boundary column
        right_mean = np.mean(out[:, 5:])   # avoid the boundary column
        assert left_mean < 1000.0, f"Left mean {left_mean:.1f} should stay near 500"
        assert right_mean > 2000.0, f"Right mean {right_mean:.1f} should stay near 3000"


# ---------------------------------------------------------------------------
# More iterations = more smoothing
# ---------------------------------------------------------------------------

class TestIterations:
    def test_more_iterations_more_smoothing(self, filter_backend):
        rng = np.random.default_rng(11)
        base = np.full((16, 16), 1000.0, dtype=np.float32)
        noise = rng.uniform(-100, 100, (16, 16)).astype(np.float32)
        frame = (base + noise).clip(1.0, 65535.0).astype(np.float32)

        f1 = SpatialFilter(alpha=0.5, delta=500.0, iterations=1, use_metal=filter_backend)
        f3 = SpatialFilter(alpha=0.5, delta=500.0, iterations=3, use_metal=filter_backend)

        out1 = _np(f1.process(mx.array(frame)))
        out3 = _np(f3.process(mx.array(frame)))

        var1 = np.var(out1)
        var3 = np.var(out3)
        assert var3 <= var1, (
            f"3 iterations should smooth more: var1={var1:.2f} var3={var3:.2f}"
        )


# ---------------------------------------------------------------------------
# hole_fill integration
# ---------------------------------------------------------------------------

class TestHoleFillIntegration:
    def test_hole_fill_mode1_fills_after_spatial(self, filter_backend):
        f = SpatialFilter(
            alpha=0.5, delta=20.0, iterations=1, hole_fill=1, use_metal=filter_backend
        )
        # Frame with a hole surrounded by valid pixels.
        data = np.full((5, 5), 1000.0, dtype=np.float32)
        data[2, 2] = 0.0  # centre hole
        inp = mx.array(data)
        out = _np(f.process(inp))
        # After spatial + hole fill, centre should be non-zero.
        assert out[2, 2] > 0.0

    def test_hole_fill_0_disabled(self, filter_backend):
        f = SpatialFilter(
            alpha=0.5, delta=20.0, iterations=1, hole_fill=0, use_metal=filter_backend
        )
        # Isolated hole far from valid pixels — spatial alone won't fill it
        # if surrounded by zeros.
        data = np.zeros((5, 5), dtype=np.float32)
        data[0, 0] = 1000.0
        inp = mx.array(data)
        out = _np(f.process(inp))
        # Centre pixel still 0 (no nearby valid source, no hole filling).
        assert out[2, 2] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Metal vs Python numerical equivalence
# ---------------------------------------------------------------------------

class TestMetalPythonEquivalence:
    """Verify that Metal kernel and Python loop produce identical results."""

    def _run_both(self, data: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        f_metal  = SpatialFilter(use_metal=True,  **kwargs)
        f_python = SpatialFilter(use_metal=False, **kwargs)
        out_metal  = _np(f_metal.process(mx.array(data)))
        out_python = _np(f_python.process(mx.array(data)))
        return out_metal, out_python

    def test_random_frame_iter1(self):
        rng = np.random.default_rng(100)
        data = rng.uniform(400, 800, (16, 16)).astype(np.float32)
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=1)
        assert np.allclose(m, p, atol=1e-4), (
            f"Max diff: {np.max(np.abs(m - p)):.2e}"
        )

    def test_random_frame_iter3(self):
        rng = np.random.default_rng(200)
        data = rng.uniform(400, 800, (16, 16)).astype(np.float32)
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=3)
        assert np.allclose(m, p, atol=1e-4), (
            f"Max diff: {np.max(np.abs(m - p)):.2e}"
        )

    def test_frame_with_invalid_pixels(self):
        rng = np.random.default_rng(300)
        data = rng.uniform(400, 800, (16, 16)).astype(np.float32)
        data[3:6, 3:6] = 0.0   # block of invalid pixels
        data[10, 10]   = 0.0   # single invalid pixel
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=2)
        assert np.allclose(m, p, atol=1e-4), (
            f"Max diff: {np.max(np.abs(m - p)):.2e}"
        )

    def test_hard_edge_frame(self):
        """Left=500, right=3000 — strong edge, small delta."""
        data = np.zeros((8, 16), dtype=np.float32)
        data[:, :8]  = 500.0
        data[:, 8:]  = 3000.0
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=2)
        assert np.allclose(m, p, atol=1e-4), (
            f"Max diff: {np.max(np.abs(m - p)):.2e}"
        )

    def test_larger_frame(self):
        """Larger frame to exercise multi-row parallelism in the Metal kernel."""
        rng = np.random.default_rng(400)
        data = rng.uniform(300, 900, (48, 64)).astype(np.float32)
        data[10:20, 20:40] = 0.0   # region of invalids
        m, p = self._run_both(data, alpha=0.5, delta=30.0, iterations=2)
        assert np.allclose(m, p, atol=1e-4), (
            f"Max diff: {np.max(np.abs(m - p)):.2e}"
        )

    def test_all_same_value(self):
        """Uniform frame: Metal and Python must agree exactly."""
        data = np.full((12, 12), 1500.0, dtype=np.float32)
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=2)
        assert np.allclose(m, p, atol=1e-4)

    def test_all_zero_frame(self):
        """Zero frame: both paths must return zero."""
        data = np.zeros((10, 10), dtype=np.float32)
        m, p = self._run_both(data, alpha=0.5, delta=20.0, iterations=2)
        assert np.all(m == 0.0)
        assert np.all(p == 0.0)

    def test_uint16_dtype_roundtrip(self):
        """Both paths must preserve uint16 dtype end-to-end."""
        data = np.full((8, 8), 1000, dtype=np.uint16)
        f_metal  = SpatialFilter(use_metal=True,  alpha=0.5, delta=20.0, iterations=1)
        f_python = SpatialFilter(use_metal=False, alpha=0.5, delta=20.0, iterations=1)
        out_m = f_metal.process(mx.array(data))
        out_p = f_python.process(mx.array(data))
        assert out_m.dtype == mx.uint16
        assert out_p.dtype == mx.uint16
        m_np = _np(out_m).astype(np.int32)
        p_np = _np(out_p).astype(np.int32)
        assert np.max(np.abs(m_np - p_np)) <= 1, (
            f"uint16 round-trip max diff: {np.max(np.abs(m_np - p_np))}"
        )
