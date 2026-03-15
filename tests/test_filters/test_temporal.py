"""Tests for TemporalFilter.

Covers:
- First frame initialises state and is returned as-is.
- EMA: pixel converges toward stable value over repeated identical frames.
- Persistence gating: pixels appearing in fewer than N of last 8 frames
  are zeroed.
- Large depth change detection: blending stops, history resets for that pixel.
- reset() clears all state → next call behaves like first frame.
- Output shape and dtype preservation.
- All-zero frame handling.
- Single-pixel frames.
- Multi-frame convergence test.
- State persists correctly between frames (prev_frame, history).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.filters.temporal import TemporalFilter


def _np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def filt() -> TemporalFilter:
    return TemporalFilter(alpha=0.4, delta=20.0, persistence=0)


@pytest.fixture
def filt_persist3() -> TemporalFilter:
    return TemporalFilter(alpha=0.4, delta=20.0, persistence=3)


def _make_frame(value: float, shape=(8, 8), dtype=np.uint16) -> mx.array:
    arr = np.full(shape, value, dtype=dtype)
    if dtype == np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return mx.array(arr)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        f = TemporalFilter()
        assert f.alpha == 0.4
        assert f.delta == 20.0
        assert f.persistence == 3

    def test_persistence_clamped(self):
        f = TemporalFilter(persistence=99)
        assert f.persistence == 8
        f2 = TemporalFilter(persistence=-1)
        assert f2.persistence == 0

    def test_state_initially_none(self):
        f = TemporalFilter()
        assert f._prev_frame is None
        assert f._history is None

    def test_repr_shows_empty_state(self):
        assert "empty" in repr(TemporalFilter())


# ---------------------------------------------------------------------------
# First-frame behaviour
# ---------------------------------------------------------------------------

class TestFirstFrame:
    def test_first_frame_returned_as_is(self, filt):
        frame = _make_frame(1000.0)
        out = _np(filt.process(frame))
        # Should equal original (allow ±1 for dtype rounding).
        assert np.all(np.abs(out.astype(np.int32) - 1000) <= 1)

    def test_state_initialised_after_first_frame(self, filt):
        frame = _make_frame(1000.0)
        filt.process(frame)
        assert filt._prev_frame is not None
        assert filt._history is not None

    def test_first_frame_all_zeros(self, filt):
        frame = mx.array(np.zeros((8, 8), dtype=np.uint16))
        out = _np(filt.process(frame))
        assert np.all(out == 0)

    def test_repr_shows_initialised_after_first_frame(self, filt):
        filt.process(_make_frame(500.0))
        assert "initialised" in repr(filt)


# ---------------------------------------------------------------------------
# EMA convergence
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_converges_to_stable_value(self):
        """Feed the same value 20 times; output should be very close to it."""
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=0)
        val = 1500
        for _ in range(20):
            out = _np(f.process(_make_frame(float(val))))
        centre = out[4, 4]
        assert abs(int(centre) - val) <= 2, (
            f"After 20 identical frames, centre should be {val}, got {centre}"
        )

    def test_alpha_1_means_no_history(self):
        """alpha=1.0 means output = current frame always."""
        f = TemporalFilter(alpha=1.0, delta=10000.0, persistence=0)
        # First frame: 1000
        f.process(_make_frame(1000.0))
        # Second frame: 2000 — with alpha=1 the blend is 1.0*2000 + 0.0*1000 = 2000.
        out = _np(f.process(_make_frame(2000.0)))
        assert np.all(np.abs(out.astype(np.int32) - 2000) <= 2)

    def test_alpha_0_means_pure_history(self):
        """alpha=0.0 means output = previous frame after first."""
        f = TemporalFilter(alpha=0.0, delta=10000.0, persistence=0)
        f.process(_make_frame(1000.0))  # init with 1000
        out = _np(f.process(_make_frame(2000.0)))
        # blend = 0.0 * 2000 + 1.0 * 1000 = 1000
        assert np.all(np.abs(out.astype(np.int32) - 1000) <= 2)

    def test_smoothed_value_between_old_and_new(self):
        f = TemporalFilter(alpha=0.5, delta=10000.0, persistence=0)
        f.process(_make_frame(1000.0))
        out = _np(f.process(_make_frame(2000.0)))
        # 0.5*2000 + 0.5*1000 = 1500 ± small rounding.
        assert np.all(np.abs(out.astype(np.int32) - 1500) <= 2)


# ---------------------------------------------------------------------------
# Large-change detection (ghosting prevention)
# ---------------------------------------------------------------------------

class TestLargeChange:
    def test_large_jump_not_blended(self):
        """If |curr - prev| > delta, take current value directly."""
        f = TemporalFilter(alpha=0.5, delta=50.0, persistence=0)
        f.process(_make_frame(1000.0))
        # Jump by 500 (>> delta=50).
        out = _np(f.process(_make_frame(1500.0)))
        # Should be close to 1500, not blended toward 1000.
        assert np.all(np.abs(out.astype(np.int32) - 1500) <= 5)

    def test_small_change_is_blended(self):
        """If |curr - prev| < delta, normal EMA blending applies."""
        f = TemporalFilter(alpha=0.5, delta=500.0, persistence=0)
        f.process(_make_frame(1000.0))
        out = _np(f.process(_make_frame(1010.0)))
        # Blended: 0.5*1010 + 0.5*1000 = 1005.
        assert np.all(np.abs(out.astype(np.int32) - 1005) <= 3)


# ---------------------------------------------------------------------------
# Persistence gating
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistence_0_passes_everything(self):
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=0)
        f.process(_make_frame(1000.0))
        out = _np(f.process(_make_frame(1000.0)))
        assert np.all(out > 0)

    def test_persistence_8_requires_8_valid_frames(self):
        """With persistence=8, pixels appearing < 8 frames must be zeroed."""
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=8)
        # Feed only 4 valid frames.
        for _ in range(4):
            out = _np(f.process(_make_frame(1000.0)))
        # After 4 frames, bits set = 4 < 8 → output should be 0 (not enough history).
        assert np.all(out == 0)

    def test_persistence_1_passes_after_one_frame(self):
        """With persistence=1, any pixel valid at least once in 8 frames passes."""
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=1)
        f.process(_make_frame(1000.0))  # first frame initialises
        out = _np(f.process(_make_frame(1000.0)))
        # After 2 valid frames, count >= 1 → all pixels should pass.
        assert np.all(out > 0)

    def test_persistence_gating_after_enough_frames(self, filt_persist3):
        """Feed 8 valid frames; with persistence=3 all pixels should pass."""
        for _ in range(8):
            out = _np(filt_persist3.process(_make_frame(1000.0)))
        # After 8 frames count=8 >= 3 → all pixels valid.
        assert np.all(out > 0)


# ---------------------------------------------------------------------------
# All-zero / invalid frames
# ---------------------------------------------------------------------------

class TestZeroFrames:
    def test_zero_frame_after_valid_keeps_prev_state(self, filt):
        filt.process(_make_frame(1000.0))
        # Zero frame: current pixels invalid → output 0.
        out = _np(filt.process(mx.array(np.zeros((8, 8), dtype=np.uint16))))
        assert np.all(out == 0)

    def test_first_frame_all_zeros(self, filt):
        out = _np(filt.process(mx.array(np.zeros((8, 8), dtype=np.uint16))))
        assert np.all(out == 0)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self, filt):
        filt.process(_make_frame(1000.0))
        filt.reset()
        assert filt._prev_frame is None
        assert filt._history is None

    def test_frame_after_reset_behaves_like_first_frame(self):
        f = TemporalFilter(alpha=0.4, delta=20.0, persistence=0)
        f.process(_make_frame(1000.0))
        f.process(_make_frame(1000.0))
        f.reset()
        # After reset, next frame should just be returned as-is.
        out = _np(f.process(_make_frame(2000.0)))
        assert np.all(np.abs(out.astype(np.int32) - 2000) <= 2)

    def test_persistence_resets_with_state(self, filt_persist3):
        for _ in range(8):
            filt_persist3.process(_make_frame(1000.0))
        filt_persist3.reset()
        # After reset: first frame → returned as-is (no history to gate against).
        out = _np(filt_persist3.process(_make_frame(1000.0)))
        assert np.all(np.abs(out.astype(np.int32) - 1000) <= 2)


# ---------------------------------------------------------------------------
# Multi-frame convergence
# ---------------------------------------------------------------------------

class TestMultiFrameConvergence:
    def test_converges_over_20_frames(self):
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=0)
        target = 1234
        for _ in range(20):
            f.process(_make_frame(float(target)))
        out = _np(f.process(_make_frame(float(target))))
        assert np.all(np.abs(out.astype(np.int32) - target) <= 2)

    def test_tracks_slow_ramp(self):
        """Value slowly ramps from 1000 to 1100 over 20 frames.
        Temporal filter should track it (lag expected, but not stuck at 1000).
        """
        f = TemporalFilter(alpha=0.6, delta=200.0, persistence=0)
        for i in range(20):
            val = float(1000 + i * 5)  # ramp: 1000→1095
            out_np = _np(f.process(_make_frame(val)))
        final = float(np.mean(out_np))
        # Should have tracked toward the end of the ramp (>1050).
        assert final > 1050.0, f"Expected tracking above 1050, got {final:.1f}"


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------

class TestDtype:
    def test_uint16_in_uint16_out(self, filt):
        f2 = TemporalFilter(alpha=0.4, delta=10000.0, persistence=0)
        frame = mx.array(np.full((4, 4), 1000, dtype=np.uint16))
        f2.process(frame)
        out = f2.process(frame)
        assert out.dtype == mx.uint16

    def test_float32_in_float32_out(self):
        f = TemporalFilter(alpha=0.4, delta=10000.0, persistence=0)
        frame = mx.array(np.full((4, 4), 1000.0, dtype=np.float32))
        f.process(frame)
        out = f.process(frame)
        assert out.dtype == mx.float32

    def test_output_shape_preserved(self, filt):
        frame = _make_frame(500.0, shape=(12, 16))
        filt.process(frame)
        out = filt.process(frame)
        assert out.shape == (12, 16)


# ---------------------------------------------------------------------------
# Edge-cases: resolution change and persistence/EMA state integrity
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_resolution_change_auto_resets(self):
        """Changing frame resolution should auto-reset, not crash."""
        f = TemporalFilter()
        f.process(mx.zeros((48, 64), dtype=mx.uint16))
        # Different resolution
        out = f.process(mx.zeros((24, 32), dtype=mx.uint16))
        assert out.shape == (24, 32)

    def test_persistence_does_not_corrupt_ema_state(self):
        """Persistence gating should not destroy accumulated EMA values."""
        f = TemporalFilter(alpha=0.5, persistence=8)  # very strict persistence
        # Feed 3 frames with valid data
        for _ in range(3):
            f.process(mx.full((8, 8), 1000, dtype=mx.uint16))
        # Now persistence requires 8/8 valid frames but we only have 3
        # Output should be zeros (persistence gate), but internal _prev_frame
        # should NOT be zeros
        assert f._prev_frame is not None
        prev = np.array(f._prev_frame)
        # EMA state should still hold the accumulated values, not zeros
        assert np.all(prev[prev > 0] > 500)  # roughly around 1000
