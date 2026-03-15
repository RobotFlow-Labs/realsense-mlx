"""Tests for FrameRecorder and FramePlayer.

All tests run without a physical camera; they exercise the recorder/player
logic directly using synthetic MLX arrays.

Test matrix
-----------
- Record N frames → verify .npz files and metadata.json are created.
- Playback → frames match recorded data exactly (MLX array round-trip).
- Metadata (intrinsics, depth_scale) is preserved through a full cycle.
- Seek to specific frame.
- Iterator protocol iterates all frames then stops.
- Empty recording (stop() with no add_frame calls) is handled gracefully.
- MLX array round-trip produces no data corruption (uint16, uint8 checked).
- start() when already active raises RuntimeError.
- add_frame() before start() raises RuntimeError.
- stop() before start() raises RuntimeError.
- FramePlayer.open() on missing directory raises FileNotFoundError.
- FramePlayer.next_frame() past end raises StopIteration.
- seek() with negative index raises ValueError.
- context-manager protocol opens/closes the player.
- color_intrinsics=None (depth-only recording) is preserved correctly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.capture.recorder import FrameRecorder, FramePlayer
from realsense_mlx.geometry.intrinsics import CameraIntrinsics


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def depth_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640,
        height=480,
        ppx=318.8,
        ppy=239.5,
        fx=383.7,
        fy=383.7,
        model="none",
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture
def color_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640,
        height=480,
        ppx=320.0,
        ppy=240.0,
        fx=615.0,
        fy=615.0,
        model="none",
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture
def depth_frame() -> mx.array:
    """Synthetic 480x640 uint16 depth frame with known, unique values."""
    rng = np.random.default_rng(42)
    arr = rng.integers(100, 8000, size=(480, 640), dtype=np.uint16)
    return mx.array(arr)


@pytest.fixture
def color_frame() -> mx.array:
    """Synthetic 480x640x3 uint8 colour frame."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    return mx.array(arr)


@pytest.fixture
def recording_dir(tmp_path: Path) -> Path:
    return tmp_path / "recording_001"


# ---------------------------------------------------------------------------
# Helper: record N frames and return (recorder summary, dir path)
# ---------------------------------------------------------------------------


def _record_n_frames(
    n: int,
    recording_dir: Path,
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics | None = None,
    depth_scale: float = 0.001,
    include_color: bool = True,
) -> dict:
    """Record *n* distinct depth (and optionally colour) frames, return summary."""
    recorder = FrameRecorder(str(recording_dir))
    recorder.start(depth_intrinsics, color_intrinsics, depth_scale=depth_scale)

    rng = np.random.default_rng(0)
    for i in range(n):
        depth_np = rng.integers(100, 9000, size=(480, 640), dtype=np.uint16)
        depth = mx.array(depth_np)
        color = None
        if include_color:
            color_np = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
            color = mx.array(color_np)
        recorder.add_frame(depth, color, timestamp=float(i * 33.3))

    return recorder.stop()


# ---------------------------------------------------------------------------
# FrameRecorder — basic lifecycle
# ---------------------------------------------------------------------------


class TestFrameRecorderLifecycle:
    def test_start_creates_directory(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        recorder = FrameRecorder(str(recording_dir))
        assert not recording_dir.exists()

        recorder.start(depth_intrinsics)
        assert recording_dir.is_dir()
        recorder.stop()

    def test_is_recording_flag(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        recorder = FrameRecorder(str(recording_dir))
        assert recorder.is_recording is False
        recorder.start(depth_intrinsics)
        assert recorder.is_recording is True
        recorder.stop()
        assert recorder.is_recording is False

    def test_double_start_raises(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        with pytest.raises(RuntimeError, match="already active"):
            recorder.start(depth_intrinsics)
        recorder.stop()

    def test_add_frame_before_start_raises(
        self, recording_dir: Path, depth_frame: mx.array
    ):
        recorder = FrameRecorder(str(recording_dir))
        with pytest.raises(RuntimeError, match="not started"):
            recorder.add_frame(depth_frame)

    def test_stop_before_start_raises(self, recording_dir: Path):
        recorder = FrameRecorder(str(recording_dir))
        with pytest.raises(RuntimeError, match="No active recording"):
            recorder.stop()

    def test_stop_returns_summary_dict(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        summary = recorder.stop()

        assert isinstance(summary, dict)
        assert "output_dir" in summary
        assert "frame_count" in summary
        assert "depth_scale" in summary
        assert "duration_s" in summary


# ---------------------------------------------------------------------------
# FrameRecorder — files on disk
# ---------------------------------------------------------------------------


class TestFrameRecorderFiles:
    def test_five_frames_creates_five_npz_files(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        color_intrinsics: CameraIntrinsics,
    ):
        summary = _record_n_frames(
            5, recording_dir, depth_intrinsics, color_intrinsics
        )

        assert summary["frame_count"] == 5
        npz_files = sorted(recording_dir.glob("frame_*.npz"))
        assert len(npz_files) == 5
        # Names must be zero-padded 6-digit indices.
        names = [f.name for f in npz_files]
        assert names == [f"frame_{i:06d}.npz" for i in range(5)]

    def test_metadata_json_created(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(3, recording_dir, depth_intrinsics)

        metadata_path = recording_dir / "metadata.json"
        assert metadata_path.exists()

        meta = json.loads(metadata_path.read_text())
        assert meta["frame_count"] == 3

    def test_empty_recording_creates_metadata_with_zero_frames(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        summary = recorder.stop()

        assert summary["frame_count"] == 0
        meta = json.loads((recording_dir / "metadata.json").read_text())
        assert meta["frame_count"] == 0
        assert meta["timestamps"] == []

    def test_depth_only_recording_no_color_key_in_npz(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)  # no color
        recorder.stop()

        data = np.load(str(recording_dir / "frame_000000.npz"))
        assert "depth" in data
        assert "color" not in data

    def test_add_frame_bad_depth_ndim_raises(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        bad_depth = mx.zeros((480, 640, 1), dtype=mx.uint16)
        with pytest.raises(ValueError, match="2-D"):
            recorder.add_frame(bad_depth)
        recorder.stop()

    def test_add_frame_bad_color_shape_raises(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        bad_color = mx.zeros((480, 640, 4), dtype=mx.uint8)  # 4 channels
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            recorder.add_frame(depth_frame, bad_color)
        recorder.stop()


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    def test_depth_intrinsics_round_trip(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)

        with FramePlayer(str(recording_dir)) as player:
            restored = player.intrinsics

        assert restored.width == depth_intrinsics.width
        assert restored.height == depth_intrinsics.height
        assert restored.fx == pytest.approx(depth_intrinsics.fx)
        assert restored.fy == pytest.approx(depth_intrinsics.fy)
        assert restored.ppx == pytest.approx(depth_intrinsics.ppx)
        assert restored.ppy == pytest.approx(depth_intrinsics.ppy)
        assert restored.model == depth_intrinsics.model
        assert restored.coeffs == pytest.approx(depth_intrinsics.coeffs)

    def test_color_intrinsics_round_trip(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        color_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics, color_intrinsics)

        with FramePlayer(str(recording_dir)) as player:
            restored = player.color_intrinsics

        assert restored is not None
        assert restored.fx == pytest.approx(color_intrinsics.fx)
        assert restored.fy == pytest.approx(color_intrinsics.fy)

    def test_color_intrinsics_none_when_not_recorded(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics, color_intrinsics=None)

        with FramePlayer(str(recording_dir)) as player:
            assert player.color_intrinsics is None

    def test_depth_scale_round_trip(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics, depth_scale=0.00025)

        with FramePlayer(str(recording_dir)) as player:
            assert player.depth_scale == pytest.approx(0.00025)

    def test_frame_count_in_player_matches_recorder(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(7, recording_dir, depth_intrinsics)

        with FramePlayer(str(recording_dir)) as player:
            assert player.frame_count == 7


# ---------------------------------------------------------------------------
# MLX array round-trip (data integrity)
# ---------------------------------------------------------------------------


class TestMLXArrayRoundTrip:
    def test_depth_uint16_exact_match(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        rng = np.random.default_rng(1337)
        original_np = rng.integers(0, 65535, size=(480, 640), dtype=np.uint16)
        depth_mx = mx.array(original_np)

        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_mx, timestamp=1.0)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            depth_back, _, _ = player.next_frame()

        recovered_np = np.array(depth_back)
        np.testing.assert_array_equal(recovered_np, original_np)

    def test_color_uint8_exact_match(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        rng = np.random.default_rng(99)
        original_np = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        color_mx = mx.array(original_np)

        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame, color_mx, timestamp=0.0)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            _, color_back, _ = player.next_frame()

        assert color_back is not None
        recovered_np = np.array(color_back)
        np.testing.assert_array_equal(recovered_np, original_np)

    def test_timestamp_preserved(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        ts = 123456.789

        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame, timestamp=ts)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            _, _, ts_back = player.next_frame()

        assert ts_back == pytest.approx(ts, abs=1e-9)

    def test_multiple_frames_data_identity(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        """Each frame must recover its unique data with no cross-contamination."""
        n = 5
        rng = np.random.default_rng(2024)
        originals = [
            rng.integers(100, 9000, size=(480, 640), dtype=np.uint16)
            for _ in range(n)
        ]

        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        for arr in originals:
            recorder.add_frame(mx.array(arr), timestamp=0.0)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            for i, original in enumerate(originals):
                depth_back, _, _ = player.next_frame()
                np.testing.assert_array_equal(
                    np.array(depth_back),
                    original,
                    err_msg=f"Frame {i} data mismatch",
                )

    def test_depth_only_playback_color_is_none(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)  # no color
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            _, color, _ = player.next_frame()

        assert color is None


# ---------------------------------------------------------------------------
# FramePlayer — seek
# ---------------------------------------------------------------------------


class TestFramePlayerSeek:
    def test_seek_and_read_specific_frame(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        n = 5
        rng = np.random.default_rng(55)
        originals = [
            rng.integers(0, 65535, size=(480, 640), dtype=np.uint16)
            for _ in range(n)
        ]

        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        for arr in originals:
            recorder.add_frame(mx.array(arr), timestamp=float(len(originals)))
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            player.seek(3)
            assert player.current_index == 3
            depth_back, _, _ = player.next_frame()

        np.testing.assert_array_equal(np.array(depth_back), originals[3])

    def test_seek_to_zero_resets_cursor(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame, timestamp=0.0)
        recorder.add_frame(depth_frame, timestamp=1.0)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            player.next_frame()
            assert player.current_index == 1
            player.seek(0)
            assert player.current_index == 0

    def test_seek_beyond_end_clamps_to_frame_count(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            player.seek(999)
            assert player.current_index == player.frame_count

    def test_seek_negative_raises_value_error(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            with pytest.raises(ValueError, match=">= 0"):
                player.seek(-1)


# ---------------------------------------------------------------------------
# FramePlayer — iterator protocol
# ---------------------------------------------------------------------------


class TestFramePlayerIterator:
    def test_iterator_yields_all_frames(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(5, recording_dir, depth_intrinsics)

        frames = []
        with FramePlayer(str(recording_dir)) as player:
            for depth, color, ts in player:
                frames.append((depth, color, ts))

        assert len(frames) == 5

    def test_iterator_from_mid_cursor(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(5, recording_dir, depth_intrinsics)

        frames = []
        with FramePlayer(str(recording_dir)) as player:
            player.seek(3)
            for item in player:
                frames.append(item)

        assert len(frames) == 2  # frames 3 and 4

    def test_iterator_on_empty_recording(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.stop()

        frames = []
        with FramePlayer(str(recording_dir)) as player:
            for item in player:
                frames.append(item)

        assert frames == []

    def test_has_frames_false_after_all_consumed(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            assert player.has_frames() is True
            player.next_frame()
            assert player.has_frames() is False

    def test_next_frame_past_end_raises_stop_iteration(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
        depth_frame: mx.array,
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        recorder.add_frame(depth_frame)
        recorder.stop()

        with FramePlayer(str(recording_dir)) as player:
            player.next_frame()
            with pytest.raises(StopIteration):
                player.next_frame()


# ---------------------------------------------------------------------------
# FramePlayer — error conditions
# ---------------------------------------------------------------------------


class TestFramePlayerErrors:
    def test_open_missing_directory_raises(self, tmp_path: Path):
        player = FramePlayer(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            player.open()

    def test_open_missing_metadata_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_rec"
        empty_dir.mkdir()
        player = FramePlayer(str(empty_dir))
        with pytest.raises(FileNotFoundError, match="metadata"):
            player.open()

    def test_access_before_open_raises_runtime_error(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)
        player = FramePlayer(str(recording_dir))
        with pytest.raises(RuntimeError, match="not open"):
            player.frame_count  # noqa: B018

    def test_has_frames_before_open_raises(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)
        player = FramePlayer(str(recording_dir))
        with pytest.raises(RuntimeError, match="not open"):
            player.has_frames()

    def test_seek_before_open_raises(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)
        player = FramePlayer(str(recording_dir))
        with pytest.raises(RuntimeError, match="not open"):
            player.seek(0)

    def test_malformed_metadata_missing_key_raises(self, tmp_path: Path):
        bad_dir = tmp_path / "bad_rec"
        bad_dir.mkdir()
        (bad_dir / "metadata.json").write_text(
            json.dumps({"frame_count": 1})  # missing depth_scale, depth_intrinsics
        )
        player = FramePlayer(str(bad_dir))
        with pytest.raises(ValueError, match="missing required key"):
            player.open()


# ---------------------------------------------------------------------------
# FramePlayer — context-manager protocol
# ---------------------------------------------------------------------------


class TestFramePlayerContextManager:
    def test_context_manager_opens_and_closes(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(2, recording_dir, depth_intrinsics)

        player = FramePlayer(str(recording_dir))
        assert player._opened is False

        with player:
            assert player._opened is True
            assert player.frame_count == 2

        assert player._opened is False

    def test_context_manager_closes_on_exception(
        self,
        recording_dir: Path,
        depth_intrinsics: CameraIntrinsics,
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)

        player = FramePlayer(str(recording_dir))
        try:
            with player:
                assert player._opened is True
                raise ValueError("test exception")
        except ValueError:
            pass

        assert player._opened is False


# ---------------------------------------------------------------------------
# FrameRecorder — repr
# ---------------------------------------------------------------------------


class TestReprs:
    def test_recorder_repr_idle(self, recording_dir: Path):
        recorder = FrameRecorder(str(recording_dir))
        r = repr(recorder)
        assert "idle" in r

    def test_recorder_repr_recording(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        recorder = FrameRecorder(str(recording_dir))
        recorder.start(depth_intrinsics)
        r = repr(recorder)
        assert "recording" in r
        recorder.stop()

    def test_player_repr_closed(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        _record_n_frames(1, recording_dir, depth_intrinsics)
        player = FramePlayer(str(recording_dir))
        r = repr(player)
        assert "closed" in r

    def test_player_repr_open(
        self, recording_dir: Path, depth_intrinsics: CameraIntrinsics
    ):
        _record_n_frames(3, recording_dir, depth_intrinsics)
        with FramePlayer(str(recording_dir)) as player:
            r = repr(player)
        assert "0/3" in r
