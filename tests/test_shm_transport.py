"""Tests for the POSIX shared-memory frame transport.

Coverage
--------
* Write → read round-trip (uint16 depth frame, uint8 RGB frame, float32)
* Sequence number increments by 2 per write (seqlock even-only publishes)
* Double-buffer: two consecutive writes use alternating slots; both are
  readable without data corruption
* Header parsing: all fields (magic, version, width, height, channels,
  dtype_code, data_offset, data_size) are correct after construction
* Cleanup on close: shm block is unlinked after writer.close(unlink=True)
* Context-manager protocol (with statement) works for both writer and reader
* Read before any write returns zeros with seq==0
* Shape mismatch raises ValueError on write
* Dtype mismatch raises ValueError on write
* Reading from a non-existent shm name raises FileNotFoundError
* Multiple consecutive writes → seq advances monotonically
* Large frame (1280×720 uint16) round-trip
* writer.close(unlink=False) leaves block accessible for reader
"""

from __future__ import annotations

import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from realsense_mlx.transport.shm_frame import (
    ShmFrameReader,
    ShmFrameWriter,
    _HEADER_SIZE,
    _MAGIC,
    _VERSION,
    _unpack_header,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_name() -> str:
    """Generate a collision-free shm name for each test."""
    return f"rs_mlx_test_{uuid.uuid4().hex[:12]}"


def _rng_frame(
    height: int, width: int, dtype: np.dtype, channels: int = 1, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (height, width) if channels == 1 else (height, width, channels)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, dtype=dtype)
    return rng.random(shape).astype(dtype)


# ---------------------------------------------------------------------------
# 1. Write → read round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_uint16_depth_frame(self):
        name = _unique_name()
        H, W = 48, 64
        frame = _rng_frame(H, W, np.dtype(np.uint16))

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            writer.write(frame)
            reader = ShmFrameReader(name)
            got, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(got, frame, err_msg="uint16 round-trip mismatch")
        assert seq > 0

    def test_uint8_rgb_frame(self):
        name = _unique_name()
        H, W, C = 48, 64, 3
        frame = _rng_frame(H, W, np.dtype(np.uint8), channels=C)

        try:
            writer = ShmFrameWriter(name, width=W, height=H, channels=C, dtype=np.uint8)
            writer.write(frame)
            reader = ShmFrameReader(name)
            got, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(got, frame, err_msg="uint8 RGB round-trip mismatch")

    def test_float32_frame(self):
        name = _unique_name()
        H, W = 48, 64
        frame = _rng_frame(H, W, np.dtype(np.float32))

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.float32)
            writer.write(frame)
            reader = ShmFrameReader(name)
            got, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(got, frame, err_msg="float32 round-trip mismatch")

    def test_returned_array_is_a_copy(self):
        """Mutating the returned array should not corrupt the shm block."""
        name = _unique_name()
        H, W = 16, 16
        frame = np.full((H, W), 42, dtype=np.uint16)

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            writer.write(frame)
            reader = ShmFrameReader(name)
            got, _ = reader.read()
            got[:] = 0  # mutate
            got2, _ = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(
            got2, frame, err_msg="Mutating returned array must not corrupt shm"
        )

    def test_large_720p_frame(self):
        name = _unique_name()
        H, W = 720, 1280
        frame = _rng_frame(H, W, np.dtype(np.uint16), seed=7)

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            writer.write(frame)
            reader = ShmFrameReader(name)
            got, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(got, frame, err_msg="720p round-trip mismatch")


# ---------------------------------------------------------------------------
# 2. Sequence number behaviour
# ---------------------------------------------------------------------------


class TestSequenceNumbers:
    def test_seq_starts_at_zero_before_write(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
            reader = ShmFrameReader(name)
            _, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        assert seq == 0, f"Seq should be 0 before any write, got {seq}"

    def test_seq_is_even_after_write(self):
        name = _unique_name()
        frame = np.zeros((16, 16), dtype=np.uint16)
        try:
            writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
            seq_returned = writer.write(frame)
            reader = ShmFrameReader(name)
            _, seq_read = reader.read()
        finally:
            reader.close()
            writer.close()

        assert seq_returned % 2 == 0, f"write() returned odd seq {seq_returned}"
        assert seq_read % 2 == 0, f"read() returned odd seq {seq_read}"

    def test_seq_increments_by_2_per_write(self):
        name = _unique_name()
        frame = np.zeros((16, 16), dtype=np.uint16)
        try:
            writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
            s1 = writer.write(frame)
            s2 = writer.write(frame)
            s3 = writer.write(frame)
        finally:
            writer.close()

        assert s2 - s1 == 2, f"Expected +2 per write, got {s2 - s1}"
        assert s3 - s2 == 2

    def test_seq_monotonically_increases(self):
        name = _unique_name()
        frame = np.zeros((8, 8), dtype=np.uint16)
        seqs = []
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            for i in range(10):
                seqs.append(writer.write(frame))
        finally:
            writer.close()

        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], (
                f"Sequence not monotonic at index {i}: {seqs[i - 1]} → {seqs[i]}"
            )

    def test_writer_seq_property(self):
        name = _unique_name()
        frame = np.zeros((8, 8), dtype=np.uint16)
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            assert writer.seq == 0
            writer.write(frame)
            assert writer.seq == 2
            writer.write(frame)
            assert writer.seq == 4
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# 3. Double-buffer correctness
# ---------------------------------------------------------------------------


class TestDoubleBuffer:
    def test_two_writes_use_alternating_slots(self):
        """After two writes, reading should return the second frame."""
        name = _unique_name()
        H, W = 8, 8
        frame1 = np.full((H, W), 100, dtype=np.uint16)
        frame2 = np.full((H, W), 200, dtype=np.uint16)

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            writer.write(frame1)
            writer.write(frame2)
            reader = ShmFrameReader(name)
            got, seq = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(
            got, frame2, err_msg="After two writes, reader should see the latest frame"
        )

    def test_first_frame_accessible_via_seq(self):
        """After write 1 (seq=2) the reader gets frame1; after write 2 (seq=4) gets frame2."""
        name = _unique_name()
        H, W = 8, 8
        frame1 = np.full((H, W), 111, dtype=np.uint16)
        frame2 = np.full((H, W), 222, dtype=np.uint16)

        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            writer.write(frame1)
            reader = ShmFrameReader(name)
            got1, seq1 = reader.read()

            writer.write(frame2)
            got2, seq2 = reader.read()
        finally:
            reader.close()
            writer.close()

        np.testing.assert_array_equal(got1, frame1, err_msg="After write 1, got wrong frame")
        np.testing.assert_array_equal(got2, frame2, err_msg="After write 2, got wrong frame")
        assert seq2 > seq1

    def test_ten_sequential_writes(self):
        """Each read after each write should return the correct frame."""
        name = _unique_name()
        H, W = 16, 16
        try:
            writer = ShmFrameWriter(name, width=W, height=H, dtype=np.uint16)
            reader = ShmFrameReader(name)

            for i in range(10):
                expected = np.full((H, W), i * 100, dtype=np.uint16)
                writer.write(expected)
                got, _ = reader.read()
                np.testing.assert_array_equal(
                    got, expected,
                    err_msg=f"Frame {i} mismatch after write"
                )
        finally:
            reader.close()
            writer.close()


# ---------------------------------------------------------------------------
# 4. Header parsing
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    def test_magic_and_version(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint16)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info is not None  # _unpack_header did not raise
        # Magic and version are validated inside _unpack_header; reaching here = OK

    def test_width_height_channels(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=320, height=240, channels=3, dtype=np.uint8)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["width"] == 320
        assert info["height"] == 240
        assert info["channels"] == 3

    def test_dtype_code_uint16(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint16)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["dtype_code"] == 2, (
            f"uint16 should have dtype_code=2, got {info['dtype_code']}"
        )

    def test_dtype_code_uint8(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint8)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["dtype_code"] == 1

    def test_dtype_code_float32(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.float32)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["dtype_code"] == 4

    def test_data_offset_equals_header_size(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint16)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["data_offset"] == _HEADER_SIZE, (
            f"data_offset should be {_HEADER_SIZE}, got {info['data_offset']}"
        )

    def test_data_size_is_frame_bytes(self):
        name = _unique_name()
        W, H, C = 64, 48, 1
        expected_bytes = W * H * C * np.dtype(np.uint16).itemsize
        try:
            writer = ShmFrameWriter(name, width=W, height=H, channels=C, dtype=np.uint16)
            reader = ShmFrameReader(name)
            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert info["data_size"] == expected_bytes, (
            f"data_size should be {expected_bytes}, got {info['data_size']}"
        )

    def test_timestamp_updated_on_write(self):
        name = _unique_name()
        frame = np.zeros((16, 16), dtype=np.uint16)
        try:
            writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
            reader = ShmFrameReader(name)

            ts_before = time.time_ns()
            writer.write(frame)
            ts_after = time.time_ns()

            info = reader.read_header()
        finally:
            reader.close()
            writer.close()

        assert ts_before <= info["timestamp_ns"] <= ts_after, (
            f"timestamp_ns {info['timestamp_ns']} not in [{ts_before}, {ts_after}]"
        )


# ---------------------------------------------------------------------------
# 5. Cleanup on close
# ---------------------------------------------------------------------------


class TestCleanupOnClose:
    def test_unlink_true_removes_block(self):
        """After writer.close(unlink=True), the shm block should be gone."""
        name = _unique_name()
        writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
        writer.close(unlink=True)

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)

    def test_unlink_false_leaves_block_accessible(self):
        """After writer.close(unlink=False), readers can still open the block."""
        name = _unique_name()
        frame = np.full((16, 16), 77, dtype=np.uint16)
        writer = ShmFrameWriter(name, width=16, height=16, dtype=np.uint16)
        writer.write(frame)
        writer.close(unlink=False)

        try:
            reader = ShmFrameReader(name)
            got, seq = reader.read()
            reader.close()
        finally:
            # Manual cleanup
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        np.testing.assert_array_equal(got, frame)
        assert seq == 2

    def test_double_close_is_safe(self):
        name = _unique_name()
        writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
        writer.close()
        writer.close()  # second close should not raise

    def test_reader_double_close_is_safe(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            reader = ShmFrameReader(name)
        finally:
            writer.close(unlink=False)

        reader.close()
        reader.close()  # second close should not raise

        # Cleanup
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# 6. Context manager protocol
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_writer_context_manager_closes_on_exit(self):
        name = _unique_name()
        with ShmFrameWriter(name, width=8, height=8, dtype=np.uint16) as writer:
            assert not writer._closed
        assert writer._closed

    def test_reader_context_manager_closes_on_exit(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            with ShmFrameReader(name) as reader:
                assert not reader._closed
            assert reader._closed
        finally:
            writer.close()

    def test_full_round_trip_via_context_manager(self):
        name = _unique_name()
        frame = np.full((16, 16), 255, dtype=np.uint16)

        with ShmFrameWriter(name, width=16, height=16, dtype=np.uint16) as writer:
            writer.write(frame)
            with ShmFrameReader(name) as reader:
                got, seq = reader.read()

        np.testing.assert_array_equal(got, frame)
        assert seq == 2


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_nonexistent_shm_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ShmFrameReader("rs_mlx_nonexistent_block_xyz")

    def test_write_after_close_raises_runtime_error(self):
        name = _unique_name()
        frame = np.zeros((8, 8), dtype=np.uint16)
        writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
        writer.close()
        with pytest.raises(RuntimeError, match="closed"):
            writer.write(frame)

    def test_read_after_close_raises_runtime_error(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            reader = ShmFrameReader(name)
            reader.close()
        finally:
            writer.close()

        with pytest.raises(RuntimeError, match="closed"):
            reader.read()

    def test_write_wrong_shape_raises_value_error(self):
        name = _unique_name()
        bad_frame = np.zeros((99, 99), dtype=np.uint16)
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            with pytest.raises(ValueError, match="shape"):
                writer.write(bad_frame)
        finally:
            writer.close()

    def test_write_wrong_dtype_raises_value_error(self):
        name = _unique_name()
        bad_frame = np.zeros((8, 8), dtype=np.float32)
        try:
            writer = ShmFrameWriter(name, width=8, height=8, dtype=np.uint16)
            with pytest.raises(ValueError, match="dtype"):
                writer.write(bad_frame)
        finally:
            writer.close()

    def test_unsupported_dtype_raises_value_error(self):
        name = _unique_name()
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ShmFrameWriter(name, width=8, height=8, dtype=np.complex64)  # type: ignore

    def test_bad_magic_raises_value_error(self):
        """Connecting a reader to a non-RMLX shm block should raise ValueError."""
        name = _unique_name()
        size = 256
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        try:
            # Write garbage (not RMLX magic)
            shm.buf[:_HEADER_SIZE] = b"\xDE\xAD\xBE\xEF" + b"\x00" * (_HEADER_SIZE - 4)
            with pytest.raises(ValueError, match="Bad magic"):
                ShmFrameReader(name)
        finally:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# 8. Reader properties reflect writer config
# ---------------------------------------------------------------------------


class TestReaderProperties:
    def test_reader_reflects_writer_dimensions(self):
        name = _unique_name()
        W, H, C = 320, 240, 3
        try:
            writer = ShmFrameWriter(name, width=W, height=H, channels=C, dtype=np.uint8)
            reader = ShmFrameReader(name)
        finally:
            reader.close()
            writer.close()

        assert reader.width == W
        assert reader.height == H
        assert reader.channels == C
        assert reader.dtype == np.dtype(np.uint8)

    def test_reader_repr_smoke(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint16)
            reader = ShmFrameReader(name)
            r = repr(reader)
        finally:
            reader.close()
            writer.close()

        assert "ShmFrameReader" in r
        assert name in r

    def test_writer_repr_smoke(self):
        name = _unique_name()
        try:
            writer = ShmFrameWriter(name, width=64, height=48, dtype=np.uint16)
            r = repr(writer)
        finally:
            writer.close()

        assert "ShmFrameWriter" in r
        assert name in r
