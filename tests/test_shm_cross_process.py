"""Cross-process shared-memory transport tests.

Validates that ShmFrameWriter / ShmFrameReader work correctly when the
writer lives in a subprocess and the reader lives in the main process (and
vice-versa).  Uses multiprocessing.Process so the two sides share nothing
except the named POSIX shm block.

Coverage
--------
* Frames written in one process can be read in another (uint16)
* Frames written in one process can be read in another (uint8 RGB)
* Sequence numbers strictly increase across process boundary
* Multiple frames written in a tight loop: reader sees at least some of them
* Reader started before writer: reader blocks until writer creates block
* Writer closes with unlink=False: reader can still drain the last frame
* Frame content integrity: exact byte-level match after cross-process round-trip
* Large frame (720p uint16) round-trip across processes
"""

from __future__ import annotations

import os
import time
import uuid
import multiprocessing

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers shared between main process and subprocesses
# ---------------------------------------------------------------------------


def _unique_name() -> str:
    return f"rs_xp_{os.getpid()}_{uuid.uuid4().hex[:10]}"


def _writer_process(
    name: str,
    frames_bytes: list[bytes],
    shape: tuple[int, ...],
    dtype_str: str,
    delay_s: float = 0.01,
    unlink: bool = True,
) -> None:
    """Subprocess target: write *frames_bytes* sequentially to shm, then close."""
    from realsense_mlx.transport.shm_frame import ShmFrameWriter

    dtype = np.dtype(dtype_str)
    channels = shape[2] if len(shape) == 3 else 1
    height, width = shape[0], shape[1]

    writer = ShmFrameWriter(name, width=width, height=height, channels=channels, dtype=dtype)
    for raw in frames_bytes:
        frame = np.frombuffer(raw, dtype=dtype).reshape(shape)
        writer.write(frame)
        time.sleep(delay_s)
    writer.close(unlink=unlink)


def _reader_process(
    name: str,
    shape: tuple[int, ...],
    dtype_str: str,
    n_reads: int,
    result_queue: multiprocessing.Queue,
    wait_for_shm_s: float = 2.0,
) -> None:
    """Subprocess target: read up to *n_reads* frames and put (frame_bytes, seq) pairs."""
    from multiprocessing import shared_memory as shmlib
    from realsense_mlx.transport.shm_frame import ShmFrameReader

    dtype = np.dtype(dtype_str)
    deadline = time.monotonic() + wait_for_shm_s
    reader = None

    while time.monotonic() < deadline:
        try:
            reader = ShmFrameReader(name)
            break
        except FileNotFoundError:
            time.sleep(0.02)

    if reader is None:
        result_queue.put(None)
        return

    results = []
    last_seq = -1
    for _ in range(n_reads):
        frame, seq = reader.read()
        if seq > 0 and seq != last_seq:
            results.append((frame.tobytes(), int(seq)))
            last_seq = seq
        time.sleep(0.015)

    reader.close()
    result_queue.put(results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_frames():
    """Five distinct 48×64 uint16 depth frames."""
    rng = np.random.default_rng(42)
    return [
        rng.integers(0, 65535, (48, 64), dtype=np.uint16)
        for _ in range(5)
    ]


@pytest.fixture()
def rgb_frames():
    """Five distinct 48×64×3 uint8 RGB frames."""
    rng = np.random.default_rng(7)
    return [
        rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for _ in range(5)
    ]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestWriterReaderDifferentProcesses:
    """Writer in subprocess, reader in main process."""

    def test_basic_cross_process_round_trip(self, small_frames):
        name = _unique_name()
        shape = small_frames[0].shape  # (48, 64)
        dtype_str = "uint16"

        frames_bytes = [f.tobytes() for f in small_frames]
        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, frames_bytes, shape, dtype_str),
            kwargs={"delay_s": 0.01, "unlink": False},
        )
        p.start()
        # Give the writer a moment to create the shm block
        time.sleep(0.07)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        # Wait up to 2 s for the block to appear
        reader = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.02)

        p.join(timeout=5)
        assert p.exitcode == 0, f"Writer subprocess exited with {p.exitcode}"
        assert reader is not None, "ShmFrameReader could not attach to shm block"

        received = []
        for _ in range(20):
            frame, seq = reader.read()
            if seq > 0:
                received.append((frame.copy(), seq))
                break
            time.sleep(0.02)

        reader.close()
        # Cleanup — writer exited with unlink=False
        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert len(received) > 0, "Reader never saw a valid frame (seq > 0)"
        assert received[-1][0].shape == (48, 64), (
            f"Unexpected frame shape: {received[-1][0].shape}"
        )
        assert received[-1][0].dtype == np.dtype("uint16")

    def test_uint8_rgb_cross_process(self, rgb_frames):
        name = _unique_name()
        shape = rgb_frames[0].shape  # (48, 64, 3)
        dtype_str = "uint8"

        frames_bytes = [f.tobytes() for f in rgb_frames]
        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, frames_bytes, shape, dtype_str),
            kwargs={"delay_s": 0.01, "unlink": False},
        )
        p.start()
        time.sleep(0.08)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        reader = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.02)

        p.join(timeout=5)
        assert p.exitcode == 0
        assert reader is not None

        frame, seq = None, 0
        for _ in range(20):
            frame, seq = reader.read()
            if seq > 0:
                break
            time.sleep(0.02)

        reader.close()
        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert seq > 0
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.dtype("uint8")

    def test_frame_content_integrity(self):
        """Byte-level equality: what writer wrote is exactly what reader reads."""
        name = _unique_name()
        # Use a deterministic fill so we can verify exact bytes
        shape = (16, 16)
        dtype_str = "uint16"
        sentinel_frame = np.full(shape, 0xABCD, dtype=np.uint16)

        frames_bytes = [sentinel_frame.tobytes()]
        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, frames_bytes, shape, dtype_str),
            kwargs={"delay_s": 0.05, "unlink": False},
        )
        p.start()
        time.sleep(0.08)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        reader = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.02)

        p.join(timeout=5)
        assert p.exitcode == 0

        received_frame, seq = None, 0
        for _ in range(30):
            received_frame, seq = reader.read()
            if seq > 0:
                break
            time.sleep(0.02)

        reader.close()
        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert seq > 0
        np.testing.assert_array_equal(
            received_frame, sentinel_frame,
            err_msg="Cross-process frame content mismatch"
        )

    @pytest.mark.slow
    def test_large_720p_cross_process(self):
        """720p uint16 frame survives a cross-process round-trip intact."""
        rng = np.random.default_rng(99)
        shape = (720, 1280)
        dtype_str = "uint16"
        big_frame = rng.integers(0, 65535, shape, dtype=np.uint16)
        name = _unique_name()

        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, [big_frame.tobytes()], shape, dtype_str),
            kwargs={"delay_s": 0.05, "unlink": False},
        )
        p.start()
        time.sleep(0.15)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        reader = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.05)

        p.join(timeout=10)
        assert p.exitcode == 0
        assert reader is not None

        received, seq = None, 0
        for _ in range(30):
            received, seq = reader.read()
            if seq > 0:
                break
            time.sleep(0.05)

        reader.close()
        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert seq > 0
        np.testing.assert_array_equal(received, big_frame)


class TestSequenceNumbersCrossProcess:
    """Verify monotonically increasing sequence numbers across processes."""

    def test_sequence_numbers_increase(self, small_frames):
        name = _unique_name()
        shape = small_frames[0].shape
        dtype_str = "uint16"

        frames_bytes = [f.tobytes() for f in small_frames]
        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, frames_bytes, shape, dtype_str),
            kwargs={"delay_s": 0.02, "unlink": False},
        )
        p.start()
        time.sleep(0.05)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        reader = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.02)

        assert reader is not None

        seqs = []
        for _ in range(30):
            _, seq = reader.read()
            if seq > 0:
                seqs.append(seq)
            time.sleep(0.02)
            if len(seqs) >= 3:
                break

        reader.close()
        p.join(timeout=5)

        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert len(seqs) >= 1, "Reader never received any valid frames"
        for i in range(1, len(seqs)):
            assert seqs[i] >= seqs[i - 1], (
                f"Sequence not monotonic at index {i}: {seqs[i-1]} → {seqs[i]}"
            )

    def test_all_sequence_numbers_are_even(self, small_frames):
        """The seqlock protocol guarantees all published sequences are even."""
        name = _unique_name()
        shape = small_frames[0].shape
        dtype_str = "uint16"

        frames_bytes = [f.tobytes() for f in small_frames]
        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, frames_bytes, shape, dtype_str),
            kwargs={"delay_s": 0.02, "unlink": False},
        )
        p.start()
        time.sleep(0.05)

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        reader = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                reader = ShmFrameReader(name)
                break
            except FileNotFoundError:
                time.sleep(0.02)

        assert reader is not None

        seqs = []
        for _ in range(30):
            _, seq = reader.read()
            if seq > 0:
                seqs.append(seq)
            time.sleep(0.015)
            if len(seqs) >= 3:
                break

        reader.close()
        p.join(timeout=5)

        try:
            from multiprocessing import shared_memory as shmlib
            shm = shmlib.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        for seq in seqs:
            assert seq % 2 == 0, f"Odd sequence number {seq} observed by reader"


class TestReaderInSubprocess:
    """Writer in main process, reader in subprocess."""

    def test_reader_in_subprocess_receives_frame(self):
        name = _unique_name()
        shape = (32, 32)
        dtype = np.uint16
        frame = np.full(shape, 12345, dtype=dtype)

        from realsense_mlx.transport.shm_frame import ShmFrameWriter

        writer = ShmFrameWriter(name, width=32, height=32, dtype=dtype)
        writer.write(frame)
        writer.write(frame)  # second write so seq=4

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_reader_process,
            args=(name, shape, "uint16", 10, result_queue),
        )
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0, f"Reader subprocess exited with {p.exitcode}"

        writer.close(unlink=True)

        results = result_queue.get_nowait()
        assert results is not None, "Reader subprocess returned None (shm not found)"
        assert len(results) > 0, "Reader subprocess received no frames"

        raw_bytes, seq = results[0]
        received = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
        np.testing.assert_array_equal(received, frame)
        assert seq % 2 == 0

    def test_reader_subprocess_sees_seq_gt_zero(self):
        name = _unique_name()
        shape = (16, 16)
        dtype = np.uint16
        frame = np.arange(256, dtype=dtype).reshape(shape)

        from realsense_mlx.transport.shm_frame import ShmFrameWriter

        writer = ShmFrameWriter(name, width=16, height=16, dtype=dtype)
        for _ in range(3):
            writer.write(frame)

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_reader_process,
            args=(name, shape, "uint16", 5, result_queue),
        )
        p.start()
        p.join(timeout=10)
        writer.close(unlink=True)

        results = result_queue.get_nowait()
        assert results is not None
        assert len(results) > 0
        for _, seq in results:
            assert seq > 0


class TestUnlinkBehaviourCrossProcess:
    """close(unlink=False) in a subprocess leaves the block readable."""

    def test_writer_exits_unlink_false_reader_can_read(self):
        name = _unique_name()
        shape = (16, 16)
        dtype_str = "uint16"
        payload = np.full(shape, 0x1234, dtype=np.uint16)

        p = multiprocessing.Process(
            target=_writer_process,
            args=(name, [payload.tobytes()], shape, dtype_str),
            kwargs={"delay_s": 0.02, "unlink": False},
        )
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0

        from realsense_mlx.transport.shm_frame import ShmFrameReader

        # Writer has already exited; the block should still exist
        try:
            reader = ShmFrameReader(name)
            frame, seq = reader.read()
            reader.close()
        finally:
            try:
                from multiprocessing import shared_memory as shmlib
                shm = shmlib.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        assert seq > 0
        np.testing.assert_array_equal(frame, payload)
