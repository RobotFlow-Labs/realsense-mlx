"""Inter-process frame transport via POSIX shared memory.

Provides a zero-copy producer/consumer channel for streaming depth (or
color) frames between OS processes on the same machine.

Usage
-----
Producer (capture process)::

    from realsense_mlx.transport import ShmFrameWriter
    import numpy as np

    writer = ShmFrameWriter("depth_stream", width=640, height=480,
                            channels=1, dtype=np.uint16)
    while capturing:
        frame = camera.get_depth()   # np.ndarray (H, W) uint16
        writer.write(frame)
    writer.close()

Consumer (processing process)::

    from realsense_mlx.transport import ShmFrameReader
    import time

    reader = ShmFrameReader("depth_stream")
    last_seq = -1
    while running:
        frame, seq = reader.read()
        if seq != last_seq:
            process(frame)
            last_seq = seq
        else:
            time.sleep(0.001)   # no new frame yet
    reader.close()

See :mod:`realsense_mlx.transport.shm_frame` for implementation details.
"""

from realsense_mlx.transport.shm_frame import ShmFrameReader, ShmFrameWriter

__all__ = ["ShmFrameWriter", "ShmFrameReader"]
