"""Shared pytest fixtures for realsense-mlx tests.

All depth fixtures return numpy uint16 arrays — tests convert to mx.array
as needed.  This keeps fixtures decoupled from the MLX device and allows
them to be used in pure-numpy sanity checks too.

Fixture summary
---------------
depth_ramp_480p       Linear ramp 0→9585 across width. Shape (480, 640).
depth_ramp_720p       Linear ramp 0→8953 across width. Shape (720, 1280).
depth_with_holes      Random depth with a rectangular and scattered holes.
depth_flat_wall       Flat wall at 1 m (1000 counts). Shape (480, 640).
color_gradient        BGR gradient image. Shape (480, 640, 3) uint8.
d415_intrinsics       Typical D415 depth stream intrinsics.
d415_color_intrinsics Typical D415 color stream intrinsics.
identity_extrinsics   Identity rigid-body transform.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Depth frame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def depth_ramp_480p() -> np.ndarray:
    """Linear depth ramp 0→9585 across width, tiled to full height.

    Returns
    -------
    np.ndarray
        Shape ``(480, 640)``, dtype ``uint16``.
        Values span ``[0, 9585]`` (640 * 15 - 15 = 9585).
    """
    row = np.arange(640, dtype=np.uint16) * np.uint16(15)
    return np.tile(row, (480, 1))


@pytest.fixture
def depth_ramp_720p() -> np.ndarray:
    """Linear depth ramp across width, tiled to full height.

    Returns
    -------
    np.ndarray
        Shape ``(720, 1280)``, dtype ``uint16``.
        Values span ``[0, 8953]`` (1280 * 7 - 7 = 8953).
    """
    row = np.arange(1280, dtype=np.uint16) * np.uint16(7)
    return np.tile(row, (720, 1))


@pytest.fixture
def depth_with_holes() -> np.ndarray:
    """Depth frame with a rectangular invalid region and scattered holes.

    Structure
    ---------
    - Background: random uint16 values in [500, 5000).
    - Rectangular hole: rows 100–199, cols 200–399 are set to 0.
    - Random holes: approximately 5 % of pixels are zeroed (seed=0).

    Returns
    -------
    np.ndarray
        Shape ``(480, 640)``, dtype ``uint16``.
    """
    rng = np.random.default_rng(0)
    depth = rng.integers(500, 5000, size=(480, 640), dtype=np.uint16)
    # Rectangular hole
    depth[100:200, 200:400] = 0
    # ~5 % random invalid pixels
    mask = rng.random((480, 640)) < 0.05
    depth[mask] = 0
    return depth


@pytest.fixture
def depth_flat_wall() -> np.ndarray:
    """Flat wall at exactly 1 metre (1000 depth counts @ 1 mm/count).

    Returns
    -------
    np.ndarray
        Shape ``(480, 640)``, dtype ``uint16``.  All pixels equal 1000.
    """
    return np.full((480, 640), 1000, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Color frame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def color_gradient() -> np.ndarray:
    """BGR color gradient image.

    Channel layout (OpenCV-compatible BGR order):
    - B: zero everywhere
    - G: increases with column index (0–213 over 640 pixels)
    - R: increases with row index (0–239 over 480 pixels)

    Returns
    -------
    np.ndarray
        Shape ``(480, 640, 3)``, dtype ``uint8``.  Contiguous C-order array.
    """
    rows = np.arange(480, dtype=np.uint8).reshape(-1, 1)
    cols = np.arange(640, dtype=np.uint8).reshape(1, -1)

    b = np.zeros((480, 640), dtype=np.uint8)
    g = (cols // 3).astype(np.uint8)           # 0..213
    r = (rows // 2).astype(np.uint8)           # 0..239

    # Stack to (480, 640, 3) in BGR order; ensure contiguous.
    return np.stack([b, g, r], axis=-1)


# ---------------------------------------------------------------------------
# Intrinsics / Extrinsics fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def d415_intrinsics():
    """Typical Intel RealSense D415 depth stream intrinsics at 640x480.

    Returns
    -------
    CameraIntrinsics
        Approximate factory-calibration values for the D415 depth sensor.
    """
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics

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
def d415_color_intrinsics():
    """Typical Intel RealSense D415 color stream intrinsics at 640x480.

    Returns
    -------
    CameraIntrinsics
        Approximate factory-calibration values for the D415 RGB sensor.
    """
    from realsense_mlx.geometry.intrinsics import CameraIntrinsics

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
def identity_extrinsics():
    """Identity rigid-body transform (depth and color frames co-located).

    Returns
    -------
    CameraExtrinsics
        Rotation = I₃, translation = 0.
    """
    from realsense_mlx.geometry.intrinsics import CameraExtrinsics

    return CameraExtrinsics(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )
