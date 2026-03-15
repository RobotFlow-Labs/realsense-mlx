"""Comprehensive PLY export tests for PointCloudGenerator.export_ply.

Coverage
--------
* Export XYZ only (no color) — verify PLY header format and vertex count
* Export XYZ+RGB — verify all six properties are present in the header
* skip_zero=True filters zero-depth points — verify reduced count
* Round-trip: export → re-read with numpy structured array → verify coordinates
* Large point cloud (720p, 1280×720) — verify no OOM or truncation
* File overwrite behaviour — second export replaces the first cleanly
* File permissions — export into a read-only directory raises PermissionError
* Binary body byte-size: XYZ only = 12 bytes/point, XYZ+RGB = 15 bytes/point
* All-zero depth with skip_zero=True → writes 0 points and a valid PLY header
* Float32 colors are normalised to uint8 correctly in output
"""

from __future__ import annotations

import os
import stat
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.geometry.intrinsics import CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pinhole_intr() -> CameraIntrinsics:
    """640×480 pinhole camera, principal point at exact centre."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="none",
    )


@pytest.fixture
def gen(pinhole_intr: CameraIntrinsics) -> PointCloudGenerator:
    return PointCloudGenerator(pinhole_intr, depth_scale=0.001)


@pytest.fixture
def hd_intr() -> CameraIntrinsics:
    """1280×720 HD pinhole camera for large-point-cloud tests."""
    return CameraIntrinsics(
        width=1280, height=720,
        ppx=640.0, ppy=360.0,
        fx=900.0, fy=900.0,
        model="none",
    )


@pytest.fixture
def hd_gen(hd_intr: CameraIntrinsics) -> PointCloudGenerator:
    return PointCloudGenerator(hd_intr, depth_scale=0.001)


# ---------------------------------------------------------------------------
# PLY parsing helpers
# ---------------------------------------------------------------------------


def _parse_header(path: Path) -> dict:
    """Parse a binary-little-endian PLY header.

    Returns a dict with:
    - ``n_vertices``    : int
    - ``properties``    : list[str]  e.g. ``["x","y","z","red","green","blue"]``
    - ``header_bytes``  : int  byte length including the final newline of end_header
    - ``raw_header``    : str  full decoded header text
    """
    with open(path, "rb") as f:
        raw = f.read()

    end_marker = b"end_header\n"
    end_pos = raw.find(end_marker)
    if end_pos == -1:
        raise ValueError("end_header not found in PLY file")

    header_bytes = end_pos + len(end_marker)
    header_text = raw[:end_pos].decode("ascii")

    n_vertices = 0
    properties: list[str] = []
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property"):
            # "property float x" → last token is name
            properties.append(line.split()[-1])

    return {
        "n_vertices": n_vertices,
        "properties": properties,
        "header_bytes": header_bytes,
        "raw_header": header_text,
    }


def _read_ply_xyz(path: Path, n_vertices: int, header_bytes: int) -> np.ndarray:
    """Read XYZ-only binary PLY body into (N, 3) float32 array."""
    with open(path, "rb") as f:
        f.seek(header_bytes)
        raw = f.read(n_vertices * 12)
    return np.frombuffer(raw, dtype=np.float32).reshape(n_vertices, 3)


def _read_ply_xyzrgb(
    path: Path,
    n_vertices: int,
    header_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ+RGB binary PLY body.

    Returns (xyz float32 (N,3), rgb uint8 (N,3)).
    """
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("r", np.uint8),
        ("g", np.uint8),
        ("b", np.uint8),
    ])
    with open(path, "rb") as f:
        f.seek(header_bytes)
        raw = f.read(n_vertices * dtype.itemsize)
    buf = np.frombuffer(raw, dtype=dtype)
    xyz = np.stack([buf["x"], buf["y"], buf["z"]], axis=-1)
    rgb = np.stack([buf["r"], buf["g"], buf["b"]], axis=-1)
    return xyz, rgb


def _flat_depth(h: int, w: int, raw_value: int) -> mx.array:
    return mx.full((h, w), raw_value, dtype=mx.uint16)


# ---------------------------------------------------------------------------
# 1. XYZ only — header and vertex count
# ---------------------------------------------------------------------------


class TestXYZOnly:
    def test_file_is_created(self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            gen.export_ply(pts, path)
            assert path.exists()

    def test_header_starts_with_ply(self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            gen.export_ply(pts, path)
            info = _parse_header(path)
            assert info["raw_header"].strip().startswith("ply"), (
                "PLY file must begin with 'ply'"
            )

    def test_header_format_binary_little_endian(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            gen.export_ply(pts, path)
            info = _parse_header(path)
            assert "binary_little_endian" in info["raw_header"]

    def test_vertex_count_in_header_matches_return_value(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            n = gen.export_ply(pts, path)
            info = _parse_header(path)
            assert info["n_vertices"] == n, (
                f"Header says {info['n_vertices']} vertices, return value is {n}"
            )

    def test_xyz_properties_present(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            gen.export_ply(pts, path)
            info = _parse_header(path)
            assert "x" in info["properties"]
            assert "y" in info["properties"]
            assert "z" in info["properties"]

    def test_no_color_properties_in_xyz_only(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            gen.export_ply(pts, path)
            info = _parse_header(path)
            for prop in ("red", "green", "blue"):
                assert prop not in info["properties"], (
                    f"Color property '{prop}' should not appear in XYZ-only export"
                )

    def test_binary_body_size_12_bytes_per_point(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """XYZ only: 3 × float32 = 12 bytes per point."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyz.ply"
            n = gen.export_ply(pts, path, skip_zero=False)
            info = _parse_header(path)
            body_bytes = path.stat().st_size - info["header_bytes"]
            assert body_bytes == n * 12, (
                f"Expected {n * 12} body bytes, got {body_bytes}"
            )


# ---------------------------------------------------------------------------
# 2. XYZ + RGB — header and vertex count
# ---------------------------------------------------------------------------


class TestXYZRGB:
    def test_all_six_properties_present(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        color = mx.zeros((H, W, 3), dtype=mx.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xyzrgb.ply"
            gen.export_ply(pts, path, colors=color)
            info = _parse_header(path)
            for prop in ("x", "y", "z", "red", "green", "blue"):
                assert prop in info["properties"], (
                    f"Expected property '{prop}' in XYZRGB header"
                )

    def test_vertex_count_matches_xyz_only(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Adding color should not change which points are included."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        color = mx.zeros((H, W, 3), dtype=mx.uint8)
        with tempfile.TemporaryDirectory() as td:
            n_xyz = gen.export_ply(pts, Path(td) / "xyz.ply")
            n_rgb = gen.export_ply(pts, Path(td) / "xyzrgb.ply", colors=color)
        assert n_xyz == n_rgb

    def test_binary_body_size_15_bytes_per_point(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """XYZ+RGB: 3×float32 + 3×uint8 = 15 bytes per point."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        color = mx.zeros((H, W, 3), dtype=mx.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rgb.ply"
            n = gen.export_ply(pts, path, colors=color, skip_zero=False)
            info = _parse_header(path)
            body_bytes = path.stat().st_size - info["header_bytes"]
            assert body_bytes == n * 15, (
                f"Expected {n * 15} body bytes, got {body_bytes}"
            )

    def test_float_color_clamped_to_uint8(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Float32 colors in [0,1] should be scaled to [0,255] uint8."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))
        # All pixels = (1.0, 0.5, 0.0) → expected (255, 127 or 128, 0)
        color_np = np.ones((H, W, 3), dtype=np.float32)
        color_np[:, :, 1] = 0.5
        color_np[:, :, 2] = 0.0
        color = mx.array(color_np)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "float_color.ply"
            n = gen.export_ply(pts, path, colors=color, skip_zero=True)
            info = _parse_header(path)
            _, rgb = _read_ply_xyzrgb(path, info["n_vertices"], info["header_bytes"])
            assert np.all(rgb[:, 0] == 255), "Red channel should be 255"
            assert np.all(rgb[:, 2] == 0), "Blue channel should be 0"
            # Green: round(0.5 * 255) — allow ±1 for rounding differences
            assert np.all(np.abs(rgb[:, 1].astype(np.int16) - 127) <= 1), (
                f"Green channel should be ~127, got {np.unique(rgb[:, 1])}"
            )


# ---------------------------------------------------------------------------
# 3. skip_zero filtering
# ---------------------------------------------------------------------------


class TestSkipZero:
    def test_skip_zero_reduces_count(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[: H // 2, :] = 1000  # top half valid
        pts = gen.generate(mx.array(depth_np))

        with tempfile.TemporaryDirectory() as td:
            n = gen.export_ply(pts, Path(td) / "partial.ply", skip_zero=True)
        assert n == (H // 2) * W, f"Expected {(H // 2) * W} points, got {n}"

    def test_all_zero_depth_produces_zero_points(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 0))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.ply"
            n = gen.export_ply(pts, path, skip_zero=True)
            assert n == 0
            info = _parse_header(path)
            assert info["n_vertices"] == 0

    def test_skip_zero_false_writes_all_pixels(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[: H // 2, :] = 1000
        pts = gen.generate(mx.array(depth_np))

        with tempfile.TemporaryDirectory() as td:
            n = gen.export_ply(pts, Path(td) / "all.ply", skip_zero=False)
        assert n == H * W

    def test_skip_zero_with_color(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Skipping zero-depth should filter both points and colors consistently."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[: H // 2, :] = 2000
        pts = gen.generate(mx.array(depth_np))

        color_np = np.zeros((H, W, 3), dtype=np.uint8)
        color_np[: H // 2, :] = [100, 150, 200]
        color = mx.array(color_np)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partial_rgb.ply"
            n = gen.export_ply(pts, path, colors=color, skip_zero=True)
            info = _parse_header(path)
            _, rgb = _read_ply_xyzrgb(path, info["n_vertices"], info["header_bytes"])

        assert n == (H // 2) * W
        # All written pixels came from the non-zero half — must be [100, 150, 200]
        np.testing.assert_array_equal(
            rgb,
            np.tile([100, 150, 200], (n, 1)),
            err_msg="Color values should match the valid (non-zero depth) half",
        )


# ---------------------------------------------------------------------------
# 4. Round-trip: export → re-read → verify coordinates
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_xyz_values_preserved(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Read back XYZ floats from the binary body and compare to MLX output."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth_raw = 2000  # 2.0 m
        pts = gen.generate(_flat_depth(H, W, depth_raw))
        mx.eval(pts)
        pts_np = np.array(pts, copy=False)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rt.ply"
            gen.export_ply(pts, path, skip_zero=False)
            info = _parse_header(path)
            xyz_read = _read_ply_xyz(path, info["n_vertices"], info["header_bytes"])

        expected = pts_np.reshape(-1, 3)
        np.testing.assert_allclose(
            xyz_read, expected, atol=1e-6,
            err_msg="Round-tripped XYZ values should match original float32 array",
        )

    def test_rgb_values_preserved(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Read back RGB bytes and compare to the original uint8 color array."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))

        rng = np.random.default_rng(99)
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        color = mx.array(color_np)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rt_rgb.ply"
            gen.export_ply(pts, path, colors=color, skip_zero=False)
            info = _parse_header(path)
            _, rgb_read = _read_ply_xyzrgb(
                path, info["n_vertices"], info["header_bytes"]
            )

        np.testing.assert_array_equal(
            rgb_read, color_np.reshape(-1, 3),
            err_msg="Round-tripped RGB values should match original uint8 array",
        )

    def test_centre_pixel_coordinates(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Centre pixel at 1 m depth: round-trip should give (0, 0, 1)."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth_np = np.zeros((H, W), dtype=np.uint16)
        # Only set centre pixel to 1000 (1.0 m)
        cy, cx = H // 2, W // 2
        depth_np[cy, cx] = 1000
        pts = gen.generate(mx.array(depth_np))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "centre.ply"
            n = gen.export_ply(pts, path, skip_zero=True)
            assert n == 1, f"Only one valid point, got {n}"
            info = _parse_header(path)
            xyz = _read_ply_xyz(path, info["n_vertices"], info["header_bytes"])

        np.testing.assert_allclose(xyz[0, 2], 1.0, atol=1e-5, err_msg="Z should be 1 m")
        np.testing.assert_allclose(xyz[0, 0], 0.0, atol=1e-5, err_msg="X should be 0")
        np.testing.assert_allclose(xyz[0, 1], 0.0, atol=1e-5, err_msg="Y should be 0")


# ---------------------------------------------------------------------------
# 5. Large point cloud (720p)
# ---------------------------------------------------------------------------


class TestLargePointCloud:
    def test_720p_xyz_export_no_oom(
        self, hd_gen: PointCloudGenerator, hd_intr: CameraIntrinsics
    ):
        """1280×720 flat wall export should complete without OOM or truncation."""
        H, W = hd_intr.height, hd_intr.width
        pts = hd_gen.generate(_flat_depth(H, W, 1500))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hd.ply"
            n = hd_gen.export_ply(pts, path, skip_zero=True)
        assert n == H * W, f"Expected {H * W} points, got {n}"

    def test_720p_xyzrgb_export_no_oom(
        self, hd_gen: PointCloudGenerator, hd_intr: CameraIntrinsics
    ):
        H, W = hd_intr.height, hd_intr.width
        pts = hd_gen.generate(_flat_depth(H, W, 1500))
        color = mx.zeros((H, W, 3), dtype=mx.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hd_rgb.ply"
            n = hd_gen.export_ply(pts, path, colors=color, skip_zero=True)
        assert n == H * W

    def test_720p_vertex_count_in_header(
        self, hd_gen: PointCloudGenerator, hd_intr: CameraIntrinsics
    ):
        H, W = hd_intr.height, hd_intr.width
        pts = hd_gen.generate(_flat_depth(H, W, 1500))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hd_count.ply"
            n = hd_gen.export_ply(pts, path, skip_zero=True)
            info = _parse_header(path)
        assert info["n_vertices"] == n


# ---------------------------------------------------------------------------
# 6. File overwrite behaviour
# ---------------------------------------------------------------------------


class TestOverwrite:
    def test_second_export_overwrites_first(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Exporting twice to the same path replaces the file cleanly."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts_far = gen.generate(_flat_depth(H, W, 3000))
        pts_near = gen.generate(_flat_depth(H, W, 500))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "overwrite.ply"
            gen.export_ply(pts_far, path)
            size_first = path.stat().st_size

            gen.export_ply(pts_near, path)
            size_second = path.stat().st_size

        # Both exports produce the same number of valid points (all H*W with
        # skip_zero=True) so file sizes should be equal
        assert size_first == size_second, (
            "Overwrite should produce a file of the same size for same-resolution input"
        )

    def test_overwrite_produces_valid_header(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "overwrite2.ply"
            gen.export_ply(pts, path)
            gen.export_ply(pts, path)  # second write
            info = _parse_header(path)

        assert info["n_vertices"] > 0
        assert info["raw_header"].strip().startswith("ply")


# ---------------------------------------------------------------------------
# 7. File permissions
# ---------------------------------------------------------------------------


class TestFilePermissions:
    @pytest.mark.skipif(os.getuid() == 0, reason="root can write anywhere")
    def test_write_to_readonly_directory_raises(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """Exporting into a read-only directory should raise PermissionError."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))

        with tempfile.TemporaryDirectory() as td:
            ro_dir = Path(td) / "readonly"
            ro_dir.mkdir()
            ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
            try:
                with pytest.raises((PermissionError, OSError)):
                    gen.export_ply(pts, ro_dir / "test.ply")
            finally:
                # Restore permissions so tempdir cleanup works
                ro_dir.chmod(stat.S_IRWXU)

    def test_parent_directory_auto_created(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """export_ply should create missing parent directories automatically."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts = gen.generate(_flat_depth(H, W, 1000))

        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "a" / "b" / "c" / "output.ply"
            assert not nested.parent.exists(), "Parent should not exist yet"
            gen.export_ply(pts, nested)
            assert nested.exists(), "File should be created including parents"


# ---------------------------------------------------------------------------
# 8. Edge-case: N-D point array input
# ---------------------------------------------------------------------------


class TestNDInputShape:
    def test_flat_n3_input_accepted(
        self, gen: PointCloudGenerator, pinhole_intr: CameraIntrinsics
    ):
        """export_ply should accept (N, 3) input as well as (H, W, 3)."""
        H, W = pinhole_intr.height, pinhole_intr.width
        pts_hwc = gen.generate(_flat_depth(H, W, 1000))
        mx.eval(pts_hwc)
        pts_n3 = mx.array(np.array(pts_hwc).reshape(-1, 3))

        with tempfile.TemporaryDirectory() as td:
            n_hwc = gen.export_ply(pts_hwc, Path(td) / "hwc.ply", skip_zero=False)
            n_n3 = gen.export_ply(pts_n3, Path(td) / "n3.ply", skip_zero=False)

        assert n_hwc == n_n3 == H * W
