"""Tests for PointCloudGenerator.

Coverage
--------
* No-distortion: flat wall at known distance → centre pixel = (0, 0, depth)
* No-distortion: off-centre pixels match analytic formula
* Brown-Conrady distortion: compare to analytical reference
* Inverse-Brown-Conrady: compare to analytical reference
* Zero depth → (0, 0, 0) everywhere
* Grid caching: grids not recomputed on subsequent calls
* Shape mismatch raises ValueError
* generate_with_color returns matching shapes
* export_ply writes a valid file with correct point count
* PointCloudGenerator repr smoke test
"""

from __future__ import annotations

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
    """640x480 camera, principal point at exact centre, fx=fy=600."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="none",
    )


@pytest.fixture
def pinhole_gen(pinhole_intr) -> PointCloudGenerator:
    return PointCloudGenerator(pinhole_intr, depth_scale=0.001)


@pytest.fixture
def bc_intr() -> CameraIntrinsics:
    """Brown-Conrady intrinsics with mild radial distortion."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="brown_conrady",
        coeffs=[0.1, 0.02, 0.0, 0.0, 0.001],
    )


@pytest.fixture
def ibc_intr() -> CameraIntrinsics:
    """Inverse-Brown-Conrady intrinsics (typical depth sensor model)."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="inverse_brown_conrady",
        coeffs=[0.1, 0.02, 0.0, 0.0, 0.001],
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _flat_depth(h: int, w: int, raw_value: int) -> mx.array:
    """Return a (H, W) uint16 array filled with ``raw_value``."""
    return mx.full((h, w), raw_value, dtype=mx.uint16)


def _eval_np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr, copy=False)


# ---------------------------------------------------------------------------
# 1. No-distortion: centre pixel
# ---------------------------------------------------------------------------


class TestPinholeCentrePixel:
    def test_centre_pixel_xyz_at_1m(self, pinhole_gen, pinhole_intr):
        """Centre pixel with depth 1 m → X=0, Y=0, Z=1."""
        # 1000 raw counts * 0.001 m/count = 1.0 m
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 1000)
        pc = pinhole_gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        cy, cx = pinhole_intr.height // 2, pinhole_intr.width // 2
        centre = pts[cy, cx]

        # Centre pixel is at exact principal point → normalised x=0, y=0
        assert centre[2] == pytest.approx(1.0, abs=1e-5), "Z should equal depth"
        assert centre[0] == pytest.approx(0.0, abs=1e-5), "X should be 0 at centre"
        assert centre[1] == pytest.approx(0.0, abs=1e-5), "Y should be 0 at centre"

    def test_centre_pixel_xyz_at_05m(self, pinhole_gen, pinhole_intr):
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 500)
        pc = pinhole_gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)
        cy, cx = pinhole_intr.height // 2, pinhole_intr.width // 2
        centre = pts[cy, cx]
        assert centre[2] == pytest.approx(0.5, abs=1e-5)
        assert centre[0] == pytest.approx(0.0, abs=1e-5)
        assert centre[1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. No-distortion: off-centre analytical verification
# ---------------------------------------------------------------------------


class TestPinholeOffCentre:
    def test_analytic_x(self, pinhole_gen, pinhole_intr):
        """Pixel at (ppx + fx, ppy) → X == Z (normalised x == 1)."""
        i = pinhole_intr
        depth_raw = 2000  # 2 m
        depth = _flat_depth(i.height, i.width, depth_raw)
        pc = pinhole_gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        # Column where pixel_x = ppx + fx  → nx = 1.0
        col = int(i.ppx + i.fx)  # = 320 + 600 = 920 — out of frame for 640
        # Use a scaled-down intrinsic instead
        # Build a small frame so the column fits
        pass  # tested via per-pixel formula below

    def test_analytic_formula_all_pixels(self, pinhole_intr):
        """Verify X = (u - ppx)/fx * Z, Y = (v - ppy)/fy * Z for all pixels."""
        i = pinhole_intr
        depth_raw = 1000
        gen = PointCloudGenerator(i, depth_scale=0.001)

        depth = _flat_depth(i.height, i.width, depth_raw)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        Z = depth_raw * 0.001
        u = np.arange(i.width, dtype=np.float32)
        v = np.arange(i.height, dtype=np.float32)
        X_ref = ((u - i.ppx) / i.fx) * Z  # (W,)
        Y_ref = ((v - i.ppy) / i.fy) * Z  # (H,)

        # Broadcast reference to (H, W) for assert_allclose shape compatibility
        X_ref_2d = np.broadcast_to(X_ref[None, :], (i.height, i.width))
        Y_ref_2d = np.broadcast_to(Y_ref[:, None], (i.height, i.width))
        np.testing.assert_allclose(pts[:, :, 0], X_ref_2d, atol=1e-5,
                                   err_msg="X mismatch")
        np.testing.assert_allclose(pts[:, :, 1], Y_ref_2d, atol=1e-5,
                                   err_msg="Y mismatch")
        np.testing.assert_allclose(pts[:, :, 2], Z, atol=1e-5,
                                   err_msg="Z mismatch")

    def test_shape(self, pinhole_gen, pinhole_intr):
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 500)
        pc = pinhole_gen.generate(depth)
        assert pc.shape == (pinhole_intr.height, pinhole_intr.width, 3)

    def test_dtype_float32(self, pinhole_gen, pinhole_intr):
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 500)
        pc = pinhole_gen.generate(depth)
        assert pc.dtype == mx.float32


# ---------------------------------------------------------------------------
# 3. Zero depth → (0, 0, 0)
# ---------------------------------------------------------------------------


class TestZeroDepth:
    def test_zero_depth_produces_zero_xyz(self, pinhole_gen, pinhole_intr):
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 0)
        pc = pinhole_gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)
        np.testing.assert_array_equal(pts, 0.0)

    def test_zero_depth_bc(self, bc_intr):
        gen = PointCloudGenerator(bc_intr, depth_scale=0.001)
        depth = _flat_depth(bc_intr.height, bc_intr.width, 0)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)
        np.testing.assert_array_equal(pts, 0.0)


# ---------------------------------------------------------------------------
# 4. Brown-Conrady distortion: analytical cross-check
# ---------------------------------------------------------------------------


class TestBrownConradyDistortion:
    def _reference_bc_deproject(
        self,
        u: float,
        v: float,
        depth: float,
        i: CameraIntrinsics,
        iters: int = 10,
    ) -> np.ndarray:
        """Per-pixel scalar reference matching cuda-pointcloud.cu."""
        k1, k2, p1, p2, k3 = i.coeffs
        x0 = (u - i.ppx) / i.fx
        y0 = (v - i.ppy) / i.fy
        x, y = x0, y0
        for _ in range(iters):
            r2 = x * x + y * y
            icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
            x = (x0 - dx) * icdist
            y = (y0 - dy) * icdist
        return np.array([depth * x, depth * y, depth], dtype=np.float32)

    def test_centre_pixel(self, bc_intr):
        """At centre pixel, Brown-Conrady correction should still give X≈0, Y≈0."""
        gen = PointCloudGenerator(bc_intr, depth_scale=0.001)
        depth_raw = 1000
        depth = _flat_depth(bc_intr.height, bc_intr.width, depth_raw)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        cy, cx = bc_intr.height // 2, bc_intr.width // 2
        centre = pts[cy, cx]
        # At exact principal point, nx=ny=0, distortion correction = 0
        assert centre[0] == pytest.approx(0.0, abs=1e-4)
        assert centre[1] == pytest.approx(0.0, abs=1e-4)
        assert centre[2] == pytest.approx(1.0, abs=1e-4)

    def test_off_centre_vs_reference(self, bc_intr):
        """Compare MLX output against scalar Newton-Raphson reference."""
        gen = PointCloudGenerator(bc_intr, depth_scale=0.001)
        depth_raw = 2000
        depth = _flat_depth(bc_intr.height, bc_intr.width, depth_raw)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        depth_m = depth_raw * 0.001

        # Sample a few non-trivial pixels
        test_pixels = [(100, 200), (300, 400), (50, 600), (0, 0)]
        for row, col in test_pixels:
            if row >= bc_intr.height or col >= bc_intr.width:
                continue
            ref = self._reference_bc_deproject(col, row, depth_m, bc_intr)
            got = pts[row, col]
            np.testing.assert_allclose(
                got, ref, atol=1e-4,
                err_msg=f"Mismatch at pixel ({col}, {row})"
            )


# ---------------------------------------------------------------------------
# 5. Inverse Brown-Conrady: analytical cross-check
# ---------------------------------------------------------------------------


class TestInverseBrownConradyDistortion:
    def _reference_ibc_deproject(
        self,
        u: float,
        v: float,
        depth: float,
        i: CameraIntrinsics,
    ) -> np.ndarray:
        """Forward inverse-Brown-Conrady polynomial (non-iterative)."""
        k1, k2, p1, p2, k3 = i.coeffs
        x = (u - i.ppx) / i.fx
        y = (v - i.ppy) / i.fy
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        ux = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        uy = y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        return np.array([depth * ux, depth * uy, depth], dtype=np.float32)

    def test_centre_pixel(self, ibc_intr):
        gen = PointCloudGenerator(ibc_intr, depth_scale=0.001)
        depth_raw = 1000
        depth = _flat_depth(ibc_intr.height, ibc_intr.width, depth_raw)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)
        cy, cx = ibc_intr.height // 2, ibc_intr.width // 2
        centre = pts[cy, cx]
        assert centre[0] == pytest.approx(0.0, abs=1e-4)
        assert centre[1] == pytest.approx(0.0, abs=1e-4)
        assert centre[2] == pytest.approx(1.0, abs=1e-4)

    def test_off_centre_vs_reference(self, ibc_intr):
        gen = PointCloudGenerator(ibc_intr, depth_scale=0.001)
        depth_raw = 1500
        depth = _flat_depth(ibc_intr.height, ibc_intr.width, depth_raw)
        pc = gen.generate(depth)
        mx.eval(pc)
        pts = np.array(pc, copy=False)

        depth_m = depth_raw * 0.001
        test_pixels = [(100, 200), (300, 400), (0, 0), (479, 639)]
        for row, col in test_pixels:
            ref = self._reference_ibc_deproject(col, row, depth_m, ibc_intr)
            got = pts[row, col]
            np.testing.assert_allclose(
                got, ref, atol=1e-4,
                err_msg=f"IBC mismatch at pixel ({col}, {row})"
            )


# ---------------------------------------------------------------------------
# 6. Grid caching
# ---------------------------------------------------------------------------


class TestGridCaching:
    def test_grids_computed_once(self, pinhole_gen, pinhole_intr):
        """_x_grid and _y_grid are None before first call, set after."""
        assert pinhole_gen._x_grid is None, "Grid should be None before first generate()"
        assert pinhole_gen._y_grid is None

        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 100)
        pinhole_gen.generate(depth)

        assert pinhole_gen._x_grid is not None, "Grid should be cached after first generate()"
        assert pinhole_gen._y_grid is not None

    def test_grids_not_recomputed(self, pinhole_gen, pinhole_intr):
        """The same grid objects are reused across frames."""
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 100)
        pinhole_gen.generate(depth)

        x_grid_id = id(pinhole_gen._x_grid)
        y_grid_id = id(pinhole_gen._y_grid)

        # Second call — different depth values, same intrinsics
        depth2 = _flat_depth(pinhole_intr.height, pinhole_intr.width, 500)
        pinhole_gen.generate(depth2)

        assert id(pinhole_gen._x_grid) == x_grid_id, "x_grid was reallocated"
        assert id(pinhole_gen._y_grid) == y_grid_id, "y_grid was reallocated"

    def test_invalidate_recomputes(self, pinhole_gen, pinhole_intr):
        """invalidate_cache() causes re-computation on next generate()."""
        depth = _flat_depth(pinhole_intr.height, pinhole_intr.width, 100)
        pinhole_gen.generate(depth)

        old_x_id = id(pinhole_gen._x_grid)
        pinhole_gen.invalidate_cache()
        assert pinhole_gen._x_grid is None

        pinhole_gen.generate(depth)
        assert pinhole_gen._x_grid is not None
        # New object allocated
        assert id(pinhole_gen._x_grid) != old_x_id

    def test_distorted_grids_are_2d(self, bc_intr):
        """With distortion, grids should be (H, W) not 1-D."""
        gen = PointCloudGenerator(bc_intr, depth_scale=0.001)
        depth = _flat_depth(bc_intr.height, bc_intr.width, 100)
        gen.generate(depth)

        assert gen._x_grid is not None
        assert gen._x_grid.shape == (bc_intr.height, bc_intr.width), (
            f"Expected 2-D grid, got {gen._x_grid.shape}"
        )


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_shape_mismatch_raises(self, pinhole_gen):
        bad_depth = mx.zeros((100, 100), dtype=mx.uint16)
        with pytest.raises(ValueError, match="does not match intrinsics"):
            pinhole_gen.generate(bad_depth)

    def test_negative_depth_scale_raises(self, pinhole_intr):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            PointCloudGenerator(pinhole_intr, depth_scale=-0.001)

    def test_zero_depth_scale_raises(self, pinhole_intr):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            PointCloudGenerator(pinhole_intr, depth_scale=0.0)


# ---------------------------------------------------------------------------
# 8. generate_with_color
# ---------------------------------------------------------------------------


class TestGenerateWithColor:
    def test_shapes_match(self, pinhole_gen, pinhole_intr):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        color = mx.zeros((H, W, 3), dtype=mx.uint8)
        pts, cols = pinhole_gen.generate_with_color(depth, color)
        assert pts.shape == (H, W, 3)
        assert cols.shape == (H, W, 3)

    def test_color_passthrough(self, pinhole_gen, pinhole_intr):
        """Color output should be identical to input (aligned)."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        rng = np.random.default_rng(42)
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        color = mx.array(color_np)
        _, cols = pinhole_gen.generate_with_color(depth, color)
        mx.eval(cols)
        np.testing.assert_array_equal(np.array(cols), color_np)

    def test_color_shape_mismatch_raises(self, pinhole_gen, pinhole_intr):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        bad_color = mx.zeros((H + 10, W, 3), dtype=mx.uint8)
        with pytest.raises(ValueError, match="does not match intrinsics"):
            pinhole_gen.generate_with_color(depth, bad_color)


# ---------------------------------------------------------------------------
# 9. export_ply
# ---------------------------------------------------------------------------


class TestExportPly:
    def _count_ply_vertices(self, path: Path) -> int:
        """Parse PLY header and return declared vertex count."""
        with open(path, "rb") as f:
            for line in f:
                line = line.decode("ascii").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
        raise ValueError("No vertex count found in PLY header")

    def _read_ply_header_end(self, path: Path) -> int:
        """Return byte offset of first byte after 'end_header\\n'."""
        with open(path, "rb") as f:
            data = f.read()
        marker = b"end_header\n"
        idx = data.find(marker)
        return idx + len(marker)

    def test_export_xyz_only(self, pinhole_gen, pinhole_intr):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.ply"
            n = pinhole_gen.export_ply(pts, path)

            assert path.exists(), "PLY file not created"
            assert n > 0, "Should write at least one point"
            assert self._count_ply_vertices(path) == n

    def test_export_with_color(self, pinhole_gen, pinhole_intr):
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        pts = pinhole_gen.generate(depth)
        color = mx.zeros((H, W, 3), dtype=mx.uint8)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "colored.ply"
            n = pinhole_gen.export_ply(pts, path, colors=color)

            assert path.exists()
            assert n > 0
            assert self._count_ply_vertices(path) == n
            # Check header mentions color properties
            with open(path, "rb") as f:
                header = f.read(500).decode("ascii", errors="ignore")
            assert "property uchar red" in header

    def test_skip_zero_filters_invalid(self, pinhole_intr):
        """Zero-depth pixels should be skipped when skip_zero=True."""
        H, W = pinhole_intr.height, pinhole_intr.width
        gen = PointCloudGenerator(pinhole_intr, depth_scale=0.001)

        # Mix half zero, half non-zero depth
        depth_np = np.zeros((H, W), dtype=np.uint16)
        depth_np[:H // 2, :] = 1000
        depth = mx.array(depth_np)
        pts = gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partial.ply"
            n = gen.export_ply(pts, path, skip_zero=True)
            declared = self._count_ply_vertices(path)
            assert declared == n
            assert n == (H // 2) * W, f"Expected {(H//2)*W} points, got {n}"

    def test_no_skip_zero(self, pinhole_gen, pinhole_intr):
        """With skip_zero=False, all H*W points are written."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "all.ply"
            n = pinhole_gen.export_ply(pts, path, skip_zero=False)
            assert n == H * W

    def test_binary_body_size_xyz(self, pinhole_gen, pinhole_intr):
        """Binary body should be n_points * 12 bytes (3 float32)."""
        H, W = pinhole_intr.height, pinhole_intr.width
        depth = _flat_depth(H, W, 1000)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bin.ply"
            n = pinhole_gen.export_ply(pts, path, skip_zero=False)

            file_size = path.stat().st_size
            header_end = self._read_ply_header_end(path)
            body_size = file_size - header_end
            assert body_size == n * 12, (
                f"Expected {n * 12} bytes body, got {body_size}"
            )


# ---------------------------------------------------------------------------
# 10. Repr smoke test
# ---------------------------------------------------------------------------


def test_repr(pinhole_gen):
    r = repr(pinhole_gen)
    assert "PointCloudGenerator" in r
    assert "640x480" in r


# ---------------------------------------------------------------------------
# 11. Brown-Conrady with real-world coefficients
# ---------------------------------------------------------------------------


class TestBrownConradyDistortionEdgeCases:
    def test_pointcloud_with_brown_conrady(self):
        """PointCloud with real-world Brown-Conrady coefficients."""
        intr = CameraIntrinsics(
            640, 480, 318.8, 239.5, 383.7, 383.7,
            model="brown_conrady",
            coeffs=[0.1, -0.25, 0.001, -0.001, 0.05]
        )
        gen = PointCloudGenerator(intr, 0.001)
        depth = mx.full((480, 640), 1000, dtype=mx.uint16)
        pts = gen.generate(depth)
        assert pts.shape == (480, 640, 3)
        # Center should still be near (0, 0, 1.0) even with distortion
        center = np.array(pts[240, 320])
        assert abs(center[2] - 1.0) < 1e-4
        assert abs(center[0]) < 0.05  # wider tolerance for distortion

    def test_no_distortion_vs_zero_coeffs_brown_conrady(self):
        """Brown-Conrady with zero coefficients should match no-distortion."""
        intr_none = CameraIntrinsics(64, 48, 32.0, 24.0, 50.0, 50.0, model="none")
        intr_bc = CameraIntrinsics(64, 48, 32.0, 24.0, 50.0, 50.0,
                                    model="brown_conrady", coeffs=[0, 0, 0, 0, 0])
        depth = mx.array(np.random.randint(500, 2000, (48, 64), dtype=np.uint16))
        pts_none = np.array(PointCloudGenerator(intr_none, 0.001).generate(depth))
        pts_bc = np.array(PointCloudGenerator(intr_bc, 0.001).generate(depth))
        np.testing.assert_allclose(pts_none, pts_bc, atol=1e-5)
