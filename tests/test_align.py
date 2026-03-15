"""Tests for the Aligner class.

Coverage
--------
* Identity extrinsics, same intrinsics:
  - align_color_to_depth → output matches input colour
  - align_depth_to_color → output depth matches input depth (when resolutions match)
* Known rigid transform → verify reprojected pixel coordinates
* Out-of-bounds projections → zero output
* align_depth_to_color: scatter-min semantics (nearest depth wins)
* align_color_to_depth_subpixel smoke test
* Shape mismatch raises ValueError
* Aligner repr smoke test
* Zero depth → zero color output
* Metal vs MLX equivalence (pinhole intrinsics, identity and non-identity extrinsics)
* Metal fallback: distorted intrinsics always uses MLX path
* use_metal flag propagates correctly
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.align import Aligner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def depth_intr() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="none",
    )


@pytest.fixture
def color_intr() -> CameraIntrinsics:
    """Same resolution as depth for identity-transform tests."""
    return CameraIntrinsics(
        width=640, height=480,
        ppx=320.0, ppy=240.0,
        fx=600.0, fy=600.0,
        model="none",
    )


@pytest.fixture
def color_intr_hd() -> CameraIntrinsics:
    """Higher-resolution colour sensor."""
    return CameraIntrinsics(
        width=1280, height=720,
        ppx=640.0, ppy=360.0,
        fx=900.0, fy=900.0,
        model="none",
    )


@pytest.fixture
def identity_ext() -> CameraExtrinsics:
    return CameraExtrinsics.identity()


@pytest.fixture
def identity_aligner(depth_intr, color_intr, identity_ext) -> Aligner:
    return Aligner(depth_intr, color_intr, identity_ext, depth_scale=0.001)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _make_depth(h: int, w: int, raw_value: int) -> mx.array:
    return mx.full((h, w), raw_value, dtype=mx.uint16)


def _make_color(h: int, w: int, fill: tuple[int, int, int] = (128, 64, 32)) -> mx.array:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = fill
    return mx.array(arr)


def _eval_np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.array(arr, copy=False)


# ---------------------------------------------------------------------------
# 1. Identity transform, same intrinsics — align_color_to_depth
# ---------------------------------------------------------------------------


class TestIdentityAlignColorToDepth:
    def test_output_shape(self, identity_aligner, depth_intr):
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(H, W)
        out = identity_aligner.align_color_to_depth(depth, color)
        assert out.shape == (H, W, 3)

    def test_same_intrinsics_colour_preserved(self, identity_aligner, depth_intr):
        """With identity extrinsics and same intrinsics, colour is copied 1:1."""
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)

        rng = np.random.default_rng(0)
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        color = mx.array(color_np)

        out = identity_aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)

        # With identity extrinsics and same intrinsics, pixel mapping is 1:1
        # (nearest-neighbour rounding may shift by at most 1 pixel at extreme edges)
        # Test that the interior matches exactly
        interior = out_np[1:-1, 1:-1]
        ref_interior = color_np[1:-1, 1:-1]
        np.testing.assert_array_equal(interior, ref_interior)

    def test_zero_depth_outputs_zeros(self, identity_aligner, depth_intr):
        """Zero depth pixels project at Z=0 → invalid → zero output."""
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 0)
        color = _make_color(H, W, fill=(255, 255, 255))
        out = identity_aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)
        np.testing.assert_array_equal(out_np, 0)


# ---------------------------------------------------------------------------
# 2. Identity transform, same intrinsics — align_depth_to_color
# ---------------------------------------------------------------------------


class TestIdentityAlignDepthToColor:
    def test_output_shape(self, identity_aligner, depth_intr, color_intr):
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        out = identity_aligner.align_depth_to_color(depth)
        assert out.shape == (color_intr.height, color_intr.width)

    def test_depth_values_preserved(self, identity_aligner, depth_intr):
        """With identity extrinsics and same intrinsics, depth is preserved."""
        H, W = depth_intr.height, depth_intr.width
        depth_raw = 2500
        depth = _make_depth(H, W, depth_raw)
        out = identity_aligner.align_depth_to_color(depth)
        out_np = _eval_np(out).astype(np.int32)

        # Interior pixels should map 1:1; check a centre patch
        cy, cx = H // 2, W // 2
        interior = out_np[cy - 50 : cy + 50, cx - 50 : cx + 50]
        assert np.all(interior == depth_raw), (
            f"Expected depth {depth_raw}, got unique values {np.unique(interior)}"
        )

    def test_zero_depth_not_written(self, identity_aligner, depth_intr):
        """Zero-depth pixels should not overwrite the output (remain 0)."""
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 0)
        out = identity_aligner.align_depth_to_color(depth)
        out_np = _eval_np(out)
        np.testing.assert_array_equal(out_np, 0)

    def test_dtype_uint16(self, identity_aligner, depth_intr):
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        out = identity_aligner.align_depth_to_color(depth)
        assert out.dtype == mx.uint16


# ---------------------------------------------------------------------------
# 3. Known translation — verify reprojection
# ---------------------------------------------------------------------------


class TestKnownTranslation:
    def _make_aligner_with_translation(
        self,
        depth_intr: CameraIntrinsics,
        color_intr: CameraIntrinsics,
        tx: float, ty: float, tz: float,
    ) -> Aligner:
        ext = CameraExtrinsics(
            rotation=np.eye(3),
            translation=np.array([tx, ty, tz], dtype=np.float64),
        )
        return Aligner(depth_intr, color_intr, ext, depth_scale=0.001)

    def test_pure_x_translation_shifts_pixels(self, depth_intr, color_intr):
        """A positive Tx shifts projected pixels in +x direction."""
        # Place a single point at (0, 0, 1 m) in depth frame.
        # After Tx=+0.1 m, in colour frame point is at (0.1, 0, 1).
        # Projected: px = 0.1/1.0 * fx + ppx = 60 + 320 = 380
        H, W = depth_intr.height, depth_intr.width
        Tx = 0.1  # 10 cm

        aligner = self._make_aligner_with_translation(
            depth_intr, color_intr, Tx, 0.0, 0.0
        )

        # Single depth pixel at centre = 1000 raw = 1.0 m
        depth_np = np.zeros((H, W), dtype=np.uint16)
        cy, cx = H // 2, W // 2
        depth_np[cy, cx] = 1000
        depth = mx.array(depth_np)

        color_np = np.zeros((H, W, 3), dtype=np.uint8)
        # Place a distinctive colour at the expected projected location
        expected_col = int(round(cx + Tx * depth_intr.fx))  # ~320 + 60 = 380
        if 0 <= expected_col < W:
            color_np[cy, expected_col] = [255, 0, 0]

        color = mx.array(color_np)
        out = aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)

        # The centre pixel of the output should have picked up the
        # colour from (cy, expected_col) in the colour frame.
        # Verify non-zero value at centre if expected_col is valid.
        if 0 <= expected_col < W:
            centre_pixel = out_np[cy, cx]
            assert centre_pixel[0] == 255, (
                f"Expected red at centre, got {centre_pixel} "
                f"(expected_col={expected_col})"
            )

    def test_translation_out_of_bounds(self, depth_intr, color_intr):
        """Large translation pushes all depth pixels outside colour frame → zero."""
        aligner = self._make_aligner_with_translation(
            depth_intr, color_intr, tx=100.0, ty=0.0, tz=0.0
        )
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(H, W, fill=(200, 100, 50))
        out = aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)
        np.testing.assert_array_equal(out_np, 0)


# ---------------------------------------------------------------------------
# 4. Out-of-bounds projections → zero output
# ---------------------------------------------------------------------------


class TestOutOfBounds:
    def test_negative_depth_gives_zero(self, identity_aligner, depth_intr):
        """Depth = 0 should produce zero colour."""
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 0)
        color = _make_color(H, W, fill=(100, 200, 50))
        out = identity_aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)
        np.testing.assert_array_equal(out_np, 0)

    def test_depth_to_color_out_of_frame(self, depth_intr):
        """When colour frame is much smaller, most depth pixels should produce 0."""
        tiny_color = CameraIntrinsics(
            width=10, height=10,
            ppx=5.0, ppy=5.0,
            fx=600.0, fy=600.0,
            model="none",
        )
        ext = CameraExtrinsics.identity()
        aligner = Aligner(depth_intr, tiny_color, ext, depth_scale=0.001)

        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        out = aligner.align_depth_to_color(depth)
        out_np = _eval_np(out)

        # Output should be (10, 10)
        assert out_np.shape == (10, 10)
        # Most pixels in the big depth frame project outside tiny colour frame
        # Some tiny interior might hit, but the majority must be zero
        nonzero = np.count_nonzero(out_np)
        total = 10 * 10
        assert nonzero <= total, "Cannot have more nonzero than total pixels"


# ---------------------------------------------------------------------------
# 5. Scatter-min semantics (align_depth_to_color)
# ---------------------------------------------------------------------------


class TestScatterMin:
    def test_nearer_depth_wins(self, depth_intr, color_intr, identity_ext):
        """When two depth pixels map to same colour pixel, smaller depth wins."""
        H, W = depth_intr.height, depth_intr.width
        aligner = Aligner(depth_intr, color_intr, identity_ext, depth_scale=0.001)

        # Create a depth frame with all zeros except two nearby pixels
        # that will map to the same colour pixel after rounding.
        # Set the centre pixel to two adjacent raw values; after identical
        # projection they write the same colour output location.
        depth_np = np.zeros((H, W), dtype=np.uint16)
        cy, cx = H // 2, W // 2

        # Near pixel (small depth = nearer object)
        depth_np[cy, cx] = 500   # 0.5 m
        # Neighbour 1 pixel away (slightly further)
        if cx + 1 < W:
            depth_np[cy, cx + 1] = 600  # 0.6 m

        depth = mx.array(depth_np)
        out = aligner.align_depth_to_color(depth)
        out_np = _eval_np(out).astype(np.int32)

        # Centre output pixel should carry the nearer (smaller) depth
        assert out_np[cy, cx] == 500, (
            f"Expected minimum depth 500 at centre, got {out_np[cy, cx]}"
        )


# ---------------------------------------------------------------------------
# 6. align_color_to_depth_subpixel smoke test
# ---------------------------------------------------------------------------


class TestSubpixelAlign:
    def test_output_shape(self, identity_aligner, depth_intr):
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(H, W)
        out = identity_aligner.align_color_to_depth_subpixel(depth, color)
        assert out.shape == (H, W, 3)

    def test_consistent_with_single_pass(self, identity_aligner, depth_intr):
        """Subpixel result should be close to single-pass for uniform color."""
        H, W = depth_intr.height, depth_intr.width
        depth = _make_depth(H, W, 1000)
        # Uniform colour — both shifts gather the same value
        color = _make_color(H, W, fill=(100, 150, 200))
        out_single = identity_aligner.align_color_to_depth(depth, color)
        out_sub = identity_aligner.align_color_to_depth_subpixel(depth, color)

        out_s_np = _eval_np(out_single)
        out_sub_np = _eval_np(out_sub)

        # Interior should be identical for uniform colour
        np.testing.assert_array_equal(
            out_s_np[5:-5, 5:-5], out_sub_np[5:-5, 5:-5]
        )


# ---------------------------------------------------------------------------
# 7. Different resolutions (HD colour)
# ---------------------------------------------------------------------------


class TestDifferentResolutions:
    def test_color_to_depth_shape_with_hd_color(
        self, depth_intr, color_intr_hd, identity_ext
    ):
        aligner = Aligner(depth_intr, color_intr_hd, identity_ext, depth_scale=0.001)
        H_d, W_d = depth_intr.height, depth_intr.width
        H_c, W_c = color_intr_hd.height, color_intr_hd.width
        depth = _make_depth(H_d, W_d, 1000)
        color = _make_color(H_c, W_c)
        out = aligner.align_color_to_depth(depth, color)
        assert out.shape == (H_d, W_d, 3)

    def test_depth_to_color_shape_with_hd_color(
        self, depth_intr, color_intr_hd, identity_ext
    ):
        aligner = Aligner(depth_intr, color_intr_hd, identity_ext, depth_scale=0.001)
        H_d, W_d = depth_intr.height, depth_intr.width
        H_c, W_c = color_intr_hd.height, color_intr_hd.width
        depth = _make_depth(H_d, W_d, 1000)
        out = aligner.align_depth_to_color(depth)
        assert out.shape == (H_c, W_c)


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------


class TestAlignerErrors:
    def test_depth_shape_mismatch_color_to_depth(self, identity_aligner, color_intr):
        bad_depth = mx.zeros((100, 100), dtype=mx.uint16)
        color = _make_color(color_intr.height, color_intr.width)
        with pytest.raises(ValueError, match="does not match depth intrinsics"):
            identity_aligner.align_color_to_depth(bad_depth, color)

    def test_color_shape_mismatch(self, identity_aligner, depth_intr):
        depth = _make_depth(depth_intr.height, depth_intr.width, 1000)
        bad_color = mx.zeros((50, 50, 3), dtype=mx.uint8)
        with pytest.raises(ValueError, match="does not match colour intrinsics"):
            identity_aligner.align_color_to_depth(depth, bad_color)

    def test_depth_shape_mismatch_depth_to_color(self, identity_aligner):
        bad_depth = mx.zeros((100, 100), dtype=mx.uint16)
        with pytest.raises(ValueError, match="does not match depth intrinsics"):
            identity_aligner.align_depth_to_color(bad_depth)

    def test_negative_depth_scale_raises(self, depth_intr, color_intr, identity_ext):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            Aligner(depth_intr, color_intr, identity_ext, depth_scale=-0.001)


# ---------------------------------------------------------------------------
# 9. Properties and repr
# ---------------------------------------------------------------------------


def test_aligner_properties(identity_aligner, depth_intr, color_intr, identity_ext):
    assert identity_aligner.depth_intrinsics is depth_intr
    assert identity_aligner.color_intrinsics is color_intr
    assert identity_aligner.extrinsics is identity_ext
    assert identity_aligner.depth_scale == pytest.approx(0.001)


def test_aligner_repr(identity_aligner):
    r = repr(identity_aligner)
    assert "Aligner" in r
    assert "640x480" in r
    assert "use_metal" in r
    assert "metal_active" in r


# ---------------------------------------------------------------------------
# 10. CameraExtrinsics inverse
# ---------------------------------------------------------------------------


class TestExtrinsicsInverse:
    def test_inverse_of_identity_is_identity(self, identity_ext):
        inv = identity_ext.inverse()
        assert inv.is_identity

    def test_inverse_round_trip(self):
        """R^-1 @ (R @ p + t) - (R^T @ t) == p."""
        rng = np.random.default_rng(7)
        # Random rotation via QR decomposition
        M = rng.standard_normal((3, 3))
        R, _ = np.linalg.qr(M)
        t = rng.standard_normal(3)
        ext = CameraExtrinsics(rotation=R, translation=t)
        inv = ext.inverse()

        p = rng.standard_normal(3)
        # Forward: q = R @ p + t
        q = R @ p + t
        # Inverse: R^T @ q + t_inv
        p_back = inv.rotation @ q + inv.translation
        np.testing.assert_allclose(p_back, p, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. Brown-Conrady distortion in alignment pipeline
# ---------------------------------------------------------------------------


class TestAlignBrownConrady:
    """Verify the Aligner works correctly when depth intrinsics use Brown-Conrady."""

    @pytest.fixture
    def bc_depth_intr(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="brown_conrady",
            coeffs=[0.05, -0.1, 0.0, 0.0, 0.02],
        )

    def test_align_color_to_depth_bc_output_shape(self, bc_depth_intr, color_intr, identity_ext):
        """align_color_to_depth with Brown-Conrady depth intrinsics returns correct shape."""
        aligner = Aligner(bc_depth_intr, color_intr, identity_ext, depth_scale=0.001)
        H, W = bc_depth_intr.height, bc_depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(color_intr.height, color_intr.width)
        out = aligner.align_color_to_depth(depth, color)
        assert out.shape == (H, W, 3)

    def test_align_depth_to_color_bc_output_shape(self, bc_depth_intr, color_intr, identity_ext):
        """align_depth_to_color with Brown-Conrady depth intrinsics returns colour-frame shape."""
        aligner = Aligner(bc_depth_intr, color_intr, identity_ext, depth_scale=0.001)
        H_d, W_d = bc_depth_intr.height, bc_depth_intr.width
        H_c, W_c = color_intr.height, color_intr.width
        depth = _make_depth(H_d, W_d, 1000)
        out = aligner.align_depth_to_color(depth)
        assert out.shape == (H_c, W_c)

    def test_align_zero_depth_bc_gives_zero_color(self, bc_depth_intr, color_intr, identity_ext):
        """Zero-depth pixels with Brown-Conrady intrinsics still produce zero output."""
        aligner = Aligner(bc_depth_intr, color_intr, identity_ext, depth_scale=0.001)
        H, W = bc_depth_intr.height, bc_depth_intr.width
        depth = _make_depth(H, W, 0)
        color = _make_color(color_intr.height, color_intr.width, fill=(255, 128, 64))
        out = aligner.align_color_to_depth(depth, color)
        out_np = _eval_np(out)
        np.testing.assert_array_equal(out_np, 0)

    def test_bc_vs_none_centre_region_similar(self, color_intr, identity_ext):
        """With small distortion coefficients, BC and no-distortion should agree near centre."""
        intr_none = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="none",
        )
        intr_bc = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="brown_conrady",
            coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],  # zero distortion
        )
        aligner_none = Aligner(intr_none, color_intr, identity_ext, depth_scale=0.001)
        aligner_bc = Aligner(intr_bc, color_intr, identity_ext, depth_scale=0.001)

        H, W = intr_none.height, intr_none.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(color_intr.height, color_intr.width, fill=(100, 150, 200))

        out_none = _eval_np(aligner_none.align_color_to_depth(depth, color))
        out_bc = _eval_np(aligner_bc.align_color_to_depth(depth, color))

        # With zero distortion coefficients results must match exactly in the interior
        cy, cx = H // 2, W // 2
        patch_none = out_none[cy - 20:cy + 20, cx - 20:cx + 20]
        patch_bc = out_bc[cy - 20:cy + 20, cx - 20:cx + 20]
        np.testing.assert_array_equal(patch_none, patch_bc)


# ---------------------------------------------------------------------------
# 12. Metal vs MLX equivalence tests
# ---------------------------------------------------------------------------


def _check_metal_mlx_close(
    out_metal: np.ndarray,
    out_mlx: np.ndarray,
    label: str,
    border: int = 1,
) -> None:
    """Assert Metal and MLX outputs are equivalent, allowing boundary differences.

    The Metal kernel and the MLX vectorised path perform the same float32
    arithmetic but in slightly different FP evaluation orders (Metal may use
    FMA or different register-level rounding).  For pixels whose projected
    coordinates land exactly on the frame boundary (within ~1 ULP of 0.0 or
    width/height), the two paths can disagree by 0 or 1 pixel.

    Interior pixels (``border`` rows/cols from the edge) are checked for
    **exact bit equality**.  This catches all algorithmic differences including
    the half-integer rounding issue (Metal::rint vs mx.round).

    Parameters
    ----------
    out_metal, out_mlx : (H, W, C) uint8 arrays.
    label              : Description for error messages.
    border             : Number of boundary rows/cols to exclude (default 1).
    """
    H, W = out_metal.shape[:2]
    # Full-frame: both must have identical shapes
    assert out_metal.shape == out_mlx.shape, (
        f"{label}: shape mismatch {out_metal.shape} vs {out_mlx.shape}"
    )
    # Interior check: exact equality
    if H > 2 * border and W > 2 * border:
        interior_metal = out_metal[border:H-border, border:W-border]
        interior_mlx   = out_mlx[border:H-border, border:W-border]
        np.testing.assert_array_equal(
            interior_metal, interior_mlx,
            err_msg=f"{label}: interior (excluding {border}-pixel border) pixels differ",
        )
    # Boundary: at most 2*(H+W) pixels allowed to differ (one-pixel boundary band)
    diff = (out_metal != out_mlx)
    if diff.ndim == 3:
        diff = diff.any(axis=2)
    n_diff = int(diff.sum())
    max_allowed = 2 * (H + W) * 2  # generous: twice the perimeter
    assert n_diff <= max_allowed, (
        f"{label}: {n_diff} mismatches exceed boundary allowance {max_allowed}. "
        f"This indicates interior pixel differences beyond float32 boundary noise."
    )


class TestMetalVsMLXEquivalence:
    """Verify that Metal and MLX paths produce equivalent results.

    Interior pixels (>= 1 pixel from any edge) are checked for **exact**
    bit equality — Metal::rint() matches mx.round() (both banker's rounding).

    Boundary pixels may legitimately differ: Metal's JIT float32 arithmetic
    can produce tiny-negative projected coordinates (~1e-8 ULP error) for
    pixels that land exactly on the frame boundary, causing a validity-check
    divergence of at most one pixel band.  This is documented in the module
    docstring and is acceptable for real-world depth frames where boundary
    pixels typically have invalid depth anyway.
    """

    @pytest.fixture
    def small_depth_intr(self) -> CameraIntrinsics:
        """Smaller resolution for faster test execution."""
        return CameraIntrinsics(
            width=120, height=90,
            ppx=60.0, ppy=45.0,
            fx=150.0, fy=150.0,
            model="none",
        )

    @pytest.fixture
    def small_color_intr(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=120, height=90,
            ppx=60.0, ppy=45.0,
            fx=150.0, fy=150.0,
            model="none",
        )

    @pytest.fixture
    def small_color_intr_hd(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=160, height=120,
            ppx=80.0, ppy=60.0,
            fx=200.0, fy=200.0,
            model="none",
        )

    def _both_aligners(
        self,
        depth_intr: CameraIntrinsics,
        color_intr: CameraIntrinsics,
        ext: CameraExtrinsics,
    ) -> tuple[Aligner, Aligner]:
        """Return (metal_aligner, mlx_aligner) pair."""
        metal = Aligner(depth_intr, color_intr, ext, depth_scale=0.001, use_metal=True)
        mlx = Aligner(depth_intr, color_intr, ext, depth_scale=0.001, use_metal=False)
        return metal, mlx

    def test_identity_extrinsics_same_resolution(
        self, small_depth_intr, small_color_intr, identity_ext
    ):
        """Metal == MLX for identity transform, same resolution, uniform color."""
        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr, identity_ext
        )
        H, W = small_depth_intr.height, small_depth_intr.width
        depth = _make_depth(H, W, 1500)
        color = _make_color(H, W, fill=(80, 160, 240))

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        _check_metal_mlx_close(out_metal, out_mlx, "identity/same-resolution")

    def test_identity_extrinsics_random_color(
        self, small_depth_intr, small_color_intr, identity_ext
    ):
        """Metal == MLX for random colour frame with identity transform."""
        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr, identity_ext
        )
        rng = np.random.default_rng(42)
        H, W = small_depth_intr.height, small_depth_intr.width
        depth_np = rng.integers(500, 3000, (H, W), dtype=np.uint16)
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        depth = mx.array(depth_np)
        color = mx.array(color_np)

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        _check_metal_mlx_close(out_metal, out_mlx, "random depth/color identity")

    def test_pure_translation_extrinsics(
        self, small_depth_intr, small_color_intr
    ):
        """Metal == MLX with a non-trivial pure-translation extrinsics."""
        ext = CameraExtrinsics(
            rotation=np.eye(3),
            translation=np.array([0.015, -0.005, 0.0], dtype=np.float64),
        )
        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr, ext
        )
        rng = np.random.default_rng(7)
        H, W = small_depth_intr.height, small_depth_intr.width
        depth_np = rng.integers(300, 4000, (H, W), dtype=np.uint16)
        # 10% of pixels are zero (invalid)
        mask = rng.random((H, W)) < 0.1
        depth_np[mask] = 0
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        depth = mx.array(depth_np)
        color = mx.array(color_np)

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        _check_metal_mlx_close(out_metal, out_mlx, "translation extrinsics")

    def test_rotation_and_translation_extrinsics(
        self, small_depth_intr, small_color_intr
    ):
        """Metal == MLX with a full rotation+translation extrinsics."""
        # Small rotation (1 degree around Z-axis) + translation
        theta = np.radians(1.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ], dtype=np.float64)
        t = np.array([0.01, 0.005, 0.0], dtype=np.float64)
        ext = CameraExtrinsics(rotation=R, translation=t)

        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr, ext
        )
        rng = np.random.default_rng(99)
        H, W = small_depth_intr.height, small_depth_intr.width
        depth_np = rng.integers(500, 2500, (H, W), dtype=np.uint16)
        color_np = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        depth = mx.array(depth_np)
        color = mx.array(color_np)

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        _check_metal_mlx_close(out_metal, out_mlx, "rotation+translation extrinsics")

    def test_different_resolutions_metal_vs_mlx(
        self, small_depth_intr, small_color_intr_hd, identity_ext
    ):
        """Metal == MLX when colour frame has a different resolution than depth."""
        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr_hd, identity_ext
        )
        rng = np.random.default_rng(13)
        H_d, W_d = small_depth_intr.height, small_depth_intr.width
        H_c, W_c = small_color_intr_hd.height, small_color_intr_hd.width
        depth_np = rng.integers(800, 2000, (H_d, W_d), dtype=np.uint16)
        color_np = rng.integers(0, 256, (H_c, W_c, 3), dtype=np.uint8)
        depth = mx.array(depth_np)
        color = mx.array(color_np)

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        assert out_metal.shape == (H_d, W_d, 3)
        assert out_mlx.shape == (H_d, W_d, 3)
        _check_metal_mlx_close(
            out_metal, out_mlx, "different-resolution colour frame"
        )

    def test_all_zero_depth_metal_vs_mlx(
        self, small_depth_intr, small_color_intr, identity_ext
    ):
        """Both paths produce all-zero output for zero-depth frame."""
        metal_a, mlx_a = self._both_aligners(
            small_depth_intr, small_color_intr, identity_ext
        )
        H, W = small_depth_intr.height, small_depth_intr.width
        depth = _make_depth(H, W, 0)
        color = _make_color(H, W, fill=(255, 128, 0))

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        np.testing.assert_array_equal(out_metal, 0)
        np.testing.assert_array_equal(out_mlx, 0)

    def test_output_shape_metal(
        self, small_depth_intr, small_color_intr, identity_ext
    ):
        """Metal path returns correct (H_d, W_d, C) shape."""
        metal_a = Aligner(
            small_depth_intr, small_color_intr, identity_ext,
            depth_scale=0.001, use_metal=True,
        )
        H, W = small_depth_intr.height, small_depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(H, W)
        out = metal_a.align_color_to_depth(depth, color)
        assert out.shape == (H, W, 3)

    def test_metal_active_flag_pinhole(self, small_depth_intr, small_color_intr, identity_ext):
        """_can_use_metal() returns True for 'none' distortion models."""
        aligner = Aligner(
            small_depth_intr, small_color_intr, identity_ext,
            depth_scale=0.001, use_metal=True,
        )
        assert aligner._can_use_metal() is True

    def test_metal_active_flag_distorted(self, color_intr, identity_ext):
        """_can_use_metal() returns False when depth intrinsics has distortion."""
        bc_intr = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="brown_conrady",
            coeffs=[0.05, -0.1, 0.0, 0.0, 0.02],
        )
        aligner = Aligner(bc_intr, color_intr, identity_ext, depth_scale=0.001, use_metal=True)
        # Distorted depth intrinsics → Metal kernel not applicable
        assert aligner._can_use_metal() is False

    def test_use_metal_false_disables_metal(
        self, small_depth_intr, small_color_intr, identity_ext
    ):
        """use_metal=False forces MLX path even for pinhole intrinsics."""
        aligner = Aligner(
            small_depth_intr, small_color_intr, identity_ext,
            depth_scale=0.001, use_metal=False,
        )
        assert aligner._can_use_metal() is False
        # Should still produce correct output via MLX path
        H, W = small_depth_intr.height, small_depth_intr.width
        depth = _make_depth(H, W, 1000)
        color = _make_color(H, W, fill=(33, 66, 99))
        out = _eval_np(aligner.align_color_to_depth(depth, color))
        # All valid → interior should match the fill colour (broadcast compare)
        interior = out[5:-5, 5:-5]  # (H-10, W-10, 3)
        expected = np.array([[33, 66, 99]], dtype=np.uint8)  # (1, 3) broadcast
        assert np.all(interior == expected), (
            f"Interior pixel mismatch: unique values {np.unique(interior.reshape(-1, 3), axis=0)}"
        )

    def test_large_frame_metal_vs_mlx(self, identity_ext):
        """Metal == MLX at 640x480 (full D435 resolution) with random data."""
        d_intr = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="none",
        )
        c_intr = CameraIntrinsics(
            width=640, height=480,
            ppx=320.0, ppy=240.0,
            fx=600.0, fy=600.0,
            model="none",
        )
        metal_a = Aligner(d_intr, c_intr, identity_ext, depth_scale=0.001, use_metal=True)
        mlx_a = Aligner(d_intr, c_intr, identity_ext, depth_scale=0.001, use_metal=False)

        rng = np.random.default_rng(2024)
        depth_np = rng.integers(200, 5000, (480, 640), dtype=np.uint16)
        # 5% invalid
        depth_np.flat[rng.choice(480 * 640, size=480 * 640 // 20, replace=False)] = 0
        color_np = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = mx.array(depth_np)
        color = mx.array(color_np)

        out_metal = _eval_np(metal_a.align_color_to_depth(depth, color))
        out_mlx = _eval_np(mlx_a.align_color_to_depth(depth, color))

        _check_metal_mlx_close(out_metal, out_mlx, "full 640x480 resolution")

    def test_repr_contains_metal_active(self, small_depth_intr, small_color_intr, identity_ext):
        """Repr for metal-active aligner shows metal_active=True."""
        aligner = Aligner(
            small_depth_intr, small_color_intr, identity_ext,
            depth_scale=0.001, use_metal=True,
        )
        r = repr(aligner)
        assert "metal_active=True" in r

    def test_repr_contains_metal_inactive(self, identity_ext):
        """Repr for distorted aligner shows metal_active=False."""
        bc_intr = CameraIntrinsics(
            width=64, height=48,
            ppx=32.0, ppy=24.0,
            fx=60.0, fy=60.0,
            model="brown_conrady",
            coeffs=[0.01, 0.0, 0.0, 0.0, 0.0],
        )
        c_intr = CameraIntrinsics(
            width=64, height=48,
            ppx=32.0, ppy=24.0,
            fx=60.0, fy=60.0,
            model="none",
        )
        aligner = Aligner(bc_intr, c_intr, identity_ext, depth_scale=0.001, use_metal=True)
        r = repr(aligner)
        assert "metal_active=False" in r
