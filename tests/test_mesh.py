"""Tests for DepthMeshGenerator and the mesh export/normal methods added to
PointCloudGenerator.

Coverage
--------
* Flat plane (all valid)  → exactly 2*(H-1)*(W-1) triangles
* Plane with hole (one vertex zeroed) → fewer triangles
* Depth discontinuity → triangles across the gap removed by edge-length filter
* compute_normals on a flat XY plane → all normals point toward the camera (+Z)
* export_obj → valid Wavefront OBJ header parseable from file
* export_ply_mesh → PLY face count in header matches return value
* DepthMeshGenerator.compute_normals mirrors PointCloudGenerator.compute_normals
* Invalid inputs raise ValueError
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from realsense_mlx.geometry.mesh import DepthMeshGenerator
from realsense_mlx.geometry.intrinsics import CameraIntrinsics
from realsense_mlx.geometry.pointcloud import PointCloudGenerator


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _flat_plane(H: int, W: int, z: float = 1.0) -> mx.array:
    """Return a (H, W, 3) organised point cloud of a flat plane at depth z.

    X = column index (float), Y = row index (float), Z = z.
    All pixels are valid (non-zero Z).
    """
    rows = np.arange(H, dtype=np.float32)
    cols = np.arange(W, dtype=np.float32)
    xx, yy = np.meshgrid(cols, rows)          # both (H, W)
    zz = np.full((H, W), z, dtype=np.float32)
    pts = np.stack([xx, yy, zz], axis=-1)     # (H, W, 3)
    return mx.array(pts)


@pytest.fixture
def mesh_gen() -> DepthMeshGenerator:
    return DepthMeshGenerator(max_edge_length=2.0)


@pytest.fixture
def pinhole_gen() -> PointCloudGenerator:
    intr = CameraIntrinsics(
        width=8, height=6,
        ppx=4.0, ppy=3.0,
        fx=1.0, fy=1.0,
        model="none",
    )
    return PointCloudGenerator(intr, depth_scale=0.001)


# ---------------------------------------------------------------------------
# 1. Flat plane — full triangle count
# ---------------------------------------------------------------------------


class TestFlatPlaneTriangleCount:
    def test_full_plane_triangle_count(self, mesh_gen):
        """All-valid flat plane → exactly 2*(H-1)*(W-1) triangles."""
        H, W = 10, 12
        pts = _flat_plane(H, W)
        verts, faces = mesh_gen.generate(pts)
        expected = 2 * (H - 1) * (W - 1)
        mx.eval(faces)
        assert faces.shape == (expected, 3), (
            f"Expected {expected} faces, got {faces.shape[0]}"
        )

    def test_vertex_count_equals_hw(self, mesh_gen):
        """Vertex array is always H*W regardless of filtering."""
        H, W = 8, 6
        pts = _flat_plane(H, W)
        verts, _ = mesh_gen.generate(pts)
        mx.eval(verts)
        assert verts.shape == (H * W, 3)

    def test_face_indices_in_range(self, mesh_gen):
        """All face indices must reference valid vertex indices."""
        H, W = 8, 6
        pts = _flat_plane(H, W)
        verts, faces = mesh_gen.generate(pts)
        mx.eval(verts, faces)
        faces_np = np.array(faces, copy=False)
        assert faces_np.min() >= 0
        assert faces_np.max() < H * W

    def test_face_dtype_int32(self, mesh_gen):
        H, W = 6, 6
        pts = _flat_plane(H, W)
        _, faces = mesh_gen.generate(pts)
        assert faces.dtype == mx.int32

    def test_minimum_grid_2x2(self, mesh_gen):
        """2×2 grid → exactly 2 triangles."""
        pts = _flat_plane(2, 2)
        _, faces = mesh_gen.generate(pts)
        mx.eval(faces)
        assert faces.shape[0] == 2


# ---------------------------------------------------------------------------
# 2. Plane with hole — fewer triangles
# ---------------------------------------------------------------------------


class TestPlaneWithHole:
    def test_single_zero_vertex_removes_triangles(self, mesh_gen):
        """Setting one vertex's depth to 0 removes all triangles that use it."""
        H, W = 5, 5
        pts_np = np.ones((H, W, 3), dtype=np.float32)
        pts_np[2, 2, :] = 0.0   # one invalid vertex

        # Triangles incident to (2,2) in the grid:
        # That pixel is used by up to 6 surrounding quads → up to 6 triangles.
        pts_orig = _flat_plane(H, W)
        _, faces_full = mesh_gen.generate(pts_orig)
        mx.eval(faces_full)
        full_count = faces_full.shape[0]

        pts_hole = mx.array(pts_np)
        _, faces_hole = mesh_gen.generate(pts_hole)
        mx.eval(faces_hole)

        assert faces_hole.shape[0] < full_count, (
            "Zeroing a vertex should remove incident triangles"
        )

    def test_entire_row_zeroed(self, mesh_gen):
        """Zeroing an entire interior row halves the available quads roughly."""
        H, W = 7, 7
        pts_np = np.ones((H, W, 3), dtype=np.float32)
        pts_np[3, :, :] = 0.0   # entire middle row zeroed
        pts = mx.array(pts_np)

        _, faces = mesh_gen.generate(pts)
        mx.eval(faces)
        # Row 3 belongs to quads in rows [2,3] and [3,4] → 2*(W-1) quads removed
        # per set; two triangle sets per quad = 2*2*(W-1) = 4*(W-1) fewer faces
        expected_max = 2 * (H - 1) * (W - 1) - 4 * (W - 1)
        assert faces.shape[0] <= expected_max


# ---------------------------------------------------------------------------
# 3. Depth discontinuity — edge-length filter
# ---------------------------------------------------------------------------


class TestDepthDiscontinuity:
    def test_large_step_removes_triangles(self):
        """A large depth step exceeding max_edge_length removes those triangles."""
        H, W = 5, 6
        gen = DepthMeshGenerator(max_edge_length=0.1)

        # Left half at Z=1.0, right half at Z=2.0 — step of 1.0 >> 0.1
        pts_np = np.ones((H, W, 3), dtype=np.float32)
        pts_np[:, W // 2:, 2] = 2.0

        pts = mx.array(pts_np)
        verts_tight, faces_tight = gen.generate(pts)
        mx.eval(faces_tight)

        # All triangles that straddle the boundary should be removed
        # With a generous max_edge_length they would all survive
        gen_loose = DepthMeshGenerator(max_edge_length=10.0)
        _, faces_loose = gen_loose.generate(pts)
        mx.eval(faces_loose)

        assert faces_tight.shape[0] < faces_loose.shape[0], (
            "Tight edge-length limit should remove boundary triangles"
        )

    def test_zero_max_edge_length_raises(self):
        with pytest.raises(ValueError, match="max_edge_length must be positive"):
            DepthMeshGenerator(max_edge_length=0.0)

    def test_negative_max_edge_length_raises(self):
        with pytest.raises(ValueError, match="max_edge_length must be positive"):
            DepthMeshGenerator(max_edge_length=-0.05)

    def test_all_within_threshold_no_filtering(self):
        """When all edge lengths are short, no triangles are removed by filter."""
        H, W = 4, 4
        gen = DepthMeshGenerator(max_edge_length=100.0)   # very generous
        pts = _flat_plane(H, W, z=1.0)
        _, faces = gen.generate(pts)
        mx.eval(faces)
        assert faces.shape[0] == 2 * (H - 1) * (W - 1)

    def test_wrong_shape_raises(self, mesh_gen):
        bad = mx.zeros((10, 10), dtype=mx.float32)
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            mesh_gen.generate(bad)

    def test_wrong_channel_count_raises(self, mesh_gen):
        bad = mx.zeros((10, 10, 4), dtype=mx.float32)
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            mesh_gen.generate(bad)


# ---------------------------------------------------------------------------
# 4. Normal computation
# ---------------------------------------------------------------------------


class TestNormalComputation:
    def test_flat_xy_plane_normals_point_toward_camera(self, mesh_gen):
        """Flat XY-plane (Z=const) — all normals should be [0, 0, ±1]."""
        H, W = 6, 6
        pts = _flat_plane(H, W, z=1.0)
        verts, faces = mesh_gen.generate(pts)
        normals = mesh_gen.compute_normals(verts, faces)

        mx.eval(normals)
        nrm_np = np.array(normals, copy=False)

        # Only check vertices that are referenced by faces
        faces_np = np.array(faces, copy=False)
        used_verts = np.unique(faces_np)
        used_normals = nrm_np[used_verts]

        # For a flat XY plane the cross product of row and column vectors
        # gives [0, 0, +1] or [0, 0, -1] depending on winding order.
        z_abs = np.abs(used_normals[:, 2])
        xy_mag = np.sqrt(used_normals[:, 0] ** 2 + used_normals[:, 1] ** 2)

        np.testing.assert_allclose(z_abs, 1.0, atol=1e-5,
                                   err_msg="Z-component of normals should be ±1")
        np.testing.assert_allclose(xy_mag, 0.0, atol=1e-5,
                                   err_msg="XY-components should be 0")

    def test_normals_are_unit_length(self, mesh_gen):
        """Per-vertex normals must have unit length for referenced vertices."""
        H, W = 8, 8
        pts = _flat_plane(H, W)
        verts, faces = mesh_gen.generate(pts)
        normals = mesh_gen.compute_normals(verts, faces)

        mx.eval(normals)
        nrm_np = np.array(normals, copy=False)
        faces_np = np.array(faces, copy=False)
        used = np.unique(faces_np)

        norms = np.linalg.norm(nrm_np[used], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                   err_msg="All normals should be unit-length")

    def test_empty_faces_returns_zero_normals(self, mesh_gen):
        """When faces is empty, compute_normals returns zero array."""
        verts = mx.zeros((10, 3), dtype=mx.float32)
        faces = mx.zeros((0, 3), dtype=mx.int32)
        normals = mesh_gen.compute_normals(verts, faces)
        mx.eval(normals)
        nrm_np = np.array(normals, copy=False)
        np.testing.assert_array_equal(nrm_np, 0.0)

    def test_pcg_compute_normals_matches_mesh_generator(self, mesh_gen, pinhole_gen):
        """PointCloudGenerator.compute_normals and DepthMeshGenerator.compute_normals
        must give identical results for the same input."""
        H, W = 6, 8
        depth_np = np.full((H, W), 1000, dtype=np.uint16)
        depth = mx.array(depth_np)
        pts = pinhole_gen.generate(depth)

        verts, faces = mesh_gen.generate(pts)
        n1 = mesh_gen.compute_normals(verts, faces)
        n2 = pinhole_gen.compute_normals(verts, faces)

        mx.eval(n1, n2)
        np.testing.assert_allclose(
            np.array(n1, copy=False),
            np.array(n2, copy=False),
            atol=1e-6,
            err_msg="Both compute_normals implementations must agree",
        )


# ---------------------------------------------------------------------------
# 5. OBJ export
# ---------------------------------------------------------------------------


class TestOBJExport:
    def _parse_obj(self, path: Path) -> dict:
        """Minimal OBJ parser: count 'v ' and 'f ' lines."""
        vertex_lines: list[list[str]] = []
        face_lines: list[list[str]] = []
        with open(path, "r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    vertex_lines.append(line.split())
                elif line.startswith("f "):
                    face_lines.append(line.split())
        return {"vertices": vertex_lines, "faces": face_lines}

    def test_obj_header_present(self, pinhole_gen):
        """OBJ file must begin with the '# Wavefront OBJ' comment."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cloud.obj"
            pinhole_gen.export_obj(pts, path)
            with open(path, "r") as f:
                first_line = f.readline()
        assert "OBJ" in first_line or "Wavefront" in first_line

    def test_obj_vertex_count(self, pinhole_gen):
        """Number of 'v' lines must equal the return value."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cloud.obj"
            n = pinhole_gen.export_obj(pts, path)
            info = self._parse_obj(path)

        assert n == H * W
        assert len(info["vertices"]) == H * W

    def test_obj_with_faces(self, mesh_gen, pinhole_gen):
        """When faces are provided, 'f' lines appear with 1-based indices."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.obj"
            pinhole_gen.export_obj(verts, path, faces=faces)
            info = self._parse_obj(path)

        mx.eval(faces)
        n_faces = np.array(faces, copy=False).shape[0]
        assert len(info["faces"]) == n_faces, (
            f"Expected {n_faces} face lines, got {len(info['faces'])}"
        )

        # Verify 1-based indexing: every index token must be >= 1
        for fl in info["faces"]:
            for tok in fl[1:]:
                assert int(tok) >= 1, f"OBJ face index must be 1-based, got {tok}"

    def test_obj_vertex_fields_are_floats(self, pinhole_gen):
        """Each 'v' line must have exactly 3 parseable float fields."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "floats.obj"
            pinhole_gen.export_obj(pts, path)
            info = self._parse_obj(path)

        for vl in info["vertices"]:
            # "v x y z" → 4 tokens; "v x y z r g b" → 7 tokens (with color)
            assert len(vl) in (4, 7), f"Unexpected vertex line: {vl}"
            for tok in vl[1:4]:
                float(tok)   # raises ValueError if not parseable

    def test_obj_with_color_has_extra_fields(self, pinhole_gen):
        """Vertex lines with color should have 7 tokens: v x y z r g b."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        color = mx.zeros((H, W, 3), dtype=mx.uint8)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "colored.obj"
            pinhole_gen.export_obj(pts, path, colors=color)
            info = self._parse_obj(path)

        for vl in info["vertices"]:
            assert len(vl) == 7, f"With color, expected 7 tokens, got {len(vl)}: {vl}"

    def test_obj_parent_dirs_created(self, pinhole_gen):
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)

        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "a" / "b" / "out.obj"
            pinhole_gen.export_obj(pts, nested)
            assert nested.exists()


# ---------------------------------------------------------------------------
# 6. PLY mesh export
# ---------------------------------------------------------------------------


def _parse_ply_header(path: Path) -> dict:
    """Parse binary-little-endian PLY header, return metadata dict."""
    with open(path, "rb") as f:
        raw = f.read()
    end_marker = b"end_header\n"
    end_pos = raw.find(end_marker)
    if end_pos == -1:
        raise ValueError("end_header not found")
    header_text = raw[:end_pos].decode("ascii")
    n_vertices = n_faces = 0
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
    return {
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "header_bytes": end_pos + len(end_marker),
        "raw_header": header_text,
    }


class TestPLYMeshExport:
    def test_face_count_in_header_matches_return(self, mesh_gen, pinhole_gen):
        """PLY header element face count must match the return value of export_ply_mesh."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.ply"
            n = pinhole_gen.export_ply_mesh(verts, faces, path)
            info = _parse_ply_header(path)

        assert info["n_faces"] == n, (
            f"Header says {info['n_faces']} faces, return = {n}"
        )

    def test_vertex_count_in_header(self, mesh_gen, pinhole_gen):
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh_vc.ply"
            pinhole_gen.export_ply_mesh(verts, faces, path)
            info = _parse_ply_header(path)

        mx.eval(verts)
        expected_verts = np.array(verts, copy=False).shape[0]
        assert info["n_vertices"] == expected_verts

    def test_ply_header_starts_with_ply(self, mesh_gen, pinhole_gen):
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hdr.ply"
            pinhole_gen.export_ply_mesh(verts, faces, path)
            info = _parse_ply_header(path)

        assert info["raw_header"].strip().startswith("ply")

    def test_ply_mesh_binary_little_endian(self, mesh_gen, pinhole_gen):
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fmt.ply"
            pinhole_gen.export_ply_mesh(verts, faces, path)
            info = _parse_ply_header(path)

        assert "binary_little_endian" in info["raw_header"]

    def test_ply_mesh_with_normals_header(self, mesh_gen, pinhole_gen):
        """When normals are supplied, PLY header must include nx/ny/nz properties."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)
        normals = mesh_gen.compute_normals(verts, faces)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "normals.ply"
            pinhole_gen.export_ply_mesh(verts, faces, path, normals=normals)
            info = _parse_ply_header(path)

        assert "nx" in info["raw_header"]
        assert "ny" in info["raw_header"]
        assert "nz" in info["raw_header"]

    def test_ply_mesh_with_colors_header(self, mesh_gen, pinhole_gen):
        """When colors are supplied, PLY header must include red/green/blue."""
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)
        colors = mx.zeros((H * W, 3), dtype=mx.uint8)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "colors.ply"
            pinhole_gen.export_ply_mesh(verts, faces, path, colors=colors)
            info = _parse_ply_header(path)

        assert "red" in info["raw_header"]
        assert "green" in info["raw_header"]
        assert "blue" in info["raw_header"]

    def test_ply_mesh_empty_faces(self, pinhole_gen):
        """Zero-face mesh must still produce a valid PLY file."""
        verts = mx.zeros((12, 3), dtype=mx.float32)
        faces = mx.zeros((0, 3), dtype=mx.int32)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.ply"
            n = pinhole_gen.export_ply_mesh(verts, faces, path)
            info = _parse_ply_header(path)

        assert n == 0
        assert info["n_faces"] == 0

    def test_ply_mesh_parent_dirs_created(self, mesh_gen, pinhole_gen):
        H, W = 6, 8
        depth = mx.full((H, W), 1000, dtype=mx.uint16)
        pts = pinhole_gen.generate(depth)
        verts, faces = mesh_gen.generate(pts)

        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "x" / "y" / "mesh.ply"
            pinhole_gen.export_ply_mesh(verts, faces, nested)
            assert nested.exists()
