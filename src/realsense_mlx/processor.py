"""End-to-end RealSense depth processing pipeline.

Chains filter → point cloud → mesh seamlessly and handles
decimation-adjusted intrinsics automatically.

The central problem this module solves
---------------------------------------
When ``DecimationFilter(scale=2)`` reduces a 480×640 frame to 240×320,
the raw ``CameraIntrinsics`` (which still describe the full-resolution lens)
produce incorrect 3-D coordinates if fed directly into
``PointCloudGenerator``.  ``RealsenseProcessor`` calls
``DecimationFilter.adjust_intrinsics`` after every ``process()`` call so the
downstream generators always receive geometrically consistent intrinsics.

Usage
-----
::

    from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    from realsense_mlx.processor import RealsenseProcessor
    import mlx.core as mx

    intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
    proc = RealsenseProcessor(intr, depth_scale=0.001)

    depth = mx.array(raw_uint16_frame)           # (480, 640)
    result = proc.process(depth)

    # result.filtered_depth  — (240, 320) uint16  (decimated & filtered)
    # result.points          — (240, 320, 3) float32  point cloud
    # result.colored_depth   — (240, 320, 3) uint8    colourised
    # result.intrinsics      — decimation-adjusted CameraIntrinsics
    # result.processing_time_ms — wall-clock milliseconds

    proc.export_ply(result, "/tmp/scene.ply")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from realsense_mlx.filters.colorizer import DepthColorizer
from realsense_mlx.filters.pipeline import DepthPipeline, PipelineConfig
from realsense_mlx.geometry.align import Aligner
from realsense_mlx.geometry.intrinsics import CameraExtrinsics, CameraIntrinsics
from realsense_mlx.geometry.mesh import DepthMeshGenerator
from realsense_mlx.geometry.pointcloud import PointCloudGenerator
from realsense_mlx.utils.depth_stats import DepthStats

__all__ = ["ProcessingResult", "RealsenseProcessor"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ProcessingResult:
    """Container for all outputs produced by :class:`RealsenseProcessor`.

    Attributes
    ----------
    filtered_depth:
        ``(H', W')`` uint16 depth frame after the full filter pipeline.
        ``H'`` and ``W'`` reflect decimation: if ``scale=2`` was used on a
        480×640 input then ``filtered_depth.shape == (240, 320)``.
    points:
        ``(H', W', 3)`` float32 XYZ point cloud in metres (camera frame),
        or ``None`` when ``enable_pointcloud=False``.
    colored_depth:
        ``(H', W', 3)`` uint8 RGB colourisation of ``filtered_depth``, or
        ``None`` when ``enable_colorize=False``.
    aligned_color:
        ``(H', W', 3)`` uint8 colour frame resampled into the depth frame,
        or ``None`` when no ``color`` frame was provided or alignment is not
        configured.
    vertices:
        ``(N, 3)`` float32 mesh vertex positions, or ``None`` when
        ``enable_mesh=False``.
    faces:
        ``(M, 3)`` int32 mesh face indices, or ``None`` when
        ``enable_mesh=False``.
    normals:
        ``(N, 3)`` float32 per-vertex normals, or ``None`` when
        ``enable_mesh=False``.
    stats:
        Dictionary of depth statistics from :class:`~realsense_mlx.utils.depth_stats.DepthStats`,
        or ``None`` when ``enable_stats=False``.
    intrinsics:
        :class:`~realsense_mlx.geometry.intrinsics.CameraIntrinsics` adjusted
        for the actual output resolution (accounts for decimation).
    processing_time_ms:
        Wall-clock time in milliseconds for the full ``process()`` call,
        including all enabled stages.
    """

    filtered_depth: mx.array
    points: Optional[mx.array] = None
    colored_depth: Optional[mx.array] = None
    aligned_color: Optional[mx.array] = None
    vertices: Optional[mx.array] = None
    faces: Optional[mx.array] = None
    normals: Optional[mx.array] = None
    stats: Optional[dict] = None
    intrinsics: Optional[CameraIntrinsics] = None
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class RealsenseProcessor:
    """End-to-end depth processing: filter → point cloud → mesh → export.

    Wraps the full RealSense MLX processing stack into a single object that
    handles decimation-adjusted intrinsics automatically.  The caller only
    needs to supply the original (full-resolution) intrinsics once; the
    processor derives decimated intrinsics on the fly and passes them to the
    ``PointCloudGenerator`` and ``Aligner``.

    Parameters
    ----------
    depth_intrinsics:
        Full-resolution depth camera intrinsics.  These are adjusted
        internally after each ``process()`` call to match the decimated
        output resolution.
    depth_scale:
        Metres per raw uint16 count.  Typical value for D-series: ``0.001``.
    color_intrinsics:
        Intrinsics of the colour sensor.  Required only when a colour frame
        is passed to ``process()`` and ``enable_alignment=True`` (the default
        when colour intrinsics are provided).
    extrinsics:
        Rigid transform from depth camera frame to colour camera frame.
        Defaults to identity when not provided.
    pipeline_config:
        Configuration for the ``DepthPipeline`` (decimation scale, filter
        parameters, etc.).  Defaults to ``PipelineConfig()`` which uses
        ``decimation_scale=2``.
    enable_pointcloud:
        When ``True`` (default), generate a ``(H', W', 3)`` XYZ point cloud.
    enable_mesh:
        When ``True``, triangulate the point cloud into a mesh.  Requires
        ``enable_pointcloud=True`` (automatically enabled).
    enable_colorize:
        When ``True`` (default), colourize the filtered depth frame.
    enable_stats:
        When ``True``, compute depth statistics and attach to the result.
    colormap:
        Name of the depth colourmap to use (see ``DepthColorizer.COLORMAPS``).
        Default ``"jet"``.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from realsense_mlx.geometry.intrinsics import CameraIntrinsics
    >>> from realsense_mlx.processor import RealsenseProcessor
    >>> intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
    >>> proc = RealsenseProcessor(intr, depth_scale=0.001)
    >>> depth = mx.zeros((480, 640), dtype=mx.uint16)
    >>> result = proc.process(depth)
    >>> result.filtered_depth.shape
    (240, 320)
    >>> result.points.shape
    (240, 320, 3)
    """

    def __init__(
        self,
        depth_intrinsics: CameraIntrinsics,
        depth_scale: float = 0.001,
        color_intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        enable_pointcloud: bool = True,
        enable_mesh: bool = False,
        enable_colorize: bool = True,
        enable_stats: bool = False,
        colormap: str = "jet",
    ) -> None:
        if depth_scale <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale}")

        self._depth_intrinsics = depth_intrinsics
        self._depth_scale = float(depth_scale)
        self._color_intrinsics = color_intrinsics
        self._extrinsics = extrinsics or CameraExtrinsics.identity()
        self.pipeline_config = pipeline_config or PipelineConfig()

        # Feature flags
        self.enable_pointcloud = enable_pointcloud
        self.enable_mesh = enable_mesh
        # Mesh requires a point cloud as input
        if enable_mesh:
            self.enable_pointcloud = True
        self.enable_colorize = enable_colorize
        self.enable_stats = enable_stats
        self.colormap = colormap

        # Build sub-components
        self._pipeline = DepthPipeline(self.pipeline_config)
        self._colorizer: Optional[DepthColorizer] = None
        self._pc_gen: Optional[PointCloudGenerator] = None
        self._mesh_gen: Optional[DepthMeshGenerator] = None
        self._aligner: Optional[Aligner] = None

        if self.enable_colorize:
            self._colorizer = DepthColorizer(
                colormap=colormap,
                depth_units=depth_scale,
            )

        if self.enable_mesh:
            self._mesh_gen = DepthMeshGenerator()

        # Point cloud generator and aligner are built lazily after the first
        # process() call because we need the actual decimated intrinsics.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        depth: mx.array,
        color: Optional[mx.array] = None,
    ) -> ProcessingResult:
        """Run the full processing pipeline on one frame.

        Parameters
        ----------
        depth:
            ``(H, W)`` uint16 raw depth frame from the sensor.
        color:
            Optional ``(H_c, W_c, 3)`` uint8 colour frame.  When provided
            and ``color_intrinsics`` was set in the constructor, the colour
            frame is aligned to the (decimated) depth frame.

        Returns
        -------
        ProcessingResult
            All requested outputs.  Fields that are disabled are ``None``.

        Raises
        ------
        ValueError
            If ``depth`` is not a 2-D array.
        """
        if depth.ndim != 2:
            raise ValueError(
                f"RealsenseProcessor.process() expects 2-D (H, W) depth, "
                f"got shape {depth.shape}"
            )

        t_start = time.perf_counter()

        # --- Step 1: filter pipeline (decimation + spatial/temporal + hole fill)
        filtered = self._pipeline.process(depth)
        mx.eval(filtered)

        # --- Step 2: derive decimation-adjusted intrinsics
        dec_filter = self._pipeline.decimation
        dec_intr = dec_filter.adjust_intrinsics(self._depth_intrinsics)

        # --- Step 3: lazily build point cloud generator with decimated intrinsics
        self._ensure_pc_gen(dec_intr)

        # --- Step 4: optional colour alignment
        aligned_color: Optional[mx.array] = None
        if color is not None and self._color_intrinsics is not None:
            aligned_color = self._align_color(filtered, color, dec_intr)

        # --- Step 5: optional point cloud
        points: Optional[mx.array] = None
        if self.enable_pointcloud and self._pc_gen is not None:
            points = self._pc_gen.generate(filtered)
            mx.eval(points)

        # --- Step 6: optional mesh
        vertices: Optional[mx.array] = None
        faces: Optional[mx.array] = None
        normals: Optional[mx.array] = None
        if self.enable_mesh and points is not None and self._mesh_gen is not None:
            vertices, faces = self._mesh_gen.generate(points)
            mx.eval(vertices, faces)
            if faces.shape[0] > 0:
                normals = self._mesh_gen.compute_normals(vertices, faces)
                mx.eval(normals)

        # --- Step 7: optional colourisation
        colored_depth: Optional[mx.array] = None
        if self.enable_colorize and self._colorizer is not None:
            colored_depth = self._colorizer.colorize(filtered)
            mx.eval(colored_depth)

        # --- Step 8: optional statistics
        stats: Optional[dict] = None
        if self.enable_stats:
            stats = DepthStats.compute(filtered, depth_scale=self._depth_scale)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return ProcessingResult(
            filtered_depth=filtered,
            points=points,
            colored_depth=colored_depth,
            aligned_color=aligned_color,
            vertices=vertices,
            faces=faces,
            normals=normals,
            stats=stats,
            intrinsics=dec_intr,
            processing_time_ms=elapsed_ms,
        )

    def export_ply(self, result: ProcessingResult, path: str) -> int:
        """Export a :class:`ProcessingResult` to a binary PLY file.

        If the result contains mesh data (``vertices`` and ``faces`` are
        populated) the mesh is exported.  Otherwise the point cloud
        (``points``) is written as a plain vertex cloud.

        Parameters
        ----------
        result:
            Output from a previous :meth:`process` call.
        path:
            Destination file path.  Parent directories are created
            automatically.

        Returns
        -------
        int
            Number of faces written (mesh export) or number of vertices
            written (point cloud export).

        Raises
        ------
        ValueError
            When neither ``points`` nor ``vertices``/``faces`` are available.
        """
        if result.vertices is not None and result.faces is not None:
            if self._pc_gen is None:
                raise ValueError(
                    "PointCloudGenerator not initialised — call process() first."
                )
            colors = result.colored_depth if result.aligned_color is None else result.aligned_color
            return self._pc_gen.export_ply_mesh(
                result.vertices,
                result.faces,
                path,
                colors=colors,
                normals=result.normals,
            )

        if result.points is not None:
            if self._pc_gen is None:
                raise ValueError(
                    "PointCloudGenerator not initialised — call process() first."
                )
            colors = result.colored_depth if result.aligned_color is None else result.aligned_color
            return self._pc_gen.export_ply(
                result.points,
                path,
                colors=colors,
            )

        raise ValueError(
            "ProcessingResult has neither points nor vertices/faces. "
            "Enable enable_pointcloud=True or enable_mesh=True."
        )

    def export_obj(self, result: ProcessingResult, path: str) -> int:
        """Export a :class:`ProcessingResult` to a Wavefront OBJ file.

        If the result contains mesh data (``vertices`` and ``faces``) the mesh
        is exported.  Otherwise the point cloud is written as vertex-only OBJ.

        Parameters
        ----------
        result:
            Output from a previous :meth:`process` call.
        path:
            Destination file path.

        Returns
        -------
        int
            Number of vertices written.

        Raises
        ------
        ValueError
            When neither ``points`` nor ``vertices``/``faces`` are available.
        """
        if self._pc_gen is None:
            raise ValueError(
                "PointCloudGenerator not initialised — call process() first."
            )

        if result.vertices is not None and result.faces is not None:
            colors = result.colored_depth if result.aligned_color is None else result.aligned_color
            return self._pc_gen.export_obj(
                result.vertices,
                path,
                faces=result.faces,
                colors=colors,
            )

        if result.points is not None:
            colors = result.colored_depth if result.aligned_color is None else result.aligned_color
            return self._pc_gen.export_obj(
                result.points,
                path,
                colors=colors,
            )

        raise ValueError(
            "ProcessingResult has neither points nor vertices/faces. "
            "Enable enable_pointcloud=True or enable_mesh=True."
        )

    def reset(self) -> None:
        """Reset temporal filter state.

        Call after a scene cut, stream restart, or resolution change.
        """
        self._pipeline.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def depth_intrinsics(self) -> CameraIntrinsics:
        """The original (full-resolution) depth intrinsics."""
        return self._depth_intrinsics

    @property
    def depth_scale(self) -> float:
        """Depth scale in metres per raw count."""
        return self._depth_scale

    @property
    def pipeline(self) -> DepthPipeline:
        """The underlying :class:`~realsense_mlx.filters.pipeline.DepthPipeline`."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_pc_gen(self, dec_intr: CameraIntrinsics) -> None:
        """Build or rebuild the point cloud generator when intrinsics change.

        This handles the case where the decimation scale changes (e.g. via
        ``pipeline.reconfigure``) by comparing the new intrinsics dimensions
        against the cached generator.
        """
        if not self.enable_pointcloud:
            return

        if self._pc_gen is None:
            self._pc_gen = PointCloudGenerator(dec_intr, self._depth_scale)
            return

        cached = self._pc_gen.intrinsics
        if cached.width != dec_intr.width or cached.height != dec_intr.height:
            # Resolution changed — rebuild (also clears cached grids)
            self._pc_gen = PointCloudGenerator(dec_intr, self._depth_scale)

    def _ensure_aligner(
        self,
        dec_intr: CameraIntrinsics,
        color_channels: int,
    ) -> Optional[Aligner]:
        """Build or rebuild the Aligner for the current decimated intrinsics.

        Returns ``None`` when no colour intrinsics have been configured.
        """
        if self._color_intrinsics is None:
            return None

        if self._aligner is None:
            self._aligner = Aligner(
                depth_intrinsics=dec_intr,
                color_intrinsics=self._color_intrinsics,
                depth_to_color_extrinsics=self._extrinsics,
                depth_scale=self._depth_scale,
            )
            return self._aligner

        # Rebuild when the depth-side intrinsics have changed (decimation scale)
        cached_d = self._aligner.depth_intrinsics
        if cached_d.width != dec_intr.width or cached_d.height != dec_intr.height:
            self._aligner = Aligner(
                depth_intrinsics=dec_intr,
                color_intrinsics=self._color_intrinsics,
                depth_to_color_extrinsics=self._extrinsics,
                depth_scale=self._depth_scale,
            )

        return self._aligner

    def _align_color(
        self,
        depth: mx.array,
        color: mx.array,
        dec_intr: CameraIntrinsics,
    ) -> Optional[mx.array]:
        """Align *color* to the decimated *depth* frame.

        Returns ``None`` on any error (e.g. wrong shape) rather than raising,
        so that the rest of the pipeline continues cleanly.
        """
        C = color.shape[2] if color.ndim == 3 else 1
        aligner = self._ensure_aligner(dec_intr, C)
        if aligner is None:
            return None

        try:
            aligned = aligner.align_color_to_depth(depth, color)
            mx.eval(aligned)
            return aligned
        except ValueError:
            # Shape mismatch (e.g. caller passed wrong colour resolution)
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dec_scale = self.pipeline_config.decimation_scale
        return (
            f"RealsenseProcessor("
            f"{self._depth_intrinsics.width}x{self._depth_intrinsics.height}, "
            f"scale={self._depth_scale}, "
            f"decimation={dec_scale}, "
            f"pc={self.enable_pointcloud}, "
            f"mesh={self.enable_mesh}, "
            f"colorize={self.enable_colorize})"
        )
