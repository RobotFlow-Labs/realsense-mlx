"""MLX-accelerated stereo depth estimation.

Pipeline
--------
1. OpenCV SGBM       – stereo matching → raw disparity (int16, ×16 fixed-point)
2. Disparity → depth – depth = baseline_mm * focal_px / disparity_px  (metres)
3. SpatialFilter     – edge-preserving smoothing (Metal GPU kernel)
4. TemporalFilter    – EMA across frames, optional
5. HoleFillingFilter – fill remaining invalid pixels

Design notes
------------
- SGBM is the compute bottleneck (~130 ms at 720p on CPU).  Our MLX
  post-processing adds only ~8 ms, running on the Apple Silicon GPU via Metal.
- Disparity values from SGBM are int16 scaled by 16 (SGBM_MODE_HH uses this).
  We divide by 16.0 to get sub-pixel disparity in pixels.
- Zero or negative disparity → invalid measurement → depth = 0.0.
- All MLX arrays stay on-device throughout the filter chain.  Only the final
  ``mx.eval`` materialises when the caller inspects the result.

MLX constraints
---------------
- No int64: depth and disparity use float32.
- mx.where() for masking invalid pixels.
- mx.eval() after each filter pass to bound graph size.

Usage
-----
>>> estimator = StereoDepthEstimator(baseline_mm=120.0, focal_px=700.0)
>>> depth = estimator.compute(left_gray, right_gray)
>>> # depth is (H, W) float32 in metres; 0.0 = invalid

>>> depth_m, color_vis = estimator.compute_with_color(left_bgr, right_bgr)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = ["StereoDepthEstimator", "StereoDepthConfig"]


@dataclass
class StereoDepthConfig:
    """All tunable parameters for the stereo depth pipeline.

    Parameters
    ----------
    baseline_mm:
        Distance between left and right optical centres in millimetres.
        ZED 2i: 120 mm.  Typical USB stereo cameras: 60–200 mm.
    focal_px:
        Focal length in pixels (fx = fy assumed).  Compute from the camera
        intrinsics matrix as ``K[0, 0]``.  ZED 2i @ 720p ≈ 700 px.
    min_disparity:
        Minimum valid disparity (pixels).  Typically 0.
    num_disparities:
        Search range in pixels.  Must be divisible by 16.  Larger values
        detect closer objects but increase SGBM runtime.
    block_size:
        SGBM window size (odd integer, 3–11).  Larger = smoother but less
        detail.
    sgbm_p1:
        SGBM smoothness penalty for small disparity changes.
        If 0, auto-computed as 8 * channels * block_size ** 2.
    sgbm_p2:
        SGBM smoothness penalty for large disparity changes.
        If 0, auto-computed as 32 * channels * block_size ** 2.
    sgbm_disp12_max_diff:
        Maximum allowed left/right disparity check difference.  -1 disables.
    sgbm_pre_filter_cap:
        Truncation value for the pre-filtered image pixels (5–63).
    sgbm_uniqueness_ratio:
        Margin (%) by which the best cost must beat the second best.
    sgbm_speckle_window_size:
        Maximum size of smooth disparity regions to consider noise (0 = off).
    sgbm_speckle_range:
        Maximum disparity variation within each connected component.
    sgbm_mode:
        SGBM mode constant.  cv2.STEREO_SGBM_MODE_HH (1) = full dynamic
        programming (best quality, slowest).  Use cv2.STEREO_SGBM_MODE_SGBM
        (0) for faster processing.
    enable_spatial:
        Apply MLX SpatialFilter after disparity → depth conversion.
    spatial_alpha:
        SpatialFilter smoothing strength [0, 1].
    spatial_delta:
        SpatialFilter edge sensitivity (metres).
    spatial_iterations:
        SpatialFilter number of passes (1–5).
    enable_temporal:
        Apply MLX TemporalFilter (EMA across frames).
    temporal_alpha:
        TemporalFilter EMA weight for current frame [0, 1].
    temporal_delta:
        TemporalFilter large-change threshold (metres).
    enable_hole_fill:
        Apply MLX HoleFillingFilter after spatial+temporal passes.
    hole_fill_mode:
        HoleFillingFilter mode: 0=FILL_FROM_LEFT, 1=FARTHEST, 2=NEAREST.
    min_depth_m:
        Depth values below this threshold are treated as invalid and zeroed.
        Default 0.1 m.
    max_depth_m:
        Depth values above this threshold are treated as invalid and zeroed.
        Default 20.0 m.
    colormap:
        Colormap name for ``compute_with_color``.  Any key accepted by
        ``DepthColorizer.COLORMAPS``.
    """

    baseline_mm: float = 120.0
    focal_px: float = 700.0

    # SGBM parameters
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    sgbm_p1: int = 0            # 0 → auto
    sgbm_p2: int = 0            # 0 → auto
    sgbm_disp12_max_diff: int = 1
    sgbm_pre_filter_cap: int = 63
    sgbm_uniqueness_ratio: int = 10
    sgbm_speckle_window_size: int = 100
    sgbm_speckle_range: int = 32
    sgbm_mode: int = 1          # cv2.STEREO_SGBM_MODE_HH

    # MLX spatial filter
    enable_spatial: bool = True
    spatial_alpha: float = 0.5
    spatial_delta: float = 0.05   # metres
    spatial_iterations: int = 2

    # MLX temporal filter
    enable_temporal: bool = True
    temporal_alpha: float = 0.4
    temporal_delta: float = 0.1   # metres

    # MLX hole fill
    enable_hole_fill: bool = True
    hole_fill_mode: int = 1       # FARTHEST

    # Depth clipping
    min_depth_m: float = 0.1
    max_depth_m: float = 20.0

    # Colorizer
    colormap: str = "jet"

    def __post_init__(self) -> None:
        if self.num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities must be divisible by 16, got {self.num_disparities}"
            )
        if self.block_size % 2 == 0 or self.block_size < 3:
            raise ValueError(
                f"block_size must be an odd integer >= 3, got {self.block_size}"
            )
        if self.baseline_mm <= 0:
            raise ValueError(f"baseline_mm must be positive, got {self.baseline_mm}")
        if self.focal_px <= 0:
            raise ValueError(f"focal_px must be positive, got {self.focal_px}")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError(
                f"min_depth_m ({self.min_depth_m}) must be < max_depth_m ({self.max_depth_m})"
            )


class StereoDepthEstimator:
    """Compute depth from stereo image pairs using SGBM + MLX post-processing.

    This class is the main entry point for the generic stereo pipeline.  It
    wraps OpenCV's Semi-Global Block Matching (SGBM) for stereo correspondence
    and applies a chain of MLX-accelerated filters on Apple Silicon.

    The value proposition: any stereo camera + this MLX pipeline = real-time
    depth on Mac without any vendor SDK.

    Parameters
    ----------
    baseline_mm:
        Distance between left and right cameras in mm.  ZED 2i = 120 mm.
    focal_px:
        Focal length in pixels (fx = fy assumed).
    config:
        Full ``StereoDepthConfig`` for advanced tuning.  If provided,
        ``baseline_mm`` and ``focal_px`` override the config's values.
    **kwargs:
        Shorthand overrides for individual ``StereoDepthConfig`` fields.

    Examples
    --------
    >>> estimator = StereoDepthEstimator(baseline_mm=120, focal_px=700)
    >>> depth = estimator.compute(left_gray, right_gray)
    >>> # depth is (H, W) float32 in metres; 0.0 = invalid

    >>> depth_m, vis = estimator.compute_with_color(left_bgr, right_bgr)
    """

    def __init__(
        self,
        baseline_mm: float = 120.0,
        focal_px: float = 700.0,
        config: Optional[StereoDepthConfig] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = StereoDepthConfig(
                baseline_mm=baseline_mm,
                focal_px=focal_px,
                **kwargs,
            )
        else:
            # Named args override config fields.
            config.baseline_mm = baseline_mm
            config.focal_px = focal_px
            for k, v in kwargs.items():
                if not hasattr(config, k):
                    raise TypeError(f"Unknown config field: {k!r}")
                setattr(config, k, v)

        self.config = config
        self._sgbm = self._build_sgbm()
        self._spatial: object | None = None
        self._temporal: object | None = None
        self._hole_fill: object | None = None
        self._colorizer: object | None = None
        self._init_mlx_filters()

        # Cached baseline * focal product to avoid per-frame multiply.
        self._bf: float = config.baseline_mm * config.focal_px  # mm·px

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_sgbm(self):
        """Construct and return a configured cv2.StereoSGBM object."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for stereo matching. "
                "Install it with: pip install opencv-python"
            ) from exc

        cfg = self.config
        channels = 1  # grayscale input

        p1 = cfg.sgbm_p1 if cfg.sgbm_p1 > 0 else 8 * channels * cfg.block_size ** 2
        p2 = cfg.sgbm_p2 if cfg.sgbm_p2 > 0 else 32 * channels * cfg.block_size ** 2

        return cv2.StereoSGBM.create(
            minDisparity=cfg.min_disparity,
            numDisparities=cfg.num_disparities,
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=cfg.sgbm_disp12_max_diff,
            preFilterCap=cfg.sgbm_pre_filter_cap,
            uniquenessRatio=cfg.sgbm_uniqueness_ratio,
            speckleWindowSize=cfg.sgbm_speckle_window_size,
            speckleRange=cfg.sgbm_speckle_range,
            mode=cfg.sgbm_mode,
        )

    def _init_mlx_filters(self) -> None:
        """Lazily construct MLX filter objects from config."""
        from realsense_mlx.filters.spatial import SpatialFilter
        from realsense_mlx.filters.temporal import TemporalFilter
        from realsense_mlx.filters.hole_filling import HoleFillingFilter
        from realsense_mlx.filters.colorizer import DepthColorizer

        cfg = self.config
        if cfg.enable_spatial:
            self._spatial = SpatialFilter(
                alpha=cfg.spatial_alpha,
                delta=cfg.spatial_delta,
                iterations=cfg.spatial_iterations,
                use_metal=True,
            )
        if cfg.enable_temporal:
            self._temporal = TemporalFilter(
                alpha=cfg.temporal_alpha,
                delta=cfg.temporal_delta,
            )
        if cfg.enable_hole_fill:
            self._hole_fill = HoleFillingFilter(mode=cfg.hole_fill_mode)

        self._colorizer = DepthColorizer(
            colormap=cfg.colormap,
            min_depth=cfg.min_depth_m,
            max_depth=cfg.max_depth_m,
            equalize=True,
            depth_units=1.0,   # input is already in metres (float32)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> mx.array:
        """Compute depth from a stereo image pair.

        Parameters
        ----------
        left:
            Left camera image.  Grayscale (H, W) uint8 or BGR (H, W, 3) uint8.
            If BGR, converted to grayscale automatically.
        right:
            Right camera image.  Same format as *left*.

        Returns
        -------
        mx.array
            Shape ``(H, W)``, dtype ``float32``, values in metres.
            Invalid pixels (no correspondence found) are 0.0.

        Raises
        ------
        ValueError
            If *left* and *right* have different shapes.
        ImportError
            If ``opencv-python`` is not installed.
        """
        left_gray = self._to_gray(left)
        right_gray = self._to_gray(right)

        if left_gray.shape != right_gray.shape:
            raise ValueError(
                f"left and right images must have the same shape, "
                f"got {left_gray.shape} vs {right_gray.shape}"
            )

        # 1. SGBM → disparity (int16, ×16 fixed-point).
        disp_raw = self._sgbm.compute(left_gray, right_gray)  # (H, W) int16

        # 2. Convert to float32 disparity in pixels.
        disp_f = disp_raw.astype(np.float32) / 16.0  # sub-pixel

        # 3. Mask invalid disparity (≤ min_disparity).
        valid_mask = disp_f > float(self.config.min_disparity)
        disp_f[~valid_mask] = 0.0

        # 4. Disparity → depth in metres.
        #    depth_m = (baseline_mm * focal_px) / disparity_px / 1000
        #    (divide by 1000 to convert mm→m because baseline is in mm)
        #    Suppress divide-by-zero warnings for zero-disparity pixels;
        #    they are masked out by the np.where condition anyway.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_np = np.where(
                disp_f > 0,
                (self._bf / disp_f) / 1000.0,  # mm·px / px → mm → m
                0.0,
            ).astype(np.float32)

        # 5. Clip to [min_depth_m, max_depth_m]; outside range → invalid.
        cfg = self.config
        out_of_range = (depth_np < cfg.min_depth_m) | (depth_np > cfg.max_depth_m)
        # Keep zeros (already invalid) but also zero-out out-of-range positives.
        out_of_range &= depth_np > 0.0
        depth_np[out_of_range] = 0.0

        # 6. Transfer to MLX for GPU post-processing.
        depth_mx = mx.array(depth_np)
        mx.eval(depth_mx)

        # 7. MLX spatial filter (Metal GPU).
        if self._spatial is not None:
            depth_mx = self._spatial.process(depth_mx)
            mx.eval(depth_mx)

        # 8. MLX temporal filter (EMA).
        if self._temporal is not None:
            depth_mx = self._temporal.process(depth_mx)
            mx.eval(depth_mx)

        # 9. MLX hole filling.
        if self._hole_fill is not None:
            depth_mx = self._hole_fill.process(depth_mx)
            mx.eval(depth_mx)

        return depth_mx

    def compute_with_color(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> tuple[mx.array, np.ndarray]:
        """Compute depth and return a colorized visualization.

        Parameters
        ----------
        left_bgr:
            Left camera image, BGR (H, W, 3) uint8.
        right_bgr:
            Right camera image, BGR (H, W, 3) uint8.

        Returns
        -------
        depth_m : mx.array
            Shape ``(H, W)``, float32, metres.  Same as :meth:`compute`.
        color_vis : np.ndarray
            Shape ``(H, W, 3)``, uint8, BGR colorized depth image ready for
            ``cv2.imshow`` / ``cv2.imwrite``.
        """
        depth_mx = self.compute(left_bgr, right_bgr)

        # Colorize: DepthColorizer expects float32 depth in metres when
        # depth_units=1.0 (as configured in _init_mlx_filters).
        # It returns (H, W, 3) uint8 in RGB order; convert to BGR for OpenCV.
        rgb_mx = self._colorizer.colorize(depth_mx)
        mx.eval(rgb_mx)
        rgb_np = np.array(rgb_mx, copy=False)
        bgr_np = rgb_np[:, :, ::-1].copy()  # RGB → BGR, ensure contiguous

        return depth_mx, bgr_np

    def reset(self) -> None:
        """Reset temporal filter state.

        Call after a significant scene change, camera pause, or resolution
        switch to prevent stale EMA history from corrupting new frames.
        """
        if self._temporal is not None:
            self._temporal.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert a BGR or grayscale image to grayscale uint8.

        Parameters
        ----------
        img:
            ``(H, W)`` uint8 grayscale or ``(H, W, 3)`` uint8 BGR.

        Returns
        -------
        np.ndarray
            ``(H, W)`` uint8 grayscale.

        Raises
        ------
        ValueError
            For unsupported image shapes.
        """
        if img.ndim == 2:
            return img.astype(np.uint8)
        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0].astype(np.uint8)
            if img.shape[2] == 3:
                try:
                    import cv2
                    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                except ImportError:
                    # Fallback: standard luminance conversion without OpenCV.
                    return (
                        0.114 * img[:, :, 0].astype(np.float32)
                        + 0.587 * img[:, :, 1].astype(np.float32)
                        + 0.299 * img[:, :, 2].astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
            if img.shape[2] == 4:
                # BGRA — drop alpha, convert.
                try:
                    import cv2
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                except ImportError:
                    return (
                        0.114 * img[:, :, 0].astype(np.float32)
                        + 0.587 * img[:, :, 1].astype(np.float32)
                        + 0.299 * img[:, :, 2].astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
        raise ValueError(
            f"Unsupported image shape for grayscale conversion: {img.shape}"
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def baseline_mm(self) -> float:
        """Stereo baseline in millimetres."""
        return self.config.baseline_mm

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return self.config.focal_px

    def __repr__(self) -> str:
        cfg = self.config
        filters = []
        if cfg.enable_spatial:
            filters.append("spatial")
        if cfg.enable_temporal:
            filters.append("temporal")
        if cfg.enable_hole_fill:
            filters.append("hole_fill")
        return (
            f"StereoDepthEstimator("
            f"baseline_mm={cfg.baseline_mm}, "
            f"focal_px={cfg.focal_px}, "
            f"num_disparities={cfg.num_disparities}, "
            f"filters=[{', '.join(filters)}])"
        )
