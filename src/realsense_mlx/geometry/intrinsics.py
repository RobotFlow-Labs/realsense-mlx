"""Camera intrinsics and extrinsics dataclasses for RealSense MLX.

Provides CameraIntrinsics and CameraExtrinsics with conversion from
pyrealsense2 objects and validation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Distortion model string tokens — mirrors rs2_distortion enum
_DISTORTION_MODEL_MAP: dict[int, str] = {
    0: "none",
    1: "modified_brown_conrady",
    2: "inverse_brown_conrady",
    3: "ftheta",
    4: "brown_conrady",
    5: "kannala_brandt4",
}

SUPPORTED_MODELS = frozenset(
    {"none", "brown_conrady", "inverse_brown_conrady", "modified_brown_conrady"}
)


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters with optional distortion.

    Attributes
    ----------
    width:  Image width in pixels.
    height: Image height in pixels.
    ppx:    Principal point x (horizontal offset from top-left, pixels).
    ppy:    Principal point y (vertical offset from top-left, pixels).
    fx:     Focal length x in pixels.
    fy:     Focal length y in pixels.
    model:  Distortion model identifier string.
            One of: ``"none"``, ``"brown_conrady"``,
            ``"inverse_brown_conrady"``, ``"modified_brown_conrady"``.
    coeffs: Five distortion coefficients [k1, k2, p1, p2, k3].

    Examples
    --------
    >>> intr = CameraIntrinsics(640, 480, 320.0, 240.0, 600.0, 600.0)
    >>> intr.focal_length_mean
    600.0
    """

    width: int
    height: int
    ppx: float
    ppy: float
    fx: float
    fy: float
    model: str = "none"
    coeffs: list[float] = field(default_factory=lambda: [0.0] * 5)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got ({self.width}, {self.height})"
            )
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(
                f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported distortion model '{self.model}'. "
                f"Supported: {sorted(SUPPORTED_MODELS)}"
            )
        if len(self.coeffs) != 5:
            raise ValueError(
                f"coeffs must have exactly 5 elements, got {len(self.coeffs)}"
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def focal_length_mean(self) -> float:
        """Arithmetic mean of fx and fy."""
        return (self.fx + self.fy) * 0.5

    @property
    def has_distortion(self) -> bool:
        """True when the model is not ``"none"`` and any coefficient is non-zero."""
        return self.model != "none" and any(c != 0.0 for c in self.coeffs)

    @property
    def fov_x_deg(self) -> float:
        """Horizontal field-of-view in degrees."""
        import math
        return math.degrees(2.0 * math.atan2(self.width * 0.5, self.fx))

    @property
    def fov_y_deg(self) -> float:
        """Vertical field-of-view in degrees."""
        import math
        return math.degrees(2.0 * math.atan2(self.height * 0.5, self.fy))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_rs2(cls, rs_intrinsics) -> "CameraIntrinsics":
        """Convert a ``pyrealsense2.intrinsics`` object to :class:`CameraIntrinsics`.

        Parameters
        ----------
        rs_intrinsics:
            A ``pyrealsense2.intrinsics`` instance obtained via e.g.
            ``profile.get_intrinsics()``.

        Returns
        -------
        CameraIntrinsics
            Equivalent dataclass instance.

        Raises
        ------
        ImportError
            If ``pyrealsense2`` is not installed.
        ValueError
            If the distortion model is unsupported.

        Examples
        --------
        >>> # intr = pipeline.get_active_profile()
        >>> #     .get_stream(rs.stream.depth)
        >>> #     .as_video_stream_profile()
        >>> #     .get_intrinsics()
        >>> # ci = CameraIntrinsics.from_rs2(intr)
        """
        # rs2_distortion is an IntEnum-like object; convert to int then look up
        try:
            model_int = int(rs_intrinsics.model)
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                f"Expected pyrealsense2.intrinsics, got {type(rs_intrinsics)}"
            ) from exc

        model_str = _DISTORTION_MODEL_MAP.get(model_int, "none")

        # Unsupported models degrade to "none" with a warning rather than raising,
        # since callers may still want geometry without distortion correction.
        if model_str not in SUPPORTED_MODELS:
            import warnings
            warnings.warn(
                f"Distortion model id={model_int} ('{model_str}') is not fully "
                "supported; distortion correction will be skipped.",
                stacklevel=2,
            )
            model_str = "none"

        return cls(
            width=int(rs_intrinsics.width),
            height=int(rs_intrinsics.height),
            ppx=float(rs_intrinsics.ppx),
            ppy=float(rs_intrinsics.ppy),
            fx=float(rs_intrinsics.fx),
            fy=float(rs_intrinsics.fy),
            model=model_str,
            coeffs=[float(c) for c in rs_intrinsics.coeffs],
        )

    @classmethod
    def make_pinhole(
        cls,
        width: int,
        height: int,
        fov_deg: float = 69.4,
    ) -> "CameraIntrinsics":
        """Create a synthetic no-distortion intrinsics from a symmetric FoV.

        Useful for testing without a physical camera.

        Parameters
        ----------
        width, height: Resolution in pixels.
        fov_deg: Horizontal field-of-view in degrees (default ≈ D435 RGB).
        """
        import math
        fx = (width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        return cls(
            width=width,
            height=height,
            ppx=width * 0.5,
            ppy=height * 0.5,
            fx=fx,
            fy=fx,
            model="none",
        )

    def __repr__(self) -> str:
        return (
            f"CameraIntrinsics({self.width}x{self.height}, "
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"pp=({self.ppx:.2f}, {self.ppy:.2f}), "
            f"model='{self.model}')"
        )


@dataclass
class CameraExtrinsics:
    """Rigid-body transform between two camera coordinate frames.

    Attributes
    ----------
    rotation:    3x3 rotation matrix (column-major in RealSense convention,
                 stored as (3,3) row-major ndarray for NumPy compatibility).
    translation: 3-element translation vector in metres.

    Notes
    -----
    The RealSense SDK stores rotation in column-major order (``rotation[0]``
    is column 0, etc.).  :meth:`from_rs2` converts this to the standard
    row-major NumPy (3,3) layout used throughout this package.

    The transform is applied as::

        p_other = R @ p_depth + t
    """

    rotation: np.ndarray     # (3, 3) float64, row-major
    translation: np.ndarray  # (3,)   float64

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64)
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"rotation must be (3,3), got {self.rotation.shape}"
            )
        if self.translation.shape != (3,):
            raise ValueError(
                f"translation must be (3,), got {self.translation.shape}"
            )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """True when rotation=I and translation=0 within float64 eps."""
        return bool(
            np.allclose(self.rotation, np.eye(3), atol=1e-9)
            and np.allclose(self.translation, 0.0, atol=1e-9)
        )

    def inverse(self) -> "CameraExtrinsics":
        """Return the inverse transform (other→depth)."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return CameraExtrinsics(rotation=R_inv, translation=t_inv)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> "CameraExtrinsics":
        """Return an identity (no-op) extrinsics."""
        return cls(rotation=np.eye(3, dtype=np.float64), translation=np.zeros(3, dtype=np.float64))

    @classmethod
    def from_rs2(cls, rs_extrinsics) -> "CameraExtrinsics":
        """Convert a ``pyrealsense2.extrinsics`` object.

        The SDK's ``rotation`` field is a flat 9-element list in column-major
        (Fortran) order.  We reshape and transpose to get the standard
        row-major (3,3) matrix.

        Parameters
        ----------
        rs_extrinsics:
            A ``pyrealsense2.extrinsics`` instance, e.g. from
            ``depth_stream.get_extrinsics_to(color_stream)``.
        """
        try:
            # SDK stores rotation as [r00, r10, r20, r01, r11, r21, r02, r12, r22]
            # i.e. column-major: reshape to (3,3) then transpose for row-major.
            rot_col_major = np.array(rs_extrinsics.rotation, dtype=np.float64).reshape(3, 3)
            rotation = rot_col_major.T  # now row-major R
            translation = np.array(rs_extrinsics.translation, dtype=np.float64)
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                f"Expected pyrealsense2.extrinsics, got {type(rs_extrinsics)}"
            ) from exc

        return cls(rotation=rotation, translation=translation)

    def __repr__(self) -> str:
        t = self.translation
        return (
            f"CameraExtrinsics(t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}], "
            f"identity={self.is_identity})"
        )
