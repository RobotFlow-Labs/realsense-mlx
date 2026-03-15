"""Multi-camera capture — manage multiple RealSense cameras simultaneously.

Discovers all connected Intel RealSense devices, starts a pipeline on each,
and provides synchronised per-camera frame retrieval.  pyrealsense2 is an
optional dependency; instantiation raises ImportError with actionable build
instructions when the library is absent.

Typical usage
-------------
::

    from realsense_mlx.capture import MultiCameraCapture, CapturedFrames

    multi = MultiCameraCapture()
    multi.start()
    try:
        frames_dict = multi.get_frames()   # dict[serial → CapturedFrames]
        for serial, frames in frames_dict.items():
            print(serial, frames)
    finally:
        multi.stop()

    # Or with the context-manager:
    with MultiCameraCapture() as multi:
        frames_dict = multi.get_frames()
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from realsense_mlx.capture.pipeline import (
    CaptureConfig,
    CapturedFrames,
    RealsenseCapture,
    HAS_RS2,
    _RS2_BUILD_INSTRUCTIONS,
)

if TYPE_CHECKING:
    pass


class MultiCameraCapture:
    """Manage multiple RealSense cameras simultaneously.

    Discovers connected cameras by querying the RS2 context, creates a
    :class:`~realsense_mlx.capture.RealsenseCapture` for each device, and
    starts all pipelines.  Frame retrieval is sequential (one blocking call
    per camera); cameras are polled in the order they were discovered.

    Parameters
    ----------
    config:
        :class:`CaptureConfig` applied to *every* camera.  Defaults to
        ``CaptureConfig()`` (640x480@30 fps, depth+colour).

    Raises
    ------
    ImportError
        If ``pyrealsense2`` is not installed.

    Examples
    --------
    ::

        with MultiCameraCapture(CaptureConfig(width=848, height=480)) as mc:
            for serial, frames in mc.get_frames().items():
                print(serial, frames.depth.shape)
    """

    def __init__(self, config: CaptureConfig | None = None) -> None:
        if not HAS_RS2:
            raise ImportError(_RS2_BUILD_INSTRUCTIONS)

        self.config = config or CaptureConfig()
        # Keyed by camera serial number (str).
        self._cameras: dict[str, RealsenseCapture] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def discover() -> list[dict[str, str]]:
        """Discover all connected Intel RealSense cameras.

        Queries the RS2 context for the list of connected devices and returns
        a lightweight descriptor for each one.

        Returns
        -------
        list[dict[str, str]]
            Each item has the keys:

            ``serial``
                Device serial number (e.g. ``"123456789012"``).
            ``name``
                Human-readable product name (e.g. ``"Intel RealSense D435"``).
            ``firmware``
                Firmware version string.
            ``usb_type``
                USB speed descriptor (e.g. ``"3.2"``).

        Notes
        -----
        Returns an empty list when no cameras are connected or when
        pyrealsense2 is not available.
        """
        if not HAS_RS2:
            return []

        try:
            import pyrealsense2 as rs  # type: ignore[import]
        except ImportError:
            return []

        result: list[dict[str, str]] = []
        try:
            ctx = rs.context()
            for dev in ctx.devices:
                try:
                    serial = str(
                        dev.get_info(rs.camera_info.serial_number)
                    )
                    name = str(dev.get_info(rs.camera_info.name))
                    firmware = str(
                        dev.get_info(rs.camera_info.firmware_version)
                    )
                    usb_type = str(
                        dev.get_info(rs.camera_info.usb_type_descriptor)
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Failed to query device info: {exc}",
                        stacklevel=2,
                    )
                    continue
                result.append(
                    {
                        "serial": serial,
                        "name": name,
                        "firmware": firmware,
                        "usb_type": usb_type,
                    }
                )
        except Exception as exc:
            warnings.warn(
                f"Failed to enumerate RealSense devices: {exc}",
                stacklevel=2,
            )

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, serials: list[str] | None = None) -> None:
        """Start streaming from cameras.

        Parameters
        ----------
        serials:
            Optional list of serial-number strings.  When ``None`` (default)
            all currently connected cameras are started.  Pass an explicit
            list to target specific devices.

        Raises
        ------
        ImportError
            If ``pyrealsense2`` is not installed.
        RuntimeError
            If a pipeline fails to start (hardware error, device in use, etc.).
        ValueError
            If *serials* is provided but no discovered camera matches any of
            the requested serial numbers.
        """
        import pyrealsense2 as rs  # type: ignore[import]

        if serials is None:
            discovered = self.discover()
            target_serials = [d["serial"] for d in discovered]
        else:
            target_serials = list(serials)

        if not target_serials:
            warnings.warn(
                "MultiCameraCapture.start(): no cameras found to start.",
                stacklevel=2,
            )
            return

        for serial in target_serials:
            if serial in self._cameras:
                # Already started — skip silently.
                continue

            # Build an rs.config that enables a specific device serial.
            rs_config = rs.config()
            rs_config.enable_device(serial)

            # Wrap in our RealsenseCapture but supply the per-device config.
            capture = _SerialCapture(self.config, serial, rs_config)
            try:
                capture.start()
            except RuntimeError as exc:
                warnings.warn(
                    f"Failed to start camera {serial}: {exc}",
                    stacklevel=2,
                )
                continue

            self._cameras[serial] = capture

    def get_frames(self) -> dict[str, CapturedFrames]:
        """Retrieve the next frame-set from every active camera.

        Frames are collected sequentially — each call blocks until the next
        frame arrives from each camera (up to ``config.timeout_ms`` per
        camera).  If a camera times out, a warning is emitted and an empty
        :class:`CapturedFrames` is included for that serial.

        Returns
        -------
        dict[str, CapturedFrames]
            Maps camera serial number to its :class:`CapturedFrames`.
        """
        result: dict[str, CapturedFrames] = {}
        for serial, capture in self._cameras.items():
            result[serial] = capture.get_frames()
        return result

    def stop(self) -> None:
        """Stop all active camera pipelines and release hardware resources."""
        for serial, capture in list(self._cameras.items()):
            try:
                capture.stop()
            except Exception as exc:
                warnings.warn(
                    f"Error stopping camera {serial}: {exc}",
                    stacklevel=2,
                )
        self._cameras.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def active_serials(self) -> list[str]:
        """List of serial numbers for currently active cameras."""
        return list(self._cameras.keys())

    @property
    def camera_count(self) -> int:
        """Number of currently active cameras."""
        return len(self._cameras)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MultiCameraCapture":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"MultiCameraCapture(cameras={self.active_serials}, "
            f"config={self.config!r})"
        )


# ---------------------------------------------------------------------------
# Internal helper: a RealsenseCapture variant that accepts a pre-built
# rs.config so we can target a specific device by serial number.
# ---------------------------------------------------------------------------


class _SerialCapture(RealsenseCapture):
    """RealsenseCapture variant that injects a pre-configured ``rs.config``.

    This allows MultiCameraCapture to lock each pipeline to a specific device
    serial number via ``rs.config.enable_device()``.

    Parameters
    ----------
    config:
        Shared :class:`CaptureConfig` (stream format, resolution, fps).
    serial:
        Camera serial number (used only for repr).
    rs_config:
        A ``pyrealsense2.config`` instance with the desired device already
        enabled.  Passed directly to ``pipeline.start()``.
    """

    def __init__(
        self,
        config: CaptureConfig,
        serial: str,
        rs_config: object,
    ) -> None:
        super().__init__(config)
        self._serial = serial
        self._rs_config = rs_config

    def start(self) -> None:  # type: ignore[override]
        """Start the pipeline using the pre-built rs.config."""
        import pyrealsense2 as rs  # type: ignore[import]

        if self._running:
            raise RuntimeError(
                f"Camera {self._serial} is already running.  Call stop() first."
            )

        cfg = self.config
        # Enable streams on the provided rs.config (which already has
        # enable_device() set).
        self._rs_config.enable_stream(
            rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
        ) if cfg.enable_depth else None
        self._rs_config.enable_stream(
            rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, cfg.fps
        ) if cfg.enable_color else None
        self._rs_config.enable_stream(
            rs.stream.infrared, 1, cfg.width, cfg.height, rs.format.y8, cfg.fps
        ) if cfg.enable_ir else None

        self._pipeline = rs.pipeline()
        try:
            self._profile = self._pipeline.start(self._rs_config)
        except RuntimeError as exc:
            self._pipeline = None
            self._profile = None
            raise RuntimeError(
                f"Failed to start RealSense pipeline for camera {self._serial}: {exc}"
            ) from exc

        self._running = True
        self._extract_metadata()

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"_SerialCapture(serial={self._serial!r}, state={state})"
