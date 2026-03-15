"""Abstract base class for MLX / CPU processing backends.

All array operations used throughout realsense-mlx route through this interface
so that the rest of the codebase stays backend-agnostic.  The ``create()``
factory is the single entry point callers should use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np

# Type alias accepted as "device array" by the interface.
ArrayLike = Any
DType = Any
Shape = Union[Tuple[int, ...], int]


class ProcessingBackend(ABC):
    """Backend-agnostic tensor operations used by realsense-mlx.

    Implementations must be stateless: every method returns a new array and
    never mutates its inputs.
    """

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    @abstractmethod
    def to_device(self, arr: np.ndarray) -> ArrayLike:
        """Copy a NumPy array to the backend device (or return a view).

        Args:
            arr: Source NumPy array.

        Returns:
            Backend array containing the same data.
        """

    @abstractmethod
    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert a backend array back to a NumPy array.

        Args:
            arr: Backend array.

        Returns:
            NumPy array with the same data and dtype.
        """

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: Shape, dtype: DType) -> ArrayLike:
        """Return a zero-filled array.

        Args:
            shape: Desired shape (tuple or int).
            dtype: Backend or NumPy dtype.

        Returns:
            Backend array of zeros.
        """

    @abstractmethod
    def ones(self, shape: Shape, dtype: DType) -> ArrayLike:
        """Return an array filled with ones."""

    @abstractmethod
    def full(self, shape: Shape, fill_value: Any, dtype: DType) -> ArrayLike:
        """Return an array filled with *fill_value*."""

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    @abstractmethod
    def cast(self, arr: ArrayLike, dtype: DType) -> ArrayLike:
        """Cast *arr* to *dtype*.

        Args:
            arr: Input backend array.
            dtype: Target dtype.

        Returns:
            New array with the requested dtype.
        """

    # ------------------------------------------------------------------
    # Shape / memory layout
    # ------------------------------------------------------------------

    @abstractmethod
    def reshape(self, arr: ArrayLike, shape: Shape) -> ArrayLike:
        """Return a view of *arr* with shape *shape*."""

    @abstractmethod
    def flatten(self, arr: ArrayLike) -> ArrayLike:
        """Return a 1-D copy of *arr*."""

    @abstractmethod
    def stack(self, arrays: list[ArrayLike], axis: int = 0) -> ArrayLike:
        """Stack a sequence of arrays along a new axis."""

    @abstractmethod
    def concatenate(self, arrays: list[ArrayLike], axis: int = 0) -> ArrayLike:
        """Join a sequence of arrays along an existing axis."""

    # ------------------------------------------------------------------
    # Element-wise operations
    # ------------------------------------------------------------------

    @abstractmethod
    def clip(self, arr: ArrayLike, a_min: Any, a_max: Any) -> ArrayLike:
        """Clip values in *arr* to the range [*a_min*, *a_max*]."""

    @abstractmethod
    def where(
        self,
        condition: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
    ) -> ArrayLike:
        """Select elements from *x* or *y* based on *condition*.

        Args:
            condition: Boolean backend array.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            Backend array of selected elements.
        """

    @abstractmethod
    def bitwise_and(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Element-wise bitwise AND."""

    @abstractmethod
    def bitwise_or(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Element-wise bitwise OR."""

    @abstractmethod
    def right_shift(self, arr: ArrayLike, shift: int) -> ArrayLike:
        """Element-wise right bit-shift."""

    @abstractmethod
    def left_shift(self, arr: ArrayLike, shift: int) -> ArrayLike:
        """Element-wise left bit-shift."""

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def minimum(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Element-wise minimum of *a* and *b*."""

    @abstractmethod
    def maximum(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Element-wise maximum of *a* and *b*."""

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    @abstractmethod
    def eval(self, arr: ArrayLike) -> None:
        """Force evaluation / materialisation of *arr* (no-op on CPU).

        MLX is lazy by default; call this to ensure computations are
        completed before timing or inspection.

        Args:
            arr: Backend array to materialise.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(backend: str = "mlx") -> "ProcessingBackend":
        """Instantiate the requested backend implementation.

        Args:
            backend: One of ``"mlx"`` (Apple-Silicon accelerated) or
                ``"cpu"`` (NumPy fallback).  Defaults to ``"mlx"``.

        Returns:
            A concrete :class:`ProcessingBackend` instance.

        Raises:
            ValueError: If *backend* is not recognised.
            ImportError: If the ``mlx`` package is not installed when
                ``backend="mlx"`` is requested.
        """
        if backend == "mlx":
            from realsense_mlx.backends.mlx_backend import MLXBackend

            return MLXBackend()
        if backend == "cpu":
            from realsense_mlx.backends.cpu_backend import CPUBackend

            return CPUBackend()
        raise ValueError(
            f"Unknown backend {backend!r}. Valid choices: 'mlx', 'cpu'."
        )

    # ------------------------------------------------------------------
    # Dunder helpers for subclass repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}()"
