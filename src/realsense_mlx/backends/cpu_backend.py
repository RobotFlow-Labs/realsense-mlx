"""CPU / NumPy fallback backend.

Provides the same :class:`~realsense_mlx.backends.base.ProcessingBackend`
interface as :class:`~realsense_mlx.backends.mlx_backend.MLXBackend` but
operates entirely on NumPy arrays.  Useful for testing on non-Apple-Silicon
machines or for comparing results against the MLX implementation.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from realsense_mlx.backends.base import ArrayLike, DType, ProcessingBackend, Shape


class CPUBackend(ProcessingBackend):
    """Concrete backend that executes on CPU via NumPy."""

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        """Return *arr* unchanged (already on CPU).

        Args:
            arr: Source NumPy array.

        Returns:
            The same array (no copy made).
        """
        return np.asarray(arr)

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Return *arr* unchanged.

        Args:
            arr: NumPy array.

        Returns:
            The same array.
        """
        return np.asarray(arr)

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def zeros(self, shape: Shape, dtype: DType = np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Shape, dtype: DType = np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def full(self, shape: Shape, fill_value: Any, dtype: DType = np.float32) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype)

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    def cast(self, arr: np.ndarray, dtype: DType) -> np.ndarray:
        return arr.astype(dtype)

    # ------------------------------------------------------------------
    # Shape / memory layout
    # ------------------------------------------------------------------

    def reshape(self, arr: np.ndarray, shape: Shape) -> np.ndarray:
        return arr.reshape(shape)

    def flatten(self, arr: np.ndarray) -> np.ndarray:
        return arr.ravel()

    def stack(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    # ------------------------------------------------------------------
    # Element-wise operations
    # ------------------------------------------------------------------

    def clip(self, arr: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(arr, a_min, a_max)

    def where(
        self,
        condition: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        return np.where(condition, x, y)

    def bitwise_and(self, a: np.ndarray, b: Any) -> np.ndarray:
        return np.bitwise_and(a, b)

    def bitwise_or(self, a: np.ndarray, b: Any) -> np.ndarray:
        return np.bitwise_or(a, b)

    def right_shift(self, arr: np.ndarray, shift: int) -> np.ndarray:
        return np.right_shift(arr, shift)

    def left_shift(self, arr: np.ndarray, shift: int) -> np.ndarray:
        return np.left_shift(arr, shift)

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------

    def minimum(self, a: np.ndarray, b: Any) -> np.ndarray:
        return np.minimum(a, b)

    def maximum(self, a: np.ndarray, b: Any) -> np.ndarray:
        return np.maximum(a, b)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def eval(self, arr: np.ndarray) -> None:  # noqa: ARG002
        """No-op: NumPy arrays are already evaluated eagerly."""
        return
