"""MLX backend implementation for Apple Silicon.

All operations delegate to ``mlx.core``.  MLX uses lazy evaluation; callers
that need results immediately should invoke :meth:`eval`.

Notes on MLX constraints respected here:
- No ``int64`` — index arithmetic uses ``int32``.
- ``mx.array.at[idx].add(val)`` is functional (returns a new array).
- Bit-shift operators use ``mx.right_shift`` / ``mx.left_shift``.
- ``mx.clip`` is used for clamping (not ``np.clip``).
- Version detection goes through ``importlib.metadata``, never
  ``mlx.__version__``.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np

import mlx.core as mx

from realsense_mlx.backends.base import ArrayLike, DType, ProcessingBackend, Shape


class MLXBackend(ProcessingBackend):
    """Concrete backend that executes on Apple Silicon via MLX."""

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to_device(self, arr: np.ndarray) -> mx.array:
        """Convert a NumPy array to an MLX array.

        Args:
            arr: Source NumPy array.

        Returns:
            ``mx.array`` with the same data and dtype.
        """
        return mx.array(arr)

    def to_numpy(self, arr: mx.array) -> np.ndarray:
        """Convert an MLX array to a NumPy array.

        Materialises the lazy graph via ``mx.eval`` before conversion.

        Args:
            arr: MLX array.

        Returns:
            NumPy array.
        """
        mx.eval(arr)
        return np.array(arr)

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def zeros(self, shape: Shape, dtype: DType = mx.float32) -> mx.array:
        return mx.zeros(shape, dtype=dtype)

    def ones(self, shape: Shape, dtype: DType = mx.float32) -> mx.array:
        return mx.ones(shape, dtype=dtype)

    def full(self, shape: Shape, fill_value: Any, dtype: DType = mx.float32) -> mx.array:
        return mx.full(shape, fill_value, dtype=dtype)

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    def cast(self, arr: mx.array, dtype: DType) -> mx.array:
        return arr.astype(dtype)

    # ------------------------------------------------------------------
    # Shape / memory layout
    # ------------------------------------------------------------------

    def reshape(self, arr: mx.array, shape: Shape) -> mx.array:
        if isinstance(shape, int):
            shape = (shape,)
        return arr.reshape(shape)

    def flatten(self, arr: mx.array) -> mx.array:
        return arr.reshape(-1)

    def stack(self, arrays: List[mx.array], axis: int = 0) -> mx.array:
        return mx.stack(arrays, axis=axis)

    def concatenate(self, arrays: List[mx.array], axis: int = 0) -> mx.array:
        return mx.concatenate(arrays, axis=axis)

    # ------------------------------------------------------------------
    # Element-wise operations
    # ------------------------------------------------------------------

    def clip(self, arr: mx.array, a_min: Any, a_max: Any) -> mx.array:
        return mx.clip(arr, a_min, a_max)

    def where(
        self,
        condition: mx.array,
        x: mx.array,
        y: mx.array,
    ) -> mx.array:
        return mx.where(condition, x, y)

    def bitwise_and(self, a: mx.array, b: Any) -> mx.array:
        # MLX supports Python ``&`` operator on integer arrays.
        return a & b

    def bitwise_or(self, a: mx.array, b: Any) -> mx.array:
        return a | b

    def right_shift(self, arr: mx.array, shift: int) -> mx.array:
        return mx.right_shift(arr, shift)

    def left_shift(self, arr: mx.array, shift: int) -> mx.array:
        return mx.left_shift(arr, shift)

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------

    def minimum(self, a: mx.array, b: Any) -> mx.array:
        return mx.minimum(a, b)

    def maximum(self, a: mx.array, b: Any) -> mx.array:
        return mx.maximum(a, b)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def eval(self, arr: mx.array) -> None:
        mx.eval(arr)
