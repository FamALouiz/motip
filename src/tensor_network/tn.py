"""Tensor network data structure."""

from dataclasses import dataclass
from typing import override

from numpy import ndarray


@dataclass
class TensorNetwork:
    """Tensor network data structure."""

    _input_indices: list[list[int]]
    _output_indices: list[int]
    _shapes: list[tuple[int, ...]]
    _size_dict: dict[int, int]
    _arrays: list[ndarray] | None

    def __post_init__(self):
        """Validate the tensor network data."""
        if len(self._input_indices) != len(self._shapes):
            raise ValueError("The number of input index lists must match the number of shapes.")
        for input_indices, shape in zip(self._input_indices, self._shapes):
            if len(input_indices) != len(shape):
                raise ValueError(
                    f"Each input index list must have the same length as its corresponding shape. "
                    f"Got {len(input_indices)} indices and shape {shape}."
                )

    @property
    def as_tuple(
        self,
    ) -> tuple[
        list[list[int]], list[int], list[tuple[int, ...]], dict[int, int], list[ndarray] | None
    ]:
        """Tuple representation of the tensor network."""
        return (
            self._input_indices,
            self._output_indices,
            self._shapes,
            self._size_dict,
            self._arrays,
        )

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the network."""
        return len(self._input_indices)

    @property
    def arrays(self) -> list[ndarray]:
        """Arrays in the tensor network."""
        if self._arrays is None:
            raise ValueError(
                "Arrays were not generated for this tensor network. Only metadata is available."
            )
        return self._arrays

    @property
    def input_indices(self) -> list[list[int]]:
        """Input indices of the tensor network."""
        return self._input_indices

    @property
    def output_indices(self) -> list[int]:
        """Output indices of the tensor network."""
        return self._output_indices

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        """Shapes of the tensors in the network."""
        return self._shapes

    @property
    def size_dict(self) -> dict[int, int]:
        """Size dictionary of the indices in the network."""
        return self._size_dict

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison for TensorNetwork."""
        if isinstance(other, TensorNetwork):
            return self.as_tuple == other.as_tuple and self._arrays == other._arrays
        elif isinstance(other, tuple):
            return self.as_tuple == other
        return NotImplemented
