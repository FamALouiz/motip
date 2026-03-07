"""Tensor network data structure."""

from dataclasses import dataclass
from typing import Sequence, TypeAlias, override

from numpy import ndarray

ContractionPath: TypeAlias = Sequence[tuple[int, int]]


@dataclass
class TensorNetwork:
    """Tensor network data structure."""

    input_indices: list[list[int]]
    output_indices: list[int]
    shapes: list[tuple[int, ...]]
    size_dict: dict[int, int]
    arrays: list[ndarray] | None

    def __post_init__(self):
        """Validate the tensor network data."""
        if len(self.input_indices) != len(self.shapes):
            raise ValueError("The number of input index lists must match the number of shapes.")
        for input_indices, shape in zip(self.input_indices, self.shapes):
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
            self.input_indices,
            self.output_indices,
            self.shapes,
            self.size_dict,
            self.arrays,
        )

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the network."""
        return len(self.input_indices)

    @property
    def arrays(self) -> list[ndarray]:
        """Arrays in the tensor network."""
        if self.arrays is None:
            raise ValueError(
                "Arrays were not generated for this tensor network. Only metadata is available."
            )
        return self.arrays

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison for TensorNetwork."""
        if isinstance(other, TensorNetwork):
            return self.as_tuple[:-1] == other.as_tuple[:-1] and all(
                (a1 == a2).all() for a1, a2 in zip(self.arrays or [], other.arrays or [])
            )
        elif isinstance(other, tuple):
            return self.as_tuple[:-1] == other[:-1] and all(
                (a1 == a2).all() for a1, a2 in zip(self.arrays or [], other[-1] or [])
            )
        return NotImplemented
