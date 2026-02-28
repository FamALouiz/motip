"""Tensor network data structure."""

from dataclasses import dataclass


@dataclass
class TensorNetwork:
    """Tensor network data structure."""

    input_indices: list[list[int]]
    output_indices: list[int]
    shapes: list[tuple[int, ...]]
    size_dict: dict[int, int]

    @property
    def as_tuple(self) -> tuple[list[list[int]], list[int], list[tuple[int, ...]], dict[int, int]]:
        """Tuple representation of the tensor network."""
        return self.input_indices, self.output_indices, self.shapes, self.size_dict

    def __eq__(self, other: object) -> bool:
        """Equality comparison for TensorNetwork."""
        if isinstance(other, TensorNetwork):
            return self.as_tuple == other.as_tuple
        elif isinstance(other, tuple):
            return self.as_tuple == other
        return NotImplemented
