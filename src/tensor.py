"""Tensor data structure."""

from dataclasses import dataclass
from typing import override

from numpy import ndarray


@dataclass
class Tensor:
    """Tensor data structure."""

    input_indices: list[int]
    shape: tuple[int, ...]
    array: ndarray | None

    def __post_init__(self) -> None:
        """Validate the tensor data."""
        if len(self.input_indices) != len(self.shape):
            raise ValueError(
                f"Each input index list must have the same length as its corresponding shape. "
                f"Got {len(self.input_indices)} indices and shape {self.shape}."
            )

    @property
    def as_tuple(self) -> tuple[list[int], tuple[int, ...], ndarray | None]:
        """Tuple representation of the tensor."""
        return (self.input_indices, self.shape, self.array)

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison for Tensor."""
        if isinstance(other, Tensor):
            return self.as_tuple[:-1] == other.as_tuple[:-1] and (
                self.array is None
                and other.array is None
                or self.array is not None
                and other.array is not None
                and (self.array == other.array).all()
            )
        elif isinstance(other, tuple):
            return self.as_tuple[:-1] == other[:-1] and (
                self.array is None
                and other[-1] is None
                or self.array is not None
                and other[-1] is not None
                and (self.array == other[-1]).all()
            )
        return NotImplemented
