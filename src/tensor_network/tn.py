"""Tensor network data structure."""

from dataclasses import dataclass
from typing import override

from numpy import ndarray

from tensor import Tensor


@dataclass(init=False)
class TensorNetwork:
    """Tensor network data structure."""

    tensors: list[Tensor]
    output_indices: list[int]
    size_dict: dict[int, int]

    def __init__(
        self,
        *,
        output_indices: list[int],
        size_dict: dict[int, int],
        tensors: list[Tensor] | None = None,
        input_indices: list[list[int]] | None = None,
        shapes: list[tuple[int, ...]] | None = None,
        tensor_arrays: list[ndarray] | None = None,
    ) -> None:
        """Initialize the tensor network from tensors or raw tensor components."""
        self.output_indices = output_indices
        self.size_dict = size_dict

        if tensors is not None:
            self.tensors = tensors
            self.__post_init__()
            return

        if input_indices is None or shapes is None:
            raise ValueError("Either tensors or both input_indices and shapes must be provided.")

        if len(input_indices) != len(shapes):
            raise ValueError("The number of input index lists must match the number of shapes.")

        if tensor_arrays is not None and len(tensor_arrays) != len(input_indices):
            raise ValueError(
                "The number of tensor arrays must match the number of input index lists."
            )

        arrays: list[ndarray | None] = []
        if tensor_arrays is not None:
            arrays.extend(tensor_arrays)
        else:
            arrays.extend([None] * len(input_indices))
        self.tensors = [
            Tensor(tensor_input_indices, shape, array)
            for tensor_input_indices, shape, array in zip(
                input_indices, shapes, arrays, strict=True
            )
        ]
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the tensor network data."""
        for tensor in self.tensors:
            if len(tensor.input_indices) != len(tensor.shape):
                raise ValueError(
                    f"Each input index list must have the same length as its corresponding shape. "
                    f"Got {len(tensor.input_indices)} indices and shape {tensor.shape}."
                )

    @property
    def input_indices(self) -> list[list[int]]:
        """Input indices for each tensor in the network."""
        return [tensor.input_indices for tensor in self.tensors]

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        """Shapes for each tensor in the network."""
        return [tensor.shape for tensor in self.tensors]

    @property
    def tensor_arrays(self) -> list[ndarray] | None:
        """Arrays for each tensor in the network."""
        if any(tensor.array is None for tensor in self.tensors):
            return None
        return [tensor.array for tensor in self.tensors if tensor.array is not None]

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
            self.tensor_arrays,
        )

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the network."""
        return len(self.tensors)

    @property
    def arrays(self) -> list[ndarray]:
        """Arrays in the tensor network."""
        if self.tensor_arrays is None:
            raise ValueError(
                "Arrays were not generated for this tensor network. Only metadata is available."
            )
        return self.tensor_arrays

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison for TensorNetwork."""
        if isinstance(other, TensorNetwork):
            return self.as_tuple[:-1] == other.as_tuple[:-1] and all(
                (a1 == a2).all()
                for a1, a2 in zip(self.tensor_arrays or [], other.tensor_arrays or [])
            )
        elif isinstance(other, tuple):
            return self.as_tuple[:-1] == other[:-1] and all(
                (a1 == a2).all() for a1, a2 in zip(self.tensor_arrays or [], other[-1] or [])
            )
        return NotImplemented
