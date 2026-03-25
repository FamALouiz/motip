"""Tensor network data structure."""

from dataclasses import dataclass
from typing import Iterator, Optional, override

from numpy import ndarray

from tensor import Tensor


class _TensorPool:
    """Pool of tensors for reuse during contraction to minimize memory allocations."""

    def __init__(self, tensors: Optional[list[Tensor]]) -> None:
        """Initialize an empty tensor pool."""
        self.__pool: list[Tensor] = []

        if tensors is not None:
            for tensor in tensors:
                self.release(tensor)

    @property
    def pool(self) -> list[Tensor]:
        """Get the list of tensors currently in the pool."""
        return self.__pool

    def get_tensor_by_shape(self, shape: tuple[int, ...]) -> Tensor:
        """Get the first tensor from the pool by shape.

        Args:
            shape: The desired shape of the tensor to retrieve.

        Returns:
            A tensor from the pool that matches the requested shape.

        Raises:
            ValueError: If no tensor of the requested shape is available in the pool.
        """
        for i, tensor in enumerate(self.__pool):
            if tensor.shape == shape:
                return self.__pool.pop(i)
        raise ValueError(f"No tensor of shape {shape} available in the pool.")

    def pop(self, idx: int) -> Tensor:
        """Pop a tensor from the pool."""
        if len(self.__pool) == 0:
            raise ValueError("No tensors available in the pool to pop.")
        return self.__pool.pop(idx)

    def release(self, tensor: Tensor) -> None:
        """Release a tensor to the pool for future reuse."""
        self.__pool.append(tensor)

    def insert(self, idx: int, tensor: Tensor) -> None:
        """Insert a tensor back into the pool at a specific index."""
        self.__pool.insert(idx, tensor)

    def __getitem__(self, index: int) -> Tensor:
        """Get a tensor at the specified index from the pool."""
        return self.__pool[index]

    def __len__(self) -> int:
        """Get the number of tensors currently in the pool."""
        return len(self.__pool)

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over the tensors in the pool."""
        return iter(self.__pool)

    def __setitem__(self, index: int, tensor: Tensor) -> None:
        """Set a tensor at the specified index in the pool."""
        self.__pool[index].array = tensor.array
        self.__pool[index].input_indices = tensor.input_indices
        self.__pool[index].shape = tensor.shape

    def __eq__(self, other: object) -> bool:
        """Equality comparison for TensorPool."""
        if isinstance(other, _TensorPool):
            return all(t1 == t2 for t1, t2 in zip(self.__pool, other.pool))
        elif isinstance(other, list):
            assert all(isinstance(t, Tensor) for t in other), (
                "All elements of the list must be Tensor instances."
            )
            return all(t1 == t2 for t1, t2 in zip(self.__pool, other))
        else:
            return NotImplemented


@dataclass(init=False)
class TensorNetwork:
    """Tensor network data structure."""

    tensors: _TensorPool
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
            self.tensors = _TensorPool(tensors)
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
        self.tensors = _TensorPool(
            [
                Tensor(tensor_input_indices, shape, array)
                for tensor_input_indices, shape, array in zip(
                    input_indices, shapes, arrays, strict=True
                )
            ]
        )
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
