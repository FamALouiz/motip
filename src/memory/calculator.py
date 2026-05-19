"""Memory calculator for the motip package."""

import math
from typing import overload

from contraction.tensor import get_contracted_indices
from memory import Memory
from tensor import Tensor
from tensor_network.tn import _TensorPool


class MemoryCalculator:
    """Memory calculator class.

    This class will be responsible for calculating the memory requirements of tensor network
    contractions.
    """

    def __init__(self) -> None:
        """Initialize the memory calculator."""
        self.__element_size_in_bytes: Memory = Memory(
            8
        )  # Default to 8 bytes (i.e. 64 bits) for double-precision floating-point numbers.

    @property
    def element_size_in_bytes(self) -> Memory:
        """Get the size of each element in the tensors in bytes."""
        return self.__element_size_in_bytes

    def set_element_size(self, element_size: int | Memory) -> "MemoryCalculator":
        """Set the size of each element in the tensors.

        This method will set the size of each element in the tensors, which is necessary for
        calculating memory requirements. The default size is 8 bytes (i.e. 64 bits) for
        double-precision floating-point numbers.

        Args:
            element_size: The size of each element (in bytes).
        """
        if isinstance(element_size, int):
            element_size_in_bytes = Memory(element_size)
        else:
            element_size_in_bytes = element_size
        self.__element_size_in_bytes = element_size_in_bytes
        return self

    def calculate_memory_for_tensor(self, tensor: Tensor) -> Memory:
        """Calculate the memory requirements for a single tensor."""
        num_elements = math.prod(tensor.shape)
        return self.__element_size_in_bytes * num_elements

    @overload
    def calculate_memory_for_tensors(self, tensors: list[Tensor]) -> Memory: ...
    @overload
    def calculate_memory_for_tensors(self, tensors: _TensorPool) -> Memory: ...
    def calculate_memory_for_tensors(self, tensors: list[Tensor] | _TensorPool) -> Memory:
        """Calculate the total memory requirements for a list of tensors."""
        total_memory = Memory(0)
        for tensor in tensors:
            total_memory += self.calculate_memory_for_tensor(tensor)
        return total_memory

    def calculate_memory_for_contraction(self, tensor_a: Tensor, tensor_b: Tensor) -> Memory:
        """Calculate the memory requirements for contracting two tensors."""
        contracted_indices = get_contracted_indices(tensor_a, tensor_b)
        free_indices_a = set(tensor_a.input_indices) - contracted_indices
        free_indices_b = set(tensor_b.input_indices) - contracted_indices
        shape_a = [tensor_a.shape[tensor_a.input_indices.index(idx)] for idx in free_indices_a]
        shape_b = [tensor_b.shape[tensor_b.input_indices.index(idx)] for idx in free_indices_b]
        result_shape = tuple(shape_a + shape_b)
        num_elements = math.prod(result_shape)
        return self.__element_size_in_bytes * num_elements
