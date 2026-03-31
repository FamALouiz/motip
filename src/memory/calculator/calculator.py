"""Memory calculator for the motip package."""

import math
from typing import overload

from memory.memory import Memory
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
