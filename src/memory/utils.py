"""Utility functions for memory operations."""

from memory.calculator import MemoryCalculator
from memory.memory import Memory, MemorySizes
from tensor_network import TensorNetwork


def get_memory_from_string(memory_str: str) -> Memory:
    """Create a Memory object from a string representation.

    Args:
        memory_str: A string like "64MB", "1GB", etc.

    Returns:
        A Memory object representing the specified memory size.

    Raises:
        ValueError: If the input string is not in a valid format.
    """
    memory_str = memory_str.strip().upper()
    for unit in reversed(MemorySizes):
        if memory_str.endswith(unit.name):
            value = float(memory_str[: -len(unit.name)].strip())
            bytes_value = int(value * unit)
            return Memory(bytes_value)
    raise ValueError(f"Invalid memory string: {memory_str}")


def get_largest_tensor_in_network(network: TensorNetwork) -> tuple[int, Memory]:
    """Get the memory requirements of the largest tensor in a tensor network.

    Args:
        network: The tensor network to analyze.

    Returns:
        A tuple containing the index of the largest tensor and its memory requirements.

    Raises:
        AssertionError: If the tensor network is empty.
    """
    assert network.tensors, "Tensor network must contain at least one tensor."
    largest_memory = MemoryCalculator().calculate_memory_for_tensor(network.tensors[0])
    largest_tensor_idx = 0
    for i, tensor in enumerate(network.tensors):
        tensor_memory = MemoryCalculator().calculate_memory_for_tensor(tensor)
        if tensor_memory > largest_memory:
            largest_memory = tensor_memory
            largest_tensor_idx = i
    return largest_tensor_idx, largest_memory
