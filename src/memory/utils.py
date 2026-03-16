"""Utility functions for memory operations."""

from copy import deepcopy

from contraction.path import ContractionPath
from contraction.tensor_network import contract_tensors_in_network
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


def get_largest_intermediate_tensor_in_contraction_path(
    network: TensorNetwork, path: ContractionPath
) -> tuple[int, Memory]:
    """Get the memory requirements of the largest intermediate tensor in a contraction path.

    This is the largest intermediate tensor that is created during the contraction process, which
    may be any of the original tensors in the network.

    Args:
        network: The tensor network to analyze.
        path: A list of pairs of tensor indices that are contracted together.

    Returns:
        A tuple containing the index of the contraction step in the path that created the largest
        intermediate tensor (AFTER performing the contraction at the given index) and its memory
        requirements. If the index is -1, it indicates that the largest tensor is one of the
        original tensors in the network.
    """
    _, largest_memory = get_largest_tensor_in_network(network)
    largest_contraction_step_idx = -1  # -1 indicates largest tensor is from the original network
    intermediate_network = deepcopy(network)

    for tensor in intermediate_network.tensors:
        print(tensor)
    print()

    for contraction_idx, contraction_pair in enumerate(path):
        intermediate_network = contract_tensors_in_network(intermediate_network, contraction_pair)
        intermediate_tensor = intermediate_network.tensors[contraction_pair[0]]
        intermediate_memory = MemoryCalculator().calculate_memory_for_tensor(intermediate_tensor)

        for tensor in intermediate_network.tensors:
            print(tensor)
        print()

        if intermediate_memory > largest_memory:
            largest_memory = intermediate_memory
            largest_contraction_step_idx = contraction_idx

    return largest_contraction_step_idx, largest_memory
