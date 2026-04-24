"""Utility functions for memory operations."""

from copy import deepcopy
from heapq import nlargest
from typing import Collection, overload

from more_itertools import one

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor_network import contract_tensors_in_network
from memory import Memory
from memory.calculator import MemoryCalculator
from tensor_network import TensorNetwork


def get_largest_k_tensors_in_network(
    network: TensorNetwork, k: int
) -> tuple[Collection[int], Collection[Memory]]:
    """Get the memory requirements of the largest k tensors in a tensor network.

    Args:
        network: The tensor network to analyze.
        k: The number of largest tensors to retrieve.

    Returns:
        A tuple containing the indices of the largest k tensors and their memory requirements.

    Raises:
        AssertionError: If the tensor network is empty or if k is not a positive integer.
    """
    assert k > 0, "k must be a positive integer."
    assert isinstance(k, int), "k must be an integer."
    assert len(network.tensors) >= k, f"Tensor network must contain at least {k} tensors."

    idx_by_tensor = {}
    for i, tensor in enumerate(network.tensors):
        idx_by_tensor[id(tensor)] = i

    largest_k_tensors = nlargest(
        k, network.tensors, key=MemoryCalculator().calculate_memory_for_tensor
    )

    largest_tensor_indices = [idx_by_tensor[id(tensor)] for tensor in largest_k_tensors]
    largest_memories = [
        MemoryCalculator().calculate_memory_for_tensor(tensor) for tensor in largest_k_tensors
    ]

    return largest_tensor_indices, largest_memories


def get_largest_tensor_in_network(network: TensorNetwork) -> tuple[int, Memory]:
    """Get the memory requirements of the largest tensor in a tensor network.

    Args:
        network: The tensor network to analyze.

    Returns:
        A tuple containing the index of the largest tensor and its memory requirements.

    Raises:
        AssertionError: If the tensor network is empty.
    """
    largest_tensor_indices, largest_memories = get_largest_k_tensors_in_network(network, k=1)
    return one(largest_tensor_indices), one(largest_memories)


@overload
def get_largest_k_intermediate_tensors_in_path(
    network: TensorNetwork, path: ContractionPath, k: int
) -> tuple[Collection[int], Collection[Memory]]: ...
@overload
def get_largest_k_intermediate_tensors_in_path(
    network: TensorNetwork, path: PersistentContractionPath, k: int
) -> tuple[Collection[int], Collection[Memory]]: ...
def get_largest_k_intermediate_tensors_in_path(
    network: TensorNetwork, path: ContractionPath | PersistentContractionPath, k: int
) -> tuple[Collection[int], Collection[Memory]]:
    """Get the memory requirements of the largest k intermediate tensors in a contraction path.

    Args:
        network: The tensor network to analyze.
        path: A list of pairs of tensor indices that are contracted together.
        k: The number of largest intermediate tensors to retrieve.

    Returns:
        A tuple containing the contraction step indices that created the largest intermediate
        tensors and their memory requirements. If an index is -1, it indicates the tensor is one
        of the original tensors in the network.

    Raises:
        AssertionError: If k is not a positive integer or there are fewer than k candidate tensors.
    """
    assert k > 0, "k must be a positive integer."
    assert isinstance(k, int), "k must be an integer."

    if isinstance(path, PersistentContractionPath):
        return _get_largest_k_intermediate_tensors_in_persistent_contraction_path(network, path, k)
    return _get_largest_k_intermediate_tensors_in_contraction_path(network, path, k)


@overload
def get_largest_intermediate_tensor_in_path(
    network: TensorNetwork, path: ContractionPath
) -> tuple[int, Memory]: ...
@overload
def get_largest_intermediate_tensor_in_path(
    network: TensorNetwork, path: PersistentContractionPath
) -> tuple[int, Memory]: ...
def get_largest_intermediate_tensor_in_path(
    network: TensorNetwork, path: ContractionPath | PersistentContractionPath
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
    largest_step_indices, largest_memories = get_largest_k_intermediate_tensors_in_path(
        network, path, k=1
    )
    return one(largest_step_indices), one(largest_memories)


def _get_largest_k_intermediate_tensors_in_contraction_path(
    network: TensorNetwork, path: ContractionPath, k: int
) -> tuple[Collection[int], Collection[Memory]]:
    calculator = MemoryCalculator()
    intermediate_tensors = [
        (-1, calculator.calculate_memory_for_tensor(tensor)) for tensor in network.tensors
    ]
    current_network = deepcopy(network)

    for step_idx, contraction_pair in enumerate(path):
        current_network = contract_tensors_in_network(current_network, contraction_pair)
        intermediate_tensor = current_network.tensors[contraction_pair[0]]
        intermediate_tensors.append(
            (step_idx, calculator.calculate_memory_for_tensor(intermediate_tensor))
        )

    return _get_largest_k_intermediate_tensors(intermediate_tensors, k)


def _get_largest_k_intermediate_tensors_in_persistent_contraction_path(
    network: TensorNetwork, path: PersistentContractionPath, k: int
) -> tuple[Collection[int], Collection[Memory]]:
    calculator = MemoryCalculator()
    intermediate_tensors = [
        (-1, calculator.calculate_memory_for_tensor(tensor)) for tensor in network.tensors
    ]

    for step_idx, contraction_pair in enumerate(path.path):
        current_network = path.get_state(step_idx + 1)
        intermediate_tensor = current_network.tensors[contraction_pair[0]]
        intermediate_tensors.append(
            (step_idx, calculator.calculate_memory_for_tensor(intermediate_tensor))
        )

    return _get_largest_k_intermediate_tensors(intermediate_tensors, k)


def _get_largest_k_intermediate_tensors(
    intermediate_tensors: list[tuple[int, Memory]], k: int
) -> tuple[Collection[int], Collection[Memory]]:
    assert len(intermediate_tensors) >= k, f"Tensor network must contain at least {k} tensors."

    largest_k_intermediate_tensors = nlargest(k, intermediate_tensors, key=lambda item: item[1])
    largest_step_indices = [step_idx for step_idx, _ in largest_k_intermediate_tensors]
    largest_memories = [memory for _, memory in largest_k_intermediate_tensors]

    return largest_step_indices, largest_memories
