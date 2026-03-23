"""Tensor permutation utilities."""

from copy import deepcopy
from typing import Sequence

import numpy as np

from contraction.path import ContractionPath, ContractionPathWithHistory
from contraction.tensor import get_contracted_indices
from contraction.tensor_network import contract_tensors_in_network
from memory.utils import get_largest_intermediate_tensor_in_contraction_path
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def permute_tensor(tensor: Tensor, permutation: Sequence[int]) -> Tensor:
    """Permute the indices of a tensor.

    This function will permute the indices of a tensor in a way that is consistent with the
    ordering of the indices in the tensor network. The specific permutation applied will depend
    on the ordering of the indices in the tensor network and the original ordering of the
    indices in the tensor.

    Args:
        tensor: The tensor to permute.
        permutation: The permutation to apply. This should be a list of integers representing the
        new order of the indices.

    Returns:
        The permuted tensor.

    Raises:
        ValueError: If the permutation is not a valid rearrangement of the tensor's indices.
    """
    if sorted(permutation) != list(range(len(tensor.input_indices))):
        raise ValueError(
            "Permutation must be a valid rearrangement of the tensor's indices. "
            f"Expected a permutation of the integers from 0 to {len(tensor.input_indices) - 1}, "
            f"got {permutation}."
        )
    return Tensor(
        input_indices=[tensor.input_indices[i] for i in permutation],
        shape=tuple(tensor.shape[i] for i in permutation),
        array=np.transpose(tensor.array, axes=permutation) if tensor.array is not None else None,
    )


def find_optimal_permutation_based_on_contraction_path(
    network: TensorNetwork, contraction_path: ContractionPath, strategy: str = "greedy"
) -> list[tuple[int, ...]]:
    """Find the optimal tensor permutation for a given contraction path.

    This function will analyze the contraction path and determine the optimal order of tensor
    indices for each intermediate tensor to minimize memory usage during contractions. The optimal
    permutation is determined by considering the sizes of the tensors being contracted and the
    resulting intermediate tensors.

    Args:
        network: The initial tensor network.
        contraction_path: The sequence of contraction pairs.
        strategy: The strategy to use for finding the optimal permutation. Defaults to "greedy".

    Returns:
        A list of optimal tensor index permutations for each intermediate tensor in the contraction
        path.
    """
    if strategy == "greedy":
        return _find_greedy_optimal_permutation(network, contraction_path)
    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def _find_greedy_optimal_permutation(
    network: TensorNetwork, contraction_path: ContractionPath
) -> list[tuple[int, ...]]:
    """Greedy strategy to find optimal tensor permutations for a contraction path."""
    largest_step_idx, _ = get_largest_intermediate_tensor_in_contraction_path(
        network, contraction_path
    )

    history = ContractionPathWithHistory.from_contraction_path(network, contraction_path)

    usage_step = {}
    current_network = deepcopy(network)
    for step, pair in enumerate(contraction_path):
        t_a = current_network.tensors[pair[0]]
        t_b = current_network.tensors[pair[1]]
        contracted_indices = get_contracted_indices(t_a, t_b)
        for idx in contracted_indices:
            if idx not in usage_step:
                usage_step[idx] = step
        current_network = contract_tensors_in_network(current_network, pair)

    for idx in current_network.output_indices:
        usage_step[idx] = len(contraction_path)

    def optimal_order_key(idx: int) -> tuple[int, int]:
        return (network.size_dict.get(idx, 0), usage_step.get(idx, 0))

    permutations = []

    for step, pair in enumerate(contraction_path):
        state_after = history.get_state(step + 1)
        intermediate_tensor = state_after.tensors[pair[0]]
        current_indices = intermediate_tensor.input_indices

        optimal_indices = sorted(current_indices, key=optimal_order_key)

        if step == largest_step_idx:
            perm = tuple(range(len(current_indices)))
        else:
            perm = tuple(current_indices.index(idx) for idx in optimal_indices)

        permutations.append(perm)

    return permutations
