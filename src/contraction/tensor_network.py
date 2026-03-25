"""Tensor network contraction utilities for the motip package."""

from copy import deepcopy

from contraction.tensor import contract_tensors
from tensor_network.tn import TensorNetwork


def contract_tensors_in_network(network: TensorNetwork, pair: tuple[int, int]) -> TensorNetwork:
    """Contract a pair of tensors in the network.

    Args:
        network: The tensor network to contract.
        pair: A tuple of two integers representing the indices of the tensors to contract.

    Returns:
        The resulting tensor network after contraction.
    """
    assert len(network) > pair[0] and len(network) > pair[1], "Tensor indices out of range."

    first_contracted_tensor = network.tensors[pair[0]]
    second_contracted_tensor = network.tensors[pair[1]]

    new_tensor = contract_tensors(first_contracted_tensor, second_contracted_tensor)

    new_tensor_network = TensorNetwork(
        tensors=[t for i, t in enumerate(network.tensors) if i not in pair],
        output_indices=network.output_indices,
        size_dict=network.size_dict,
    )
    new_tensor_network.tensors.insert(pair[0], new_tensor)

    return new_tensor_network


def contract_network(
    network: TensorNetwork, contraction_path: list[tuple[int, int]]
) -> TensorNetwork:
    """Contract a tensor network according to a given contraction path.

    Args:
        network: The tensor network to contract.
        contraction_path: A list of tuples, where each tuple contains the indices of the tensors to
            contract at each step.

    Returns:
        The resulting tensor network after performing all contractions in the path.
    """
    current_network = deepcopy(network)
    for pair in contraction_path:
        current_network = contract_tensors_in_network(current_network, pair)
    return current_network
