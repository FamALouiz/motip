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
    assert network.num_tensors > pair[0] and network.num_tensors > pair[1], (
        "Tensor indices out of range."
    )

    new_tensor_network = deepcopy(network)
    first_contracted_tensor = network.tensors[pair[0]]
    second_contracted_tensor = network.tensors[pair[1]]

    for idx in sorted(pair, reverse=True):
        new_tensor_network.tensors.pop(idx)

    new_tensor = contract_tensors(first_contracted_tensor, second_contracted_tensor)

    new_tensor_network.tensors.insert(pair[0], new_tensor)

    return new_tensor_network
