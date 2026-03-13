"""Tensor network contraction utilities."""

from copy import deepcopy

from tensor import Tensor
from tensor_network import TensorNetwork


def contract_pair(network: TensorNetwork, pair: tuple[int, int]) -> TensorNetwork:
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

    contracted_indices = set(network.tensors[pair[0]].input_indices) & set(
        network.tensors[pair[1]].input_indices
    )
    new_tensor_indices = (
        set(network.tensors[pair[0]].input_indices) | set(network.tensors[pair[1]].input_indices)
    ) - contracted_indices
    new_tensor_shape = tuple(network.size_dict[index] for index in new_tensor_indices)

    for idx in sorted(pair, reverse=True):
        new_tensor_network.tensors.pop(idx)

    new_tensor_network.tensors.insert(
        pair[0], Tensor(list(new_tensor_indices), new_tensor_shape, None)
    )

    return new_tensor_network
