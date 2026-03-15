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

    first_contracted_tensor = network.tensors[pair[0]]
    second_contracted_tensor = network.tensors[pair[1]]

    new_tensor_network = deepcopy(network)

    contracted_indices = set(first_contracted_tensor.input_indices) & set(
        second_contracted_tensor.input_indices
    )
    new_tensor_indices = (
        set(first_contracted_tensor.input_indices) | set(second_contracted_tensor.input_indices)
    ) - contracted_indices

    for idx in sorted(pair, reverse=True):
        new_tensor_network.tensors.pop(idx)

    ordered_new_indices = []
    indicies_placed_so_far = set()

    for idx in first_contracted_tensor.input_indices:
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            indicies_placed_so_far.update(ordered_new_indices)

    for idx in second_contracted_tensor.input_indices:
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            indicies_placed_so_far.update(ordered_new_indices)

    new_tensor_shape = tuple(network.size_dict[index] for index in ordered_new_indices)

    new_tensor_network.tensors.insert(pair[0], Tensor(ordered_new_indices, new_tensor_shape, None))

    return new_tensor_network
