"""Tensor network contraction utilities."""

from copy import deepcopy

from tensor import Tensor
from tensor_network import TensorNetwork


def contract_tensors(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    """Contract two tensors together.

    Args:
        tensor_a: The first tensor to contract.
        tensor_b: The second tensor to contract.

    Returns:
        The resulting tensor after contraction.
    """
    new_tensor_indices = get_indices_after_contraction(tensor_a, tensor_b)

    ordered_new_indices = []
    new_tensor_shape = []
    indicies_placed_so_far = set()

    for idx, shape in zip(tensor_a.input_indices, tensor_a.shape, strict=True):
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            new_tensor_shape.append(shape)
            indicies_placed_so_far.update(ordered_new_indices)

    for idx, shape in zip(tensor_b.input_indices, tensor_b.shape, strict=True):
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            new_tensor_shape.append(shape)
            indicies_placed_so_far.update(ordered_new_indices)

    return Tensor(ordered_new_indices, tuple(new_tensor_shape), None)


def contract_pair_of_tensors_in_network(
    network: TensorNetwork, pair: tuple[int, int]
) -> TensorNetwork:
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

    for idx in sorted(pair, reverse=True):
        network.tensors.pop(idx)

    new_tensor = contract_tensors(first_contracted_tensor, second_contracted_tensor)
    new_tensor_network = deepcopy(network)

    new_tensor_network.tensors.insert(pair[0], new_tensor)

    return new_tensor_network


def get_contracted_indices(tensor_a: Tensor, tensor_b: Tensor) -> set[int]:
    """Get the set of contracted indices between two tensors."""
    contracted_indices = set(tensor_a.input_indices) & set(tensor_b.input_indices)

    return contracted_indices


def get_indices_after_contraction(tensor_a: Tensor, tensor_b: Tensor) -> set[int]:
    """Get the set of indices that will be present in the new tensor after contraction."""
    contracted_indices = get_contracted_indices(tensor_a, tensor_b)
    new_tensor_indices = (
        set(tensor_a.input_indices) | set(tensor_b.input_indices)
    ) - contracted_indices

    return new_tensor_indices
