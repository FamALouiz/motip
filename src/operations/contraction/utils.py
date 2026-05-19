"""Util functions for contraction operations."""

from copy import deepcopy

from operations.contraction import TensorContractionOperation
from operations.contraction.path import ContractionPath
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor
from tensor_network.tn import TensorNetwork


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


def contract_tensors(tensor_a: Tensor, tensor_b: Tensor, sliced_indices: list[int]) -> Tensor:
    """Contract two tensors together.

    Args:
        tensor_a: The first tensor to contract.
        tensor_b: The second tensor to contract.
        sliced_indices: The indices to slice.

    Returns:
        The resulting tensor after contraction.
    """
    contract_tensors = TensorContractionOperation(sliced_indices)
    return contract_tensors.apply(
        tensor_operation_result_from_tensor(tensor_a),
        tensor_operation_result_from_tensor(tensor_b),
    ).tensor


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

    new_tensor = contract_tensors(first_contracted_tensor, second_contracted_tensor, [])

    new_tensor_network = TensorNetwork(
        tensors=[t for i, t in enumerate(network.tensors) if i not in pair],
        output_indices=network.output_indices,
        size_dict=network.size_dict,
    )
    new_tensor_network.tensors.insert(pair[0], new_tensor)

    return new_tensor_network


def contract_network(network: TensorNetwork, contraction_path: ContractionPath) -> TensorNetwork:
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
