"""Contraction utils for tensor networks."""

from copy import deepcopy

from operations.base import TensorOperation
from operations.contraction.path import ContractionPath
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


def apply_operations_to_network(
    tensors: list[Tensor],
    operations: list[TensorOperation],
    contraction_path: ContractionPath,
    use_tccg: bool = True,
    use_hptt: bool = True,
) -> Tensor:
    """Apply a sequence of operations to a network of tensors.

    It is assumed that the operations are ordered such that the first len(tensors) operations are
    only permutations applied to the initial tensors. The function, then, follows the contraction
    path, applying contractions and subsequent permutations at each step, until only one tensor
    remains.
    """
    network_tensors = deepcopy(tensors)

    num_initial_tensors = len(tensors)
    for i in range(num_initial_tensors):
        network_tensors[i] = (
            operations[i].apply(tensor_operation_result_from_tensor(network_tensors[i])).tensor
        )

    for step, contraction_pair in enumerate(contraction_path):
        pair_idx_0, pair_idx_1 = contraction_pair
        tensor_a = network_tensors[pair_idx_0]
        tensor_b = network_tensors[pair_idx_1]

        contraction_op_idx = num_initial_tensors + 2 * step
        contracted_tensor = (
            operations[contraction_op_idx]
            .apply(
                tensor_operation_result_from_tensor(tensor_a),
                tensor_operation_result_from_tensor(tensor_b),
                use_tccg=use_tccg,
            )
            .tensor
        )

        perm_op_idx = num_initial_tensors + 2 * step + 1
        permuted_tensor = (
            operations[perm_op_idx]
            .apply(tensor_operation_result_from_tensor(contracted_tensor), use_hptt=use_hptt)
            .tensor
        )

        if pair_idx_0 > pair_idx_1:
            network_tensors.pop(pair_idx_0)
            network_tensors.pop(pair_idx_1)
        else:
            network_tensors.pop(pair_idx_1)
            network_tensors.pop(pair_idx_0)

        network_tensors.insert(min(pair_idx_0, pair_idx_1), permuted_tensor)

    return network_tensors[0]
