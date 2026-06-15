"""Contraction utils for tensor networks."""

from copy import deepcopy
from typing import Sequence

from operations.base import TensorOperation
from operations.contraction.path import ContractionPath
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


def apply_operations_to_network(
    tensors: Sequence[Tensor],
    operations: Sequence[TensorOperation],
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
    network_tensors = list(deepcopy(tensors))

    num_initial_tensors = len(tensors)
    for i in range(num_initial_tensors):
        network_tensors[i] = (
            operations[i].apply(tensor_operation_result_from_tensor(network_tensors[i])).tensor
        )
        print(f"Applying operation {i + 1}")

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
                use_tccg=use_tccg,  # type: ignore[arg-type]
            )
            .tensor
        )

        print(f"Applying operation {contraction_op_idx + 1}")

        perm_op_idx = num_initial_tensors + 2 * step + 1
        permuted_tensor = (
            operations[perm_op_idx]
            .apply(tensor_operation_result_from_tensor(contracted_tensor), use_hptt=use_hptt)  # type: ignore[arg-type]
            .tensor
        )

        print(f"Applying operation {perm_op_idx + 1}")

        if pair_idx_0 > pair_idx_1:
            network_tensors.pop(pair_idx_0)
            network_tensors.pop(pair_idx_1)
        else:
            network_tensors.pop(pair_idx_1)
            network_tensors.pop(pair_idx_0)

        network_tensors.insert(pair_idx_0, permuted_tensor)

    return network_tensors[0]
