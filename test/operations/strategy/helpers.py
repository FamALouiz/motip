"""Helpers for strategy operation tests."""

from operations.base import TensorOperation
from operations.contraction import TensorContractionOperation
from operations.contraction.path import ContractionPath
from operations.permutation import Permutation, TensorPermutationOperation


def extract_strategy_permutations(
    operations: list[TensorOperation],
    initial_tensor_count: int,
    contraction_path: ContractionPath,
) -> tuple[list[Permutation], list[Permutation]]:
    """Extract permutation values from a strategy operation list."""
    assert len(operations) == initial_tensor_count + 2 * len(contraction_path)

    initial_permutations: list[Permutation] = []
    for operation in operations[:initial_tensor_count]:
        assert isinstance(operation, TensorPermutationOperation)
        initial_permutations.append(tuple(operation.permutation))

    intermediate_permutations: list[Permutation] = []
    offset = initial_tensor_count
    for step in range(len(contraction_path)):
        contraction_operation = operations[offset + 2 * step]
        permutation_operation = operations[offset + 2 * step + 1]

        assert isinstance(contraction_operation, TensorContractionOperation)
        assert contraction_operation.sliced_indices == []
        assert isinstance(permutation_operation, TensorPermutationOperation)
        intermediate_permutations.append(tuple(permutation_operation.permutation))

    return initial_permutations, intermediate_permutations
