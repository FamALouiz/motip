"""Canonical contracted-first permutation strategy for tensor network contraction."""

from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor import get_contracted_indices
from memory import Memory
from memory.calculator.calculator import MemoryCalculator
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.strategy.common import (
    apply_layout_to_tensor,
    build_tree_maps,
    get_result_layout_from_current_step,
    sort_indices_by_size,
)
from permutation.utils import to_identity_permutation
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def _get_contracted_first_layout(
    current_tensor: Tensor,
    sibling_tensor: Tensor,
    size_dict: dict[int, int],
) -> list[int]:
    """Build a canonical contracted-first layout for a tensor.

    Args:
        current_tensor: The current tensor.
        sibling_tensor: The sibling tensor.
        size_dict: A mapping from index id to dimension size.

    Returns:
        A canonical contracted-first layout.
    """
    contracted = get_contracted_indices(current_tensor, sibling_tensor)
    free = set(current_tensor.input_indices) - contracted

    contracted_sorted = sort_indices_by_size(contracted, size_dict)
    free_sorted = sort_indices_by_size(free, size_dict)
    return contracted_sorted + free_sorted


class CanonicalContractedFirstPermutationStrategy(IPermutationStrategy):
    """Canonical contracted-first permutation strategy.

    This strategy always places contracted indices before free indices for
    tensors participating in a contraction.
    """

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
    ) -> tuple[list[Permutation], list[Permutation]]:
        """Find canonical contracted-first permutations.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            Canonical contracted-first permutations for initial and intermediate tensors.
        """
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        _, leaf_to_node, _ = build_tree_maps(persistent_path)

        initial_permutations: list[Permutation] = [
            to_identity_permutation(tensor) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        for leaf_pos, leaf_node in leaf_to_node.items():
            if leaf_node.parent is None or leaf_node.parent.contraction_step is None:
                continue

            step = leaf_node.parent.contraction_step
            left_tensor = persistent_path.get_state(step).tensors[contraction_path[step][0]]
            right_tensor = persistent_path.get_state(step).tensors[contraction_path[step][1]]

            current_tensor = left_tensor if leaf_node.parent.left is leaf_node else right_tensor
            sibling_tensor = right_tensor if leaf_node.parent.left is leaf_node else left_tensor

            layout = _get_contracted_first_layout(
                current_tensor,
                sibling_tensor,
                network.size_dict,
            )
            initial_permutations[leaf_pos] = apply_layout_to_tensor(
                network.tensors[leaf_pos],
                layout,
            )

        for step in range(persistent_path.num_steps):
            result_tensor = persistent_path.get_state(step + 1).tensors[contraction_path[step][0]]
            result_layout = get_result_layout_from_current_step(
                step,
                persistent_path,
                network.size_dict,
                left_first=False,
            )
            intermediate_permutations.append(apply_layout_to_tensor(result_tensor, result_layout))

        return initial_permutations, intermediate_permutations

    @staticmethod
    def __calculate_memory_for_path(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        peak: bool = True,
    ) -> Memory:
        """Calculate memory usage for a contraction path.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.
            peak: If True, return peak memory; if False, return total memory.

        Returns:
            Memory usage based on peak or total calculation.
        """
        memory_calculator = MemoryCalculator()
        current_memory = memory_calculator.calculate_memory_for_tensors(network.tensors)
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        result_memory = current_memory

        for idx, contraction_pair in enumerate(contraction_path):
            tensor_a, tensor_b = (
                persistent_path.get_state(idx).tensors[contraction_pair[0]],
                persistent_path.get_state(idx).tensors[contraction_pair[1]],
            )
            tensor_a_memory = memory_calculator.calculate_memory_for_tensor(tensor_a)
            tensor_b_memory = memory_calculator.calculate_memory_for_tensor(tensor_b)
            current_memory += tensor_a_memory + tensor_b_memory

            if peak:
                result_memory = max(result_memory, current_memory)

            current_memory -= (
                tensor_a_memory + tensor_b_memory
            )  # Remove the original forms of the permuted tensors
            contraction_memory = memory_calculator.calculate_memory_for_contraction(
                tensor_a, tensor_b
            )

            current_memory += contraction_memory  # Add the newly created result tensor

            if peak:
                result_memory = max(result_memory, current_memory)
            else:
                result_memory += tensor_a_memory + tensor_b_memory + contraction_memory
            current_memory -= (
                tensor_a_memory + tensor_b_memory
            )  # Remove permuted tensors from memory (since they are consumed)

        return result_memory

    @staticmethod
    @override
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the peak memory usage for a given contraction path and tensor permutations."""
        return CanonicalContractedFirstPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=True
        )

    @staticmethod
    @override
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the total memory movement for a contraction path and tensor permutations."""
        return CanonicalContractedFirstPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=False
        )
