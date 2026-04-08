"""Local optimal permutation strategy for tensor network contraction."""

from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor import get_contracted_indices
from contraction.tree import ContractionTree, ContractionTreeNode
from memory.calculator.calculator import MemoryCalculator
from memory.memory import Memory
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.strategy.common import get_step_tensors, sort_indices_by_size
from permutation.utils import to_permutation
from tensor_network.tn import TensorNetwork


def _get_optimal_layout_for_tensor(
    is_left: bool,
    contraction_step: int,
    persistent_path: PersistentContractionPath,
    size_dict: dict[int, int],
) -> list[int]:
    left_tensor, right_tensor, _ = get_step_tensors(persistent_path, contraction_step)
    tensor_at_step = left_tensor if is_left else right_tensor
    sibling_tensor = right_tensor if is_left else left_tensor

    contracted = get_contracted_indices(tensor_at_step, sibling_tensor)
    free = set(tensor_at_step.input_indices) - contracted

    contracted_sorted = sort_indices_by_size(contracted, size_dict)
    free_sorted = sort_indices_by_size(free, size_dict)

    if is_left:
        return free_sorted + contracted_sorted
    return contracted_sorted + free_sorted


class LocalOptimalPermutationStrategy(IPermutationStrategy):
    """Local optimal permutation strategy for tensor network contraction.

    This strategy determines the optimal permutation for each intermediate contraction step
    by sorting the contracted and free indices based on their sizes.
    """

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork, contraction_path: ContractionPath
    ) -> tuple[list[Permutation], list[Permutation]]:
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        persistent_tree = ContractionTree.from_contraction_path(persistent_path)

        initial_permutations: list[Permutation] = [
            tuple(range(len(tensor.input_indices))) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        leaf_to_node: dict[int, ContractionTreeNode] = {}
        stack = [persistent_tree.root]
        while stack:
            node = stack.pop()
            if node.initial_tensor_position is not None:
                leaf_to_node[node.initial_tensor_position] = node
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        for leaf_pos, leaf_node in leaf_to_node.items():
            if leaf_node.parent is not None and leaf_node.parent.contraction_step is not None:
                contraction_step = leaf_node.parent.contraction_step
                is_left = leaf_node.parent.left is leaf_node

                optimal_layout = _get_optimal_layout_for_tensor(
                    is_left,
                    contraction_step,
                    persistent_path,
                    network.size_dict,
                )

                permutation = to_permutation(
                    network.tensors[leaf_pos].input_indices, optimal_layout
                )
                initial_permutations[leaf_pos] = permutation

        for step in range(persistent_path.num_steps):
            left_tensor, right_tensor, result_tensor = get_step_tensors(persistent_path, step)

            contracted = get_contracted_indices(left_tensor, right_tensor)
            left_free = set(left_tensor.input_indices) - contracted
            right_free = set(right_tensor.input_indices) - contracted

            contracted_sorted = sort_indices_by_size(contracted, network.size_dict)
            left_free_sorted = sort_indices_by_size(left_free, network.size_dict)
            right_free_sorted = sort_indices_by_size(right_free, network.size_dict)

            result_layout = left_free_sorted + contracted_sorted + right_free_sorted
            result_layout = [idx for idx in result_layout if idx in result_tensor.input_indices]

            permutation = to_permutation(result_tensor.input_indices, result_layout)
            intermediate_permutations.append(permutation)

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

            contraction_memory = memory_calculator.calculate_memory_for_contraction(
                tensor_a, tensor_b
            )
            current_memory += contraction_memory
            if peak:
                result_memory = max(result_memory, current_memory)
            else:
                result_memory += tensor_a_memory + tensor_b_memory + contraction_memory
            current_memory -= (
                tensor_a_memory + tensor_b_memory
            )  # Remove the original forms of the permuted tensors

        return result_memory

    @staticmethod
    @override
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        return LocalOptimalPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=True
        )

    @staticmethod
    @override
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        return LocalOptimalPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=False
        )
