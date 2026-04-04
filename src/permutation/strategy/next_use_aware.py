"""Next-use-aware permutation strategy for tensor network contraction."""

from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from memory.calculator.calculator import MemoryCalculator
from memory.memory import Memory
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.strategy.common import (
    apply_layout_to_tensor,
    build_tree_maps,
    get_input_layout_for_parent_use,
    sort_indices_by_size,
    to_identity_permutation,
)
from tensor_network.tn import TensorNetwork


class NextUseAwarePermutationStrategy(IPermutationStrategy):
    """Next-use-aware permutation strategy.

    This strategy chooses each tensor's layout based on the way it will be used
    in its next contraction instead of using only the current contraction step.
    """

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
    ) -> tuple[list[Permutation], list[Permutation]]:
        """Find next-use-aware permutations.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            Next-use-aware permutations for initial and intermediate tensors.
        """
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        _, leaf_to_node, step_to_node = build_tree_maps(persistent_path)

        initial_permutations: list[Permutation] = [
            to_identity_permutation(tensor) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        for leaf_pos, leaf_node in leaf_to_node.items():
            target_layout = get_input_layout_for_parent_use(
                leaf_node,
                persistent_path,
                network.size_dict,
            )
            if target_layout is None:
                continue

            initial_permutations[leaf_pos] = apply_layout_to_tensor(
                network.tensors[leaf_pos],
                target_layout,
            )

        for step in range(persistent_path.num_steps):
            node = step_to_node[step]
            result_tensor = persistent_path.get_state(step + 1).tensors[contraction_path[step][0]]

            target_layout = get_input_layout_for_parent_use(
                node,
                persistent_path,
                network.size_dict,
            )

            if target_layout is None:
                target_layout = sort_indices_by_size(
                    result_tensor.input_indices,
                    network.size_dict,
                )

            intermediate_permutations.append(apply_layout_to_tensor(result_tensor, target_layout))

        return initial_permutations, intermediate_permutations

    @staticmethod
    @override
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate peak memory usage for the next-use-aware strategy.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            The peak memory usage.
        """
        memory_calculator = MemoryCalculator()
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        initial_permutations, intermediate_permutations = (
            NextUseAwarePermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        current_memory = memory_calculator.calculate_memory_for_tensors(network.tensors)
        peak_memory = current_memory

        live_tensor_sources: list[tuple[str, int]] = [
            ("initial", idx) for idx in range(len(network.tensors))
        ]
        initial_tensor_permuted = [False for _ in network.tensors]

        for step, (left_pos, right_pos) in enumerate(contraction_path):
            state_before = persistent_path.get_state(step)
            left_tensor = state_before.tensors[left_pos]
            right_tensor = state_before.tensors[right_pos]

            left_source = live_tensor_sources[left_pos]
            right_source = live_tensor_sources[right_pos]

            left_memory = memory_calculator.calculate_memory_for_tensor(left_tensor)
            right_memory = memory_calculator.calculate_memory_for_tensor(right_tensor)

            if left_source[0] == "initial":
                initial_idx = left_source[1]
                identity = tuple(range(len(left_tensor.input_indices)))
                if (
                    not initial_tensor_permuted[initial_idx]
                    and initial_permutations[initial_idx] != identity
                ):
                    current_memory += left_memory
                    peak_memory = max(peak_memory, current_memory)
                    current_memory -= left_memory
                    initial_tensor_permuted[initial_idx] = True

            if right_source[0] == "initial":
                initial_idx = right_source[1]
                identity = tuple(range(len(right_tensor.input_indices)))
                if (
                    not initial_tensor_permuted[initial_idx]
                    and initial_permutations[initial_idx] != identity
                ):
                    current_memory += right_memory
                    peak_memory = max(peak_memory, current_memory)
                    current_memory -= right_memory
                    initial_tensor_permuted[initial_idx] = True

            result_tensor = persistent_path.get_state(step + 1).tensors[left_pos]
            result_memory = memory_calculator.calculate_memory_for_tensor(result_tensor)

            current_memory += result_memory
            peak_memory = max(peak_memory, current_memory)

            current_memory -= left_memory + right_memory

            identity_result = tuple(range(len(result_tensor.input_indices)))
            if intermediate_permutations[step] != identity_result:
                current_memory += result_memory
                peak_memory = max(peak_memory, current_memory)
                current_memory -= result_memory

            live_tensor_sources[left_pos] = ("intermediate", step)
            del live_tensor_sources[right_pos]

        return peak_memory

    @staticmethod
    @override
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate total memory movement for the next-use-aware strategy.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            The total memory movement.
        """
        memory_calculator = MemoryCalculator()
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        initial_permutations, intermediate_permutations = (
            NextUseAwarePermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        total_memory = Memory(0)

        live_tensor_sources: list[tuple[str, int]] = [
            ("initial", idx) for idx in range(len(network.tensors))
        ]
        initial_tensor_permuted = [False for _ in network.tensors]

        for step, (left_pos, right_pos) in enumerate(contraction_path):
            state_before = persistent_path.get_state(step)
            left_tensor = state_before.tensors[left_pos]
            right_tensor = state_before.tensors[right_pos]

            left_source = live_tensor_sources[left_pos]
            right_source = live_tensor_sources[right_pos]

            left_memory = memory_calculator.calculate_memory_for_tensor(left_tensor)
            right_memory = memory_calculator.calculate_memory_for_tensor(right_tensor)

            if left_source[0] == "initial":
                initial_idx = left_source[1]
                identity = tuple(range(len(left_tensor.input_indices)))
                if (
                    not initial_tensor_permuted[initial_idx]
                    and initial_permutations[initial_idx] != identity
                ):
                    total_memory += left_memory + left_memory
                    initial_tensor_permuted[initial_idx] = True

            if right_source[0] == "initial":
                initial_idx = right_source[1]
                identity = tuple(range(len(right_tensor.input_indices)))
                if (
                    not initial_tensor_permuted[initial_idx]
                    and initial_permutations[initial_idx] != identity
                ):
                    total_memory += right_memory + right_memory
                    initial_tensor_permuted[initial_idx] = True

            result_tensor = persistent_path.get_state(step + 1).tensors[left_pos]
            result_memory = memory_calculator.calculate_memory_for_tensor(result_tensor)

            total_memory += left_memory + right_memory + result_memory

            identity_result = tuple(range(len(result_tensor.input_indices)))
            if intermediate_permutations[step] != identity_result:
                total_memory += result_memory + result_memory

            live_tensor_sources[left_pos] = ("intermediate", step)
            del live_tensor_sources[right_pos]

        return total_memory
