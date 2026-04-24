"""Preserve-layout permutation strategy for tensor network contraction."""

from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from memory import Memory
from memory.calculator.calculator import MemoryCalculator
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.utils import to_identity_permutation
from tensor_network.tn import TensorNetwork


class PreserveLayoutPermutationStrategy(IPermutationStrategy):
    """Preserve-layout permutation strategy.

    This strategy never changes the layout of any initial or intermediate tensor.
    """

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
    ) -> tuple[list[Permutation], list[Permutation]]:
        """Return identity permutations for all tensors.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            Identity permutations for all initial and intermediate tensors.
        """
        initial_permutations = [to_identity_permutation(tensor) for tensor in network.tensors]
        intermediate_permutations: list[Permutation] = [
            tuple() for _ in range(len(contraction_path))
        ]
        return initial_permutations, intermediate_permutations

    @staticmethod
    @override
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate peak memory usage for the preserve-layout strategy.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            The peak memory usage.
        """
        memory_calculator = MemoryCalculator()
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)

        live_tensors = list(network.tensors)
        current_memory = memory_calculator.calculate_memory_for_tensors(live_tensors)
        peak_memory = current_memory

        for step, (left_pos, right_pos) in enumerate(contraction_path):
            state_before = persistent_path.get_state(step)
            left_tensor = state_before.tensors[left_pos]
            right_tensor = state_before.tensors[right_pos]

            result_memory = memory_calculator.calculate_memory_for_contraction(
                left_tensor, right_tensor
            )

            current_memory += result_memory
            peak_memory = max(peak_memory, current_memory)

            left_memory = memory_calculator.calculate_memory_for_tensor(left_tensor)
            right_memory = memory_calculator.calculate_memory_for_tensor(right_tensor)
            current_memory -= left_memory + right_memory

        return peak_memory

    @staticmethod
    @override
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate total memory movement for the preserve-layout strategy.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.

        Returns:
            The total memory movement.
        """
        memory_calculator = MemoryCalculator()
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)

        total_memory = Memory(0)

        for step, (left_pos, right_pos) in enumerate(contraction_path):
            state_before = persistent_path.get_state(step)
            left_tensor = state_before.tensors[left_pos]
            right_tensor = state_before.tensors[right_pos]

            left_memory = memory_calculator.calculate_memory_for_tensor(left_tensor)
            right_memory = memory_calculator.calculate_memory_for_tensor(right_tensor)
            result_memory = memory_calculator.calculate_memory_for_contraction(
                left_tensor, right_tensor
            )

            total_memory += left_memory + right_memory + result_memory

        return total_memory
