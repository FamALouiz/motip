"""Tests for the NextUseAwarePermutationStrategy memory estimates."""

from contraction.path import ContractionPath
from memory import Memory
from permutation.strategy.next_use_aware import NextUseAwarePermutationStrategy
from tensor_network import TensorNetwork


class TestNextUseAwarePermutationStrategyMemory:
    """Tests for next-use-aware memory estimations."""

    def test_get_peak_and_total_memory_single_step_without_permutation_copy(self) -> None:
        """Peak and total memory are correct when no tensor needs a permutation copy."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3), (3, 4)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = NextUseAwarePermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = NextUseAwarePermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(208)
        assert total_memory == Memory(352)

    def test_get_peak_and_total_memory_single_step_with_permutation_copy(self) -> None:
        """Peak and total memory include the copy required for the first use layout."""
        network = TensorNetwork(
            input_indices=[[1, 0], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(3, 2), (3, 4)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = NextUseAwarePermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = NextUseAwarePermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(208)
        assert total_memory == Memory(400)
