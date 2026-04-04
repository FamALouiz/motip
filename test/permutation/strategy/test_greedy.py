"""Tests for the GreedyPermutationStrategy."""

from contraction.path import ContractionPath
from memory import Memory
from permutation.strategy.greedy import GreedyPermutationStrategy
from tensor_network import TensorNetwork


class TestGreedyPermutationStrategyMemory:
    """Tests for greedy memory estimations."""

    def test_get_peak_memory_single_step_equals_manual_value(self) -> None:
        """Peak memory is correct for one contraction with greedy skip behavior."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3), (3, 4)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = GreedyPermutationStrategy.get_peak_memory(network, contraction_path)

        assert peak_memory == Memory(256)

    def test_get_peak_memory_skips_largest_tensor_permutation(self) -> None:
        """Peak memory reflects largest-tensor skip behavior in greedy strategy."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(2, 3), (3, 5), (5, 7)],
            size_dict={0: 2, 1: 3, 2: 5, 3: 7},
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        peak_memory = GreedyPermutationStrategy.get_peak_memory(network, contraction_path)

        assert peak_memory == Memory(720)

    def test_get_total_memory_matches_manual_value(self) -> None:
        """Total memory movement is deterministic for a fixed two-step chain."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(2, 3), (3, 5), (5, 7)],
            size_dict={0: 2, 1: 3, 2: 5, 3: 7},
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        total_memory = GreedyPermutationStrategy.get_total_memory(network, contraction_path)

        assert total_memory == Memory(1168)

    def test_get_peak_and_total_memory_single_tensor_network(self) -> None:
        """No-contraction path returns initial memory for both peak and total."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3, 4)],
            output_indices=[0, 1, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = []

        peak_memory = GreedyPermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = GreedyPermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(192)
        assert total_memory == Memory(192)
