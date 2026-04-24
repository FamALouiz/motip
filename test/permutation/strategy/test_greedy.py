"""Tests for the greedy permutation strategy."""

import pytest

from contraction.path import ContractionPath
from memory import Memory
from permutation.strategy.greedy import GreedyPermutationStrategy
from tensor_network import TensorNetwork


@pytest.fixture
def single_step_network() -> TensorNetwork:
    """A simple network with two tensors and one contraction step."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2]],
        size_dict={0: 2, 1: 3, 2: 4},
        shapes=[(2, 3), (3, 4)],
        output_indices=[0, 2],
        tensor_arrays=None,
    )


@pytest.fixture
def three_tensor_chain_network() -> TensorNetwork:
    """A simple chain with two contraction steps."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2], [2, 3]],
        size_dict={0: 2, 1: 3, 2: 4, 3: 5},
        shapes=[(2, 3), (3, 4), (4, 5)],
        output_indices=[0, 3],
        tensor_arrays=None,
    )


@pytest.fixture
def multi_shared_index_network() -> TensorNetwork:
    """A network where the contracted tensors share multiple indices."""
    return TensorNetwork(
        input_indices=[[0, 1, 4], [1, 2, 4]],
        size_dict={0: 7, 1: 2, 2: 5, 4: 3},
        shapes=[(7, 2, 3), (2, 5, 3)],
        output_indices=[0, 2],
        tensor_arrays=None,
    )


@pytest.fixture
def peak_chain_network() -> TensorNetwork:
    """A chain where the largest tensor appears before the final contraction."""
    return TensorNetwork(
        input_indices=[[0, 1, 4], [1, 2], [2, 3]],
        size_dict={0: 9, 1: 2, 2: 3, 3: 4, 4: 8},
        shapes=[(9, 2, 8), (2, 3), (3, 4)],
        output_indices=[0, 4, 3],
        tensor_arrays=None,
    )


class TestGreedyPermutationStrategy:
    """Tests for greedy permutation layouts."""

    def test_find_optimal_permutation_single_step(self, single_step_network: TensorNetwork) -> None:
        """The strategy keeps the peak tensor in its existing GEMM-friendly layout."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = GreedyPermutationStrategy.find_optimal_permutation(
            network, contraction_path
        )

        assert initial_perms == [(0, 1), (0, 1)]
        assert intermediate_perms == [(0, 1)]

    def test_find_optimal_permutation_three_tensor_chain(
        self, three_tensor_chain_network: TensorNetwork
    ) -> None:
        """The strategy propagates compatible layouts through a simple chain."""
        network = three_tensor_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        initial_perms, intermediate_perms = GreedyPermutationStrategy.find_optimal_permutation(
            network, contraction_path
        )

        assert initial_perms == [(0, 1), (0, 1), (0, 1)]
        assert intermediate_perms == [(0, 1), (0, 1)]

    def test_find_optimal_permutation_multiple_contracted_indices(
        self, multi_shared_index_network: TensorNetwork
    ) -> None:
        """The strategy keeps the peak layout and sorts the sibling to match it."""
        network = multi_shared_index_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = GreedyPermutationStrategy.find_optimal_permutation(
            network, contraction_path
        )

        assert initial_perms == [(0, 1, 2), (0, 2, 1)]
        assert intermediate_perms == [(0, 1)]

    def test_find_optimal_permutation_peak_chain(self, peak_chain_network: TensorNetwork) -> None:
        """The strategy freezes the largest intermediate and adapts surrounding layouts."""
        network = peak_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        initial_perms, intermediate_perms = GreedyPermutationStrategy.find_optimal_permutation(
            network, contraction_path
        )

        assert initial_perms == [(2, 0, 1), (0, 1), (0, 1)]
        assert intermediate_perms == [(1, 0, 2), (1, 0, 2)]

    def test_find_optimal_permutation_single_tensor_network(self) -> None:
        """A network with no contractions keeps the identity layout."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3, 4)],
            output_indices=[0, 1, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = []

        initial_perms, intermediate_perms = GreedyPermutationStrategy.find_optimal_permutation(
            network, contraction_path
        )

        assert initial_perms == [(0, 1, 2)]
        assert intermediate_perms == []


class TestGreedyPermutationStrategyMemory:
    """Tests for greedy memory estimations."""

    def test_get_peak_memory_single_step_equals_manual_value(
        self, single_step_network: TensorNetwork
    ) -> None:
        """Peak memory is correct for one contraction with greedy skip behavior."""
        network = single_step_network
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
