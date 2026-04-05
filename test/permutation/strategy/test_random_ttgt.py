"""Tests for random TTGT-style memory estimations."""

from typing import Sequence

import pytest

from contraction.path import ContractionPath
from contraction.tensor import get_contracted_indices
from memory import Memory
from permutation.strategy.random_ttgt import RandomTTGTPermutationStrategy
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
    """A three-tensor chain network with two contraction steps."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2], [2, 3]],
        size_dict={0: 2, 1: 3, 2: 5, 3: 7},
        shapes=[(2, 3), (3, 5), (5, 7)],
        output_indices=[0, 3],
        tensor_arrays=None,
    )


@pytest.fixture
def two_tensor_multi_free_network() -> TensorNetwork:
    """A two-tensor network with multiple free indices on both sides."""
    return TensorNetwork(
        input_indices=[[0, 1, 2], [2, 3, 4]],
        size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
        shapes=[(2, 3, 4), (4, 5, 6)],
        output_indices=[0, 1, 3, 4],
        tensor_arrays=None,
    )


def _to_layout(input_indices: list[int], permutation: Sequence[int]) -> list[int]:
    """Apply a permutation to an index layout."""
    return [input_indices[idx] for idx in permutation]


class TestRandomTTGTPermutationStrategy:
    """Tests for random TTGT-style permutation generation."""

    def test_find_optimal_permutation_single_step_is_ttgt_valid(
        self, single_step_network: TensorNetwork
    ) -> None:
        """Input permutations keep TTGT grouping for left and right tensors."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = RandomTTGTPermutationStrategy.find_optimal_permutation(
            network, contraction_path, seed=1
        )

        left_tensor = network.tensors[0]
        right_tensor = network.tensors[1]
        contracted = get_contracted_indices(left_tensor, right_tensor)

        left_layout = _to_layout(left_tensor.input_indices, initial_perms[0])
        right_layout = _to_layout(right_tensor.input_indices, initial_perms[1])

        left_free_count = len(set(left_tensor.input_indices) - contracted)
        right_contracted_count = len(contracted)

        assert len(initial_perms) == 2
        assert len(intermediate_perms) == 1
        assert set(left_layout[:left_free_count]).isdisjoint(contracted)
        assert set(left_layout[left_free_count:]) == contracted
        assert set(right_layout[:right_contracted_count]) == contracted
        assert set(right_layout[right_contracted_count:]).isdisjoint(contracted)

    def test_find_optimal_permutation_same_seed_produces_same_output(
        self, two_tensor_multi_free_network: TensorNetwork
    ) -> None:
        """Repeated calls with the same seed produce identical permutations."""
        network = two_tensor_multi_free_network
        contraction_path: ContractionPath = [(0, 1)]

        first_initial, first_intermediate = RandomTTGTPermutationStrategy.find_optimal_permutation(
            network, contraction_path, seed=42
        )
        second_initial, second_intermediate = (
            RandomTTGTPermutationStrategy.find_optimal_permutation(
                network, contraction_path, seed=42
            )
        )

        assert first_initial == second_initial
        assert first_intermediate == second_intermediate

    def test_find_optimal_permutation_different_seed_can_change_output(
        self, two_tensor_multi_free_network: TensorNetwork
    ) -> None:
        """Different seeds can produce different random TTGT layouts."""
        network = two_tensor_multi_free_network
        contraction_path: ContractionPath = [(0, 1)]

        seed_zero_output = RandomTTGTPermutationStrategy.find_optimal_permutation(
            network, contraction_path, seed=0
        )
        seed_one_output = RandomTTGTPermutationStrategy.find_optimal_permutation(
            network, contraction_path, seed=1
        )

        assert seed_zero_output != seed_one_output

    def test_find_optimal_permutation_two_tensor_multi_free_seed_one_expected_values(
        self, two_tensor_multi_free_network: TensorNetwork
    ) -> None:
        """Seeded random TTGT permutations are stable for a multi-free-index network."""
        network = two_tensor_multi_free_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = RandomTTGTPermutationStrategy.find_optimal_permutation(
            network, contraction_path, seed=1
        )

        assert initial_perms == [(1, 0, 2), (0, 2, 1)]
        assert intermediate_perms == [(3, 2, 0, 1)]


class TestRandomTTGTPermutationStrategyMemory:
    """Tests for random TTGT-style memory estimations."""

    def test_get_peak_and_total_memory_single_step_seed_one(
        self, single_step_network: TensorNetwork
    ) -> None:
        """Peak and total memory are correct for a fixed random seed."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=1
        )
        total_memory = RandomTTGTPermutationStrategy.get_total_memory(
            network, contraction_path, seed=1
        )

        assert peak_memory == Memory(288)
        assert total_memory == Memory(352)

    def test_get_peak_and_total_memory_two_step_chain_seed_zero(
        self, three_tensor_chain_network: TensorNetwork
    ) -> None:
        """Seeded random behavior gives stable memory values for a two-step chain."""
        network = three_tensor_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        peak_memory = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=0
        )
        total_memory = RandomTTGTPermutationStrategy.get_total_memory(
            network, contraction_path, seed=0
        )

        assert peak_memory == Memory(528)
        assert total_memory == Memory(1168)

    def test_get_peak_memory_changes_with_seed_for_chain(
        self, three_tensor_chain_network: TensorNetwork
    ) -> None:
        """Different seeds can change the sampled peak memory path."""
        network = three_tensor_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        peak_seed_zero = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=0
        )
        peak_seed_one = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=1
        )

        assert peak_seed_zero == Memory(528)
        assert peak_seed_one == Memory(640)
        assert peak_seed_zero != peak_seed_one

    def test_get_peak_and_total_memory_same_seed_are_reproducible(
        self, three_tensor_chain_network: TensorNetwork
    ) -> None:
        """Repeated memory evaluations with the same seed are deterministic."""
        network = three_tensor_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        first_peak = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=7
        )
        first_total = RandomTTGTPermutationStrategy.get_total_memory(
            network, contraction_path, seed=7
        )

        second_peak = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=7
        )
        second_total = RandomTTGTPermutationStrategy.get_total_memory(
            network, contraction_path, seed=7
        )

        assert first_peak == second_peak
        assert first_total == second_total

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

        peak_memory = RandomTTGTPermutationStrategy.get_peak_memory(
            network, contraction_path, seed=42
        )
        total_memory = RandomTTGTPermutationStrategy.get_total_memory(
            network, contraction_path, seed=42
        )

        assert peak_memory == Memory(192)
        assert total_memory == Memory(192)
