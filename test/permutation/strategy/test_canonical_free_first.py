"""Tests for the canonical free-first permutation strategy."""

import pytest

from contraction.path import ContractionPath
from memory import Memory
from permutation.strategy.canonical_free_first import (
    CanonicalFreeFirstPermutationStrategy,
)
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


class TestCanonicalFreeFirstPermutationStrategy:
    """Tests for canonical free-first permutation layouts."""

    def test_find_optimal_permutation_single_step(self, single_step_network: TensorNetwork) -> None:
        """The strategy keeps free indices first on a simple pair."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            CanonicalFreeFirstPermutationStrategy.find_optimal_permutation(
                network, contraction_path
            )
        )

        assert initial_perms == [(0, 1), (1, 0)]
        assert intermediate_perms == [(0, 1)]

    def test_find_optimal_permutation_three_tensor_chain(
        self, three_tensor_chain_network: TensorNetwork
    ) -> None:
        """The strategy stays consistent across a two-step contraction chain."""
        network = three_tensor_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        initial_perms, intermediate_perms = (
            CanonicalFreeFirstPermutationStrategy.find_optimal_permutation(
                network, contraction_path
            )
        )

        assert initial_perms == [(0, 1), (1, 0), (1, 0)]
        assert intermediate_perms == [(0, 1), (0, 1)]

    def test_find_optimal_permutation_multiple_contracted_indices(
        self, multi_shared_index_network: TensorNetwork
    ) -> None:
        """The strategy sorts free indices before contracted indices by size."""
        network = multi_shared_index_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            CanonicalFreeFirstPermutationStrategy.find_optimal_permutation(
                network, contraction_path
            )
        )

        assert initial_perms == [(0, 1, 2), (1, 0, 2)]
        assert intermediate_perms == [(0, 1)]

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

        initial_perms, intermediate_perms = (
            CanonicalFreeFirstPermutationStrategy.find_optimal_permutation(
                network, contraction_path
            )
        )

        assert initial_perms == [(0, 1, 2)]
        assert intermediate_perms == []


class TestCanonicalFreeFirstPermutationStrategyMemory:
    """Tests for canonical free-first memory estimations."""

    def test_get_peak_and_total_memory_single_step(
        self, single_step_network: TensorNetwork
    ) -> None:
        """Peak and total memory are correct for one contraction."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = CanonicalFreeFirstPermutationStrategy.get_peak_memory(
            network, contraction_path
        )
        total_memory = CanonicalFreeFirstPermutationStrategy.get_total_memory(
            network, contraction_path
        )

        assert peak_memory == Memory(288)
        assert total_memory == Memory(352)
