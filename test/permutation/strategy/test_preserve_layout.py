"""Tests for the preserve-layout permutation strategy."""

import pytest

from contraction.path import ContractionPath
from memory import Memory
from permutation.strategy.preserve_layout import PreserveLayoutPermutationStrategy
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
def two_step_chain_network() -> TensorNetwork:
    """A three-tensor chain with two contraction steps."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2], [2, 3]],
        size_dict={0: 2, 1: 3, 2: 5, 3: 7},
        shapes=[(2, 3), (3, 5), (5, 7)],
        output_indices=[0, 3],
        tensor_arrays=None,
    )


@pytest.fixture
def outer_product_network() -> TensorNetwork:
    """A two-tensor network with no shared indices (outer product)."""
    return TensorNetwork(
        input_indices=[[0, 1], [2, 3]],
        size_dict={0: 2, 1: 3, 2: 5, 3: 7},
        shapes=[(2, 3), (5, 7)],
        output_indices=[0, 1, 2, 3],
        tensor_arrays=None,
    )


class TestPreserveLayoutPermutationStrategy:
    """Tests for preserve-layout permutation outputs."""

    def test_find_optimal_permutation_single_step_returns_identities(
        self, single_step_network: TensorNetwork
    ) -> None:
        """The strategy returns identity permutations for one-step contraction."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            PreserveLayoutPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert initial_perms == [(0, 1), (0, 1)]
        assert intermediate_perms == [()]

    def test_find_optimal_permutation_two_step_chain_counts_and_identities(
        self, two_step_chain_network: TensorNetwork
    ) -> None:
        """Identity permutations are returned for all initial and intermediate tensors."""
        network = two_step_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        initial_perms, intermediate_perms = (
            PreserveLayoutPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 3
        assert len(intermediate_perms) == 2
        assert initial_perms == [(0, 1), (0, 1), (0, 1)]
        assert intermediate_perms == [(), ()]


class TestPreserveLayoutPermutationStrategyMemory:
    """Tests for preserve-layout memory estimations."""

    def test_get_peak_and_total_memory_single_step(
        self, single_step_network: TensorNetwork
    ) -> None:
        """Peak and total memory are correct when layouts are preserved."""
        network = single_step_network
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = PreserveLayoutPermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = PreserveLayoutPermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(208)
        assert total_memory == Memory(208)

    def test_get_peak_and_total_memory_two_step_chain(
        self, two_step_chain_network: TensorNetwork
    ) -> None:
        """Peak and total memory are correct for a deterministic two-step chain."""
        network = two_step_chain_network
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        peak_memory = PreserveLayoutPermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = PreserveLayoutPermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(528)
        assert total_memory == Memory(720)

    def test_get_peak_and_total_memory_outer_product(
        self, outer_product_network: TensorNetwork
    ) -> None:
        """Outer-product contraction uses expected memory values when preserving layouts."""
        network = outer_product_network
        contraction_path: ContractionPath = [(0, 1)]

        peak_memory = PreserveLayoutPermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = PreserveLayoutPermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(2008)
        assert total_memory == Memory(2008)

    def test_get_peak_and_total_memory_single_tensor_network(self) -> None:
        """No-contraction path keeps peak at initial memory and total movement at zero."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3, 4)],
            output_indices=[0, 1, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = []

        peak_memory = PreserveLayoutPermutationStrategy.get_peak_memory(network, contraction_path)
        total_memory = PreserveLayoutPermutationStrategy.get_total_memory(network, contraction_path)

        assert peak_memory == Memory(192)
        assert total_memory == Memory(0)
