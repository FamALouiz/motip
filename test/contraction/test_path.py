"""Tests for contraction path utilities."""

from copy import deepcopy

import pytest

from contraction.path import PersistentContractionPath
from contraction.tensor_network import contract_tensors_in_network
from tensor_network import TensorNetwork


class TestPersistentContractionPath:
    """Tests for contraction path history simulation and state access."""

    def test_history_length_must_match_path_length_plus_one(self) -> None:
        """Test history length validation."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            size_dict={0: 2, 1: 3, 2: 4},
            tensor_arrays=None,
        )

        with pytest.raises(
            ValueError,
            match="History must contain the initial state and one state per contraction.",
        ):
            PersistentContractionPath(path=[(0, 1)], history=[network])

    def test_from_contraction_path_with_empty_path(self) -> None:
        """Test history creation when no contractions are performed."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            size_dict={0: 2, 1: 3, 2: 4},
            tensor_arrays=None,
        )

        history = PersistentContractionPath.from_contraction_path(network, [])

        assert history.path == []
        assert history.num_steps == 0
        assert len(history.history) == 1
        assert history.initial_state == network
        assert history.final_state == network

    def test_from_contraction_path_simulates_all_steps(self) -> None:
        """Test all network states are generated in contraction order."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(2, 3), (3, 4), (4, 5)],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            tensor_arrays=None,
        )
        path = ((0, 1), (0, 1))
        initial_network = deepcopy(network)

        with_history = PersistentContractionPath.from_contraction_path(network, path)
        expected_after_first = contract_tensors_in_network(initial_network, (0, 1))
        expected_after_second = contract_tensors_in_network(expected_after_first, (0, 1))

        assert with_history.path == [(0, 1), (0, 1)]
        assert with_history.num_steps == 2
        assert len(with_history.history) == 3
        assert with_history.get_state(0) == initial_network
        assert with_history.get_state(1) == expected_after_first
        assert with_history.get_state(2) == expected_after_second
        assert with_history.initial_state == initial_network
        assert with_history.final_state == expected_after_second

    @pytest.mark.parametrize("step", [-1, 2])
    def test_get_state_raises_for_out_of_range_step(self, step: int) -> None:
        """Test bounds checking for state access."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            size_dict={0: 2, 1: 3, 2: 4},
            tensor_arrays=None,
        )
        with_history = PersistentContractionPath.from_contraction_path(network, [(0, 1)])

        with pytest.raises(IndexError, match="Step out of range."):
            with_history.get_state(step)

    def test_get_state_returns_deep_copy(self) -> None:
        """Test that mutating returned states does not mutate stored history."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(2, 3), (3, 4), (4, 5)],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            tensor_arrays=None,
        )
        with_history = PersistentContractionPath.from_contraction_path(network, [(0, 1), (0, 1)])

        state = with_history.get_state(1)
        state.tensors[0].input_indices[0] = 999

        assert with_history.get_state(1).tensors[0].input_indices[0] != 999

    def test_from_contraction_path_raises_for_invalid_pair(self) -> None:
        """Test invalid contraction pairs propagate underlying contraction errors."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            size_dict={0: 2, 1: 3, 2: 4},
            tensor_arrays=None,
        )

        with pytest.raises(AssertionError, match="Tensor indices out of range."):
            PersistentContractionPath.from_contraction_path(network, [(0, 2)])

    def test_history_shares_uncontracted_tensor_references_between_steps(self) -> None:
        """Test that uncontracted tensors are shared by reference across history states."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [3, 4]],
            output_indices=[0, 2, 3, 4],
            shapes=[(2, 3), (3, 4), (5, 6)],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
            tensor_arrays=None,
        )

        with_history = PersistentContractionPath.from_contraction_path(network, [(0, 1)])

        # Tensor at index 2 in step 0 is not contracted and should be reused in step 1.
        with_history.history[0].tensors[2].input_indices[0] = 999

        assert with_history.history[1].tensors[1].input_indices[0] == 999
