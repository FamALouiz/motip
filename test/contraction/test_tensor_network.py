"""Test the tensor network contraction utilities."""

import pytest

from contraction.tensor_network import contract_network, contract_tensors_in_network
from tensor_network import TensorNetwork


class TestTensorNetworkContraction:
    """Test the tensor network contraction utilities."""

    def test_contract_pair_of_tensors_in_network(self) -> None:
        """Test contract_pair_of_tensors_in_network."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        contracted_tn = contract_tensors_in_network(tn, (0, 1))

        assert contracted_tn.input_indices == [[0, 2], [2, 3]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert contracted_tn.shapes == [(2, 4), (4, 5)]

    def test_contract_pair_of_tensors_in_network_no_shared_indices(self) -> None:
        """Test network contraction when tensors have no shared indices."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (4, 5)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )

        contracted_tn = contract_tensors_in_network(tn, (0, 1))

        assert contracted_tn.input_indices == [[0, 1, 2, 3]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert contracted_tn.shapes == [(2, 3, 4, 5)]

    def test_contract_pair_of_tensors_in_network_all_shared_indices(self) -> None:
        """Test network contraction when tensors have all shared indices."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [0, 1]],
            size_dict={0: 2, 1: 3},
            shapes=[(2, 3), (2, 3)],
            output_indices=[0],
            tensor_arrays=None,
        )

        contracted_tn = contract_tensors_in_network(tn, (0, 1))

        assert contracted_tn.input_indices == [[]]
        assert contracted_tn.size_dict == {0: 2, 1: 3}
        assert contracted_tn.shapes == [()]

    def test_contract_pair_of_tensors_in_network_invalid_indices(self) -> None:
        """Test network contraction when the indices are invalid."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        with pytest.raises(AssertionError, match="Tensor indices out of range."):
            contract_tensors_in_network(tn, (0, 3))

    def test_contract_pair_of_tensors_in_network_preserves_order(self) -> None:
        """Test network contraction preserves order of uncontracted indices."""
        tn = TensorNetwork(
            input_indices=[[0, 3, 2], [2, 1, 4]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
            shapes=[(2, 5, 4), (4, 3, 6)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        contracted_tn = contract_tensors_in_network(tn, (0, 1))

        assert contracted_tn.input_indices == [[0, 3, 1, 4]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
        assert contracted_tn.shapes == [(2, 5, 3, 6)]

    def test_end_to_end_two_step_contraction(self) -> None:
        """Test end-to-end two-step contraction behaviour."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        first_step = contract_tensors_in_network(tn, (0, 1))
        assert first_step.input_indices == [[0, 2], [2, 3]]

        second_step = contract_tensors_in_network(first_step, (0, 1))

        assert second_step.input_indices == [[0, 3]]
        assert second_step.shapes == [(2, 5)]

    def test_contraction_does_not_mutate_input_network(self) -> None:
        """Test that contracting tensors does not mutate the input network."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        _ = contract_tensors_in_network(tn, (0, 1))

        assert tn.input_indices == [[0, 1], [1, 2], [2, 3]]
        assert tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert tn.shapes == [(2, 3), (3, 4), (4, 5)]


class TestContractNetwork:
    """Test the contract_network function."""

    def test_contract_network_end_to_end(self) -> None:
        """Test end-to-end contraction of a network."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

        contraction_path = [(0, 1), (0, 1)]

        contracted_tn = contract_network(tn, contraction_path)

        assert contracted_tn.input_indices == [[0, 3]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert contracted_tn.shapes == [(2, 5)]
