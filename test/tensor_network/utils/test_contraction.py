"""Tests for the tensor network contraction utilities."""

import pytest

from tensor_network import TensorNetwork
from tensor_network.utils.contraction import contract_pair


class TestTensorNetworkContraction:
    """Test the tensor network contraction utilities."""

    def test_contract_pair(self):
        """Test the contract_pair function."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )
        contracted_tn = contract_pair(tn, (0, 1))
        assert contracted_tn.input_indices == [[0, 2], [2, 3]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert contracted_tn.shapes == [(2, 4), (4, 5)]

    def test_contract_pair_no_shared_indices(self):
        """Test the contract_pair function when the tensors have no shared indices."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (4, 5)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )
        contracted_tn = contract_pair(tn, (0, 1))
        assert contracted_tn.input_indices == [[0, 1, 2, 3]]
        assert contracted_tn.size_dict == {0: 2, 1: 3, 2: 4, 3: 5}
        assert contracted_tn.shapes == [(2, 3, 4, 5)]

    def test_contract_pair_all_shared_indices(self):
        """Test the contract_pair function when the tensors have all shared indices."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [0, 1]],
            size_dict={0: 2, 1: 3},
            shapes=[(2, 3), (2, 3)],
            output_indices=[0],
            tensor_arrays=None,
        )
        contracted_tn = contract_pair(tn, (0, 1))
        assert contracted_tn.input_indices == [[]]
        assert contracted_tn.size_dict == {0: 2, 1: 3}
        assert contracted_tn.shapes == [()]

    def test_contract_pair_invalid_indices(self):
        """Test the contract_pair function when the indices are invalid."""
        tn = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )
        with pytest.raises(AssertionError, match="Tensor indices out of range."):
            contract_pair(tn, (0, 3))
