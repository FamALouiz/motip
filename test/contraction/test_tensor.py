"""Tests for the tensor contraction utilities."""

from contraction.tensor import (
    contract_tensors,
    get_contracted_indices,
    get_indices_after_contraction,
)
from tensor import Tensor


class TestGetContractedIndices:
    """Tests for get_contracted_indices."""

    def test_get_contracted_indices(self) -> None:
        """Test get_contracted_indices."""
        tensor_a = Tensor([0, 1, 2], (2, 3, 4), None)
        tensor_b = Tensor([2, 3, 1], (4, 5, 3), None)

        contracted_indices = get_contracted_indices(tensor_a, tensor_b)

        assert contracted_indices == {1, 2}

    def test_get_contracted_indices_no_shared_indices(self) -> None:
        """Test get_contracted_indices when there are no shared indices."""
        tensor_a = Tensor([0, 1], (2, 3), None)
        tensor_b = Tensor([2, 3], (4, 5), None)

        contracted_indices = get_contracted_indices(tensor_a, tensor_b)

        assert contracted_indices == set()


class TestGetIndicesAfterContraction:
    """Tests for get_indices_after_contraction."""

    def test_get_indices_after_contraction(self) -> None:
        """Test get_indices_after_contraction."""
        tensor_a = Tensor([0, 1, 2], (2, 3, 4), None)
        tensor_b = Tensor([2, 3, 1], (4, 5, 3), None)

        new_indices = get_indices_after_contraction(tensor_a, tensor_b)

        assert new_indices == {0, 3}

    def test_get_indices_after_contraction_no_shared_indices(self) -> None:
        """Test get_indices_after_contraction when there are no shared indices."""
        tensor_a = Tensor([0, 1], (2, 3), None)
        tensor_b = Tensor([2, 3], (4, 5), None)

        new_indices = get_indices_after_contraction(tensor_a, tensor_b)

        assert new_indices == {0, 1, 2, 3}


class TestContractTensors:
    """Tests for contract_tensors."""

    def test_contract_tensors(self) -> None:
        """Test contract_tensors."""
        tensor_a = Tensor([0, 3, 2], (2, 5, 4), None)
        tensor_b = Tensor([2, 1, 4], (4, 3, 6), None)

        contracted_tensor = contract_tensors(tensor_a, tensor_b)

        assert contracted_tensor.input_indices == [0, 3, 1, 4]
        assert contracted_tensor.shape == (2, 5, 3, 6)
        assert contracted_tensor.array is None

    def test_contract_tensors_no_shared_indices(self) -> None:
        """Test contract_tensors when there are no shared indices."""
        tensor_a = Tensor([0, 1], (2, 3), None)
        tensor_b = Tensor([2, 3], (4, 5), None)

        contracted_tensor = contract_tensors(tensor_a, tensor_b)

        assert contracted_tensor.input_indices == [0, 1, 2, 3]
        assert contracted_tensor.shape == (2, 3, 4, 5)
        assert contracted_tensor.array is None
