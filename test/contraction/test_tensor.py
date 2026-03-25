"""Tests for the tensor contraction utilities."""

import numpy as np
import pytest

from contraction.tensor import (
    contract_tensors,
    get_contracted_indices,
    get_indices_after_contraction,
)
from tensor import Tensor


def _contract_arrays(tensor_a: Tensor, tensor_b: Tensor) -> np.ndarray:
    """Helper function to contract the arrays of two tensors."""
    assert tensor_a.array is not None and tensor_b.array is not None, (
        "Both tensors must have arrays to contract."
    )

    contracted_indices = get_contracted_indices(tensor_a, tensor_b)
    axes_a = tuple(tensor_a.input_indices.index(idx) for idx in contracted_indices)
    axes_b = tuple(tensor_b.input_indices.index(idx) for idx in contracted_indices)
    new_array = np.tensordot(tensor_a.array, tensor_b.array, axes=(axes_a, axes_b))

    return new_array


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

    @pytest.mark.parametrize(
        ("tensor_a", "tensor_b"),
        [
            pytest.param(
                Tensor([0, 3, 2], (2, 5, 4), None),
                Tensor([2, 1, 4], (4, 3, 6), None),
                id="no arrays",
            ),
            pytest.param(
                Tensor([0, 3, 2], (2, 5, 4), np.random.random((2, 5, 4))),
                Tensor([2, 1, 4], (4, 3, 6), np.random.random((4, 3, 6))),
                id="with arrays",
            ),
            pytest.param(
                Tensor([0, 3, 2], (2, 5, 4), np.random.random((2, 5, 4))),
                Tensor([2, 1, 4], (4, 3, 6), None),
                id="first with array, second without",
            ),
            pytest.param(
                Tensor([0, 3, 2], (2, 5, 4), None),
                Tensor([2, 1, 4], (4, 3, 6), np.random.random((4, 3, 6))),
                id="second with array, first without",
            ),
        ],
    )
    def test_contract_tensors(self, tensor_a: Tensor, tensor_b: Tensor) -> None:
        """Test contract_tensors."""
        contracted_tensor = contract_tensors(tensor_a, tensor_b)

        assert contracted_tensor.input_indices == [0, 3, 1, 4]
        assert contracted_tensor.shape == (2, 5, 3, 6)

        if tensor_a.array is not None and tensor_b.array is not None:
            expected_array = _contract_arrays(tensor_a, tensor_b)

            assert contracted_tensor.array is not None
            assert np.allclose(contracted_tensor.array, expected_array)
        else:
            assert contracted_tensor.array is None

    @pytest.mark.parametrize(
        ("tensor_a", "tensor_b"),
        [
            pytest.param(
                Tensor([0, 1], (2, 5), None),
                Tensor([2, 3], (4, 3), None),
                id="no arrays",
            ),
            pytest.param(
                Tensor([0, 1], (2, 5), np.random.random((2, 5))),
                Tensor([2, 3], (4, 3), np.random.random((4, 3))),
                id="with arrays",
            ),
            pytest.param(
                Tensor([0, 1], (2, 5), np.random.random((2, 5))),
                Tensor([2, 3], (4, 3), None),
                id="first with array, second without",
            ),
            pytest.param(
                Tensor([0, 1], (2, 5), None),
                Tensor([2, 3], (4, 3), np.random.random((4, 3))),
                id="second with array, first without",
            ),
        ],
    )
    def test_contract_tensors_no_shared_indices(self, tensor_a: Tensor, tensor_b: Tensor) -> None:
        """Test contract_tensors when there are no shared indices."""
        contracted_tensor = contract_tensors(tensor_a, tensor_b)

        assert contracted_tensor.input_indices == [0, 1, 2, 3]
        assert contracted_tensor.shape == (2, 5, 4, 3)

        if tensor_a.array is not None and tensor_b.array is not None:
            expected_array = np.tensordot(tensor_a.array, tensor_b.array, axes=0)

            assert contracted_tensor.array is not None
            assert np.allclose(contracted_tensor.array, expected_array)
        else:
            assert contracted_tensor.array is None
