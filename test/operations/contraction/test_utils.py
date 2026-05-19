"""Tests for the tensor and tensor network contraction utilities."""

import numpy as np
import pytest

from operations.contraction import get_contracted_indices, get_indices_after_contraction
from operations.contraction.utils import contract_tensors, contract_tensors_in_network
from tensor import Tensor
from tensor_network import TensorNetwork


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
        contracted_tensor = contract_tensors(tensor_a, tensor_b, [])

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
        contracted_tensor = contract_tensors(tensor_a, tensor_b, [])

        assert contracted_tensor.input_indices == [0, 1, 2, 3]
        assert contracted_tensor.shape == (2, 5, 4, 3)

        if tensor_a.array is not None and tensor_b.array is not None:
            expected_array = np.tensordot(tensor_a.array, tensor_b.array, axes=0)

            assert contracted_tensor.array is not None
            assert np.allclose(contracted_tensor.array, expected_array)
        else:
            assert contracted_tensor.array is None


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
