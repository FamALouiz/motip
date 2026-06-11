"""Tests for contraction utils."""

import numpy as np

from operations.contraction import TensorContractionOperation
from operations.contraction.path import ContractionPath
from operations.permutation import TensorPermutationOperation
from tensor import Tensor
from tensor_network.utils.contraction import apply_operations_to_network


class TestApplyOperationsToNetwork:
    """Tests for the apply_operations_to_network function."""

    def test_apply_only_initial_permutations(self) -> None:
        """Test applying only initial permutations to the tensors without any contractions."""
        a = Tensor([0, 1], (2, 3), np.arange(6).reshape(2, 3).astype(float))
        b = Tensor([1, 2], (3, 4), np.arange(12).reshape(3, 4).astype(float))
        ops = [TensorPermutationOperation([1, 0]), TensorPermutationOperation([1, 0])]
        result = apply_operations_to_network([a, b], ops, [])
        assert isinstance(result, Tensor)
        assert result.input_indices == [1, 0]
        assert result.shape == (3, 2)
        assert np.array_equal(result.array, np.transpose(a.array, (1, 0)))

    def test_apply_contraction_and_permutation(self) -> None:
        """Test applying a contraction followed by a permutation to the resulting tensor."""
        a = Tensor([0, 1], (2, 3), np.arange(6).reshape(2, 3).astype(float))
        b = Tensor([1, 2], (3, 4), np.arange(12).reshape(3, 4).astype(float))
        initial_ops = [TensorPermutationOperation([0, 1]), TensorPermutationOperation([0, 1])]
        contraction_op = TensorContractionOperation([])
        perm_after = TensorPermutationOperation([1, 0])
        ops = initial_ops + [contraction_op, perm_after]
        path: ContractionPath = [(0, 1)]
        result = apply_operations_to_network([a, b], ops, path, use_tccg=False)
        expected_contracted = np.tensordot(a.array, b.array, axes=((1,), (0,)))
        expected_permuted = np.transpose(expected_contracted, (1, 0))
        assert result.input_indices == [2, 0]
        assert result.shape == expected_permuted.shape
        assert np.array_equal(result.array, expected_permuted)
