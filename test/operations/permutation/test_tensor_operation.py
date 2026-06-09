"""Tests for the tensor permutation operation."""

import numpy as np
import pytest

from operations.permutation import TensorPermutationOperation
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


class TestPermuteTensor:
    """Tests for permute_tensor behavior."""

    def test_permute_tensor_reorders_indices_and_shape(self) -> None:
        """Test that a valid permutation reorders input indices and shape."""
        tensor = Tensor(
            input_indices=[10, 11, 12],
            shape=(2, 3, 4),
            array=np.arange(24).reshape(2, 3, 4).astype(np.float64),
        )

        permuted = (
            TensorPermutationOperation([2, 0, 1])
            .apply(tensor_operation_result_from_tensor(tensor))
            .tensor
        )

        assert permuted.input_indices == [12, 10, 11]
        assert permuted.shape == (4, 2, 3)
        assert tensor.array is not None and permuted.array is not None
        assert (permuted.array == np.transpose(tensor.array, axes=[2, 0, 1])).all()

    @pytest.mark.parametrize(
        "permutation",
        [
            [0, 0, 1],
            [0, 1],
            [0, 1, 3],
            [-1, 0, 1],
        ],
    )
    def test_invalid_permutation_raises_value_error(self, permutation: list[int]) -> None:
        """Test invalid permutations raise ValueError."""
        tensor = Tensor(
            input_indices=[0, 1, 2],
            shape=(2, 3, 4),
            array=None,
        )

        with pytest.raises(
            ValueError,
            match="Permutation must be a valid rearrangement of the tensor's indices",
        ):
            TensorPermutationOperation(permutation).apply(
                tensor_operation_result_from_tensor(tensor)
            )

    def test_identity_permutation_returns_same_metadata(self) -> None:
        """Test identity permutation preserves input indices and shape."""
        tensor = Tensor(
            input_indices=[4, 5, 6],
            shape=(7, 8, 9),
            array=None,
        )

        permuted = (
            TensorPermutationOperation([0, 1, 2])
            .apply(tensor_operation_result_from_tensor(tensor))
            .tensor
        )

        assert permuted.input_indices == tensor.input_indices
        assert permuted.shape == tensor.shape
        assert permuted.array is None
