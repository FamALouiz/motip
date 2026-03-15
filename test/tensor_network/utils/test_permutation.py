"""Tests for the tensor permutation utilities."""

import numpy as np
import pytest

from tensor import Tensor
from tensor_network.utils.permutation import permute_tensor


class TestPermuteTensor:
    """Tests for permute_tensor behavior."""

    def test_permute_tensor_reorders_indices_and_shape(self) -> None:
        """Test that a valid permutation reorders input indices and shape."""
        tensor = Tensor(
            input_indices=[10, 11, 12],
            shape=(2, 3, 4),
            array=np.arange(24).reshape(2, 3, 4),
        )

        permuted = permute_tensor(tensor, [2, 0, 1])

        assert permuted.input_indices == [12, 10, 11]
        assert permuted.shape == (4, 2, 3)
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
            permute_tensor(tensor, permutation)

    def test_identity_permutation_returns_same_metadata(self) -> None:
        """Test identity permutation preserves input indices and shape."""
        tensor = Tensor(
            input_indices=[4, 5, 6],
            shape=(7, 8, 9),
            array=None,
        )

        permuted = permute_tensor(tensor, [0, 1, 2])

        assert permuted.input_indices == tensor.input_indices
        assert permuted.shape == tensor.shape
        assert permuted.array is None
