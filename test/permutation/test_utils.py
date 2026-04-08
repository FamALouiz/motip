"""Tests for permutation utilities."""

import pytest

from permutation.strategy.common import apply_layout_to_tensor
from permutation.utils import is_identity, to_identity_permutation
from tensor import Tensor


class TestIsIdentity:
    """Tests for the is_identity function."""

    def test_is_identity(self) -> None:
        """Test that is_identity correctly identifies identity and non-identity permutations."""
        assert is_identity([0, 1, 2])
        assert not is_identity([1, 0, 2])


class TestToIdentityPermutation:
    """Test converting a tensor's input indices to the identity permutation."""

    @pytest.mark.parametrize(
        "input_indices",
        [
            [0, 1, 2],
            [2, 0, 1],
        ],
    )
    def test_to_identity_permutation(self, input_indices: list[int]) -> None:
        """Test that the input indices are returned in sorted order."""
        tensor = Tensor(input_indices=input_indices, shape=(2, 3, 4), array=None)
        result = to_identity_permutation(tensor)
        assert result == (0, 1, 2)


class TestApplyLayoutToTensor:
    """Test applying a target layout to a tensor's input indices."""

    def test_apply_layout(self) -> None:
        """Test that the correct permutation is returned for a given layout."""
        tensor = Tensor(input_indices=[0, 1, 2], shape=(2, 3, 4), array=None)
        result = apply_layout_to_tensor(tensor, [2, 0, 1])
        assert result == (2, 0, 1)

    def test_apply_layout_with_missing_index_should_raise(self) -> None:
        """Test that a ValueError is raised if the layout is missing an index from the tensor."""
        tensor = Tensor(input_indices=[0, 1, 2], shape=(2, 3, 4), array=None)
        with pytest.raises(ValueError):
            apply_layout_to_tensor(tensor, [0, 1])
