"""Tests for permutation utilities."""

from permutation.utils import is_identity


class TestIsIdentity:
    """Tests for the is_identity function."""

    def test_is_identity(self) -> None:
        """Test that is_identity correctly identifies identity and non-identity permutations."""
        assert is_identity([0, 1, 2])
        assert not is_identity([1, 0, 2])
