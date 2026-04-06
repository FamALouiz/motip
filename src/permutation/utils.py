"""Permutation utilities."""

from permutation import Permutation


def is_identity(permutation: Permutation) -> bool:
    """Check if the given permutation is the identity permutation."""
    if isinstance(permutation, list):
        permutation = tuple(permutation)
    return permutation == tuple(range(len(permutation)))
