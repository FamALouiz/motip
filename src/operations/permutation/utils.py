"""Permutation utilities."""

from operations.base import TensorOperationResult
from operations.permutation import Permutation, TensorPermutationOperation
from tensor import Tensor


def is_identity(permutation: Permutation) -> bool:
    """Check if the given permutation is the identity permutation."""
    if isinstance(permutation, list):
        permutation = tuple(permutation)
    return permutation == tuple(range(len(permutation)))


def to_identity_permutation(tensor: Tensor) -> Permutation:
    """Return the identity permutation for a tensor.

    Args:
        tensor: The tensor.

    Returns:
        The identity permutation.
    """
    return tuple(range(len(tensor.input_indices)))


def to_permutation(current_indices: list[int], desired_layout: list[int]) -> Permutation:
    """Convert desired index layout to a permutation over current indices."""
    if len(desired_layout) != len(current_indices):
        raise ValueError(
            "Desired layout must contain exactly the same number of indices as current layout. "
            f"Got {len(desired_layout)} desired vs {len(current_indices)} current."
        )
    if sorted(desired_layout) != sorted(current_indices):
        raise ValueError(
            "Desired layout must be a rearrangement of current layout. "
            f"Got desired layout {desired_layout} and current layout {current_indices}."
        )
    return tuple(current_indices.index(idx) for idx in desired_layout)


def permute_tensor(tensor: Tensor, permutation: Permutation) -> Tensor:
    """Permute the indices of a tensor.

    This function will permute the indices of a tensor in a way that is consistent with the
    ordering of the indices in the tensor network. The specific permutation applied will depend
    on the ordering of the indices in the tensor network and the original ordering of the
    indices in the tensor.

    Args:
        tensor: The tensor to permute.
        permutation: The permutation to apply. This should be a list of integers representing the
        new order of the indices.

    Returns:
        The permuted tensor.

    Raises:
        ValueError: If the permutation is not a valid rearrangement of the tensor's indices.
    """
    tensor_permutation_operation = TensorPermutationOperation(permutation)
    return tensor_permutation_operation.apply(
        TensorOperationResult(tensor=tensor)
    ).tensor  # TODO: Use dummy tensor operation formation
