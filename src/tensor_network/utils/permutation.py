"""Tensor permutation utilities."""

import numpy as np

from tensor import Tensor


def permute_tensor(tensor: Tensor, permutation: list[int]) -> Tensor:
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
    if sorted(permutation) != list(range(len(tensor.input_indices))):
        raise ValueError(
            "Permutation must be a valid rearrangement of the tensor's indices. "
            f"Expected a permutation of the integers from 0 to {len(tensor.input_indices) - 1}, "
            f"got {permutation}."
        )
    return Tensor(
        input_indices=[tensor.input_indices[i] for i in permutation],
        shape=tuple(tensor.shape[i] for i in permutation),
        array=np.transpose(tensor.array, axes=permutation) if tensor.array is not None else None,
    )
