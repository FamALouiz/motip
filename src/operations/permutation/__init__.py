"""Permutation operation for tensors."""

from typing import Any, Sequence, TypeAlias

import numpy as np

from operations.base import TensorOperation, TensorOperationResult
from tensor import Tensor

Permutation: TypeAlias = Sequence[int]


def _permute_tensor(tensor: Tensor, permutation: Permutation) -> Tensor:
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


class TensorPermutationOperation(TensorOperation):
    """Class to represent a tensor permutation operation."""

    def __init__(self, permutation: Permutation) -> None:
        """Initialize the tensor permutation operation."""
        self.__permutation = permutation

    def apply(
        self, input: TensorOperationResult, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> TensorOperationResult:
        """Apply the tensor permutation operation."""
        output = _permute_tensor(input.tensor, self.__permutation)
        return TensorOperationResult(tensor=output)
