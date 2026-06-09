"""Permutation operation for tensors."""

from typing import Any, Sequence, TypeAlias

import hptt

from operations.base import TensorOperation, TensorOperationResult
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor

Permutation: TypeAlias = Sequence[int]


def _permute_tensor(tensor: Tensor, permutation: Permutation) -> Tensor:
    if sorted(permutation) != list(range(len(tensor.input_indices))):
        raise ValueError(
            "Permutation must be a valid rearrangement of the tensor's indices. "
            f"Expected a permutation of the integers from 0 to {len(tensor.input_indices) - 1}, "
            f"got {permutation}."
        )
    return Tensor(
        input_indices=[tensor.input_indices[i] for i in permutation],
        shape=tuple(tensor.shape[i] for i in permutation),
        array=hptt.transpose(tensor.array, axes=permutation) if tensor.array is not None else None,
    )


class TensorPermutationOperation(TensorOperation):
    """Class to represent a tensor permutation operation."""

    def __init__(self, permutation: Permutation) -> None:
        """Initialize the tensor permutation operation."""
        self.__permutation = permutation

    def apply(
        self, *inputs: TensorOperationResult, **kwargs: dict[str, Any]
    ) -> TensorOperationResult:
        """Apply the tensor permutation operation."""
        assert len(inputs) == 1, "TensorPermutationOperation expects exactly one input tensor."
        output = _permute_tensor(inputs[0].tensor, self.__permutation)
        return tensor_operation_result_from_tensor(output)
