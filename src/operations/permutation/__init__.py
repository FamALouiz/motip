"""Permutation operation for tensors."""

from typing import Any, Sequence, TypeAlias

import hptt
import numpy as np

from operations.base import TensorOperation, TensorOperationResult
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor

Permutation: TypeAlias = Sequence[int]


def _permute_tensor(tensor: Tensor, permutation: Permutation, use_hptt: bool = False) -> Tensor:
    if sorted(permutation) != list(range(len(tensor.input_indices))):
        raise ValueError(
            "Permutation must be a valid rearrangement of the tensor's indices. "
            f"Expected a permutation of the integers from 0 to {len(tensor.input_indices) - 1}, "
            f"got {permutation}."
        )
    permuted_array = None
    if tensor.array is not None:
        if use_hptt:
            permuted_array = hptt.transpose(tensor.array, axes=permutation)
        else:
            permuted_array = np.transpose(tensor.array, axes=permutation)
    return Tensor(
        input_indices=[tensor.input_indices[i] for i in permutation],
        shape=tuple(tensor.shape[i] for i in permutation),
        array=permuted_array,
    )


class TensorPermutationOperation(TensorOperation):
    """Class to represent a tensor permutation operation."""

    def __init__(self, permutation: Permutation) -> None:
        """Initialize the tensor permutation operation."""
        self.__permutation = permutation

    @property
    def permutation(self) -> Permutation:
        """Return the permutation applied by this operation."""
        return self.__permutation

    def apply(
        self, *inputs: TensorOperationResult, **kwargs: dict[str, Any]
    ) -> TensorOperationResult:
        """Apply the tensor permutation operation."""
        use_hptt = kwargs.get("use_hptt", False)
        assert isinstance(use_hptt, bool)
        assert len(inputs) == 1, "TensorPermutationOperation expects exactly one input tensor."
        output = _permute_tensor(inputs[0].tensor, self.__permutation, use_hptt=use_hptt)
        return tensor_operation_result_from_tensor(output)

    def __str__(self) -> str:
        """Return a string representation of the tensor permutation operation."""
        return f"TensorPermutationOperation(permutation={self.permutation})"

    def __repr__(self) -> str:
        """Return a string representation of the tensor permutation operation."""
        return self.__str__()
