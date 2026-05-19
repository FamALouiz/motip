"""Contraction operations."""

from typing import Any

import numpy as np

from operations.base import TensorOperation, TensorOperationResult
from operations.contraction.utils import get_contracted_indices, get_indices_after_contraction
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


def _contract_tensor_arrays(tensor_a: Tensor, tensor_b: Tensor) -> np.ndarray:
    assert tensor_a.array is not None and tensor_b.array is not None, (
        "Both tensors must have arrays to contract."
    )

    contracted_indices = get_contracted_indices(tensor_a, tensor_b)
    axes_a = tuple(tensor_a.input_indices.index(idx) for idx in contracted_indices)
    axes_b = tuple(tensor_b.input_indices.index(idx) for idx in contracted_indices)
    contracted_array = np.tensordot(tensor_a.array, tensor_b.array, axes=(axes_a, axes_b))

    return contracted_array


def _contract_tensors(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    new_tensor_indices = get_indices_after_contraction(tensor_a, tensor_b)

    ordered_new_indices = []
    new_tensor_shape = []
    indicies_placed_so_far = set()

    for idx, shape in zip(tensor_a.input_indices, tensor_a.shape, strict=True):
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            new_tensor_shape.append(shape)
            indicies_placed_so_far.update(ordered_new_indices)

    for idx, shape in zip(tensor_b.input_indices, tensor_b.shape, strict=True):
        if idx in new_tensor_indices and idx not in indicies_placed_so_far:
            ordered_new_indices.append(idx)
            new_tensor_shape.append(shape)
            indicies_placed_so_far.update(ordered_new_indices)

    new_array = None
    if tensor_a.array is not None and tensor_b.array is not None:
        new_array = _contract_tensor_arrays(tensor_a, tensor_b)

    return Tensor(ordered_new_indices, tuple(new_tensor_shape), new_array)


class TensorContractionOperation(TensorOperation):
    """Class to represent a tensor contraction operation."""

    def __init__(self, sliced_indices: list[int]) -> None:
        """Initialize the tensor contraction operation."""
        self.__sliced_indices = sliced_indices  # TODO: include sliced indices in the operation

    def apply(
        self,
        *inputs: TensorOperationResult,
        **kwargs: dict[str, Any],
    ) -> TensorOperationResult:
        """Apply the tensor contraction operation."""
        if len(inputs) < 2:
            raise ValueError("At least two tensors are required for contraction.")
        if len(inputs) > 2:
            raise NotImplementedError("Contraction of more than two tensors is not implemented.")
        return tensor_operation_result_from_tensor(
            _contract_tensors(inputs[0].tensor, inputs[1].tensor)
        )
