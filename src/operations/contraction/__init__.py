"""Contraction operations."""

import traceback
from typing import Any

import numpy as np

from operations.base import TensorOperation, TensorOperationResult
from operations.contraction.tccg_interface import execute_tccg_contraction
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


def get_contracted_indices(tensor_a: Tensor, tensor_b: Tensor) -> set[int]:
    """Get the set of contracted indices between two tensors."""
    contracted_indices = set(tensor_a.input_indices) & set(tensor_b.input_indices)

    return contracted_indices


def get_indices_after_contraction(tensor_a: Tensor, tensor_b: Tensor) -> set[int]:
    """Get the set of indices that will be present in the new tensor after contraction."""
    contracted_indices = get_contracted_indices(tensor_a, tensor_b)
    new_tensor_indices = (
        set(tensor_a.input_indices) | set(tensor_b.input_indices)
    ) - contracted_indices

    return new_tensor_indices


def _contract_tensor_arrays(tensor_a: Tensor, tensor_b: Tensor) -> np.ndarray:
    assert tensor_a.array is not None and tensor_b.array is not None, (
        "Both tensors must have arrays to contract."
    )

    contracted_indices = get_contracted_indices(tensor_a, tensor_b)
    contracted_indices_list = [idx for idx in tensor_a.input_indices if idx in contracted_indices]

    a_contract_axes = [tensor_a.input_indices.index(idx) for idx in contracted_indices_list]
    b_contract_axes = [tensor_b.input_indices.index(idx) for idx in contracted_indices_list]

    a_free_axes = [
        i for i, idx in enumerate(tensor_a.input_indices) if idx not in contracted_indices
    ]
    b_free_axes = [
        i for i, idx in enumerate(tensor_b.input_indices) if idx not in contracted_indices
    ]

    a_transposed = np.transpose(tensor_a.array, axes=(*a_free_axes, *a_contract_axes))
    b_transposed = np.transpose(tensor_b.array, axes=(*b_contract_axes, *b_free_axes))

    a_shape_free = tuple(tensor_a.shape[i] for i in a_free_axes)
    a_shape_contract = tuple(tensor_a.shape[i] for i in a_contract_axes)
    b_shape_free = tuple(tensor_b.shape[i] for i in b_free_axes)

    a_matrix = a_transposed.reshape((-1, int(np.prod(a_shape_contract, dtype=int))))
    b_matrix = b_transposed.reshape((int(np.prod(a_shape_contract, dtype=int)), -1))

    contracted_matrix = a_matrix @ b_matrix
    result_shape = (*a_shape_free, *b_shape_free)

    return contracted_matrix.reshape(result_shape)


def _contract_tensors(tensor_a: Tensor, tensor_b: Tensor, use_tccg: bool = False) -> Tensor:
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
        if use_tccg:
            try:
                new_array = execute_tccg_contraction(
                    tensor_a,
                    tensor_b,
                    ordered_new_indices,
                    tuple(new_tensor_shape),
                )
            except Exception as _:
                traceback.print_exc()
                print("Error occurred with TCCG compilation... falling back to normal contraction")
                new_array = _contract_tensor_arrays(tensor_a, tensor_b)
        else:
            new_array = _contract_tensor_arrays(tensor_a, tensor_b)

    return Tensor(ordered_new_indices, tuple(new_tensor_shape), new_array)


class TensorContractionOperation(TensorOperation):
    """Class to represent a tensor contraction operation."""

    def __init__(self, sliced_indices: list[int]) -> None:
        """Initialize the tensor contraction operation."""
        self.__sliced_indices = sliced_indices  # TODO: include sliced indices in the operation

    @property
    def sliced_indices(self) -> list[int]:
        """Return the sliced indices used by this operation."""
        return self.__sliced_indices

    def apply(
        self,
        *inputs: TensorOperationResult,
        **kwargs: dict[str, Any],
    ) -> TensorOperationResult:
        """Apply the tensor contraction operation."""
        use_tccg = kwargs.get("use_tccg", False)
        assert isinstance(use_tccg, bool)
        if len(inputs) < 2:
            raise ValueError("At least two tensors are required for contraction.")
        if len(inputs) > 2:
            raise NotImplementedError("Contraction of more than two tensors is not implemented.")

        return tensor_operation_result_from_tensor(
            _contract_tensors(inputs[0].tensor, inputs[1].tensor, use_tccg=use_tccg)
        )

    def __str__(self) -> str:
        """Return a string representation of the tensor contraction operation."""
        return f"TensorContractionOperation(sliced_indices={self.sliced_indices})"

    def __repr__(self) -> str:
        """Return a string representation of the tensor contraction operation."""
        return self.__str__()
