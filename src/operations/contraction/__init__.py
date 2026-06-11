"""Contraction operations."""

import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from operations.base import TensorOperation, TensorOperationResult
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


def _contract_tensors_using_tccg(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    contracted_indices = get_contracted_indices(tensor_a, tensor_b)

    ordered_final_indicies = []
    for idx in tensor_a.input_indices:
        if idx not in contracted_indices:
            ordered_final_indicies.append(idx)
    for idx in tensor_b.input_indices:
        if idx not in contracted_indices:
            ordered_final_indicies.append(idx)

    _generate_tccg_file(tensor_a, tensor_b, ordered_final_indicies)

    return NotImplemented  # TODO


def _generate_tccg_file(
    tensor_a: Tensor, tensor_b: Tensor, ordered_final_indicies: list[int]
) -> None:

    def _idx_to_char(n: int) -> str:
        result = ""
        n += 1
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(ord("A") + remainder) + result
        return result.lower()

    combined_size_dict = {}
    for idx, size in zip(tensor_a.input_indices, tensor_a.shape, strict=True):
        combined_size_dict[_idx_to_char(idx)] = size
    for idx, size in zip(tensor_b.input_indices, tensor_b.shape, strict=True):
        combined_size_dict[_idx_to_char(idx)] = size
    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "tccg_input.tccg"), "w") as f:
            f.write(
                f"C[{','.join(map(lambda x: _idx_to_char(x), ordered_final_indicies))}] = A[{','.join(map(lambda x: _idx_to_char(x), tensor_a.input_indices))}] * B[{','.join(map(lambda x: _idx_to_char(x), tensor_b.input_indices))}]\n"  # noqa: E501
            )
            for key, value in combined_size_dict.items():
                f.write(f"{key} = {value}\n")
        subprocess.run(
            [
                "tccg",
                os.path.join(temp_dir, "tccg_input.tccg"),
                "--noLoG",
                "--compiler",
                "g++",
                "--numThreads",
                str(os.cpu_count()),
                "--verbose",
            ],
            check=True,
        )


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
        if len(inputs) < 2:
            raise ValueError("At least two tensors are required for contraction.")
        if len(inputs) > 2:
            raise NotImplementedError("Contraction of more than two tensors is not implemented.")

        if use_tccg:
            return tensor_operation_result_from_tensor(
                _contract_tensors_using_tccg(inputs[0].tensor, inputs[1].tensor)
            )
        else:
            return tensor_operation_result_from_tensor(
                _contract_tensors(inputs[0].tensor, inputs[1].tensor)
            )
