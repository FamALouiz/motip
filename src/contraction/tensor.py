"""Tensor network contraction utilities."""

import numpy as np

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


def contract_tensors(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    """Contract two tensors together.

    Args:
        tensor_a: The first tensor to contract.
        tensor_b: The second tensor to contract.

    Returns:
        The resulting tensor after contraction.
    """
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
