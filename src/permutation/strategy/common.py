"""Shared helpers for permutation strategy baselines."""

from __future__ import annotations

from typing import Collection

from contraction.path import PersistentContractionPath
from contraction.tensor import get_contracted_indices
from contraction.tree import ContractionTree, ContractionTreeNode
from permutation import Permutation
from permutation.utils import to_permutation
from tensor import Tensor


def sort_indices_by_size(indices: Collection[int], size_dict: dict[int, int]) -> list[int]:
    """Sort tensor indices by dimension size.

    Args:
        indices: The indices to sort.
        size_dict: A mapping from index id to dimension size.

    Returns:
        The indices sorted in ascending order of dimension size.

    Raises:
        ValueError: If size_dict does not contain sizes for all indices.
    """
    for idx in indices:
        if idx not in size_dict:
            raise ValueError(f"Size dict is missing size for index {idx}")
    return sorted(indices, key=lambda idx: size_dict[idx])


def get_step_tensors(
    persistent_path: PersistentContractionPath, step: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the tensors involved in a contraction step.

    Args:
        persistent_path: The persistent contraction path.
        step: The contraction step index.

    Returns:
        A tuple containing the left input tensor, right input tensor, and result tensor.
    """
    left_pos, right_pos = persistent_path.path[step]
    before = persistent_path.get_state(step)
    after = persistent_path.get_state(step + 1)
    return before.tensors[left_pos], before.tensors[right_pos], after.tensors[left_pos]


def build_tree_maps(
    persistent_path: PersistentContractionPath,
) -> tuple[ContractionTree, dict[int, ContractionTreeNode], dict[int, ContractionTreeNode]]:
    """Build lookup maps for contraction tree nodes.

    Args:
        persistent_path: The persistent contraction path.

    Returns:
        A tuple containing the contraction tree, a map from initial tensor positions
        to leaf nodes, and a map from contraction step indices to internal nodes.
    """
    tree = ContractionTree.from_contraction_path(persistent_path)
    leaf_to_node: dict[int, ContractionTreeNode] = {}
    step_to_node: dict[int, ContractionTreeNode] = {}

    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node.initial_tensor_position is not None:
            leaf_to_node[node.initial_tensor_position] = node
        if node.contraction_step is not None:
            step_to_node[node.contraction_step] = node
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    return tree, leaf_to_node, step_to_node


def get_input_layout_for_parent_use(
    node: ContractionTreeNode,
    persistent_path: PersistentContractionPath,
    size_dict: dict[int, int],
) -> list[int] | None:
    """Return the preferred layout of a tensor when used in its parent contraction.

    Args:
        node: The leaf or internal node representing the tensor.
        persistent_path: The persistent contraction path.
        size_dict: A mapping from index id to dimension size.

    Returns:
        The preferred input layout for the tensor at its parent contraction, or None
        if the node has no parent.
    """
    if node.parent is None or node.parent.contraction_step is None:
        return None

    contraction_step = node.parent.contraction_step
    left_tensor, right_tensor, _ = get_step_tensors(persistent_path, contraction_step)

    is_left = node.parent.left is node
    tensor_at_step = left_tensor if is_left else right_tensor
    sibling_tensor = right_tensor if is_left else left_tensor

    contracted = get_contracted_indices(tensor_at_step, sibling_tensor)
    free = set(tensor_at_step.input_indices) - contracted

    contracted_sorted = sort_indices_by_size(contracted, size_dict)
    free_sorted = sort_indices_by_size(free, size_dict)

    if is_left:
        return free_sorted + contracted_sorted
    return contracted_sorted + free_sorted


def get_result_layout_from_current_step(
    step: int,
    persistent_path: PersistentContractionPath,
    size_dict: dict[int, int],
    left_first: bool = True,
) -> list[int]:
    """Build a canonical result layout from the current contraction step.

    Args:
        step: The contraction step index.
        persistent_path: The persistent contraction path.
        size_dict: A mapping from index id to dimension size.
        left_first: Whether to place the free indices of the left tensor before
            the free indices of the right tensor.

    Returns:
        The canonical result layout for the contraction result tensor.
    """
    left_tensor, right_tensor, result_tensor = get_step_tensors(persistent_path, step)

    contracted = get_contracted_indices(left_tensor, right_tensor)
    left_free = set(left_tensor.input_indices) - contracted
    right_free = set(right_tensor.input_indices) - contracted

    left_free_sorted = sort_indices_by_size(left_free, size_dict)
    right_free_sorted = sort_indices_by_size(right_free, size_dict)

    result_layout = (
        left_free_sorted + right_free_sorted if left_first else right_free_sorted + left_free_sorted
    )
    return [idx for idx in result_layout if idx in result_tensor.input_indices]


def apply_layout_to_tensor(tensor: Tensor, layout: list[int]) -> Permutation:
    """Convert a target layout into a permutation.

    Args:
        tensor: The tensor whose permutation is needed.
        layout: The target index layout.

    Returns:
        The permutation mapping the tensor's current layout to the target layout.
    """
    return to_permutation(tensor.input_indices, layout)
