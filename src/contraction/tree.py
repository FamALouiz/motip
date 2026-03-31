"""Contraction tree data structure and related utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from contraction.path import ContractionPath, PersistentContractionPath


@dataclass(slots=True)
class ContractionTreeNode:
    """Immutable node in a binary contraction tree.

    Leaf nodes represent original tensors by their position in the initial network.
    Internal nodes represent contraction operations between two children.
    """

    left: ContractionTreeNode | None = None
    right: ContractionTreeNode | None = None
    parent: ContractionTreeNode | None = None
    initial_tensor_position: int | None = None
    contraction_step: int | None = None

    def __post_init__(self) -> None:
        """Validate that the node is either a leaf or an internal contraction."""
        is_leaf = self.left is None and self.right is None

        if is_leaf:
            if self.initial_tensor_position is None:
                raise ValueError("Leaf nodes must define an initial tensor position.")
            if self.initial_tensor_position < 0:
                raise ValueError("Initial tensor position must be non-negative.")
            if self.contraction_step is not None:
                raise ValueError("Leaf nodes cannot define a contraction step.")
            return

        if self.left is None or self.right is None:
            raise ValueError("Internal nodes must have exactly two children.")
        if self.initial_tensor_position is not None:
            raise ValueError("Internal nodes cannot define an initial tensor position.")
        if self.contraction_step is None:
            raise ValueError("Internal nodes must define a contraction step.")
        if self.contraction_step < 0:
            raise ValueError("Contraction step must be non-negative.")

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (original tensor)."""
        return self.left is None and self.right is None


@dataclass
class ContractionTree:
    """Binary tree that mirrors contraction operations in a path."""

    root: ContractionTreeNode
    path: ContractionPath
    num_leaves: int

    @property
    def num_steps(self) -> int:
        """Number of contraction operations represented by this tree."""
        return len(self.path)

    @property
    def final_output(self) -> ContractionTreeNode:
        """The final contraction output node (root)."""
        return self.root

    @overload
    @staticmethod
    def from_contraction_path(
        path: ContractionPath,
    ) -> "ContractionTree": ...
    @overload
    @staticmethod
    def from_contraction_path(
        path: PersistentContractionPath,
    ) -> "ContractionTree": ...
    @staticmethod
    def from_contraction_path(
        path: PersistentContractionPath | ContractionPath,
    ) -> "ContractionTree":
        """Build a binary contraction tree from a persistent path.

        The tree follows the same index semantics as `contract_tensors_in_network`:
        at each step `(i, j)`, tensors at positions `i` and `j` are contracted and
        the resulting tensor is inserted back at `i`.
        """
        if not isinstance(path, PersistentContractionPath):
            raise NotImplementedError(
                "from_contraction_path with a raw ContractionPath is not implemented. "
                "Please create a PersistentContractionPath first using "
                "PersistentContractionPath.from_contraction_path(network, path)."
            )
        if not path.history:
            raise ValueError("Persistent contraction path history cannot be empty.")

        initial_tensor_count = len(path.history[0])
        if initial_tensor_count == 0:
            raise ValueError("Cannot build a contraction tree from an empty tensor network.")

        if len(path.history[-1]) != 1:
            raise ValueError(
                "Contraction tree requires a complete path ending in exactly one tensor."
            )

        active_nodes = [
            ContractionTreeNode(initial_tensor_position=index)
            for index in range(initial_tensor_count)
        ]

        for step, pair in enumerate(path.path):
            if len(active_nodes) != len(path.history[step]):
                raise ValueError(
                    "Persistent path history is inconsistent with path contraction steps."
                )

            left_index, right_index = pair
            if left_index == right_index:
                raise ValueError("Contraction pair must contain two distinct tensor indices.")
            if left_index < 0 or right_index < 0:
                raise ValueError("Contraction indices must be non-negative.")
            if left_index >= len(active_nodes) or right_index >= len(active_nodes):
                raise ValueError("Contraction indices are out of range for current step.")

            parent = ContractionTreeNode(
                left=active_nodes[left_index],
                right=active_nodes[right_index],
                contraction_step=step,
            )

            active_nodes[left_index].parent = parent
            active_nodes[right_index].parent = parent

            active_nodes = [node for index, node in enumerate(active_nodes) if index not in pair]
            active_nodes.insert(left_index, parent)

            if len(active_nodes) != len(path.history[step + 1]):
                raise ValueError(
                    "Persistent path history is inconsistent with resulting tensor counts."
                )

        if len(active_nodes) != 1:
            raise ValueError("Contraction path did not collapse to a single output tensor.")

        return ContractionTree(
            root=active_nodes[0],
            path=tuple(path.path),
            num_leaves=initial_tensor_count,
        )
