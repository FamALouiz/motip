"""Tests for persistent contraction tree construction."""

import pytest

from contraction.path import PersistentContractionPath
from contraction.tree import PersistentContractionTree
from tensor_network import TensorNetwork


class TestPersistentContractionTree:
    """Test contraction-tree creation from persistent contraction paths."""

    @staticmethod
    def _example_network() -> TensorNetwork:
        """Create a small deterministic network for path/tree tests."""
        return TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )

    def test_from_contraction_path_builds_binary_tree(self) -> None:
        """Tree root should represent the final contraction and keep child structure."""
        persistent_path = PersistentContractionPath.from_contraction_path(
            self._example_network(), [(0, 1), (0, 1)]
        )

        tree = PersistentContractionTree.from_contraction_path(persistent_path)
        left_branch = tree.final_output.left
        right_leaf = tree.final_output.right

        assert tree.num_leaves == 3
        assert tree.num_steps == 2
        assert tree.path == ((0, 1), (0, 1))
        assert tree.final_output.contraction_step == 1

        assert left_branch is not None
        assert right_leaf is not None
        assert left_branch.is_leaf is False
        assert left_branch.contraction_step == 0
        assert right_leaf.is_leaf is True
        assert right_leaf.initial_tensor_position == 2

        assert left_branch.left is not None
        assert left_branch.right is not None
        assert left_branch.left.initial_tensor_position == 0
        assert left_branch.right.initial_tensor_position == 1

        assert left_branch.parent is tree.final_output
        assert right_leaf.parent is tree.final_output
        assert right_leaf.parent is tree.final_output

    def test_from_contraction_path_rejects_non_complete_contraction(self) -> None:
        """Tree creation should fail when final state has more than one tensor."""
        persistent_path = PersistentContractionPath.from_contraction_path(
            self._example_network(), [(0, 1)]
        )

        with pytest.raises(
            ValueError,
            match="requires a complete path ending in exactly one tensor",
        ):
            PersistentContractionTree.from_contraction_path(persistent_path)

    def test_from_contraction_path_rejects_invalid_pair_indices(self) -> None:
        """Tree creation should fail when a pair index is out of range."""
        invalid_persistent_path = PersistentContractionPath(
            path=[(0, 3), (0, 1)],
            history=[
                self._example_network(),
                self._example_network(),
                TensorNetwork(
                    input_indices=[[0, 3]],
                    size_dict={0: 2, 1: 3, 2: 4, 3: 5},
                    shapes=[(2, 5)],
                    output_indices=[0, 3],
                    tensor_arrays=None,
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match="Contraction indices are out of range for current step",
        ):
            PersistentContractionTree.from_contraction_path(invalid_persistent_path)
