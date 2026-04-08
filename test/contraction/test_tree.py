"""Tests for persistent contraction tree construction."""

from typing import Any, cast

import pytest

from contraction.path import PersistentContractionPath
from contraction.tree import ContractionTree
from tensor_network import TensorNetwork


@pytest.fixture()
def example_network() -> TensorNetwork:
    """Create a small deterministic network for path/tree tests."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2], [2, 3]],
        size_dict={0: 2, 1: 3, 2: 4, 3: 5},
        shapes=[(2, 3), (3, 4), (4, 5)],
        output_indices=[0, 3],
        tensor_arrays=None,
    )


class TestContractionTree:
    """Test contraction-tree creation from persistent contraction paths."""

    def test_from_raw_contraction_path_builds_binary_tree(self) -> None:
        """Tree can be built directly from a complete raw contraction path."""
        tree = ContractionTree.from_contraction_path(((0, 1), (0, 1)))
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

    def test_from_raw_contraction_path_rejects_empty_path(self) -> None:
        """Raw paths cannot be empty because the initial tensor count is undefined."""
        with pytest.raises(
            ValueError,
            match="Contraction path cannot be empty when building from a raw ContractionPath",
        ):
            ContractionTree.from_contraction_path(())

    @pytest.mark.parametrize(
        "path",
        [
            ((0, 0),),
            ((-1, 0),),
            ((0, -1),),
        ],
    )
    def test_from_contraction_path_rejects_invalid_pair_shape(
        self, path: tuple[tuple[int, int], ...]
    ) -> None:
        """Validation rejects invalid pair values before type-specific tree construction."""
        with pytest.raises(
            ValueError,
            match=(
                "Contraction pair must contain two distinct tensor indices.|"
                "Contraction indices must be non-negative."
            ),
        ):
            ContractionTree.from_contraction_path(path)

    def test_from_contraction_path_rejects_non_pair_step(self) -> None:
        """Validation rejects contraction steps that are not pairs."""
        malformed_path = cast(Any, [(0, 1, 2)])

        with pytest.raises(
            ValueError,
            match="Each contraction step must be a pair of tensor indices",
        ):
            ContractionTree.from_contraction_path(malformed_path)

    def test_from_contraction_path_builds_binary_tree(self, example_network: TensorNetwork) -> None:
        """Tree root should represent the final contraction and keep child structure."""
        persistent_path = PersistentContractionPath.from_contraction_path(
            example_network, [(0, 1), (0, 1)]
        )

        tree = ContractionTree.from_contraction_path(persistent_path)
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

    def test_from_contraction_path_rejects_non_complete_contraction(
        self, example_network: TensorNetwork
    ) -> None:
        """Tree creation should fail when final state has more than one tensor."""
        persistent_path = PersistentContractionPath.from_contraction_path(example_network, [(0, 1)])

        with pytest.raises(
            ValueError,
            match="requires a complete path ending in exactly one tensor",
        ):
            ContractionTree.from_contraction_path(persistent_path)

    def test_from_contraction_path_rejects_inconsistent_history_step_size(
        self, example_network: TensorNetwork
    ) -> None:
        """Persistent paths must keep tensor counts aligned with active nodes at each step."""
        inconsistent_path = PersistentContractionPath(
            path=[(0, 1), (0, 1)],
            history=[
                example_network,
                example_network,
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
            match="Persistent path history is inconsistent with resulting tensor counts",
        ):
            ContractionTree.from_contraction_path(inconsistent_path)

    def test_from_contraction_path_rejects_invalid_pair_indices(
        self, example_network: TensorNetwork
    ) -> None:
        """Tree creation should fail when a pair index is out of range."""
        invalid_persistent_path = PersistentContractionPath(
            path=[(0, 3), (0, 1)],
            history=[
                example_network,
                example_network,
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
            ContractionTree.from_contraction_path(invalid_persistent_path)

    def test_from_raw_contraction_path_rejects_out_of_range_indices(self) -> None:
        """Raw path construction fails when a step references a missing active tensor."""
        with pytest.raises(
            ValueError,
            match="Contraction indices are out of range for current step",
        ):
            ContractionTree.from_contraction_path(((0, 1), (0, 2)))
