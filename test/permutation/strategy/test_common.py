"""Tests for common functions used in permutation strategies."""

from typing import Collection

import pytest

from contraction.path import PersistentContractionPath
from contraction.tree import ContractionTree
from permutation.strategy.common import (
    build_tree_maps,
    get_input_layout_for_parent_use,
    get_result_layout_from_current_step,
    get_step_tensors,
    sort_indices_by_size,
)
from tensor_network.tn import TensorNetwork


class TestSortIndicesBySize:
    """Test sorting indices by size."""

    @pytest.mark.parametrize(
        ("size_dict", "indices"),
        [
            ({0: 10, 1: 2, 2: 5, 3: 10, 4: 2}, {0, 1, 2}),
            ({0: 10, 1: 2, 2: 5}, [0, 1, 2]),
        ],
    )
    def test_sort_indices(self, size_dict: dict[int, int], indices: Collection[int]) -> None:
        """Test that indices are sorted in ascending order of size."""
        result = sort_indices_by_size(indices, size_dict)
        assert result == [1, 2, 0]

    def test_sort_indices_with_mismatching_dict_should_raise(self) -> None:
        """Test that a ValueError is raised if size_dict does not contain sizes for all indices."""
        indices = {0, 1, 2, 3}
        size_dict = {0: 10, 1: 2, 2: 5}
        with pytest.raises(ValueError):
            sort_indices_by_size(indices, size_dict)


class TestGetStepTensors:
    """Test retrieving tensors involved in a contraction step."""

    def test_get_step_tensors(self) -> None:
        """Test that the correct tensors are returned for a given contraction step."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        left, right, result = get_step_tensors(persistent_path, 0)
        assert left.input_indices == [0, 1]
        assert right.input_indices == [1, 2]
        assert result.input_indices == [0, 2]

    def test_get_step_tensors_invalid_step_should_raise(self) -> None:
        """Test that an IndexError is raised if the step index is out of bounds."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        with pytest.raises(IndexError):
            get_step_tensors(persistent_path, 1)


class TestBuildTreeMaps:
    """Test building lookup maps for contraction tree nodes."""

    def test_build_tree_maps(self) -> None:
        """Test that the contraction tree and lookup maps are built correctly."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        tree, leaf_to_node, step_to_node = build_tree_maps(persistent_path)
        assert isinstance(tree, ContractionTree)
        assert len(leaf_to_node) == 2
        assert len(step_to_node) == 1
        assert 0 in leaf_to_node
        assert 1 in leaf_to_node
        assert 0 in step_to_node

    def test_build_tree_maps_with_empty_path(self) -> None:
        """Test that the function can handle an empty contraction path."""
        network = TensorNetwork(
            input_indices=[[0, 1]],
            size_dict={0: 2, 1: 3},
            output_indices=[0, 1],
            shapes=[(2, 3)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [])
        tree, leaf_to_node, step_to_node = build_tree_maps(persistent_path)
        assert isinstance(tree, ContractionTree)
        assert len(leaf_to_node) == 1
        assert len(step_to_node) == 0
        assert 0 in leaf_to_node


class TestGetInputLayoutForParentUse:
    """Test retrieving the preferred layout of a tensor when used in its parent contraction."""

    def test_get_input_layout_no_parent(self) -> None:
        """Test that None is returned for a tensor that is not used in any parent contraction."""
        network = TensorNetwork(
            input_indices=[[0, 1]],
            size_dict={0: 2, 1: 3},
            output_indices=[0, 1],
            shapes=[(2, 3)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [])
        tree = ContractionTree.from_contraction_path(persistent_path)
        result = get_input_layout_for_parent_use(tree.root, persistent_path, {0: 2, 1: 3})
        assert result is None

    def test_get_input_layout_with_parent(self) -> None:
        """Test that the correct layout is returned for a tensor used in a parent contraction."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        _, leaf_to_node, _ = build_tree_maps(persistent_path)
        size_dict = {0: 2, 1: 3, 2: 4}
        result_left = get_input_layout_for_parent_use(leaf_to_node[0], persistent_path, size_dict)
        assert result_left == [0, 1]
        result_right = get_input_layout_for_parent_use(leaf_to_node[1], persistent_path, size_dict)
        assert result_right == [1, 2]

    def test_get_input_layout_with_parent_and_missing_size_dict_entry_should_raise(self) -> None:
        """Test that a ValueError is raised if size_dict is missing an entry for an index."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            output_indices=[0, 2],
            shapes=[(2, 3), (3, 4)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        _, leaf_to_node, _ = build_tree_maps(persistent_path)
        size_dict = {0: 2, 1: 3}  # Missing size for index 2
        with pytest.raises(ValueError):
            get_input_layout_for_parent_use(leaf_to_node[1], persistent_path, size_dict)


class TestGetResultLayoutFromCurrentStep:
    """Test getting the preferred layout when used in its parent contraction."""

    @pytest.mark.parametrize("left_first", [True, False], ids=["left_first", "right_first"])
    def test_get_result_layout(self, left_first: bool) -> None:
        """Test that the correct layout is returned."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 4, 1: 3, 2: 2},
            output_indices=[0, 2],
            shapes=[(4, 3), (3, 2)],
            tensor_arrays=None,
        )
        persistent_path = PersistentContractionPath.from_contraction_path(network, [(0, 1)])
        result = get_result_layout_from_current_step(
            0, persistent_path, {0: 4, 1: 3, 2: 2}, left_first=left_first
        )
        assert result == [0, 2] if left_first else [2, 0]
