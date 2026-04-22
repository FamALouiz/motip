"""Test the TensorNetwork class."""

import numpy as np
import pytest
from more_itertools import one

from tensor import Tensor
from tensor_network import TensorNetwork
from tensor_network.tn import _TensorPool


class TestTensorNetwork:
    """Test the TensorNetwork class with various parameter configurations."""

    def test_tensor_network_as_tuple(self) -> None:
        """Test that the as_tuple method returns the correct shape."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4), (4, 5)],
            tensor_arrays=None,
        )

        expected_tuple = (
            [[0, 1], [1, 2]],
            [0],
            [(3, 4), (4, 5)],
            {0: 3, 1: 4, 2: 5},
            None,
        )
        assert network.as_tuple == expected_tuple

    @pytest.mark.parametrize(
        "other",
        [
            TensorNetwork(
                input_indices=[[0, 1], [1, 2]],
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
                shapes=[(3, 4), (4, 5)],
                tensor_arrays=[np.ones((3, 4)), np.ones((4, 5))],
            ),
            (
                [[0, 1], [1, 2]],
                [0],
                [(3, 4), (4, 5)],
                {0: 3, 1: 4, 2: 5},
                [np.ones((3, 4)), np.ones((4, 5))],
            ),
        ],
    )
    def test_tensor_network_equality_succeeds(self, other: object) -> None:
        """Test the equality comparison of TensorNetwork."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4), (4, 5)],
            tensor_arrays=[np.ones((3, 4)), np.ones((4, 5))],
        )

        assert network == other

    @pytest.mark.parametrize(
        "other",
        [
            TensorNetwork(
                input_indices=[[0], [1, 2]],
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
                shapes=[(3,), (4, 5)],
                tensor_arrays=[np.ones((3,)), np.ones((4, 5))],
            ),
            (
                [[0], [1, 2]],
                [0],
                [(3,), (4, 5)],
                {0: 3, 1: 4, 2: 5},
                None,
            ),
            "not a tensor network",
        ],
    )
    def test_tensor_network_equality_fails(self, other: object) -> None:
        """Test that the equality comparison fails for inequivalent and non-TN objects."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4), (4, 5)],
            tensor_arrays=None,
        )

        assert network != other

    def test_tensor_network_arrays_property(self) -> None:
        """Test that the arrays property raises an error when arrays are not generated."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4), (4, 5)],
            tensor_arrays=None,
        )

        with pytest.raises(
            ValueError,
            match="Arrays were not generated for this tensor network. Only metadata is available.",
        ):
            _ = network.arrays

    def test_tensor_network_properties(self) -> None:
        """Test the properties of the TensorNetwork class."""
        random_tensor_1 = np.random.rand(3, 4)
        random_tensor_2 = np.random.rand(4, 5)
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4), (4, 5)],
            tensor_arrays=[random_tensor_1, random_tensor_2],
        )

        assert len(network) == 2
        assert network.input_indices == [[0, 1], [1, 2]]
        assert network.output_indices == [0]
        assert network.shapes == [(3, 4), (4, 5)]
        assert network.size_dict == {0: 3, 1: 4, 2: 5}
        assert network.arrays == [random_tensor_1, random_tensor_2]

    def test_mismatching_information_should_raise_error(self) -> None:
        """Test that mismatching information raises an error."""
        with pytest.raises(ValueError):
            TensorNetwork(
                input_indices=[[0, 1], [1, 2]],
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
                shapes=[(3, 4)],
                tensor_arrays=[np.random.rand(3, 4), np.random.rand(4, 5)],
            )

    def test_initialization_from_tensors(self) -> None:
        """Test that TensorNetwork can be initialized directly from tensor objects."""
        tensors = [
            Tensor([0, 1], (3, 4), np.ones((3, 4))),
            Tensor([1, 2], (4, 5), np.ones((4, 5))),
        ]

        network = TensorNetwork(
            tensors=tensors,
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
        )

        assert network.tensors == tensors
        assert network.input_indices == [[0, 1], [1, 2]]
        assert network.shapes == [(3, 4), (4, 5)]

    def test_initialization_without_required_tensor_data_should_raise_error(self) -> None:
        """Test that initialization fails without tensors and without raw tensor components."""
        with pytest.raises(
            ValueError,
            match="Either tensors or both input_indices and shapes must be provided.",
        ):
            TensorNetwork(
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
            )

    def test_as_graph_creates_correct_nodes(self) -> None:
        """Test that as_graph creates a node for each tensor."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
        )

        graph = network.as_graph

        assert set(graph.nodes()) == {0, 1, 2}
        assert len(graph.nodes()) == 3

    def test_as_graph_creates_edges_for_shared_indices(self) -> None:
        """Test that as_graph creates edges between tensors that share indices."""
        # Tensor 0: indices [0, 1]
        # Tensor 1: indices [1, 2] - shares index 1 with tensor 0
        # Tensor 2: indices [2, 3] - shares index 2 with tensor 1
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
        )

        graph = network.as_graph

        assert graph.has_edge(0, 1)
        assert graph.has_edge(1, 2)
        assert not graph.has_edge(0, 2)

    def test_as_graph_edge_has_correct_index_attribute(self) -> None:
        """Test that edge attributes contain the correct shared index."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3), (3, 4)],
        )

        graph = network.as_graph

        assert one(graph[0][1]["index"]) == 1

    def test_as_graph_multiple_shared_indices(self) -> None:
        """Test that multiple shared indices between two tensors create multiple edges."""
        # Tensor 0: indices [0, 1, 2]
        # Tensor 1: indices [1, 2, 3] - shares indices 1 and 2 with tensor 0
        network = TensorNetwork(
            input_indices=[[0, 1, 2], [1, 2, 3]],
            output_indices=[0, 3],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3, 4), (3, 4, 5)],
        )

        graph = network.as_graph

        # The graph itself has only 1 edge, but this is a dense edge with multiple indices
        edges_between_0_1 = graph.edges(data=True)
        assert one(edges_between_0_1)[2]["index"] == [1, 2]

    def test_as_graph_disconnected_tensors_have_no_edges(self) -> None:
        """Test that tensors with no shared indices are not connected."""
        # Tensor 0: indices [0, 1]
        # Tensor 1: indices [2, 3] - no shared indices with tensor 0
        network = TensorNetwork(
            input_indices=[[0, 1], [2, 3]],
            output_indices=[0, 3],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (4, 5)],
        )

        graph = network.as_graph

        assert set(graph.nodes()) == {0, 1}
        assert not graph.has_edge(0, 1)

    def test_as_graph_single_tensor(self) -> None:
        """Test as_graph with a single tensor (no possible connections)."""
        network = TensorNetwork(
            input_indices=[[0, 1]],
            output_indices=[0],
            size_dict={0: 2, 1: 3},
            shapes=[(2, 3)],
        )

        graph = network.as_graph

        assert len(graph.nodes()) == 1
        assert 0 in graph.nodes()
        assert len(graph.edges()) == 0

    def test_as_graph_empty_network(self) -> None:
        """Test as_graph with an empty tensor network."""
        network = TensorNetwork(
            input_indices=[],
            output_indices=[],
            size_dict={},
            shapes=[],
        )

        graph = network.as_graph

        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0


class TestTensorPool:
    """Test the internal tensor pool behavior used by TensorNetwork."""

    @staticmethod
    def _make_tensor(indices: list[int], shape: tuple[int, ...]) -> Tensor:
        """Create a tensor with deterministic values for stable assertions."""
        return Tensor(indices, shape, np.ones(shape))

    def test_initialization_with_tensors_releases_to_pool(self) -> None:
        """Test that tensors provided at init are available in the pool."""
        t1 = self._make_tensor([0, 1], (2, 3))
        t2 = self._make_tensor([1, 2], (3, 4))

        pool = _TensorPool([t1, t2])

        assert len(pool) == 2
        assert pool.pool == [t1, t2]

    def test_get_tensor_by_shape_succeeds_and_removes_from_pool(self) -> None:
        """Test getting a tensor by shape and removing it from the pool."""
        t1 = self._make_tensor([0], (2,))
        t2 = self._make_tensor([1, 2], (3, 4))
        pool = _TensorPool([t1, t2])

        got = pool.get_tensor_by_shape((3, 4))

        assert got == t2
        assert len(pool) == 1
        assert pool[0] == t1

    def test_get_tensor_by_shape_raises_when_missing(self) -> None:
        """Test that getting a non-existing shape raises ValueError."""
        pool = _TensorPool([self._make_tensor([0], (2,))])

        with pytest.raises(ValueError, match="No tensor of shape"):
            pool.get_tensor_by_shape((9, 9))

    def test_pop_succeeds_and_raises_on_empty_pool(self) -> None:
        """Test pop behavior for non-empty and empty pools."""
        tensor = self._make_tensor([0], (2,))
        pool = _TensorPool([tensor])

        popped = pool.pop(0)
        assert popped == tensor
        assert len(pool) == 0

        with pytest.raises(ValueError, match="No tensors available in the pool to pop"):
            pool.pop(0)

    def test_release_insert_and_iteration(self) -> None:
        """Test release/insert plus __iter__ and __getitem__."""
        t1 = self._make_tensor([0], (2,))
        t2 = self._make_tensor([1], (3,))
        pool = _TensorPool(None)

        pool.release(t1)
        pool.insert(0, t2)

        assert len(pool) == 2
        assert pool[0] == t2
        assert pool[1] == t1
        assert list(iter(pool)) == [t2, t1]

    def test_setitem_updates_tensor_contents_in_place(self) -> None:
        """Test __setitem__ updates tensor fields instead of replacing object identity."""
        original = Tensor([0], (2,), np.array([1.0, 1.0]))
        replacement = Tensor([1, 2], (3, 4), np.ones((3, 4)))
        pool = _TensorPool([original])
        original_id = id(pool[0])

        pool[0] = replacement

        assert id(pool[0]) == original_id
        assert pool[0].input_indices == [1, 2]
        assert pool[0].shape == (3, 4)
        assert (pool[0].array == np.ones((3, 4))).all()  # type: ignore

    def test_tensor_pool_equality_with_pool_and_list(self) -> None:
        """Test equality behavior for pool-to-pool and pool-to-list comparisons."""
        t1 = self._make_tensor([0, 1], (2, 3))
        t2 = self._make_tensor([1, 2], (3, 4))

        pool_a = _TensorPool([t1, t2])
        pool_b = _TensorPool([self._make_tensor([0, 1], (2, 3)), self._make_tensor([1, 2], (3, 4))])

        assert pool_a == pool_b
        assert pool_a == [t1, t2]

    def test_tensor_pool_equality_with_invalid_type_returns_not_equal(self) -> None:
        """Test equality against unrelated type."""
        pool = _TensorPool([self._make_tensor([0], (2,))])

        assert pool != "not a pool"
