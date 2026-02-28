"""Test the TensorNetwork class."""

import pytest
from tensor_network import TensorNetwork

import numpy as np


class TestTensorNetwork:
    """Test the TensorNetwork class with various parameter configurations."""

    def test_tensor_network_as_tuple(self):
        """Test that the as_tuple method returns the correct shape."""
        network = TensorNetwork(
            _input_indices=[[0, 1], [1, 2]],
            _output_indices=[0],
            _size_dict={0: 3, 1: 4, 2: 5},
            _shapes=[(3, 4), (4, 5)],
            _arrays=None,
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
                _input_indices=[[0, 1], [1, 2]],
                _output_indices=[0],
                _size_dict={0: 3, 1: 4, 2: 5},
                _shapes=[(3, 4), (4, 5)],
                _arrays=[np.ones((3, 4)), np.ones((4, 5))],
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
    def test_tensor_network_equality_succeeds(self, other):
        """Test the equality comparison of TensorNetwork."""
        network = TensorNetwork(
            _input_indices=[[0, 1], [1, 2]],
            _output_indices=[0],
            _size_dict={0: 3, 1: 4, 2: 5},
            _shapes=[(3, 4), (4, 5)],
            _arrays=[np.ones((3, 4)), np.ones((4, 5))],
        )

        assert network == other

    @pytest.mark.parametrize(
        "other",
        [
            TensorNetwork(
                _input_indices=[[0], [1, 2]],
                _output_indices=[0],
                _size_dict={0: 3, 1: 4, 2: 5},
                _shapes=[(3,), (4, 5)],
                _arrays=[np.ones((3,)), np.ones((4, 5))],
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
    def test_tensor_network_equality_fails(self, other):
        """Test that the equality comparison fails for inequivalent and non-TN objects."""
        network = TensorNetwork(
            _input_indices=[[0, 1], [1, 2]],
            _output_indices=[0],
            _size_dict={0: 3, 1: 4, 2: 5},
            _shapes=[(3, 4), (4, 5)],
            _arrays=None,
        )

        assert network != other

    def test_tensor_network_arrays_property(self):
        """Test that the arrays property raises an error when arrays are not generated."""
        network = TensorNetwork(
            _input_indices=[[0, 1], [1, 2]],
            _output_indices=[0],
            _size_dict={0: 3, 1: 4, 2: 5},
            _shapes=[(3, 4), (4, 5)],
            _arrays=None,
        )

        with pytest.raises(
            ValueError,
            match="Arrays were not generated for this tensor network. Only metadata is available.",
        ):
            _ = network.arrays

    def test_tensor_network_properties(self):
        """Test the properties of the TensorNetwork class."""
        random_tensor_1 = np.random.rand(3, 4)
        random_tensor_2 = np.random.rand(4, 5)
        network = TensorNetwork(
            _input_indices=[[0, 1], [1, 2]],
            _output_indices=[0],
            _size_dict={0: 3, 1: 4, 2: 5},
            _shapes=[(3, 4), (4, 5)],
            _arrays=[random_tensor_1, random_tensor_2],
        )

        assert network.num_tensors == 2
        assert network.input_indices == [[0, 1], [1, 2]]
        assert network.output_indices == [0]
        assert network.shapes == [(3, 4), (4, 5)]
        assert network.size_dict == {0: 3, 1: 4, 2: 5}
        assert network.arrays == [random_tensor_1, random_tensor_2]

    def test_mismatching_information_should_raise_error(self):
        """Test that mismatching information raises an error."""
        with pytest.raises(ValueError):
            TensorNetwork(
                _input_indices=[[0, 1], [1, 2]],
                _output_indices=[0],
                _size_dict={0: 3, 1: 4, 2: 5},
                _shapes=[(3, 4)],
                _arrays=[np.random.rand(3, 4), np.random.rand(4, 5)],
            )
