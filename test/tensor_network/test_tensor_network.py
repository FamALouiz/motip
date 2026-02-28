"""Test the TensorNetwork class."""

import pytest
from tensor_network import TensorNetwork


class TestTensorNetwork:
    """Test the TensorNetwork class with various parameter configurations."""

    def test_tensor_network_as_tuple(self):
        """Test that the as_tuple method returns the correct shape."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4)],
        )

        expected_tuple = (
            [[0, 1], [1, 2]],
            [0],
            [(3, 4)],
            {0: 3, 1: 4, 2: 5},
        )
        assert network.as_tuple == expected_tuple

    @pytest.mark.parametrize(
        "other",
        [
            TensorNetwork(
                input_indices=[[0, 1], [1, 2]],
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
                shapes=[(3, 4)],
            ),
            (
                [[0, 1], [1, 2]],
                [0],
                [(3, 4)],
                {0: 3, 1: 4, 2: 5},
            ),
        ],
    )
    def test_tensor_network_equality_succeeds(self, other):
        """Test the equality comparison of TensorNetwork."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4)],
        )

        assert network == other

    @pytest.mark.parametrize(
        "other",
        [
            TensorNetwork(
                input_indices=[[0], [1, 2]],
                output_indices=[0],
                size_dict={0: 3, 1: 4, 2: 5},
                shapes=[(3, 4)],
            ),
            (
                [[0], [1, 2]],
                [0],
                [(3, 4)],
                {0: 3, 1: 4, 2: 5},
            ),
            "not a tensor network",
        ],
    )
    def test_tensor_network_equality_fails(self, other):
        """Test that the equality comparison fails for inequivalent and non-TN objects."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            output_indices=[0],
            size_dict={0: 3, 1: 4, 2: 5},
            shapes=[(3, 4)],
        )

        assert network != other
