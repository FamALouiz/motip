"""Test the Tensor class."""

import numpy as np
import pytest

from tensor import Tensor


class TestTensor:
    """Test the Tensor class with various parameter configurations."""

    def test_tensor_as_tuple(self) -> None:
        """Test that the as_tuple method returns the correct shape."""
        tensor = Tensor(
            input_indices=[0, 1],
            shape=(3, 4),
            array=None,
        )

        expected_tuple = ([0, 1], (3, 4), None)
        assert tensor.as_tuple == expected_tuple

    @pytest.mark.parametrize(
        "other",
        [
            Tensor(
                input_indices=[0, 1],
                shape=(3, 4),
                array=np.ones((3, 4)),
            ),
            ([0, 1], (3, 4), np.ones((3, 4))),
        ],
    )
    def test_tensor_equality_succeeds(self, other: object) -> None:
        """Test the equality comparison of Tensor."""
        tensor = Tensor(
            input_indices=[0, 1],
            shape=(3, 4),
            array=np.ones((3, 4)),
        )

        assert tensor == other

    @pytest.mark.parametrize(
        "other",
        [
            Tensor(
                input_indices=[0],
                shape=(3,),
                array=np.ones((3,)),
            ),
            ([0], (3,), None),
            "not a tensor",
        ],
    )
    def test_tensor_equality_fails(self, other: object) -> None:
        """Test that the equality comparison fails for inequivalent and non-Tensor objects."""
        tensor = Tensor(
            input_indices=[0, 1],
            shape=(3, 4),
            array=None,
        )

        assert tensor != other

    def test_mismatching_information_should_raise_error(self) -> None:
        """Test that mismatching information raises an error."""
        with pytest.raises(ValueError):
            Tensor(
                input_indices=[0, 1],
                shape=(3,),
                array=np.random.rand(3, 4),
            )
