"""Tests for the DummyOperation tensor operation."""

from operations.base import TensorOperationResult
from operations.dummy import DummyOperation
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


class TestDummyOperation:
    """Tests for the DummyOperation tensor operation."""

    def test_dummy_operation_returns_first_input(self) -> None:
        """Test that the DummyOperation returns the first input tensor."""
        tensor_1 = Tensor(input_indices=[0, 1], shape=(2, 3), array=None)
        tensor_2 = Tensor(input_indices=[1, 2], shape=(3, 4), array=None)

        result = DummyOperation().apply(
            tensor_operation_result_from_tensor(tensor_1),
            tensor_operation_result_from_tensor(tensor_2),
        )

        assert isinstance(result, TensorOperationResult)
        assert result.tensor == tensor_1
