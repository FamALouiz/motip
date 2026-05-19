"""Test module for tensor operation utils."""

from operations.base import TensorOperationResult
from operations.utils import tensor_operation_result_from_tensor
from tensor import Tensor


class TestTensorOperationResultFromTensor:
    """Tests for tensor_operation_result_from_tensor behavior."""

    def test_tensor_operation_result_from_tensor_creates_result_with_given_tensor(self) -> None:
        """Test that the function creates a TensorOperationResult with the given tensor."""
        tensor = Tensor(input_indices=[0, 1], shape=(2, 4), array=None)
        result = tensor_operation_result_from_tensor(tensor)
        assert isinstance(result, TensorOperationResult)
        assert result.tensor == tensor
