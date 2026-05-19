"""Operation utilities."""

from operations.base import TensorOperationResult
from tensor import Tensor


def tensor_operation_result_from_tensor(tensor: Tensor) -> TensorOperationResult:
    """Create a TensorOperationResult from a tensor."""
    return TensorOperationResult(tensor=tensor)
