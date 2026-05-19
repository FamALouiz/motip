"""Dummy operation for testing."""

from typing import Any

from operations.base import TensorOperation, TensorOperationResult


class DummyOperation(TensorOperation):
    """A dummy operation that does nothing to the tensor."""

    def apply(
        self, input: TensorOperationResult, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> TensorOperationResult:
        """Apply the dummy operation, which simply returns the input tensor."""
        return input
