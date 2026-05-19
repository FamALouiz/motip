"""Base classes for tensor operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tensor import Tensor


@dataclass
class TensorOperationResult:
    """Class to hold the result of a tensor operation."""

    tensor: Tensor


class TensorOperation(ABC):
    """Abstract base class for tensor operations."""

    @abstractmethod
    def apply(
        self, *inputs: TensorOperationResult, **kwargs: dict[str, Any]
    ) -> TensorOperationResult:
        """Apply the tensor operation."""
