"""Permutation strategies for tensor networks."""

from typing import Protocol, runtime_checkable

from memory import Memory
from operations.base import TensorOperation
from operations.contraction.path import ContractionPath
from tensor_network.tn import TensorNetwork


@runtime_checkable
class IStrategy(Protocol):
    """Interface for defining a permutation strategy."""

    @staticmethod
    def find_optimal_permutation(
        network: TensorNetwork, contraction_path: ContractionPath
    ) -> list[TensorOperation]:
        """Find the tensor operations for a given contraction path.

        Args:
            network: The tensor network for which to find the optimal permutation.
            contraction_path: The contraction path for which to find the optimal permutation.

        Returns:
            A list of tensor operations representing permutations and contractions.
        """
        ...

    @staticmethod
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the peak memory usage for a given contraction path and tensor permutations."""
        ...

    @staticmethod
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the total memory movement for a contraction path and tensor permutations."""
        ...
