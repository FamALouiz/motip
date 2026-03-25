"""Permutation strategies for tensor networks."""

from typing import Protocol, runtime_checkable

from contraction.path import ContractionPath
from tensor_network.tn import TensorNetwork


@runtime_checkable
class IPermutationStrategy(Protocol):
    """Interface for defining a permutation strategy."""

    @staticmethod
    def find_optimal_permutation(
        network: TensorNetwork, contraction_path: ContractionPath
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Find the optimal tensor permutation for a given contraction path.

        Args:
            network: The tensor network for which to find the optimal permutation.
            contraction_path: The contraction path for which to find the optimal permutation.

        Returns:
            A tuple of 2 components:
                - A list of optimal tensor index permutations for each initial tensor in the network
                - A list of optimal tensor index permutations for each intermediate tensor in the
            contraction path
        """
        ...
