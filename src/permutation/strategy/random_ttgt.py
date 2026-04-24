"""Random TTGT-style permutation strategy for tensor network contraction."""

from __future__ import annotations

import random
from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor import get_contracted_indices
from memory import Memory
from memory.calculator.calculator import MemoryCalculator
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.strategy.common import (
    apply_layout_to_tensor,
    build_tree_maps,
    get_step_tensors,
)
from permutation.utils import to_identity_permutation
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def _shuffle_group(indices: set[int] | list[int], rng: random.Random) -> list[int]:
    """Shuffle a collection of indices.

    Args:
        indices: The indices to shuffle.
        rng: The random number generator.

    Returns:
        A shuffled list of indices.
    """
    values = list(indices)
    rng.shuffle(values)
    return values


def _get_random_ttgt_input_layout(
    current_tensor: Tensor,
    sibling_tensor: Tensor,
    is_left: bool,
    rng: random.Random,
) -> list[int]:
    """Generate a random TTGT-valid input layout.

    Args:
        current_tensor: The current tensor.
        sibling_tensor: The sibling tensor.
        is_left: Whether the tensor is the left input tensor.
        rng: The random number generator.

    Returns:
        A random TTGT-valid layout.
    """
    tensor_stub = current_tensor
    sibling_stub = sibling_tensor

    contracted = get_contracted_indices(tensor_stub, sibling_stub)
    free = set(current_tensor.input_indices) - contracted

    contracted_shuffled = _shuffle_group(contracted, rng)
    free_shuffled = _shuffle_group(free, rng)

    if is_left:
        return free_shuffled + contracted_shuffled
    return contracted_shuffled + free_shuffled


def _get_random_result_layout(
    step: int,
    persistent_path: PersistentContractionPath,
    rng: random.Random,
) -> list[int]:
    """Generate a random valid layout for a contraction result tensor.

    Args:
        step: The contraction step index.
        persistent_path: The persistent contraction path.
        rng: The random number generator.

    Returns:
        A random result layout.
    """
    left_tensor, right_tensor, result_tensor = get_step_tensors(persistent_path, step)

    contracted = get_contracted_indices(left_tensor, right_tensor)
    left_free = list(set(left_tensor.input_indices) - contracted)
    right_free = list(set(right_tensor.input_indices) - contracted)

    rng.shuffle(left_free)
    rng.shuffle(right_free)

    if rng.choice((True, False)):
        layout = left_free + right_free
    else:
        layout = right_free + left_free

    return [idx for idx in layout if idx in result_tensor.input_indices]


class RandomTTGTPermutationStrategy(IPermutationStrategy):
    """Random TTGT-style permutation strategy.

    This strategy samples random permutations that preserve TTGT-valid grouping.
    """

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        seed: int = 0,
    ) -> tuple[list[Permutation], list[Permutation]]:
        """Generate random TTGT-style permutations.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.
            seed: The random seed.

        Returns:
            Random TTGT-style permutations for initial and intermediate tensors.
        """
        rng = random.Random(seed)
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        _, leaf_to_node, _ = build_tree_maps(persistent_path)

        initial_permutations: list[Permutation] = [
            to_identity_permutation(tensor) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        for leaf_pos, leaf_node in leaf_to_node.items():
            if leaf_node.parent is None or leaf_node.parent.contraction_step is None:
                continue

            step = leaf_node.parent.contraction_step
            left_tensor, right_tensor, _ = get_step_tensors(persistent_path, step)

            is_left = leaf_node.parent.left is leaf_node
            current_tensor = left_tensor if is_left else right_tensor
            sibling_tensor = right_tensor if is_left else left_tensor

            layout = _get_random_ttgt_input_layout(
                current_tensor,
                sibling_tensor,
                is_left,
                rng,
            )
            initial_permutations[leaf_pos] = apply_layout_to_tensor(
                network.tensors[leaf_pos],
                layout,
            )

        for step in range(persistent_path.num_steps):
            _, _, result_tensor = get_step_tensors(persistent_path, step)
            layout = _get_random_result_layout(step, persistent_path, rng)
            intermediate_permutations.append(apply_layout_to_tensor(result_tensor, layout))

        return initial_permutations, intermediate_permutations

    @staticmethod
    def __calculate_memory_for_path(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        peak: bool = True,
        seed: int = 0,
    ) -> Memory:
        """Calculate memory usage for a contraction path.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.
            peak: If True, return peak memory; if False, return total memory.
            seed: The random seed.

        Returns:
            Memory usage based on peak or total calculation.
        """
        rng = random.Random(seed)
        memory_calculator = MemoryCalculator()
        current_memory = memory_calculator.calculate_memory_for_tensors(network.tensors)
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        result_memory = current_memory

        for idx, contraction_pair in enumerate(contraction_path):
            tensor_a, tensor_b = (
                persistent_path.get_state(idx).tensors[contraction_pair[0]],
                persistent_path.get_state(idx).tensors[contraction_pair[1]],
            )
            should_shuffle_a = rng.choice((True, False))
            should_shuffle_b = rng.choice((True, False))

            tensor_a_memory = memory_calculator.calculate_memory_for_tensor(tensor_a)
            tensor_b_memory = memory_calculator.calculate_memory_for_tensor(tensor_b)
            current_memory += tensor_a_memory if should_shuffle_a else Memory(0)
            current_memory += tensor_b_memory if should_shuffle_b else Memory(0)

            if peak:
                result_memory = max(result_memory, current_memory)

            current_memory -= (tensor_a_memory if should_shuffle_a else Memory(0)) + (
                tensor_b_memory if should_shuffle_b else Memory(0)
            )  # Remove the original forms of the permuted tensors
            contraction_memory = memory_calculator.calculate_memory_for_contraction(
                tensor_a, tensor_b
            )

            current_memory += contraction_memory  # Add the newly created result tensor

            if peak:
                result_memory = max(result_memory, current_memory)
            else:
                result_memory += tensor_a_memory + tensor_b_memory + contraction_memory

            current_memory -= (
                tensor_a_memory + tensor_b_memory
            )  # Remove the tensors since they are consumed

        return result_memory

    @staticmethod
    @override
    def get_peak_memory(
        network: TensorNetwork, contraction_path: ContractionPath, seed: int = 0
    ) -> Memory:
        """Calculate the peak memory usage for a given contraction path and tensor permutations."""
        return RandomTTGTPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=True, seed=seed
        )

    @staticmethod
    @override
    def get_total_memory(
        network: TensorNetwork, contraction_path: ContractionPath, seed: int = 0
    ) -> Memory:
        """Calculate the total memory movement for a contraction path and tensor permutations."""
        return RandomTTGTPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=False, seed=seed
        )
