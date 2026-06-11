"""Greedy permutation strategy."""

from typing import override

from memory import Memory
from memory.calculator import MemoryCalculator
from memory.utils import (
    get_largest_intermediate_tensor_in_path,
    get_largest_k_intermediate_tensors_in_path,
    get_largest_k_tensors_in_network,
)
from operations.base import TensorOperation
from operations.contraction import get_contracted_indices
from operations.contraction.path import ContractionPath, PersistentContractionPath
from operations.contraction.tree import ContractionTreeNode
from operations.permutation import Permutation
from operations.permutation.utils import to_permutation
from operations.strategy import IStrategy
from operations.strategy.common import (
    build_tree_maps,
    get_step_tensors,
    sort_indices_by_layout,
    sort_indices_by_size,
)
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def _get_node_tensor(
    network: TensorNetwork,
    persistent_path: PersistentContractionPath,
    node: ContractionTreeNode,
) -> Tensor:
    """Get the tensor represented by a contraction-tree node."""
    initial_pos = node.initial_tensor_position
    if initial_pos is not None:
        return network.tensors[initial_pos]

    contraction_step = node.contraction_step
    if contraction_step is None:
        raise ValueError("Internal node must define a contraction step.")

    _, _, result = get_step_tensors(
        persistent_path,
        contraction_step,
    )
    return result


def _has_left_then_right_free_order(
    desired_free: list[int], left_free: set[int], right_free: set[int]
) -> bool:
    """Check whether desired free-index order can be realized without output permutation."""
    seen_right = False
    for idx in desired_free:
        if idx in right_free:
            seen_right = True
        elif idx in left_free and seen_right:
            return False
    return True


def _select_slice_indices_for_interleaving(
    desired_free: list[int], left_free: set[int], right_free: set[int]
) -> set[int]:
    """Pick indices to treat as sliced when desired free order is interleaved.

    We keep the longest subsequence matching the representable pattern `L*R*`
    and mark all remaining indices as sliced.
    """
    sides = ["L" if idx in left_free else "R" for idx in desired_free]
    n = len(sides)

    best_keep = -1
    best_switch = 0
    prefix_left_counts = [0] * (n + 1)
    suffix_right_counts = [0] * (n + 1)

    for i in range(n):
        prefix_left_counts[i + 1] = prefix_left_counts[i] + (1 if sides[i] == "L" else 0)

    for i in range(n - 1, -1, -1):
        suffix_right_counts[i] = suffix_right_counts[i + 1] + (1 if sides[i] == "R" else 0)

    for switch in range(n + 1):
        keep = prefix_left_counts[switch] + suffix_right_counts[switch]
        if keep > best_keep:
            best_keep = keep
            best_switch = switch

    sliced: set[int] = set()
    for i, idx in enumerate(desired_free):
        keep = (i < best_switch and sides[i] == "L") or (i >= best_switch and sides[i] == "R")
        if not keep:
            sliced.add(idx)

    return sliced


def _find_peak_target_layout(
    persistent_path: PersistentContractionPath,
    peak_node: ContractionTreeNode,
    size_dict: dict[int, int],
    desired_layout_by_node_id: dict[int, list[int]] | None = None,
) -> list[int]:
    """Infer the target layout for the peak tensor using its ancestor contraction.

    If the peak has a parent contraction, the layout is derived from the parent
    contraction and, where available, from previously computed desired layouts.
    """
    peak_parent = peak_node.parent
    peak_tensor = _get_node_tensor(
        persistent_path.initial_state,
        persistent_path,
        peak_node,
    )

    if peak_node.is_leaf:
        return peak_tensor.input_indices

    if peak_parent is None:
        return sort_indices_by_size(
            set(peak_tensor.input_indices),
            size_dict,
        )

    parent_step = peak_parent.contraction_step
    if parent_step is None:
        return sort_indices_by_size(
            set(peak_tensor.input_indices),
            size_dict,
        )

    left_tensor, right_tensor, _ = get_step_tensors(
        persistent_path,
        parent_step,
    )
    is_left_child = peak_parent.left is peak_node
    sibling_tensor = right_tensor if is_left_child else left_tensor

    contracted = get_contracted_indices(peak_tensor, sibling_tensor)
    free = set(peak_tensor.input_indices) - contracted

    desired_parent_layout = None
    desired_sibling_layout = None
    if desired_layout_by_node_id is not None:
        desired_parent_layout = desired_layout_by_node_id.get(id(peak_parent))
        desired_sibling_layout = desired_layout_by_node_id.get(
            id(peak_parent.right if is_left_child else peak_parent.left)
        )

    if desired_parent_layout is not None or desired_sibling_layout is not None:
        if desired_parent_layout is not None:
            free_sorted = sort_indices_by_layout(free, desired_parent_layout)
        else:
            free_sorted = sort_indices_by_size(free, size_dict)

        if desired_sibling_layout is not None:
            contracted_sorted = sort_indices_by_layout(contracted, desired_sibling_layout)
        else:
            contracted_sorted = sort_indices_by_size(contracted, size_dict)
    else:
        free_sorted = sort_indices_by_size(free, size_dict)
        contracted_sorted = sort_indices_by_size(contracted, size_dict)

    if is_left_child:
        return free_sorted + contracted_sorted

    return contracted_sorted + free_sorted


def _find_top_k_peak_nodes(
    network: TensorNetwork,
    persistent_path: PersistentContractionPath,
    leaf_to_node: dict[int, ContractionTreeNode],
    step_to_node: dict[int, ContractionTreeNode],
    k: int,
) -> list[ContractionTreeNode]:
    candidate_count = len(network.tensors) + persistent_path.num_steps
    k = min(k, candidate_count)

    original_indices, original_memories = get_largest_k_tensors_in_network(
        network, len(network.tensors)
    )
    intermediate_step_indices, intermediate_memories = get_largest_k_intermediate_tensors_in_path(
        network,
        persistent_path,
        candidate_count,
    )

    candidates: list[tuple[ContractionTreeNode, Memory]] = []
    candidates.extend(
        (leaf_to_node[idx], memory) for idx, memory in zip(original_indices, original_memories)
    )
    for step_idx, memory in zip(intermediate_step_indices, intermediate_memories):
        if step_idx == -1:
            continue
        candidate = step_to_node.get(step_idx)
        if candidate is not None:
            candidates.append((candidate, memory))

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)

    selected: list[ContractionTreeNode] = []
    seen_ids: set[int] = set()
    for node, _ in candidates:
        node_id = id(node)
        if node_id in seen_ids:
            continue
        selected.append(node)
        seen_ids.add(node_id)
        if len(selected) == k:
            break

    if not selected:
        raise ValueError("No peak nodes found, cannot apply greedy strategy.")

    return sorted(selected, key=lambda node: node.is_leaf, reverse=True)


class GreedyPermutationStrategy(IStrategy):
    """Greedy strategy for finding optimal tensor permutations for a contraction path."""

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        k: int = 1,
    ) -> list[TensorOperation]:
        """Find optimal initial and intermediate permutations for a contraction path.

        The greedy strategy identifies the largest k tensors in the contraction path and treats
        them as "peaks". It then plans permutations to ensure that these peak tensors are in
        GEMM-friendly layouts, and recursively plans compatible layouts for all other tensors
        in the contraction tree such that the top k do not need to be permuted.

        The complexity is O(n_tensors * path_length + Σ (tensor_i_indices * log(tensor_i_indices))).

        Args:
            network (TensorNetwork): The input tensor network.
            contraction_path (ContractionPath): The contraction path.
            k (int, optional): The number of top tensors to freeze. Defaults to 1.

        Raises:
            ValueError: If an internal node in the contraction tree does not define a contraction
            step.

        Returns:
            tuple[list[Permutation], list[Permutation]]:
            A tuple containing two lists:
                - The first list contains the optimal permutations for the initial tensors.
                - The second list contains the optimal permutations for the intermediate tensors
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k > len(network.tensors) + len(contraction_path):
            raise ValueError(
                "k must not exceed the total number of tensors (initial tensors and intermediate "
                "tensors produced)"
            )
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)

        initial_permutations: list[Permutation] = [
            tuple(range(len(tensor.input_indices))) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        if persistent_path.num_steps == 0:
            return initial_permutations, intermediate_permutations

        for step in range(persistent_path.num_steps):
            _, _, result_tensor = get_step_tensors(persistent_path, step)
            intermediate_permutations.append(tuple(range(len(result_tensor.input_indices))))

        contraction_tree, leaf_to_node, step_to_node = build_tree_maps(persistent_path)

        peak_nodes = _find_top_k_peak_nodes(
            network,
            persistent_path,
            leaf_to_node,
            step_to_node,
            k,
        )

        for node in peak_nodes:
            if node.is_leaf:
                raise NotImplementedError(
                    "Greedy strategy does not currently support freezing leaf nodes."
                )

        frozen_node_ids = {id(node) for node in peak_nodes}
        desired_layout_by_node_id: dict[int, list[int]] = {}
        for node in peak_nodes:
            desired_layout_by_node_id[id(node)] = _find_peak_target_layout(
                persistent_path,
                node,
                network.size_dict,
                desired_layout_by_node_id,
            )

        def __set_node_layout(node: ContractionTreeNode, target_layout: list[int]) -> None:
            """Store permutation for a node tensor and remember desired layout for recursion."""
            current_tensor = _get_node_tensor(
                network,
                persistent_path,
                node,
            )
            permutation = to_permutation(current_tensor.input_indices, target_layout)

            leaf_pos = node.initial_tensor_position
            if leaf_pos is not None:
                initial_permutations[leaf_pos] = permutation
                return

            step = node.contraction_step
            if step is None:
                raise ValueError("Internal node must define a contraction step.")
            intermediate_permutations[step] = permutation
            desired_layout_by_node_id[id(node)] = target_layout

        def __plan_node(node: ContractionTreeNode) -> None:
            """Recursively plan GEMM-friendly layouts across the full contraction tree."""
            left_node = node.left
            right_node = node.right
            step = node.contraction_step

            if left_node is None or right_node is None or step is None:
                return

            left_tensor, right_tensor, result_tensor = get_step_tensors(
                persistent_path,
                step,
            )

            contracted = get_contracted_indices(left_tensor, right_tensor)
            left_free = set(left_tensor.input_indices) - contracted
            right_free = set(right_tensor.input_indices) - contracted
            contracted_sorted = sort_indices_by_size(
                contracted,
                network.size_dict,
            )

            desired_layout = desired_layout_by_node_id.get(id(node))
            desired_free: list[int]
            if desired_layout is not None:
                desired_free = [
                    idx for idx in desired_layout if idx in left_free or idx in right_free
                ]
            else:
                desired_free = sort_indices_by_size(
                    left_free,
                    network.size_dict,
                ) + sort_indices_by_size(
                    right_free,
                    network.size_dict,
                )

            if _has_left_then_right_free_order(
                desired_free,
                left_free,
                right_free,
            ):
                sliced_indices: set[int] = set()
            else:
                sliced_indices = _select_slice_indices_for_interleaving(
                    desired_free,
                    left_free,
                    right_free,
                )

            left_sliced = sliced_indices & left_free
            right_sliced = sliced_indices & right_free
            left_remaining = left_free - left_sliced
            right_remaining = right_free - right_sliced

            left_sliced_sorted = sort_indices_by_size(
                left_sliced,
                network.size_dict,
            )
            right_sliced_sorted = sort_indices_by_size(
                right_sliced,
                network.size_dict,
            )
            left_remaining_sorted = sort_indices_by_layout(
                left_remaining,
                desired_free,
            )
            right_remaining_sorted = sort_indices_by_layout(
                right_remaining,
                desired_free,
            )

            left_target = left_sliced_sorted + left_remaining_sorted + contracted_sorted
            right_target = contracted_sorted + right_sliced_sorted + right_remaining_sorted
            result_target = (
                left_sliced_sorted
                + left_remaining_sorted
                + right_sliced_sorted
                + right_remaining_sorted
            )

            if id(left_node) not in frozen_node_ids:
                __set_node_layout(left_node, left_target)
            if id(right_node) not in frozen_node_ids:
                __set_node_layout(right_node, right_target)

            if id(node) not in frozen_node_ids:
                intermediate_permutations[step] = to_permutation(
                    result_tensor.input_indices,
                    result_target,
                )
                desired_layout_by_node_id[id(node)] = result_target

            __plan_node(left_node)
            __plan_node(right_node)

        __plan_node(contraction_tree.root)

        return initial_permutations, intermediate_permutations

    @staticmethod
    def __calculate_memory_for_path(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        peak: bool = True,
    ) -> Memory:
        """Calculate memory usage for a contraction path.

        Args:
            network: The tensor network.
            contraction_path: The contraction path.
            peak: If True, return peak memory; if False, return total memory.

        Returns:
            Memory usage based on peak or total calculation.
        """
        memory_calculator = MemoryCalculator()
        current_memory = memory_calculator.calculate_memory_for_tensors(network.tensors)
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        _, largest_memory = get_largest_intermediate_tensor_in_path(
            network,
            persistent_path,
        )
        result_memory = current_memory

        for idx, contraction_pair in enumerate(contraction_path):
            tensor_a, tensor_b = (
                persistent_path.get_state(idx).tensors[contraction_pair[0]],
                persistent_path.get_state(idx).tensors[contraction_pair[1]],
            )
            tensor_a_memory = memory_calculator.calculate_memory_for_tensor(tensor_a)
            tensor_b_memory = memory_calculator.calculate_memory_for_tensor(tensor_b)
            current_memory += tensor_a_memory + tensor_b_memory

            if tensor_a_memory >= largest_memory or tensor_b_memory >= largest_memory:
                current_memory -= max(tensor_a_memory, tensor_b_memory)  # The largest intermediate
                # tensor will not be permuted

            if peak:
                result_memory = max(result_memory, current_memory)

            contraction_memory = memory_calculator.calculate_memory_for_contraction(
                tensor_a, tensor_b
            )
            current_memory += contraction_memory
            if peak:
                result_memory = max(result_memory, current_memory)
            else:
                result_memory += tensor_a_memory + tensor_b_memory + contraction_memory
            current_memory -= (
                tensor_a_memory + tensor_b_memory
            )  # Remove the original forms of the permuted tensors

        return result_memory

    @staticmethod
    @override
    def get_peak_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the peak memory usage for a given contraction path and tensor permutations."""
        return GreedyPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=True
        )

    @staticmethod
    @override
    def get_total_memory(network: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the total memory movement for a contraction path and tensor permutations."""
        return GreedyPermutationStrategy.__calculate_memory_for_path(
            network, contraction_path, peak=False
        )
