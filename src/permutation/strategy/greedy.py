"""Greedy permutation strategy."""

from typing import override

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor import get_contracted_indices
from contraction.tree import ContractionTree, ContractionTreeNode
from memory.calculator.calculator import MemoryCalculator
from memory.memory import Memory
from memory.utils import get_largest_intermediate_tensor_in_contraction_path
from permutation import Permutation
from permutation.strategy import IPermutationStrategy
from permutation.utils import to_permutation
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def _sort_indices_by_size(indices: set[int] | list[int], size_dict: dict[int, int]) -> list[int]:
    """Sort indices by their dimension sizes (smallest first)."""
    return sorted(indices, key=lambda idx: size_dict[idx])


def _get_step_tensors(
    persistent_path: PersistentContractionPath, step: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Get left, right, and result tensors for a contraction step."""
    left_pos, right_pos = persistent_path.path[step]
    before = persistent_path.get_state(step)
    after = persistent_path.get_state(step + 1)

    return (
        before.tensors[left_pos],
        before.tensors[right_pos],
        after.tensors[left_pos],
    )


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

    _, _, result = _get_step_tensors(
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


def _node_walk(root: ContractionTreeNode) -> list[ContractionTreeNode]:
    """Return all nodes in the contraction tree using DFS order."""
    stack = [root]
    nodes: list[ContractionTreeNode] = []
    while stack:
        node = stack.pop()
        nodes.append(node)
        left = node.left
        right = node.right
        if right is not None:
            stack.append(right)
        if left is not None:
            stack.append(left)
    return nodes


def _find_peak_target_layout(
    tree: ContractionTree,
    persistent_path: PersistentContractionPath,
    peak_node: ContractionTreeNode,
    size_dict: dict[int, int],
) -> list[int]:
    """Infer the target layout for the peak tensor using its ancestor contraction.

    If the peak is not a leaf/root, we inspect its immediate parent contraction and
    place free and contracted indices in GEMM-friendly groups.
    """
    peak_parent = peak_node.parent
    peak_tensor = _get_node_tensor(
        persistent_path.initial_state,
        persistent_path,
        peak_node,
    )

    if peak_parent is None:
        return _sort_indices_by_size(
            set(peak_tensor.input_indices),
            size_dict,
        )

    parent_step = peak_parent.contraction_step
    if parent_step is None:
        return _sort_indices_by_size(
            set(peak_tensor.input_indices),
            size_dict,
        )

    left_tensor, right_tensor, _ = _get_step_tensors(
        persistent_path,
        parent_step,
    )
    is_left_child = peak_parent.left is peak_node
    peak_tensor_at_parent = left_tensor if is_left_child else right_tensor
    sibling_tensor = right_tensor if is_left_child else left_tensor

    contracted = get_contracted_indices(peak_tensor_at_parent, sibling_tensor)
    free = set(peak_tensor_at_parent.input_indices) - contracted
    free_sorted = _sort_indices_by_size(free, size_dict)
    contracted_sorted = _sort_indices_by_size(contracted, size_dict)

    if is_left_child:
        return free_sorted + contracted_sorted

    return contracted_sorted + free_sorted


class GreedyPermutationStrategy(IPermutationStrategy):
    """Greedy strategy for finding optimal tensor permutations for a contraction path."""

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork,
        contraction_path: ContractionPath,
        k: int = 1,
    ) -> tuple[list[Permutation], list[Permutation]]:
        """Find optimal initial and intermediate permutations for a contraction path.

        The greedy strategy identifies the largest k tensors in the contraction path and treats
        them as "peaks". It then plans permutations to ensure that these peak tensors are in
        GEMM-friendly layouts, and recursively plans compatible layouts for all other tensors
        in the contraction tree such that the top k do not need to be permuted.

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
        persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
        contraction_tree = ContractionTree.from_contraction_path(persistent_path)

        initial_permutations: list[Permutation] = [
            tuple(range(len(tensor.input_indices))) for tensor in network.tensors
        ]
        intermediate_permutations: list[Permutation] = []

        for step in range(persistent_path.num_steps):
            _, _, result_tensor = _get_step_tensors(persistent_path, step)
            intermediate_permutations.append(tuple(range(len(result_tensor.input_indices))))

        step_to_node: dict[int, ContractionTreeNode] = {}
        leaf_to_node: dict[int, ContractionTreeNode] = {}
        for node in _node_walk(contraction_tree.root):
            contraction_step = node.contraction_step
            leaf_pos = node.initial_tensor_position
            if contraction_step is not None:
                step_to_node[contraction_step] = node
            if leaf_pos is not None:
                leaf_to_node[leaf_pos] = node

        largest_step_idx, _ = get_largest_intermediate_tensor_in_contraction_path(
            network,
            contraction_path,
        )
        if largest_step_idx >= 0:
            peak_node = step_to_node[largest_step_idx]
        else:
            largest_initial_idx = max(
                range(len(network.tensors)),
                key=lambda i: max(network.tensors[i].shape) if network.tensors[i].shape else 1,
            )
            peak_node = leaf_to_node[largest_initial_idx]

        desired_layout_by_node_id: dict[int, list[int]] = {
            id(peak_node): _find_peak_target_layout(
                contraction_tree,
                persistent_path,
                peak_node,
                network.size_dict,
            )
        }

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

            left_tensor, right_tensor, result_tensor = _get_step_tensors(
                persistent_path,
                step,
            )

            contracted = get_contracted_indices(left_tensor, right_tensor)
            left_free = set(left_tensor.input_indices) - contracted
            right_free = set(right_tensor.input_indices) - contracted
            contracted_sorted = _sort_indices_by_size(
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
                desired_free = _sort_indices_by_size(
                    left_free,
                    network.size_dict,
                ) + _sort_indices_by_size(
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

            left_sliced_sorted = _sort_indices_by_size(
                left_sliced,
                network.size_dict,
            )
            right_sliced_sorted = _sort_indices_by_size(
                right_sliced,
                network.size_dict,
            )
            left_remaining_sorted = _sort_indices_by_size(
                left_remaining,
                network.size_dict,
            )
            right_remaining_sorted = _sort_indices_by_size(
                right_remaining,
                network.size_dict,
            )

            left_target = left_sliced_sorted + left_remaining_sorted + contracted_sorted
            right_target = contracted_sorted + right_sliced_sorted + right_remaining_sorted
            result_target = (
                left_sliced_sorted
                + left_remaining_sorted
                + right_sliced_sorted
                + right_remaining_sorted
            )

            if id(left_node) != id(peak_node):
                __set_node_layout(left_node, left_target)
            if id(right_node) != id(peak_node):
                __set_node_layout(right_node, right_target)

            if id(node) != id(peak_node):
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
        _, largest_memory = get_largest_intermediate_tensor_in_contraction_path(
            network,
            contraction_path,
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
