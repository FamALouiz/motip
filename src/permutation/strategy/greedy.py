"""Greedy permutation strategy."""

from typing import override

from contraction.path import ContractionPath, ContractionPathWithHistory
from contraction.tensor import get_contracted_indices
from memory.utils import get_largest_intermediate_tensor_in_contraction_path
from permutation.strategy import IPermutationStrategy
from tensor import Tensor
from tensor_network.tn import TensorNetwork


def _index_sort_key(network: TensorNetwork, idx: int) -> tuple[int, int]:
    """Sort by index dimension first, then index id for deterministic ordering."""
    return (network.size_dict.get(idx, 0), idx)


def _build_gemm_friendly_output_layout(
    network: TensorNetwork, tensor_a: Tensor, tensor_b: Tensor, contracted: set[int]
) -> list[int]:
    """Build an output layout as [free(A), free(B)] with each block size-sorted."""
    free_a = [idx for idx in tensor_a.input_indices if idx not in contracted]
    free_b = [idx for idx in tensor_b.input_indices if idx not in contracted]
    sorted_free_a = sorted(free_a, key=lambda idx: _index_sort_key(network, idx))
    sorted_free_b = sorted(free_b, key=lambda idx: _index_sort_key(network, idx))
    return [*sorted_free_a, *sorted_free_b]


def _align_layout_to_target(
    current_indices: list[int], preferred_layout: list[int], fallback_layout: list[int]
) -> list[int]:
    """Align a target layout to available indices while preserving target relative order."""
    current_set = set(current_indices)
    preferred_subset = [idx for idx in preferred_layout if idx in current_set]
    already_placed = set(preferred_subset)
    fallback_subset = [
        idx for idx in fallback_layout if idx in current_set and idx not in already_placed
    ]
    return [*preferred_subset, *fallback_subset]


def _needs_slice_fallback(
    desired_layout: list[int], a_indices: list[int], b_indices: list[int]
) -> bool:
    """Detect if desired output interleaves A/B blocks in a way not realizable by permutation.

    With current contraction semantics, output index order is produced as all surviving indices
    from tensor A followed by all surviving indices from tensor B.
    """
    a_set = set(a_indices)
    b_set = set(b_indices)
    desired_a = [idx for idx in desired_layout if idx in a_set and idx not in b_set]
    desired_b = [idx for idx in desired_layout if idx in b_set and idx not in a_set]
    return desired_layout != [*desired_a, *desired_b]


def _build_slice_first_layout(
    network: TensorNetwork,
    desired_layout: list[int],
    tensor_a: Tensor,
    tensor_b: Tensor,
    contracted: set[int],
) -> list[int]:
    """Fallback order when strict target layout cannot be achieved by permutation alone.

    Heuristic:
    1) Move problematic (interleaving) indices to the front (slice-first), size-sorted.
    2) Place remaining free indices in GEMM-friendly order [free(A), free(B)], each size-sorted.
    """
    free_a = [idx for idx in tensor_a.input_indices if idx not in contracted]
    free_b = [idx for idx in tensor_b.input_indices if idx not in contracted]

    free_a_set = set(free_a)
    free_b_set = set(free_b)
    last_a_pos = max((i for i, idx in enumerate(desired_layout) if idx in free_a_set), default=-1)
    problematic_from_b = [
        idx for i, idx in enumerate(desired_layout) if idx in free_b_set and i < last_a_pos
    ]
    problematic_set = set(problematic_from_b)

    sliced = sorted(problematic_set, key=lambda idx: _index_sort_key(network, idx))
    remaining_a = [idx for idx in free_a if idx not in problematic_set]
    remaining_b = [idx for idx in free_b if idx not in problematic_set]
    sorted_remaining_a = sorted(remaining_a, key=lambda idx: _index_sort_key(network, idx))
    sorted_remaining_b = sorted(remaining_b, key=lambda idx: _index_sort_key(network, idx))

    return [*sliced, *sorted_remaining_a, *sorted_remaining_b]


def _to_permutation(current_indices: list[int], desired_layout: list[int]) -> tuple[int, ...]:
    """Convert desired index layout to a permutation over current indices."""
    if len(desired_layout) != len(current_indices):
        raise ValueError(
            "Desired layout must contain exactly the same number of indices as current layout. "
            f"Got {len(desired_layout)} desired vs {len(current_indices)} current."
        )
    return tuple(current_indices.index(idx) for idx in desired_layout)


def _infer_peak_target_layout(
    network: TensorNetwork,
    contraction_path: ContractionPath,
    history: ContractionPathWithHistory,
    largest_step_idx: int,
) -> list[int]:
    """Infer desired layout of the peak intermediate tensor from its next contraction step."""
    if largest_step_idx < 0:
        return []

    peak_pair = contraction_path[largest_step_idx]
    peak_tensor = history.get_state(largest_step_idx + 1).tensors[peak_pair[0]]
    peak_indices = peak_tensor.input_indices

    if largest_step_idx >= len(contraction_path) - 1:
        return sorted(peak_indices, key=lambda idx: _index_sort_key(network, idx))

    next_pair = contraction_path[largest_step_idx + 1]
    next_state_before = history.get_state(largest_step_idx + 1)
    next_tensor_a = next_state_before.tensors[next_pair[0]]
    next_tensor_b = next_state_before.tensors[next_pair[1]]
    next_contracted = get_contracted_indices(next_tensor_a, next_tensor_b)
    next_base_layout = _build_gemm_friendly_output_layout(
        network, next_tensor_a, next_tensor_b, next_contracted
    )

    return _align_layout_to_target(
        current_indices=peak_indices,
        preferred_layout=next_base_layout,
        fallback_layout=sorted(peak_indices, key=lambda idx: _index_sort_key(network, idx)),
    )


class GreedyPermutationStrategy(IPermutationStrategy):
    """Greedy strategy for finding optimal tensor permutations for a contraction path."""

    @staticmethod
    @override
    def find_optimal_permutation(
        network: TensorNetwork, contraction_path: ContractionPath
    ) -> list[tuple[int, ...]]:
        """Greedy strategy to find optimal tensor permutations for a contraction path.

        The strategy is peak-memory aware and traverses the contraction path backwards. Steps after
        the peak are freely optimized for GEMM- and stride-friendly layouts. The peak step itself
        is kept unpermuted to reduce the need for copying the largest intermediate tensor twice.
        Steps before the peak are shaped to make the peak tensor naturally emerge in thw desired
        layout; when a desire layout is not realizable by pure permutation, a slice-first fallback
        order is used.
        """
        num_steps = len(contraction_path)
        if num_steps == 0:
            return []

        largest_step_idx, _ = get_largest_intermediate_tensor_in_contraction_path(
            network, contraction_path
        )

        if largest_step_idx == -1:
            raise NotImplementedError(
                "Cannot determine optimal permutation without a valid largest intermediate tensor."
            )

        history = ContractionPathWithHistory.from_contraction_path(network, contraction_path)
        permutations: list[tuple[int, ...]] = [tuple()] * num_steps

        peak_target_layout = _infer_peak_target_layout(
            network, contraction_path, history, largest_step_idx
        )

        for step in range(num_steps - 1, -1, -1):
            pair = contraction_path[step]
            state_before = history.get_state(step)
            state_after = history.get_state(step + 1)

            tensor_a = state_before.tensors[pair[0]]
            tensor_b = state_before.tensors[pair[1]]
            intermediate_tensor = state_after.tensors[pair[0]]

            contracted = get_contracted_indices(tensor_a, tensor_b)
            base_layout = _build_gemm_friendly_output_layout(
                network, tensor_a, tensor_b, contracted
            )

            if step > largest_step_idx:
                desired_layout = base_layout
            elif step == largest_step_idx:
                desired_layout = _align_layout_to_target(
                    current_indices=intermediate_tensor.input_indices,
                    preferred_layout=peak_target_layout or base_layout,
                    fallback_layout=base_layout,
                )
                # Keep peak intermediate tensor in-place and instead shape prior contractions.
                permutations[step] = tuple(range(len(intermediate_tensor.input_indices)))
                continue
            else:
                desired_layout = _align_layout_to_target(
                    current_indices=intermediate_tensor.input_indices,
                    preferred_layout=peak_target_layout,
                    fallback_layout=base_layout,
                )

                if _needs_slice_fallback(
                    desired_layout, tensor_a.input_indices, tensor_b.input_indices
                ):
                    desired_layout = _build_slice_first_layout(
                        network=network,
                        desired_layout=desired_layout,
                        tensor_a=tensor_a,
                        tensor_b=tensor_b,
                        contracted=contracted,
                    )

            permutations[step] = _to_permutation(
                current_indices=intermediate_tensor.input_indices,
                desired_layout=desired_layout,
            )

        return permutations
