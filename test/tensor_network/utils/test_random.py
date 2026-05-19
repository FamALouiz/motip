"""Tests for the random tensor network generation utility."""

from copy import deepcopy

import cotengra as ctg

from operations.contraction.path import ContractionPath
from operations.contraction.utils import contract_tensors_in_network
from tensor_network import TensorNetwork
from tensor_network.utils.random import generate_random_tn


def _contract_network(network: TensorNetwork, contraction_path: ContractionPath) -> TensorNetwork:
    """Contract a tensor network according to a given contraction path.

    Args:
        network: The tensor network to contract.
        contraction_path: A list of tuples, where each tuple contains the indices of the tensors to
            contract at each step.

    Returns:
        The resulting tensor network after performing all contractions in the path.
    """
    current_network = deepcopy(network)
    for pair in contraction_path:
        current_network = contract_tensors_in_network(current_network, pair)
    return current_network


class TestRandomTNGeneration:
    """Tests for the random tensor network generation utility."""

    def test_random_tn_generation(self) -> None:
        """Test the generation of a random tensor network."""
        num_tensors = 5
        average_rank = 3
        max_dim = 4
        num_output_indices = 2
        tn = generate_random_tn(
            num_tensors=num_tensors,
            average_rank=average_rank,
            max_dim=max_dim,
            seed=42,
            generate_arrays=True,
            num_output_indices=num_output_indices,
        )
        contraction_path = ctg.array_contract_tree(
            inputs=tn.input_indices,
            output=tn.output_indices,
            size_dict=tn.size_dict,
            shapes=tn.shapes,
        ).get_path()

        final_network = _contract_network(tn, contraction_path)

        assert isinstance(tn, TensorNetwork)
        assert len(tn.input_indices) == num_tensors
        assert len(tn.output_indices) == num_output_indices
        assert final_network.output_indices == tn.output_indices
        assert len(final_network) == 1
        assert len(final_network.tensors[0].input_indices) == num_output_indices
        assert all(isinstance(size, int) and size > 0 for size in tn.size_dict.values())
        assert all(isinstance(shape, tuple) for shape in tn.shapes)
        assert tn.arrays is not None

    def test_same_seed_produces_same_tn(self) -> None:
        """Test that using the same seed produces the same tensor network."""
        tn1 = generate_random_tn(
            num_tensors=5,
            average_rank=3,
            max_dim=4,
            seed=42,
            generate_arrays=False,
            num_output_indices=2,
        )
        tn2 = generate_random_tn(
            num_tensors=5,
            average_rank=3,
            max_dim=4,
            seed=42,
            generate_arrays=False,
            num_output_indices=2,
        )
        tn3 = generate_random_tn(
            num_tensors=5,
            average_rank=3,
            max_dim=4,
            seed=44,
            generate_arrays=True,
            num_output_indices=2,
        )
        tn4 = generate_random_tn(
            num_tensors=5,
            average_rank=3,
            max_dim=4,
            seed=44,
            generate_arrays=True,
            num_output_indices=2,
        )

        assert tn1 == tn2
        assert tn3 == tn4
