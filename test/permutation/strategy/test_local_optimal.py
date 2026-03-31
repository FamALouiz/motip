"""Tests for the local optimal permutation strategy."""

from contraction.path import ContractionPath
from permutation.strategy.local_optimal import LocalOptimalPermutationStrategy
from tensor_network import TensorNetwork


class TestLocalOptimalPermutationStrategy:
    """Test the local optimal permutation strategy for GEMM-friendly layouts."""

    def test_simple_contraction_gemm_friendly_layout(self) -> None:
        """Test that contraction results are in GEMM-friendly layout."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3), (3, 4)],
            output_indices=[0, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 2
        assert len(intermediate_perms) == 1

        assert initial_perms[0] == (0, 1)
        assert initial_perms[1] == (0, 1)
        assert intermediate_perms[0] == (0, 1)

    def test_three_tensor_chain_contraction(self) -> None:
        """Test contraction path with three tensors forming a chain."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (3, 4), (4, 5)],
            output_indices=[0, 3],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1), (0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 3
        assert len(intermediate_perms) == 2

    def test_permutations_are_valid_rearrangements(self) -> None:
        """Test that all generated permutations are valid rearrangements."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2], [2, 3, 4]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
            shapes=[(2, 3, 4), (4, 5, 6)],
            output_indices=[0, 1, 3, 4],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        for perm in initial_perms:
            assert sorted(perm) == list(range(len(perm)))

        for perm in intermediate_perms:
            assert sorted(perm) == list(range(len(perm)))

    def test_contraction_with_multiple_free_indices(self) -> None:
        """Test contraction with multiple free indices on left and right tensors."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2], [2, 3, 4]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
            shapes=[(2, 3, 4), (4, 5, 6)],
            output_indices=[0, 1, 3, 4],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(intermediate_perms) == 1
        result_permutation = intermediate_perms[0]
        assert len(result_permutation) == 4

    def test_contraction_no_shared_indices(self) -> None:
        """Test contraction between tensors with no shared indices (outer product)."""
        network = TensorNetwork(
            input_indices=[[0, 1], [2, 3]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5},
            shapes=[(2, 3), (4, 5)],
            output_indices=[0, 1, 2, 3],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(intermediate_perms) == 1
        assert len(intermediate_perms[0]) == 4

    def test_returns_correct_number_of_permutations(self) -> None:
        """Test that the strategy returns correct number of permutations."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3], [3, 4]],
            size_dict={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
            shapes=[(2, 3), (3, 4), (4, 5), (5, 6)],
            output_indices=[0, 4],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1), (0, 1), (0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 4
        assert len(intermediate_perms) == 3

    def test_single_tensor_network(self) -> None:
        """Test with a single tensor (no contraction)."""
        network = TensorNetwork(
            input_indices=[[0, 1, 2]],
            size_dict={0: 2, 1: 3, 2: 4},
            shapes=[(2, 3, 4)],
            output_indices=[0, 1, 2],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = []

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 1
        assert len(intermediate_perms) == 0
        assert initial_perms[0] == (0, 1, 2)

    def test_initial_tensors_optimized_for_first_use(self) -> None:
        """Test that initial tensors are permuted based on their parent contraction."""
        network = TensorNetwork(
            input_indices=[[0, 1, 9], [1, 2, 3]],
            size_dict={0: 5, 1: 2, 2: 10, 3: 3, 9: 8},
            shapes=[(5, 2, 8), (2, 10, 3)],
            output_indices=[0, 9, 2, 3],
            tensor_arrays=None,
        )
        contraction_path: ContractionPath = [(0, 1)]

        initial_perms, intermediate_perms = (
            LocalOptimalPermutationStrategy.find_optimal_permutation(network, contraction_path)
        )

        assert len(initial_perms) == 2
        applied_indices_0 = [network.tensors[0].input_indices[i] for i in initial_perms[0]]
        applied_indices_1 = [network.tensors[1].input_indices[i] for i in initial_perms[1]]

        assert sorted(applied_indices_0) == sorted(network.tensors[0].input_indices)
        assert sorted(applied_indices_1) == sorted(network.tensors[1].input_indices)
