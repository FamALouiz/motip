"""Tests for the random tensor network generation utility."""

from tensor_network.utils.random import generate_random_tn
from tensor_network import TensorNetwork


class TestRandomTNGeneration:
    """Tests for the random tensor network generation utility."""

    def test_random_tn_generation(self):
        """Test the generation of a random tensor network."""
        tn = generate_random_tn(num_tensors=5, max_rank=3, max_dim=4, seed=42, generate_arrays=True)

        assert isinstance(tn, TensorNetwork)
        assert len(tn.input_indices) == 5
        assert len(tn.output_indices) <= 5
        assert all(isinstance(size, int) and size > 0 for size in tn.size_dict.values())
        assert all(isinstance(shape, tuple) for shape in tn.shapes)
        assert tn.arrays is not None

    def test_same_seed_produces_same_tn(self):
        """Test that using the same seed produces the same tensor network."""
        tn1 = generate_random_tn(
            num_tensors=5, max_rank=3, max_dim=4, seed=42, generate_arrays=False
        )
        tn2 = generate_random_tn(
            num_tensors=5, max_rank=3, max_dim=4, seed=42, generate_arrays=False
        )
        tn3 = generate_random_tn(
            num_tensors=5, max_rank=3, max_dim=4, seed=43, generate_arrays=True
        )
        tn4 = generate_random_tn(
            num_tensors=5, max_rank=3, max_dim=4, seed=43, generate_arrays=True
        )

        assert tn1 == tn2
        assert tn3 == tn4
