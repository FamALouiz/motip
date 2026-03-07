"""Random tensor network generation utility."""

from tensor_network import TensorNetwork
from tensor_network.builder import TensorNetworkBuilder


def generate_random_tn(
    num_tensors: int, average_rank: int, max_dim: int, seed: int = 42, generate_arrays: bool = False
) -> TensorNetwork:
    """Generate a random tensor network."""
    builder = (
        TensorNetworkBuilder()
        .with_number_of_tensors(num_tensors)
        .with_average_number_of_indices_per_tensor(average_rank)
        .with_max_dimension_size(max_dim)
        .with_seed(seed)
    )
    if generate_arrays:
        builder = builder.with_generate_arrays()

    return builder.build()
