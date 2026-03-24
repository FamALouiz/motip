"""Random tensor network generation utility."""

from tensor_network import TensorNetwork
from tensor_network.builder import TensorNetworkBuilder


def generate_random_tn(
    num_tensors: int,
    average_rank: int,
    max_dim: int,
    num_output_indices: int,
    seed: int = 42,
    generate_arrays: bool = False,
) -> TensorNetwork:
    """Generate a random tensor network.

    Args:
        num_tensors: Number of tensors in the network.
        average_rank: Average number of indices per tensor.
        max_dim: Maximum dimension size for any index.
        num_output_indices: Number of output indices in the network.
        seed: Random seed for reproducibility.
        generate_arrays: Whether to generate random arrays for the tensors.

    Returns:
        A randomly generated TensorNetwork instance.
    """
    builder = (
        TensorNetworkBuilder()
        .with_number_of_tensors(num_tensors)
        .with_average_number_of_indices_per_tensor(average_rank)
        .with_max_dimension_size(max_dim)
        .with_number_of_output_indices(num_output_indices)
        .with_seed(seed)
    )
    if generate_arrays:
        builder = builder.with_generate_arrays()

    return builder.build()
