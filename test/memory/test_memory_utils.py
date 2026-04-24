"""Tests for the memory utilities."""

import pytest

from contraction.path import ContractionPath, PersistentContractionPath
from contraction.tensor_network import contract_tensors_in_network
from memory.calculator import MemoryCalculator
from memory.utils import (
    get_largest_intermediate_tensor_in_path,
    get_largest_tensor_in_network,
)
from tensor_network import TensorNetwork


@pytest.fixture
def sample_network() -> TensorNetwork:
    """Create a deterministic tensor network with known contraction outcomes."""
    return TensorNetwork(
        input_indices=[[0, 1], [2, 3], [1, 2]],
        output_indices=[0, 3],
        shapes=[(2, 3), (5, 7), (3, 5)],
        size_dict={0: 2, 1: 3, 2: 5, 3: 7},
        tensor_arrays=None,
    )


class TestLargestTensorInNetwork:
    """Tests for get_largest_tensor_in_network behavior."""

    def test_largest_tensor_in_network(self, sample_network: TensorNetwork) -> None:
        """Test that the largest tensor in the network is correctly identified."""
        expected_largest_tensor_idx = 1
        expected_largest_memory = MemoryCalculator().calculate_memory_for_tensor(
            sample_network.tensors[expected_largest_tensor_idx]
        )

        largest_tensor_idx, largest_memory = get_largest_tensor_in_network(sample_network)

        assert largest_tensor_idx == expected_largest_tensor_idx
        assert largest_memory == expected_largest_memory

    def test_empty_network_raises_assertion_error(self) -> None:
        """Test that an empty tensor network raises an AssertionError."""
        empty_network = TensorNetwork(
            input_indices=[],
            output_indices=[],
            shapes=[],
            size_dict={},
            tensor_arrays=None,
        )

        with pytest.raises(
            AssertionError, match="Tensor network must contain at least one tensor."
        ):
            get_largest_tensor_in_network(empty_network)


class TestLargestIntermediateTensorInContractionPath:
    """Tests for get_largest_intermediate_tensor_in_contraction_path behavior."""

    @pytest.mark.parametrize(
        "path",
        [
            [(1, 2), (0, 1)],
            PersistentContractionPath.from_contraction_path(
                TensorNetwork(
                    input_indices=[[0, 1], [1, 2], [2, 3]],
                    output_indices=[0, 3],
                    shapes=[(10, 10), (10, 2), (2, 2)],
                    size_dict={0: 10, 1: 10, 2: 2, 3: 2},
                    tensor_arrays=None,
                ),
                [(1, 2), (0, 1)],
            ),
        ],
    )
    def test_returns_minus_one_when_largest_tensor_is_initial(
        self, path: PersistentContractionPath | ContractionPath
    ) -> None:
        """Test largest tensor being one of the original tensors."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(10, 10), (10, 2), (2, 2)],
            size_dict={0: 10, 1: 10, 2: 2, 3: 2},
            tensor_arrays=None,
        )
        largest_idx, largest_memory = get_largest_intermediate_tensor_in_path(network, path)

        expected_memory = MemoryCalculator().calculate_memory_for_tensor(network.tensors[0])
        assert largest_idx == -1
        assert largest_memory == expected_memory

    @pytest.mark.parametrize(
        "path",
        [
            [(0, 1), (0, 1)],
            PersistentContractionPath.from_contraction_path(
                TensorNetwork(
                    input_indices=[[0, 1], [2, 3], [1, 2]],
                    output_indices=[0, 3],
                    shapes=[(10, 10), (8, 8), (10, 8)],
                    size_dict={0: 10, 1: 10, 2: 8, 3: 8},
                    tensor_arrays=None,
                ),
                [(0, 1), (0, 1)],
            ),
        ],
    )
    def test_returns_intermediate_step_when_largest_tensor_is_intermediate(
        self, path: PersistentContractionPath | ContractionPath
    ) -> None:
        """Test largest tensor being created at a non-final contraction step."""
        network = TensorNetwork(
            input_indices=[[0, 1], [2, 3], [1, 2]],
            output_indices=[0, 3],
            shapes=[(10, 10), (8, 8), (10, 8)],
            size_dict={0: 10, 1: 10, 2: 8, 3: 8},
            tensor_arrays=None,
        )

        largest_idx, largest_memory = get_largest_intermediate_tensor_in_path(network, path)

        after_first_contraction = contract_tensors_in_network(network, (0, 1))
        expected_memory = MemoryCalculator().calculate_memory_for_tensor(
            after_first_contraction.tensors[0]
        )
        assert largest_idx == 0
        assert largest_memory == expected_memory

    @pytest.mark.parametrize(
        "path",
        [
            [(0, 1), (0, 1)],
            PersistentContractionPath.from_contraction_path(
                TensorNetwork(
                    input_indices=[[0, 1], [1, 2], [3, 4]],
                    output_indices=[0, 2, 3, 4],
                    shapes=[(10, 2), (2, 10), (9, 9)],
                    size_dict={0: 10, 1: 2, 2: 10, 3: 9, 4: 9},
                    tensor_arrays=None,
                ),
                [(0, 1), (0, 1)],
            ),
        ],
    )
    def test_returns_last_step_when_largest_tensor_is_final_tensor(
        self, path: PersistentContractionPath | ContractionPath
    ) -> None:
        """Test largest tensor being the final tensor after all contractions."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [3, 4]],
            output_indices=[0, 2, 3, 4],
            shapes=[(10, 2), (2, 10), (9, 9)],
            size_dict={0: 10, 1: 2, 2: 10, 3: 9, 4: 9},
            tensor_arrays=None,
        )
        largest_idx, largest_memory = get_largest_intermediate_tensor_in_path(network, path)

        after_first_contraction = contract_tensors_in_network(network, (0, 1))
        after_second_contraction = contract_tensors_in_network(after_first_contraction, (0, 1))
        expected_memory = MemoryCalculator().calculate_memory_for_tensor(
            after_second_contraction.tensors[0]
        )
        assert largest_idx == 1
        assert largest_memory == expected_memory
