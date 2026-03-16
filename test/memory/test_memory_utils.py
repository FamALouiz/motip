"""Tests for the memory utilities."""

import pytest

from contraction.tensor_network import contract_tensors_in_network
from memory.calculator import MemoryCalculator
from memory.utils import (
    get_largest_intermediate_tensor_in_contraction_path,
    get_largest_tensor_in_network,
    get_memory_from_string,
)
from tensor_network import TensorNetwork
from tensor_network.utils.contraction import contract_pair_of_tensors_in_network


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


class TestMemoryFromString:
    """Tests for Memory.from_string parsing behavior."""

    @pytest.mark.parametrize(
        ("memory_str", "expected_bytes"),
        [
            ("1B", 1),
            ("1 KB", 1024),
            ("1.5 MB", int(1.5 * 1024**2)),
            ("2 gb", 2 * 1024**3),
            (" 3 TB ", 3 * 1024**4),
            ("0.5 pb", int(0.5 * 1024**5)),
            ("1 EB", 1024**6),
            ("1 ZB", 1024**7),
            ("1 YB", 1024**8),
        ],
    )
    def test_from_string_valid_inputs(self, memory_str: str, expected_bytes: int) -> None:
        """Test valid memory string parsing for all units."""
        memory = get_memory_from_string(memory_str)

        assert memory.to_bytes == expected_bytes

    @pytest.mark.parametrize(
        "memory_str",
        [
            "",
            "   ",
            "1024",
            "abc",
            "1 XB",
            "1.2.3 MB",
        ],
    )
    def test_from_string_invalid_inputs_raise_value_error(self, memory_str: str) -> None:
        """Test invalid memory strings raise ValueError."""
        with pytest.raises(ValueError):
            get_memory_from_string(memory_str)


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

    def test_returns_minus_one_when_largest_tensor_is_initial(self) -> None:
        """Test largest tensor being one of the original tensors."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [2, 3]],
            output_indices=[0, 3],
            shapes=[(10, 10), (10, 2), (2, 2)],
            size_dict={0: 10, 1: 10, 2: 2, 3: 2},
            tensor_arrays=None,
        )
        path = [(1, 2), (0, 1)]

        largest_idx, largest_memory = get_largest_intermediate_tensor_in_contraction_path(
            network, path
        )

        expected_memory = MemoryCalculator().calculate_memory_for_tensor(network.tensors[0])
        assert largest_idx == -1
        assert largest_memory == expected_memory

    def test_returns_intermediate_step_when_largest_tensor_is_intermediate(self) -> None:
        """Test largest tensor being created at a non-final contraction step."""
        network = TensorNetwork(
            input_indices=[[0, 1], [2, 3], [1, 2]],
            output_indices=[0, 3],
            shapes=[(10, 10), (8, 8), (10, 8)],
            size_dict={0: 10, 1: 10, 2: 8, 3: 8},
            tensor_arrays=None,
        )
        path = [(0, 1), (0, 1)]

        largest_idx, largest_memory = get_largest_intermediate_tensor_in_contraction_path(
            network, path
        )

        after_first_contraction = contract_tensors_in_network(network, path[0])
        expected_memory = MemoryCalculator().calculate_memory_for_tensor(
            after_first_contraction.tensors[0]
        )
        assert largest_idx == 0
        assert largest_memory == expected_memory

    def test_returns_last_step_when_largest_tensor_is_final_tensor(self) -> None:
        """Test largest tensor being the final tensor after all contractions."""
        network = TensorNetwork(
            input_indices=[[0, 1], [1, 2], [3, 4]],
            output_indices=[0, 2, 3, 4],
            shapes=[(10, 2), (2, 10), (9, 9)],
            size_dict={0: 10, 1: 2, 2: 10, 3: 9, 4: 9},
            tensor_arrays=None,
        )
        path = [(0, 1), (0, 1)]

        largest_idx, largest_memory = get_largest_intermediate_tensor_in_contraction_path(
            network, path
        )

        after_first_contraction = contract_tensors_in_network(network, path[0])
        after_second_contraction = contract_tensors_in_network(after_first_contraction, path[1])
        expected_memory = MemoryCalculator().calculate_memory_for_tensor(
            after_second_contraction.tensors[0]
        )
        assert largest_idx == 1
        assert largest_memory == expected_memory
