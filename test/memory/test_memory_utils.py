"""Tests for the memory utilities."""

import pytest

from memory.calculator import MemoryCalculator
from memory.utils import get_largest_tensor_in_network, get_memory_from_string
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
    def test_from_string_valid_inputs(self, memory_str: str, expected_bytes: int):
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
    def test_from_string_invalid_inputs_raise_value_error(self, memory_str: str):
        """Test invalid memory strings raise ValueError."""
        with pytest.raises(ValueError):
            get_memory_from_string(memory_str)


class TestLargestTensorInNetwork:
    """Tests for get_largest_tensor_in_network behavior."""

    def test_largest_tensor_in_network(self, sample_network: TensorNetwork):
        """Test that the largest tensor in the network is correctly identified."""
        expected_largest_tensor_idx = 1
        expected_largest_memory = MemoryCalculator().calculate_memory_for_tensor(
            sample_network.tensors[expected_largest_tensor_idx]
        )

        largest_tensor_idx, largest_memory = get_largest_tensor_in_network(sample_network)

        assert largest_tensor_idx == expected_largest_tensor_idx
        assert largest_memory == expected_largest_memory

    def test_empty_network_raises_assertion_error(self):
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
