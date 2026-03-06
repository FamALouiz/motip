"""Tests for the memory utilities."""

from memory.utils import get_memory_from_string
import pytest


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
