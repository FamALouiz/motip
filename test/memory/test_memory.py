"""Tests for the Memory class."""

import pytest

from memory import Memory


class TestMemory:
    """Tests for the Memory class."""

    def test_init(self):
        """Test initialization of the Memory class."""
        memory = Memory(1024)
        assert memory.to_bytes == 1024

    def test_init_negative(self):
        """Test initialization with a negative value."""
        with pytest.raises(ValueError):
            Memory(-1)

    def test_init_zero(self):
        """Test initialization with zero."""
        memory = Memory(0)
        assert memory.to_bytes == 0


class TestMemoryConversions:
    """Tests for numeric conversions in the Memory class."""

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0),
            (1, 1),
            (1024, 1024),
            (10 * 1024**3, 10 * 1024**3),
        ],
    )
    def test_to_bytes(self, bytes_value: int, expected: int):
        """Test conversion to bytes."""
        memory = Memory(bytes_value)

        assert memory.to_bytes == expected

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1, 1 / 1024),
            (1024, 1.0),
            (1536, 1.5),
        ],
    )
    def test_to_kilobytes(self, bytes_value: int, expected: float):
        """Test conversion to kilobytes."""
        memory = Memory(bytes_value)

        assert memory.to_kilobytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**2, 1.0),
            (3 * 1024**2, 3.0),
            (1024**3, 1024.0),
        ],
    )
    def test_to_megabytes(self, bytes_value: int, expected: float):
        """Test conversion to megabytes."""
        memory = Memory(bytes_value)

        assert memory.to_megabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**3, 1.0),
            (5 * 1024**3, 5.0),
            (1024**4, 1024.0),
        ],
    )
    def test_to_gigabytes(self, bytes_value: int, expected: float):
        """Test conversion to gigabytes."""
        memory = Memory(bytes_value)

        assert memory.to_gigabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**4, 1.0),
            (7 * 1024**4, 7.0),
            (1024**5, 1024.0),
        ],
    )
    def test_to_terabytes(self, bytes_value: int, expected: float):
        """Test conversion to terabytes."""
        memory = Memory(bytes_value)

        assert memory.to_terabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**5, 1.0),
            (2 * 1024**5, 2.0),
            (1024**6, 1024.0),
        ],
    )
    def test_to_petabytes(self, bytes_value: int, expected: float):
        """Test conversion to petabytes."""
        memory = Memory(bytes_value)

        assert memory.to_petabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**6, 1.0),
            (9 * 1024**6, 9.0),
            (1024**7, 1024.0),
        ],
    )
    def test_to_exabytes(self, bytes_value: int, expected: float):
        """Test conversion to exabytes."""
        memory = Memory(bytes_value)

        assert memory.to_exabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**7, 1.0),
            (3 * 1024**7, 3.0),
            (1024**8, 1024.0),
        ],
    )
    def test_to_zettabytes(self, bytes_value: int, expected: float):
        """Test conversion to zettabytes."""
        memory = Memory(bytes_value)

        assert memory.to_zettabytes == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, 0.0),
            (1024**8, 1.0),
            (11 * 1024**8, 11.0),
        ],
    )
    def test_to_yottabytes(self, bytes_value: int, expected: float):
        """Test conversion to yottabytes."""
        memory = Memory(bytes_value)

        assert memory.to_yottabytes == pytest.approx(expected)


class TestMemoryStringRepresentation:
    """Tests for string representation of Memory."""

    @pytest.mark.parametrize(
        ("bytes_value", "expected"),
        [
            (0, "0 B"),
            (1, "1.00 B"),
            (1023, "1023.00 B"),
            (1024, "1.00 KB"),
            (1536, "1.50 KB"),
            (1024**2, "1.00 MB"),
            (1024**3, "1.00 GB"),
            (1024**4, "1.00 TB"),
            (1024**5, "1.00 PB"),
            (1024**6, "1.00 EB"),
            (1024**7, "1.00 ZB"),
            (1024**8, "1.00 YB"),
        ],
    )
    def test_str_representation(self, bytes_value: int, expected: str):
        """Test string representation for values across all units."""
        memory = Memory(bytes_value)

        assert str(memory) == expected


class TestMemoryOperations:
    """Tests for arithmetic operations on Memory."""

    @pytest.mark.parametrize(
        ("memory1", "memory2", "expected"),
        [
            (Memory(1024), Memory(2048), Memory(3072)),
            (Memory(0), Memory(1024), Memory(1024)),
            (Memory(512), Memory(512), Memory(1024)),
        ],
    )
    def test_addition(self, memory1: Memory, memory2: Memory, expected: Memory):
        """Test addition of two Memory instances."""
        result = memory1 + memory2
        result_rev = memory2 + memory1

        assert result == expected
        assert result_rev == expected

    @pytest.mark.parametrize(
        ("memory", "factor", "expected"),
        [
            (Memory(1024), 3, Memory(3072)),
            (Memory(0), 5, Memory(0)),
            (Memory(512), 2, Memory(1024)),
        ],
    )
    def test_multiplication(self, memory: Memory, factor: int, expected: Memory):
        """Test multiplication of Memory by an integer."""
        result = memory * factor
        result_rev = factor * memory

        assert result == expected
        assert result_rev == expected

    @pytest.mark.parametrize(
        ("memory", "divisor", "expected"),
        [
            (Memory(3072), 3, Memory(1024)),
            (Memory(1025), 2, Memory(512)),
            (Memory(0), 5, Memory(0)),
        ],
    )
    def test_true_division(self, memory: Memory, divisor: int, expected: Memory):
        """Test true division of Memory by an integer."""
        result = memory / divisor

        assert result == expected

    @pytest.mark.parametrize(
        ("memory", "divisor", "expected"),
        [
            (Memory(3072), 3, Memory(1024)),
            (Memory(1025), 2, Memory(512)),
            (Memory(0), 5, Memory(0)),
        ],
    )
    def test_floor_division(self, memory: Memory, divisor: int, expected: Memory):
        """Test floor division of Memory by an integer."""
        result = memory // divisor

        assert result == expected

    @pytest.mark.parametrize(
        ("memory", "divisor"),
        [
            (Memory(1024), 0),
            (Memory(1), 0),
        ],
    )
    def test_division_by_zero_raises(self, memory: Memory, divisor: int):
        """Test that division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            _ = memory / divisor

        with pytest.raises(ZeroDivisionError):
            _ = memory // divisor
