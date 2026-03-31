"""Tests for the MemoryCalculator class."""

from collections.abc import Iterator

import pytest
from _pytest.fixtures import FixtureRequest

from contraction.path import ContractionPath
from memory import Memory
from memory.calculator import MemoryCalculator
from tensor import Tensor
from tensor_network import TensorNetwork
from tensor_network.tn import _TensorPool


@pytest.fixture(params=[1, 4, 8])
def element_size(request: FixtureRequest) -> int:
    """Create different element sizes."""
    return request.param


@pytest.fixture(autouse=True)
def reset_element_size() -> Iterator[None]:
    """Reset element size between tests."""
    MemoryCalculator().set_element_size(8)
    yield
    MemoryCalculator().set_element_size(8)


@pytest.fixture
def sample_network() -> TensorNetwork:
    """Create a deterministic tensor network with known contraction outcomes."""
    return TensorNetwork(
        input_indices=[[0, 1], [1, 2], [2, 3]],
        output_indices=[0, 3],
        shapes=[(2, 3), (3, 5), (5, 7)],
        size_dict={0: 2, 1: 3, 2: 5, 3: 7},
        tensor_arrays=None,
    )


@pytest.fixture
def sample_path() -> ContractionPath:
    """Contraction path for the sample network."""
    return [(0, 1), (0, 1)]


class TestMemoryCalculatorElementSize:
    """Tests for element-size configuration behavior."""

    def test_default_element_size_is_8_bytes(self) -> None:
        """Test default element size."""
        calculator = MemoryCalculator()

        assert calculator.element_size_in_bytes == Memory(8)

    def test_set_element_size_with_int(self) -> None:
        """Test setting element size with int input."""
        calculator = MemoryCalculator()

        result = calculator.set_element_size(4)

        assert result is calculator
        assert calculator.element_size_in_bytes == Memory(4)

    def test_set_element_size_with_memory(self) -> None:
        """Test setting element size with Memory input."""
        calculator = MemoryCalculator()
        element_size = Memory(16)

        result = calculator.set_element_size(element_size)

        assert result is calculator
        assert calculator.element_size_in_bytes == element_size

    def test_set_element_size_negative_int_raises_value_error(self) -> None:
        """Test negative int element size fails."""
        calculator = MemoryCalculator()

        with pytest.raises(ValueError, match="Memory value cannot be negative"):
            calculator.set_element_size(-1)

    def test_element_size_is_not_shared_across_instances(self) -> None:
        """Test class-level element size is reflected by all instances."""
        calculator1 = MemoryCalculator().set_element_size(2)
        calculator2 = MemoryCalculator()

        assert calculator1.element_size_in_bytes == Memory(2)
        assert calculator2.element_size_in_bytes == Memory(8)


class TestMemoryCalculatorTensorMemory:
    """Tests for memory calculation of individual tensors."""

    def test_calculate_memory_for_tensor(self, element_size: int) -> None:
        """Test memory calculation for a single tensor."""
        calculator = MemoryCalculator().set_element_size(element_size)
        tensor = Tensor(input_indices=[0, 1], shape=(3, 4), array=None)

        memory = calculator.calculate_memory_for_tensor(tensor)

        expected_memory = Memory(4 * 3 * element_size)  # element size * num elements
        assert memory == expected_memory

    def test_calculate_memory_for_tensor_with_zero_elements(self, element_size: int) -> None:
        """Test memory calculation for a tensor with zero elements."""
        calculator = MemoryCalculator().set_element_size(element_size)
        tensor = Tensor(input_indices=[0], shape=(0,), array=None)

        memory = calculator.calculate_memory_for_tensor(tensor)

        assert memory == Memory(0)

    @pytest.mark.parametrize(
        "tensors",
        (
            [
                Tensor(input_indices=[0, 1], shape=(3, 4), array=None),
                Tensor(input_indices=[2], shape=(5,), array=None),
            ],
            _TensorPool(
                [
                    Tensor(input_indices=[0, 1], shape=(3, 4), array=None),
                    Tensor(input_indices=[2], shape=(5,), array=None),
                ]
            ),
        ),
    )
    def test_calculate_memory_for_tensors(
        self, tensors: list[Tensor] | _TensorPool, element_size: int
    ) -> None:
        """Test memory calculation for a list of tensors."""
        calculator = MemoryCalculator().set_element_size(element_size)

        total_memory = calculator.calculate_memory_for_tensors(tensors)

        expected_memory = Memory(
            4 * 3 * element_size + 5 * element_size
        )  # sum of individual tensor memories
        assert total_memory == expected_memory

    def test_calculate_memory_for_contraction(self, element_size: int) -> None:
        """Test memory calculation for contracting two tensors."""
        calculator = MemoryCalculator().set_element_size(element_size)
        tensor_a = Tensor(input_indices=[0, 1], shape=(3, 4), array=None)
        tensor_b = Tensor(input_indices=[1, 2], shape=(4, 5), array=None)

        contraction_memory = calculator.calculate_memory_for_contraction(tensor_a, tensor_b)

        expected_memory = Memory(3 * 5 * element_size)  # memory of the resulting tensor
        assert contraction_memory == expected_memory

    def test_memory_for_contraction_with_no_contracted_indices(self, element_size: int) -> None:
        """Test memory calculation for contracting tensors with no shared indices."""
        calculator = MemoryCalculator().set_element_size(element_size)
        tensor_a = Tensor(input_indices=[0], shape=(3,), array=None)
        tensor_b = Tensor(input_indices=[1], shape=(4,), array=None)

        contraction_memory = calculator.calculate_memory_for_contraction(tensor_a, tensor_b)

        expected_memory = Memory(3 * 4 * element_size)  # memory of the resulting tensor
        assert contraction_memory == expected_memory
