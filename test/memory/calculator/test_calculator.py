"""Tests for the MemoryCalculator class."""

from copy import deepcopy

import pytest
from _pytest.fixtures import FixtureRequest

from memory import Memory
from memory.calculator import MemoryCalculator
from tensor import Tensor
from tensor_network import TensorNetwork


@pytest.fixture(params=[1, 4, 8])
def element_size(request: FixtureRequest):
    """Create different element sizes."""
    return request.param


@pytest.fixture(autouse=True)
def reset_element_size():
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
def sample_path() -> list[tuple[int, int]]:
    """Contraction path for the sample network."""
    return [(0, 1), (0, 1)]


class TestMemoryCalculatorElementSize:
    """Tests for element-size configuration behavior."""

    def test_default_element_size_is_8_bytes(self):
        """Test default element size."""
        calculator = MemoryCalculator()

        assert calculator.element_size_in_bytes == Memory(8)

    def test_set_element_size_with_int(self):
        """Test setting element size with int input."""
        calculator = MemoryCalculator()

        result = calculator.set_element_size(4)

        assert result is calculator
        assert calculator.element_size_in_bytes == Memory(4)

    def test_set_element_size_with_memory(self):
        """Test setting element size with Memory input."""
        calculator = MemoryCalculator()
        element_size = Memory(16)

        result = calculator.set_element_size(element_size)

        assert result is calculator
        assert calculator.element_size_in_bytes == element_size

    def test_set_element_size_negative_int_raises_value_error(self):
        """Test negative int element size fails."""
        calculator = MemoryCalculator()

        with pytest.raises(ValueError, match="Memory value cannot be negative"):
            calculator.set_element_size(-1)

    def test_element_size_is_not_shared_across_instances(self):
        """Test class-level element size is reflected by all instances."""
        calculator1 = MemoryCalculator().set_element_size(2)
        calculator2 = MemoryCalculator()

        assert calculator1.element_size_in_bytes == Memory(2)
        assert calculator2.element_size_in_bytes == Memory(8)


class TestMemoryCalculatorTensorMemory:
    """Tests for memory calculation of individual tensors."""

    def test_calculate_memory_for_tensor(self, element_size):
        """Test memory calculation for a single tensor."""
        calculator = MemoryCalculator().set_element_size(element_size)
        tensor = Tensor(input_indices=[0, 1], shape=(3, 4), array=None)

        memory = calculator.calculate_memory_for_tensor(tensor)

        expected_memory = Memory(4 * 3 * element_size)  # element size * num elements
        assert memory == expected_memory


class TestMemoryCalculatorPeakMemory:
    """Tests for peak-memory calculation."""

    def test_calculate_peak_memory_for_known_network(
        self, sample_network, sample_path, element_size
    ):
        """Test peak memory against hand-computed expected value."""
        calculator = MemoryCalculator().set_element_size(element_size)

        peak_memory = calculator.calculate_peak_memory(sample_network, sample_path)

        assert peak_memory == Memory(66) * element_size

    def test_calculate_peak_memory_empty_path_equals_initial_memory(
        self, sample_network, element_size
    ):
        """Test peak memory with no contractions."""
        calculator = MemoryCalculator().set_element_size(element_size)

        peak_memory = calculator.calculate_peak_memory(sample_network, [])

        assert peak_memory == Memory(56) * element_size

    def test_calculate_peak_memory_does_not_mutate_input_network(self, sample_network, sample_path):
        """Test peak memory calculation leaves input network unchanged."""
        calculator = MemoryCalculator().set_element_size(1)
        original = deepcopy(sample_network)

        _ = calculator.calculate_peak_memory(sample_network, sample_path)

        assert sample_network == original


class TestMemoryCalculatorTotalMemory:
    """Tests for total-memory calculation."""

    def test_calculate_total_memory_for_known_network(
        self, sample_network, sample_path, element_size
    ):
        """Test total memory against hand-computed expected value."""
        calculator = MemoryCalculator().set_element_size(element_size)

        total_memory = calculator.calculate_total_memory(sample_network, sample_path)

        assert total_memory == Memory(80) * element_size

    def test_calculate_total_memory_empty_path_equals_initial_memory(
        self, sample_network, element_size
    ):
        """Test total memory with no contractions."""
        calculator = MemoryCalculator().set_element_size(element_size)

        total_memory = calculator.calculate_total_memory(sample_network, [])

        assert total_memory == Memory(56) * element_size

    def test_calculate_total_memory_does_not_mutate_input_network(
        self, sample_network, sample_path
    ):
        """Test total memory calculation leaves input network unchanged."""
        calculator = MemoryCalculator().set_element_size(1)
        original = deepcopy(sample_network)

        _ = calculator.calculate_total_memory(sample_network, sample_path)

        assert sample_network == original


class TestMemoryCalculatorErrorHandling:
    """Tests for error propagation and unsupported behaviors."""

    @pytest.mark.parametrize("method_name", ["calculate_peak_memory", "calculate_total_memory"])
    def test_invalid_contraction_index_raises_index_error(self, sample_network, method_name: str):
        """Test out-of-range contraction indices raise errors."""
        calculator = MemoryCalculator()
        method = getattr(calculator, method_name)

        with pytest.raises(IndexError):
            method(sample_network, [(0, 3)])

    @pytest.mark.parametrize("method_name", ["calculate_peak_memory", "calculate_total_memory"])
    def test_missing_index_size_in_size_dict_raises_key_error(
        self, sample_network, method_name: str
    ):
        """Test missing index in size dict raises KeyError during contraction."""
        calculator = MemoryCalculator()
        malformed_network = TensorNetwork(
            input_indices=sample_network.input_indices,
            output_indices=sample_network.output_indices,
            shapes=sample_network.shapes,
            size_dict={0: 2, 1: 3, 2: 5},
            tensor_arrays=None,
        )
        method = getattr(calculator, method_name)

        with pytest.raises(KeyError):
            method(malformed_network, [(1, 2)])

    def test_repeated_tensor_index_in_pair_raises_value_error_peak_memory(self, sample_network):
        """Test invalid contraction pair with same tensor index for peak memory."""
        calculator = MemoryCalculator()
        with pytest.raises(ValueError):
            calculator.calculate_peak_memory(sample_network, [(0, 0)])

    def test_repeated_tensor_index_in_pair_raises_value_error_total_memory(self, sample_network):
        """Test invalid contraction pair with same tensor index for total memory."""
        calculator = MemoryCalculator()
        with pytest.raises(ValueError):
            calculator.calculate_total_memory(sample_network, [(0, 0)])

    def test_peak_memory_with_disk_writeback_not_implemented(self, sample_network, sample_path):
        """Test unsupported peak-memory writeback mode."""
        calculator = MemoryCalculator()

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            calculator.calculate_peak_memory_with_disk_writeback(sample_network, sample_path)

    def test_total_memory_with_disk_writeback_not_implemented(self, sample_network, sample_path):
        """Test unsupported total-memory writeback mode."""
        calculator = MemoryCalculator()

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            calculator.calculate_total_memory_with_disk_writeback(sample_network, sample_path)
