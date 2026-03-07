"""Test the TensorNetworkBuilder class."""

import numpy as np
import pytest

from tensor_network import TensorNetwork
from tensor_network.builder import TensorNetworkBuilder


def make_required_builder() -> TensorNetworkBuilder:
    """Create a TensorNetworkBuilder with only the required parameters set."""
    return (
        TensorNetworkBuilder()
        .with_number_of_tensors(6)
        .with_average_number_of_indices_per_tensor(7)
    )


def assert_required_network_from_builder(network: TensorNetwork) -> None:
    """Assert that the required builder creates a valid tensor network."""
    assert isinstance(network, TensorNetwork)  # guard assertion
    assert len(network.input_indices) == 6

    total_indices = sum(len(tensor_input_indices) for tensor_input_indices in network.input_indices)
    expected_total = 6 * 7
    std_dev = np.sqrt(expected_total)
    assert (
        abs(total_indices - expected_total) <= std_dev
    )  # withing 1 std dev of expected total since it is not guaranteed


class TestWithNumberOfTensors:
    """Test the with_number_of_tensors method of TensorNetworkBuilder."""

    def test_default_unset_value_fails_validation(self):
        """Test that the default unset value for number of tensors fails validation."""
        builder = TensorNetworkBuilder().with_average_number_of_indices_per_tensor(2)

        with pytest.raises(ValueError, match="Number of tensors must be set."):
            builder.build()

    def test_set_value_succeeds(self):
        """Test that setting the number of tensors succeeds."""
        network = (
            TensorNetworkBuilder()
            .with_number_of_tensors(7)
            .with_average_number_of_indices_per_tensor(2)
            .build()
        )

        assert network.input_indices is not None
        assert len(network.input_indices) == 7

    @pytest.mark.parametrize("value", [0, -1])
    def test_validation(self, value: int):
        """Test that invalid values for number of tensors fail validation."""
        builder = (
            TensorNetworkBuilder()
            .with_number_of_tensors(value)
            .with_average_number_of_indices_per_tensor(2)
        )

        with pytest.raises(ValueError, match="Number of tensors must be positive."):
            builder.build()


class TestWithAverageNumberOfIndicesPerTensor:
    """Test the with_average_number_of_indices_per_tensor method of TensorNetworkBuilder."""

    def test_default_unset_value_fails_validation(self):
        """Test that the default unset value for average no of idxs per tensor fails validation."""
        builder = TensorNetworkBuilder().with_number_of_tensors(2)

        with pytest.raises(ValueError, match="Average number of indices per tensor must be set."):
            builder.build()

    def test_set_value_succeeds(self):
        """Test that setting the average number of indices per tensor succeeds."""
        network = (
            TensorNetworkBuilder()
            .with_number_of_tensors(3)
            .with_average_number_of_indices_per_tensor(4)
            .build()
        )

        total_indices = sum(
            len(tensor_input_indices) for tensor_input_indices in network.input_indices
        )

        assert network.input_indices is not None
        expected_total = 3 * 4
        std_dev = np.sqrt(expected_total)
        assert (
            abs(total_indices - expected_total) <= std_dev
        )  # withing 1 std dev of expected total since it is not guaranteed

    @pytest.mark.parametrize("value", [0, -1])
    def test_validation(self, value: int):
        """Test that invalid values for average number of indices per tensor fail validation."""
        builder = (
            TensorNetworkBuilder()
            .with_number_of_tensors(3)
            .with_average_number_of_indices_per_tensor(value)
        )

        with pytest.raises(
            ValueError, match="Average number of indices per tensor must be positive."
        ):
            builder.build()


class TestWithNumberOfOutputIndices:
    """Test the with_number_of_output_indices method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for number of output indices succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the number of output indices succeeds."""
        network = make_required_builder().with_number_of_output_indices(2).build()

        assert_required_network_from_builder(network)
        assert network.output_indices is not None
        assert len(network.output_indices) == 2

    def test_validation(self):
        """Test that invalid values for number of output indices fail validation."""
        builder = (
            make_required_builder()
            .with_number_of_inner_hyper_indices(0)
            .with_number_of_output_indices(-1)
        )

        with pytest.raises(ValueError):
            builder.build()

    def test_valid_output_indices_with_outer_hyper_indices(self):
        """Test that valid values for number of output indices with outer hyper indices succeed."""
        network = (
            make_required_builder()
            .with_number_of_output_indices(2)
            .with_number_of_outer_hyper_indices(3)
            .build()
        )

        assert_required_network_from_builder(network)
        assert network.output_indices is not None
        assert len(network.output_indices) == 5


class TestWithNumberOfInnerHyperIndices:
    """Test the with_number_of_inner_hyper_indices method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for number of inner hyper indices succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the number of inner hyper indices succeeds."""
        network = make_required_builder().with_number_of_inner_hyper_indices(6).build()

        assert network.output_indices is not None

        # Check how many tests have bonds to other tensors (i.e. are inner hyper indices)
        total_inner_hyper_indices = sum(1 for tensor in network.input_indices if len(tensor) > 0)
        assert total_inner_hyper_indices == 6

    @pytest.mark.parametrize("value", [-1])
    def test_validation(self, value: int):
        """Test that invalid values for number of inner hyper indices fail validation."""
        builder = make_required_builder().with_number_of_inner_hyper_indices(value)

        with pytest.raises(ValueError):
            builder.build()


class TestWithNumberOfOuterHyperIndices:
    """Test the with_number_of_outer_hyper_indices method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for number of outer hyper indices succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the number of outer hyper indices succeeds."""
        network = make_required_builder().with_number_of_outer_hyper_indices(3).build()

        assert network.output_indices is not None
        assert len(network.output_indices) == 3

    @pytest.mark.parametrize("value", [-1])
    def test_validation(self, value: int):
        """Test that invalid values for number of outer hyper indices fail validation."""
        builder = make_required_builder().with_number_of_outer_hyper_indices(value)

        with pytest.raises(ValueError):
            builder.build()


class TestWithMinDimensionSize:
    """Test the with_min_dimension_size method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for minimum dimension size succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the minimum dimension size succeeds."""
        network = make_required_builder().with_min_dimension_size(3).build()

        assert all(len(tensor) >= 3 for tensor in network.input_indices)

    @pytest.mark.parametrize("value", [0, -1])
    def test_validation(self, value: int):
        """Test that invalid values for minimum dimension size fail validation."""
        builder = make_required_builder().with_min_dimension_size(value)

        with pytest.raises(ValueError):
            builder.build()


class TestWithMaxDimensionSize:
    """Test the with_max_dimension_size method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for maximum dimension size succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the maximum dimension size succeeds."""
        network = make_required_builder().with_max_dimension_size(9).build()

        assert all(len(tensor) <= 9 for tensor in network.input_indices)

    @pytest.mark.parametrize("value", [0, -1])
    def test_validation(self, value: int):
        """Test that invalid values for maximum dimension size fail validation."""
        builder = make_required_builder().with_max_dimension_size(value)

        with pytest.raises(ValueError):
            builder.build()


class TestWithSeed:
    """Test the with_seed method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for seed succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

    def test_set_value_succeeds(self):
        """Test that setting the seed succeeds."""
        network = make_required_builder().with_seed(123).build()

        assert_required_network_from_builder(network)

    @pytest.mark.parametrize("value", [-1])
    def test_validation(self, value: int):
        """Test that invalid values for seed fail validation."""
        builder = make_required_builder().with_seed(value)

        with pytest.raises(ValueError):
            builder.build()

    def test_same_seed_produces_same_network(self):
        """Test that using the same seed produces the same network."""
        builder = make_required_builder().with_seed(123)
        network1 = builder.build()
        network2 = builder.build()

        assert network1 == network2


class TestWithGenerateArrays:
    """Test the with_generate_arrays method of TensorNetworkBuilder."""

    def test_default_unset_value_succeeds(self):
        """Test that the default unset value for generate arrays succeeds."""
        network = make_required_builder().build()

        assert_required_network_from_builder(network)

        with pytest.raises(ValueError, match="Arrays were not generated for this tensor network."):
            _ = network.arrays

    def test_set_value_succeeds(self):
        """Test that setting the generate arrays flag succeeds."""
        network = make_required_builder().with_generate_arrays().build()

        assert_required_network_from_builder(network)
        assert network.arrays is not None

    def test_arrays_are_consistent_with_shapes_and_size_dict(self):
        """Test that the generated arrays are consistent with the shapes and size dict."""
        network = make_required_builder().with_generate_arrays().build()

        assert network.arrays is not None
        assert len(network.arrays) == len(network.input_indices)

        for tensor_input_indices, shape, array in zip(
            network.input_indices, network.shapes, network.arrays
        ):
            expected_shape = tuple(network.size_dict[idx] for idx in tensor_input_indices)
            assert shape == expected_shape
            assert array.shape == expected_shape

    def test_same_seed_produces_same_arrays(self):
        """Test that using the same seed produces the same arrays."""
        builder = make_required_builder().with_generate_arrays().with_seed(123)
        network1 = builder.build()
        network2 = builder.build()

        assert len(network1.arrays) == len(network2.arrays)
        for a1, a2 in zip(network1.arrays, network2.arrays):
            assert (a1 == a2).all()


class TestOverallEdgeCases:
    """Test overall edge cases for TensorNetworkBuilder."""

    def test_build_succeeds_with_min_greater_than_max(self):
        """Test that building fails with dimension constraints that are contradictory."""
        builder = (
            TensorNetworkBuilder()
            .with_number_of_tensors(2)
            .with_average_number_of_indices_per_tensor(2)
            .with_min_dimension_size(5)
            .with_max_dimension_size(2)
        )

        with pytest.raises(ValueError):
            builder.build()

    @pytest.mark.parametrize(
        "configure",
        [
            lambda builder: (
                builder.with_number_of_tensors(2)
                .with_average_number_of_indices_per_tensor(2)
                .with_number_of_inner_hyper_indices(0)
                .with_number_of_output_indices(-1)
            ),
            lambda builder: (
                builder.with_number_of_tensors(2)
                .with_average_number_of_indices_per_tensor(2)
                .with_seed(-1)
                .with_min_dimension_size(2)
            ),
        ],
    )
    def test_build_fails_for_composed_invalid_values(self, configure):
        """Test that building fails when multiple invalid values are set together."""
        builder = TensorNetworkBuilder()
        configure(builder)

        with pytest.raises(ValueError):
            builder.build()

    def test_builder_succeeds(self):
        """Test that the builder succeeds with valid parameters."""
        network = (
            TensorNetworkBuilder()
            .with_number_of_tensors(4)
            .with_average_number_of_indices_per_tensor(3)
            .with_number_of_output_indices(2)
            .with_number_of_inner_hyper_indices(1)
            .with_number_of_outer_hyper_indices(1)
            .with_min_dimension_size(2)
            .with_max_dimension_size(5)
            .with_seed(42)
            .build()
        )

        assert isinstance(network, TensorNetwork)  # guard assertion
        assert len(network.input_indices) == 4
        assert len(network.output_indices) == 3  # 2 output + 1 outer hyper
        assert all(len(tensor) >= 2 for tensor in network.input_indices)
        assert all(len(tensor) <= 5 for tensor in network.input_indices)
