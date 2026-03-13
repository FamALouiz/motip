"""Tensor network builder."""

import cotengra as ctg
from numpy import inf, random

from tensor import Tensor
from tensor_network import TensorNetwork


class TensorNetworkBuilder:
    """Tensor network builder with a fluent interface for setting generation parameters."""

    UNSET = -inf

    def __init__(self):
        """Initialize builder with unset values."""
        self.__number_of_tensors = self.UNSET
        self.__number_of_output_indices = self.UNSET
        self.__average_number_of_indices_per_tensor = self.UNSET
        self.__number_of_inner_hyper_indices = self.UNSET
        self.__number_of_outer_hyper_indices = self.UNSET
        self.__min_dimension_size = self.UNSET
        self.__max_dimension_size = self.UNSET
        self.__seed = self.UNSET
        self.__generate_arrays = False

    def with_number_of_tensors(self, number_of_tensors: int) -> "TensorNetworkBuilder":
        """Set the number of tensors in the generated tensor network."""
        self.__number_of_tensors = number_of_tensors
        return self

    def with_average_number_of_indices_per_tensor(
        self, average_number_of_indices_per_tensor: int
    ) -> "TensorNetworkBuilder":
        """Set the average number of indices per tensor in the generated tensor network."""
        self.__average_number_of_indices_per_tensor = average_number_of_indices_per_tensor
        return self

    def with_number_of_output_indices(
        self, number_of_output_indices: int
    ) -> "TensorNetworkBuilder":
        """Set the number of output indices in the generated tensor network."""
        self.__number_of_output_indices = number_of_output_indices
        return self

    def with_number_of_inner_hyper_indices(
        self, number_of_inner_hyper_indices: int
    ) -> "TensorNetworkBuilder":
        """Set the number of inner hyper indices in the generated tensor network."""
        self.__number_of_inner_hyper_indices = number_of_inner_hyper_indices
        return self

    def with_number_of_outer_hyper_indices(
        self, number_of_outer_hyper_indices: int
    ) -> "TensorNetworkBuilder":
        """Set the number of outer hyper indices in the generated tensor network."""
        self.__number_of_outer_hyper_indices = number_of_outer_hyper_indices
        return self

    def with_min_dimension_size(self, min_dimension_size: int) -> "TensorNetworkBuilder":
        """Set the minimum dimension size of indices in the generated tensor network."""
        self.__min_dimension_size = min_dimension_size
        return self

    def with_max_dimension_size(self, max_dimension_size: int) -> "TensorNetworkBuilder":
        """Set the maximum dimension size of indices in the generated tensor network."""
        self.__max_dimension_size = max_dimension_size
        return self

    def with_seed(self, seed: int) -> "TensorNetworkBuilder":
        """Set the seed for random generation of the tensor network."""
        self.__seed = seed
        return self

    def with_generate_arrays(self) -> "TensorNetworkBuilder":
        """Set whether to generate random arrays for the tensor network."""
        self.__generate_arrays = True
        return self

    def __validate(self) -> None:
        self.__validate_number_of_tensors()
        self.__validate_average_number_of_indices_per_tensor()
        self.__validate_number_of_output_indices()
        self.__validate_number_of_inner_hyper_indices()
        self.__validate_number_of_outer_hyper_indices()
        self.__validate_min_dimension_size()
        self.__validate_max_dimension_size()
        self.__validate_seed()

    def __validate_number_of_tensors(self) -> None:
        if self.__number_of_tensors == self.UNSET:
            raise ValueError("Number of tensors must be set.")
        if self.__number_of_tensors <= 0:
            raise ValueError("Number of tensors must be positive.")
        assert self.__number_of_tensors > 0

    def __validate_average_number_of_indices_per_tensor(self) -> None:
        if self.__average_number_of_indices_per_tensor == self.UNSET:
            raise ValueError("Average number of indices per tensor must be set.")
        if self.__average_number_of_indices_per_tensor <= 0:
            raise ValueError("Average number of indices per tensor must be positive.")
        assert self.__average_number_of_indices_per_tensor > 0, ValueError(
            "Average number of indices per tensor must be positive."
        )

    def __validate_number_of_output_indices(self) -> None:
        if self.__number_of_output_indices != self.UNSET and self.__number_of_output_indices < 0:
            raise ValueError("Number of output indices must be non-negative.")

    def __validate_number_of_inner_hyper_indices(self) -> None:
        if (
            self.__number_of_inner_hyper_indices != self.UNSET
            and self.__number_of_inner_hyper_indices < 0
        ):
            raise ValueError("Number of inner hyper indices must be non-negative.")

    def __validate_number_of_outer_hyper_indices(self) -> None:
        if (
            self.__number_of_outer_hyper_indices != self.UNSET
            and self.__number_of_outer_hyper_indices < 0
        ):
            raise ValueError("Number of outer hyper indices must be non-negative.")

    def __validate_min_dimension_size(self) -> None:
        if self.__min_dimension_size == self.UNSET:
            return
        if self.__min_dimension_size <= 0:
            raise ValueError("Minimum dimension size must be positive.")

    def __validate_max_dimension_size(self) -> None:
        if self.__max_dimension_size != self.UNSET and self.__max_dimension_size <= 0:
            raise ValueError("Maximum dimension size must be positive.")

    def __validate_seed(self) -> None:
        if self.__seed != self.UNSET and self.__seed < 0:
            raise ValueError("Seed must be non-negative.")

    def build(self) -> TensorNetwork:
        """Build the tensor network with the specified parameters.

        Raises:
            ValueError: If any required parameter is not set or if any parameter is invalid.

        Returns:
            TensorNetwork: The generated tensor network.
        """
        self.__validate()

        generation_info = {
            "n": self.__number_of_tensors,
            "reg": self.__average_number_of_indices_per_tensor,
            "n_out": self.__number_of_output_indices,
            "n_hyper_in": self.__number_of_inner_hyper_indices,
            "n_hyper_out": self.__number_of_outer_hyper_indices,
            "d_min": self.__min_dimension_size,
            "d_max": self.__max_dimension_size,
            "seed": self.__seed if self.__seed != self.UNSET else 0,
        }
        generation_info = {k: v for k, v in generation_info.items() if v != self.UNSET}

        input_indices, output_indices, shapes, size_dict = ctg.utils.rand_equation(
            **generation_info,
        )

        assert isinstance(input_indices, list)
        assert isinstance(output_indices, list)
        assert isinstance(shapes, list)
        assert isinstance(size_dict, dict)

        if self.__generate_arrays:
            random.seed(self.__seed if self.__seed != self.UNSET else 0)
            self.__arrays = [random.rand(*shape) for shape in shapes]

        arrays = self.__arrays if self.__generate_arrays else [None] * len(input_indices)
        tensors = [
            Tensor(tensor_input_indices, shape, array)
            for tensor_input_indices, shape, array in zip(input_indices, shapes, arrays)
        ]

        resulting_tn = TensorNetwork(
            output_indices=output_indices,
            tensors=tensors,
            size_dict=size_dict,
        )

        return resulting_tn
