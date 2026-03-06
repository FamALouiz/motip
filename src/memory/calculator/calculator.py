"""Memory calculator for the motip package."""

from tensor_network import TensorNetwork
from memory import Memory
import math


class MemoryCalculator:
    """Memory calculator class.

    This class will be responsible for calculating the memory requirements of tensor network
    contractions.
    """

    __element_size_in_bytes: Memory = Memory(
        8
    )  # Default to 8 bytes (i.e. 64 bits) for double-precision floating-point numbers.

    @property
    def element_size_in_bytes(self) -> Memory:
        """Get the size of each element in the tensors in bytes."""
        return MemoryCalculator.__element_size_in_bytes

    def set_element_size(self, element_size: int | Memory) -> "MemoryCalculator":
        """Set the size of each element in the tensors.

        This method will set the size of each element in the tensors, which is necessary for
        calculating memory requirements. The default size is 8 bytes (i.e. 64 bits) for
        double-precision floating-point numbers.

        Args:
            element_size: The size of each element (in bytes).
        """
        if isinstance(element_size, int):
            element_size_in_bytes = Memory(element_size)
        else:
            element_size_in_bytes = element_size
        MemoryCalculator.__element_size_in_bytes = element_size_in_bytes
        return self

    def __calculate_intial_memory_requirements(self, network: TensorNetwork) -> Memory:
        """Calculate the initial memory requirements of a tensor network.

        This method will calculate the initial memory requirements of a tensor network by summing
        the sizes of all tensors in the network. This method assumes that all tensors are intially
        loaded in memory.

        Args:
            network: The tensor network for which to calculate memory requirements.

        Returns:
            The initial memory requirements.
        """
        total_elements = sum(math.prod(tensor_shape) for tensor_shape in network.shapes)
        return MemoryCalculator.__element_size_in_bytes * total_elements  # type: ignore

    def calculate_peak_memory(
        self, network: TensorNetwork, contraction_path: list[tuple[int, int]]
    ) -> Memory:
        """Calculate the peak memory requirements of a tensor network contraction.

        This method will calculate the memory requirements of a tensor network contraction by
        simulating the contraction process and keeping track of the intermediate tensor sizes. This
        method assumes that all tensors are intially loaded in memory and that the contraction
        process is performed without writing intermediate results to disk.
        The memory requirements are calculated based on the maximum size (i.e. the peak memory
        usage) during the contraction process.

        Args:
            network: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The memory requirements.
        """
        peak_memory = self.__calculate_intial_memory_requirements(network)

        return peak_memory

    def calculate_total_memory(
        self, network: TensorNetwork, contraction_path: list[tuple[int, int]]
    ) -> Memory:
        """Calculate the total memory requirements of a tensor network.

        This method will calculate the total memory requirements of a tensor network by summing
        the sizes of all tensors in the network. This method assumes that all tensors are intially
        loaded in memory and that no intermediate results are written to disk.

        Args:
            network: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The total memory requirements.
        """
        raise NotImplementedError("Memory calculation is not yet implemented.")

    def calculate_peak_memory_with_disk_writeback(
        self, network: TensorNetwork, contraction_path: list[tuple[int, int]]
    ) -> Memory:
        """Calculate the peak memory requirements of a tn contraction with disk writeback.

        This method will calculate the memory requirements of a tensor network contraction by
        simulating the contraction process and keeping track of the intermediate tensor sizes. This
        method assumes that all tensors are intially loaded in memory and that the contraction
        process is performed with writing intermediate results to disk when necessary. The memory
        requirements are calculated based on the maximum size (i.e. the peak memory usage) during
        the contraction process.

        Args:
            network: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The memory requirements.
        """
        raise NotImplementedError("Memory calculation is not yet implemented.")

    def calculate_total_memory_with_disk_writeback(
        self, network: TensorNetwork, contraction_path: list[tuple[int, int]]
    ) -> Memory:
        """Calculate the total memory requirements of a tn contraction with disk writeback.

        This method will calculate the total memory requirements of a tensor network by summing
        the sizes of all tensors in the network. This method assumes that all tensors are intially
        loaded in memory and that intermediate results are written to disk when necessary.

        Args:
            network: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The total memory requirements.
        """
        raise NotImplementedError("Memory calculation is not yet implemented.")
