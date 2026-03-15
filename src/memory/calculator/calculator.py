"""Memory calculator for the motip package."""

import math
from copy import deepcopy

from memory.memory import Memory
from tensor import Tensor
from tensor_network import ContractionPath, TensorNetwork
from tensor_network.utils.contraction import contract_pair


class MemoryCalculator:
    """Memory calculator class.

    This class will be responsible for calculating the memory requirements of tensor network
    contractions.
    """

    def __init__(self):
        """Initialize the memory calculator."""
        self.__element_size_in_bytes: Memory = Memory(
            8
        )  # Default to 8 bytes (i.e. 64 bits) for double-precision floating-point numbers.

    @property
    def element_size_in_bytes(self) -> Memory:
        """Get the size of each element in the tensors in bytes."""
        return self.__element_size_in_bytes

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
        self.__element_size_in_bytes = element_size_in_bytes
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
        total_memory = sum(
            self.calculate_memory_for_tensor(tensor).bytes for tensor in network.tensors
        )
        return Memory(total_memory)

    def __calculate_memory_for_contraction_pair(
        self, network: TensorNetwork, contraction_pair: tuple[int, int]
    ) -> Memory:
        """Calculate the memory requirements for a single contraction pair."""
        if contraction_pair[0] == contraction_pair[1]:
            raise ValueError("Contraction pair cannot consist of the same tensor index.")
        contracted_indices = set(network.input_indices[contraction_pair[0]]) & set(
            network.input_indices[contraction_pair[1]]
        )

        new_tensor_indices = (
            set(network.input_indices[contraction_pair[0]])
            | set(network.input_indices[contraction_pair[1]])
        ) - contracted_indices

        new_tensor_shape = tuple(network.size_dict[index] for index in new_tensor_indices)
        new_tensor_elements = math.prod(new_tensor_shape)

        return self.__element_size_in_bytes * new_tensor_elements

    def __calculate_memory_for_unused_tensors(
        self, network: TensorNetwork, contraction_pair: tuple[int, int]
    ) -> Memory:
        """Calculate the memory for tensors that are no longer used after a contraction."""
        total_memory_to_remove = sum(
            self.calculate_memory_for_tensor(network.tensors[tensor_idx]).bytes
            for tensor_idx in contraction_pair
        )

        return Memory(total_memory_to_remove)

    def calculate_peak_memory(self, tn: TensorNetwork, contraction_path: ContractionPath) -> Memory:
        """Calculate the peak memory requirements of a tensor network contraction.

        This method will calculate the memory requirements of a tensor network contraction by
        simulating the contraction process and keeping track of the intermediate tensor sizes. This
        method assumes that all tensors are intially loaded in memory and that the contraction
        process is performed without writing intermediate results to disk.
        The memory requirements are calculated based on the maximum size (i.e. the peak memory
        usage) during the contraction process.

        Args:
            tn: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The memory requirements.
        """
        network: TensorNetwork = deepcopy(tn)
        total_memory: Memory = self.__calculate_intial_memory_requirements(network)
        peak_memory: Memory = total_memory

        for contraction_pair in contraction_path:
            total_memory += self.__calculate_memory_for_contraction_pair(network, contraction_pair)
            peak_memory = max(peak_memory, total_memory)
            total_memory -= self.__calculate_memory_for_unused_tensors(network, contraction_pair)
            network = contract_pair(network, contraction_pair)

        return peak_memory

    def calculate_total_memory(
        self, tn: TensorNetwork, contraction_path: ContractionPath
    ) -> Memory:
        """Calculate the total memory requirements of a tensor network.

        This method will calculate the total memory requirements of a tensor network by summing
        the sizes of all tensors in the network. This method assumes that all tensors are intially
        loaded in memory and that no intermediate results are written to disk.

        Args:
            tn: The tensor network for which to calculate memory requirements.
            contraction_path: A list of pairs of tensor indices that are contracted together.

        Returns:
            The total memory requirements.
        """
        network: TensorNetwork = deepcopy(tn)
        total_memory: Memory = self.__calculate_intial_memory_requirements(network)

        for contraction_pair in contraction_path:
            total_memory += self.__calculate_memory_for_contraction_pair(network, contraction_pair)
            network = contract_pair(network, contraction_pair)

        return total_memory

    def calculate_peak_memory_with_disk_writeback(
        self, network: TensorNetwork, contraction_path: ContractionPath
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
        self, network: TensorNetwork, contraction_path: ContractionPath
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

    def calculate_memory_for_tensor(self, tensor: Tensor) -> Memory:
        """Calculate the memory requirements for a single tensor."""
        num_elements = math.prod(tensor.shape)
        return self.__element_size_in_bytes * num_elements
