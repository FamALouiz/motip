"""Contraction path file."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeAlias

from contraction.tensor_network import contract_tensors_in_network
from tensor_network import TensorNetwork

ContractionPath: TypeAlias = Sequence[tuple[int, int]]


@dataclass
class PersistentContractionPath:
    """Contraction path with all intermediate tensor network states."""

    path: ContractionPath
    history: list[TensorNetwork]

    def __post_init__(self) -> None:
        """Validate path and history consistency."""
        if len(self.history) != len(self.path) + 1:
            raise ValueError(
                "History must contain the initial state and one state per contraction."
            )

    @classmethod
    def from_contraction_path(
        cls, network: TensorNetwork, contraction_path: ContractionPath
    ) -> "PersistentContractionPath":
        """Create contraction history by simulating all contractions in order."""
        path = list(contraction_path)
        current_network = deepcopy(network)
        history = [current_network]

        for pair in path:
            current_network = contract_tensors_in_network(current_network, pair)
            history.append(current_network)

        return cls(path=path, history=history)

    def get_state(self, step: int) -> TensorNetwork:
        """Get the tensor network state at a given step.

        Args:
            step: The step index to retrieve the state for. Must be between 0 and
            num_steps (inclusive). Step 0 corresponds to the initial state before any contractions.

        Returns:
            The tensor network state at the given step.

        Raises:
            IndexError: If the step index is out of range.
        """
        if step < 0 or step >= len(self.history):
            raise IndexError("Step out of range.")
        return deepcopy(self.history[step])

    @property
    def initial_state(self) -> TensorNetwork:
        """Initial tensor network before any contraction."""
        return self.get_state(0)

    @property
    def final_state(self) -> TensorNetwork:
        """Final tensor network after all contractions."""
        return self.get_state(len(self.history) - 1)

    @property
    def num_steps(self) -> int:
        """Number of contraction steps."""
        return len(self.path)
