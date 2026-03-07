"""Memory class for the motip package."""

from dataclasses import dataclass
from enum import IntEnum


class MemorySizes(IntEnum):
    """Memory size units and their corresponding byte values."""

    B = 1
    KB = 1024
    MB = 1024**2
    GB = 1024**3
    TB = 1024**4
    PB = 1024**5
    EB = 1024**6
    ZB = 1024**7
    YB = 1024**8


@dataclass
class Memory:
    """Memory information in bytes."""

    bytes: int

    def __post_init__(self):
        """Validate the memory value."""
        if not isinstance(self.bytes, int):
            raise TypeError("Memory value must be an integer.")
        if self.bytes < 0:
            raise ValueError("Memory value cannot be negative.")

    def __str__(self) -> str:
        """Return a human-readable string representation of the memory."""
        for unit in reversed(MemorySizes):
            if self.bytes >= MemorySizes(unit):
                value = self.bytes / MemorySizes(unit)
                return f"{value:.2f} {unit.name}"
        return f"{self.bytes} B"

    def __add__(self, other: "Memory") -> "Memory":
        """Add two Memory instances."""
        if not isinstance(other, Memory):
            return NotImplemented
        return Memory(self.bytes + other.bytes)

    def __radd__(self, other: "Memory") -> "Memory":
        """Add two Memory instances (reflected)."""
        return self.__add__(other)

    def __sub__(self, other: "Memory") -> "Memory":
        """Subtract one Memory instance from another."""
        if not isinstance(other, Memory):
            return NotImplemented
        if self.bytes < other.bytes:
            raise ValueError("Resulting Memory cannot be negative.")
        return Memory(self.bytes - other.bytes)

    def __rsub__(self, other: "Memory") -> "Memory":
        """Subtract one Memory instance from another (reflected)."""
        if not isinstance(other, Memory):
            return NotImplemented
        if other.bytes < self.bytes:
            raise ValueError("Resulting Memory cannot be negative.")
        return Memory(other.bytes - self.bytes)

    def __mul__(self, other: int) -> "Memory":
        """Multiply Memory by an integer."""
        if not isinstance(other, int):
            return NotImplemented
        return Memory(self.bytes * other)

    def __rmul__(self, other: int) -> "Memory":
        """Multiply Memory by an integer (reflected)."""
        return self.__mul__(other)

    def __truediv__(self, other: int) -> "Memory":
        """Divide Memory by an integer."""
        if not isinstance(other, int):
            return NotImplemented
        return Memory(self.bytes // other)

    def __floordiv__(self, other: int) -> "Memory":
        """Floor divide Memory by an integer."""
        if not isinstance(other, int):
            return NotImplemented
        return Memory(self.bytes // other)

    @property
    def to_bytes(self) -> int:
        """Return the memory in bytes."""
        return self.bytes

    @property
    def to_kilobytes(self) -> float:
        """Return the memory in kilobytes."""
        return self.bytes / MemorySizes.KB

    @property
    def to_megabytes(self) -> float:
        """Return the memory in megabytes."""
        return self.bytes / MemorySizes.MB

    @property
    def to_gigabytes(self) -> float:
        """Return the memory in gigabytes."""
        return self.bytes / MemorySizes.GB

    @property
    def to_terabytes(self) -> float:
        """Return the memory in terabytes."""
        return self.bytes / MemorySizes.TB

    @property
    def to_petabytes(self) -> float:
        """Return the memory in petabytes."""
        return self.bytes / MemorySizes.PB

    @property
    def to_exabytes(self) -> float:
        """Return the memory in exabytes."""
        return self.bytes / MemorySizes.EB

    @property
    def to_zettabytes(self) -> float:
        """Return the memory in zettabytes."""
        return self.bytes / MemorySizes.ZB

    @property
    def to_yottabytes(self) -> float:
        """Return the memory in yottabytes."""
        return self.bytes / MemorySizes.YB
