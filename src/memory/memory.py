"""Memory class for the motip package."""


class Memory:
    """Memory information in bytes."""

    __MEMORY_CONVERSION = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "EB": 1024**6,
        "ZB": 1024**7,
        "YB": 1024**8,
    }

    def __init__(self, bytes: int) -> None:
        """Initialize the memory."""
        self.__bytes = bytes

    def __str__(self) -> str:
        """Return a human-readable string representation of the memory."""
        for unit in reversed(self.__MEMORY_CONVERSION):
            if self.__bytes >= self.__MEMORY_CONVERSION[unit]:
                value = self.__bytes / self.__MEMORY_CONVERSION[unit]
                return f"{value:.2f} {unit}"
        return f"{self.__bytes} B"

    def to_bytes(self) -> int:
        """Return the memory in bytes."""
        return self.__bytes

    def to_kilobytes(self) -> float:
        """Return the memory in kilobytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["KB"]

    def to_megabytes(self) -> float:
        """Return the memory in megabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["MB"]

    def to_gigabytes(self) -> float:
        """Return the memory in gigabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["GB"]

    def to_terabytes(self) -> float:
        """Return the memory in terabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["TB"]

    def to_petabytes(self) -> float:
        """Return the memory in petabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["PB"]

    def to_exabytes(self) -> float:
        """Return the memory in exabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["EB"]

    def to_zettabytes(self) -> float:
        """Return the memory in zettabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["ZB"]

    def to_yottabytes(self) -> float:
        """Return the memory in yottabytes."""
        return self.__bytes / self.__MEMORY_CONVERSION["YB"]

    @staticmethod
    def from_string(memory_str: str) -> "Memory":
        """Create a Memory object from a string representation."""
        memory_str = memory_str.strip().upper()
        for unit in reversed(Memory.__MEMORY_CONVERSION):
            if memory_str.endswith(unit):
                value = float(memory_str[: -len(unit)].strip())
                bytes_value = int(value * Memory.__MEMORY_CONVERSION[unit])
                return Memory(bytes_value)
        raise ValueError(f"Invalid memory string: {memory_str}")
