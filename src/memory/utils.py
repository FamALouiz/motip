"""Utility functions for memory operations."""

from memory import Memory, MemorySizes


def get_memory_from_string(memory_str: str) -> Memory:
    """Create a Memory object from a string representation."""
    memory_str = memory_str.strip().upper()
    for unit in reversed(MemorySizes):
        if memory_str.endswith(unit.name):
            value = float(memory_str[: -len(unit.name)].strip())
            bytes_value = int(value * unit)
            return Memory(bytes_value)
    raise ValueError(f"Invalid memory string: {memory_str}")
