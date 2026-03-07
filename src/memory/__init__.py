"""Memory module for the motip package."""

from importlib import import_module

from memory.memory import Memory, MemorySizes

__all__ = ["Memory", "MemorySizes", "utils", "calculator"]


def __getattr__(name: str):
    """Lazily import submodules."""
    if name == "utils":
        return import_module("memory.utils")
    if name == "calculator":
        return import_module("memory.calculator")
    raise AttributeError(f"module 'memory' has no attribute '{name}'")
