"""Tensor network module."""

from importlib import import_module

from tensor_network.tn import ContractionPath, TensorNetwork

__all__ = ["TensorNetwork", "ContractionPath", "builder"]


def __getattr__(name: str):
    """Lazily import submodules."""
    if name == "builder":
        return import_module("tensor_network.builder")
    raise AttributeError(f"module 'tensor_network' has no attribute '{name}'")
