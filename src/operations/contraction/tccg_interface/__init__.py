"""TCCG interface for tensor contraction operations."""

import os

import numpy as np

from operations.contraction.tccg_interface.discoverer import TCCGDiscoverer
from operations.contraction.tccg_interface.generator import TCCGGenerator
from operations.contraction.tccg_interface.pybind_compiler import TCCGPyBind11Compiler
from operations.contraction.tccg_interface.runtime import TCCGRuntime
from tensor import Tensor


def execute_tccg_contraction(
    tensor_a: Tensor,
    tensor_b: Tensor,
    ordered_new_indices: list[int],
    new_tensor_shape: tuple[int, ...],
    tccg_impl_dir: str = "tccg_implementations",
) -> np.ndarray:
    """High-level API for TCCG contraction execution.

    Args:
        tensor_a: First input tensor array.
        tensor_b: Second input tensor array.
        ordered_new_indices: Indices for output tensor.
        new_tensor_shape: Shape of output tensor.
        tccg_impl_dir: Path to tccg_implementations directory.

    Returns:
        Output tensor array in Fortran order.
    """
    TCCGGenerator().generate_tccg_file(tensor_a, tensor_b, ordered_new_indices, tccg_impl_dir)

    discoverer = TCCGDiscoverer(tccg_impl_dir, tensor_a.array.dtype, tensor_b.array.dtype)
    fn_name, param_count, cpp_path, has_work = discoverer.discover()

    dtype_str = "float"  # if tensor_a.array.dtype == np.float32 else "double"
    compiler = TCCGPyBind11Compiler(cpp_path, fn_name, param_count, dtype_str, has_work)
    so_path = compiler.compile()

    runtime = TCCGRuntime(so_path, fn_name, param_count == 7, tensor_a.array.dtype)

    tensor_a_f = np.asfortranarray(tensor_a.array)
    tensor_b_f = np.asfortranarray(tensor_b.array)
    output = runtime.execute_contraction(
        tensor_a_f, tensor_b_f, ordered_new_indices, new_tensor_shape
    )

    os.rmdir(tccg_impl_dir)

    return output
