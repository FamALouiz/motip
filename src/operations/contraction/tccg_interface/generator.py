"""TCCG cpp generator."""

import os
import subprocess
from importlib import util
from pathlib import Path

import numpy as np

from tensor import Tensor

# ruff: noqa: E501
if not util.find_spec("tccg"):
    raise EnvironmentError("tccg must be installed and TCCG_ROOT should be set")


class TCCGGenerator:
    """Generates TCCG .cpp files for given tensor contraction patterns."""

    @staticmethod
    def _idx_to_char(n: int) -> str:
        result = ""
        n += 1
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(ord("A") + remainder) + result
        return result.lower()

    @staticmethod
    def _infer_tccg_float_type(dtype: np.dtype) -> str:
        if dtype == np.float32:
            return "float"
        if dtype == np.float64:
            return "double"
        raise ValueError(f"Unsupported dtype for TCCG generation: {dtype}")

    @staticmethod
    def _convert_index(index: int) -> str:
        return TCCGGenerator._idx_to_char(index) if isinstance(index, int) else str(index)

    @staticmethod
    def generate_tccg_file(
        tensor_a: Tensor,
        tensor_b: Tensor,
        ordered_new_indices: list[int],
        tccg_impl_dir: str = "tccg_implementations",
    ) -> None:
        """Generate TCCG .cpp file for the given contraction pattern.

        Args:
            tensor_a: First input tensor array.
            tensor_b: Second input tensor array.
            ordered_new_indices: Indices for output tensor.
            tensor_a_indices: Input indices for tensor_a.
            tensor_b_indices: Input indices for tensor_b.
            tccg_impl_dir: Path to tccg_implementations directory.
        """
        implementation_dir = Path(tccg_impl_dir)
        implementation_dir.mkdir(parents=True, exist_ok=True)

        if tensor_a.array.dtype != tensor_b.array.dtype:
            raise ValueError(
                f"Tensor dtypes must match for TCCG generation: "
                f"{tensor_a.array.dtype} != {tensor_b.array.dtype}"
            )

        combined_size: dict[str, int] = {}
        for idx, size in zip(tensor_a.input_indices, tensor_a.shape, strict=True):
            combined_size[TCCGGenerator._convert_index(idx)] = int(size)
        for idx, size in zip(tensor_b.input_indices, tensor_b.shape, strict=True):
            key = TCCGGenerator._convert_index(idx)
            if key in combined_size and combined_size[key] != int(size):
                raise ValueError(
                    f"Mismatched sizes for index {idx}: {combined_size[key]} != {size}"
                )
            combined_size[key] = int(size)

        output_indices = [TCCGGenerator._convert_index(idx) for idx in ordered_new_indices]
        input_a_indices = [TCCGGenerator._convert_index(idx) for idx in tensor_a.input_indices]
        input_b_indices = [TCCGGenerator._convert_index(idx) for idx in tensor_b.input_indices]

        tccg_input_path = implementation_dir / "tccg_input.tccg"
        with open(tccg_input_path, "w") as handle:
            handle.write(
                f"C[{','.join(output_indices)}] = A[{','.join(input_a_indices)}] * B[{','.join(input_b_indices)}]\n"
            )
            for key, value in combined_size.items():
                handle.write(f"{key} = {value}\n")

        compiled_command = [
            "tccg",
            str(tccg_input_path),
            "--noLoG",
            "--compiler",
            "g++",
            "--maxImplementations",
            "1",
            "--numThreads",
            str(os.cpu_count()),
            "--verbose",
        ]
        subprocess.run(compiled_command, check=True, capture_output=True)
