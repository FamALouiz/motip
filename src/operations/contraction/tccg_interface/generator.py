"""TCCG cpp generator."""

import os
import subprocess
from pathlib import Path

import numpy as np

# ruff: noqa: E501


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
    def _clean_generated_artifacts(tccg_impl_dir: Path) -> None:
        for pattern in ["*.cpp", "*.so", "*.o", "*.tccg"]:
            for file in tccg_impl_dir.glob(pattern):
                if file.name not in {"ttgemmt.cpp", "ttgemmt.hpp"}:
                    file.unlink()

    @staticmethod
    def _infer_tccg_float_type(dtype: np.dtype) -> str:
        if dtype == np.float32:
            return "float"
        if dtype == np.float64:
            return "double"
        raise ValueError(f"Unsupported dtype for TCCG generation: {dtype}")

    @staticmethod
    def generate_tccg_file(
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        ordered_new_indices: list[int],
        tensor_a_indices: list[int] | None = None,
        tensor_b_indices: list[int] | None = None,
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
        TCCGGenerator._clean_generated_artifacts(implementation_dir)

        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"Tensor dtypes must match for TCCG generation: "
                f"{tensor_a.dtype} != {tensor_b.dtype}"
            )

        tensor_a_indices = (
            list(range(tensor_a.ndim)) if tensor_a_indices is None else tensor_a_indices
        )
        tensor_b_indices = (
            list(range(tensor_b.ndim)) if tensor_b_indices is None else tensor_b_indices
        )

        if len(tensor_a_indices) != tensor_a.ndim:
            raise ValueError("tensor_a_indices must match tensor_a.ndim")
        if len(tensor_b_indices) != tensor_b.ndim:
            raise ValueError("tensor_b_indices must match tensor_b.ndim")

        def convert_index(index: int) -> str:
            return TCCGGenerator._idx_to_char(index) if isinstance(index, int) else str(index)

        combined_size: dict[str, int] = {}
        for idx, size in zip(tensor_a_indices, tensor_a.shape, strict=True):
            combined_size[convert_index(idx)] = int(size)
        for idx, size in zip(tensor_b_indices, tensor_b.shape, strict=True):
            key = convert_index(idx)
            if key in combined_size and combined_size[key] != int(size):
                raise ValueError(
                    f"Mismatched sizes for index {idx}: {combined_size[key]} != {size}"
                )
            combined_size[key] = int(size)

        output_indices = [convert_index(idx) for idx in ordered_new_indices]
        input_a_indices = [convert_index(idx) for idx in tensor_a_indices]
        input_b_indices = [convert_index(idx) for idx in tensor_b_indices]

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
        subprocess.run(
            compiled_command,
            check=True,
        )
