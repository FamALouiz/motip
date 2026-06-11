"""TCCG discoverer."""

import re
from pathlib import Path

import numpy as np


class TCCGDiscoverer:
    """Discovers and parses TCCG-generated function signatures."""

    def __init__(self, tccg_impl_dir: str, tensor_a_dtype: np.dtype, tensor_b_dtype: np.dtype):
        """Initialize the discoverer with target directory and array dtypes.

        Args:
            tccg_impl_dir: Path to tccg_implementations directory.
            tensor_a_dtype: Data type of first tensor array.
            tensor_b_dtype: Data type of second tensor array.
        """
        self.tccg_impl_dir = Path(tccg_impl_dir)
        self.tensor_a_dtype = tensor_a_dtype
        self.tensor_b_dtype = tensor_b_dtype

    def _infer_tccg_float_type(self) -> str:
        """Infer TCCG float type from tensor dtypes.

        Returns:
            Either 'float' for float32 or 'double' for float64.

        Raises:
            ValueError: If tensors have different or unsupported dtypes.
        """
        if self.tensor_a_dtype != self.tensor_b_dtype:
            raise ValueError(
                f"Tensor dtypes must match. Got {self.tensor_a_dtype} and {self.tensor_b_dtype}."
            )

        if self.tensor_a_dtype == np.float32:
            return "float"
        elif self.tensor_a_dtype == np.float64:
            return "double"
        else:
            raise ValueError(f"Unsupported dtype for TCCG: {self.tensor_a_dtype}")

    def _parse_function_signature(self, cpp_content: str) -> tuple[str, int, bool]:
        """Parse generated .cpp to extract function name and parameter count.

        Args:
            cpp_content: Content of generated .cpp file.

        Returns:
            Tuple of (function_name, parameter_count).

        Raises:
            ValueError: If function signature cannot be parsed.
        """
        pattern = r"int\s+(\w+)\s*\((.*?)\)"
        pattern_for_gett = r"void\s+(\w+)\s*\((.*?)\)"
        match = re.search(pattern, cpp_content)
        match_with_gett = re.search(pattern_for_gett, cpp_content)
        if not match and not match_with_gett:
            raise ValueError("Could not parse function signature from generated .cpp")

        match = match if match else match_with_gett
        assert match

        fn_name = match.group(1)
        params_str = match.group(2)
        cleaned_params = [p for p in params_str.split(",") if p.strip()]
        param_count = len(cleaned_params)
        has_work = any("work_" in p for p in cleaned_params)
        return fn_name, param_count, has_work

    def discover(self) -> tuple[str, int, str, bool]:
        """Clean, generate TCCG, and discover function signature.

        Returns:
            Tuple of (function_name, parameter_count, cpp_path).
        """
        for cpp_file in self.tccg_impl_dir.glob("*.cpp"):
            with open(cpp_file, "r") as f:
                content = f.read()
                fn_name, param_count, has_work = self._parse_function_signature(content)
                return fn_name, param_count, str(cpp_file), has_work

        raise FileNotFoundError(
            f"No generated .cpp file found in {self.tccg_impl_dir}. "
            "Ensure _generate_tccg_file ran successfully."
        )
