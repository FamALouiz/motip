"""TCCG runtime integration."""

import importlib.util

import numpy as np


class TCCGRuntime:
    """Loads and executes compiled TCCG kernels."""

    def __init__(self, so_path: str, fn_name: str, has_work: bool, dtype: np.dtype):
        """Initialize the runtime.

        Args:
            so_path: Path to compiled .so module.
            fn_name: Name of the TCCG function in the module.
            has_work: Whether function signature includes work_ parameter.
            dtype: NumPy dtype for the tensors.
        """
        self.so_path = so_path
        self.fn_name = fn_name
        self.has_work = has_work
        self.dtype = dtype
        self.module = None

    def _load_module(self) -> None:
        """Load compiled extension module dynamically."""
        if self.module is None:
            spec = importlib.util.spec_from_file_location("tccg_kernel", self.so_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec from {self.so_path}")
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def execute_contraction(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        ordered_new_indices: list[int],
        new_tensor_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Execute TCCG contraction with workspace management.

        Args:
            tensor_a: First input tensor in Fortran order.
            tensor_b: Second input tensor in Fortran order.
            ordered_new_indices: Indices for output tensor (unused).
            new_tensor_shape: Shape of output tensor.

        Returns:
            Output tensor array in Fortran order.
        """
        self._load_module()

        output = np.zeros(new_tensor_shape, dtype=self.dtype, order="F")
        alpha = self.dtype.type(1.0)
        beta = self.dtype.type(0.0)

        if self.has_work:
            work_buf = np.empty(0, dtype=self.dtype, order="F")
            workspace_bytes = getattr(self.module, self.fn_name)(
                tensor_a, tensor_b, output, alpha, beta, work_buf
            )

            if workspace_bytes > 0:
                work_buf = np.zeros(
                    workspace_bytes // self.dtype().itemsize, dtype=self.dtype, order="F"
                )
                getattr(self.module, self.fn_name)(
                    tensor_a, tensor_b, output, alpha, beta, work_buf
                )
        else:
            getattr(self.module, self.fn_name)(tensor_a, tensor_b, output, alpha, beta)

        return output
