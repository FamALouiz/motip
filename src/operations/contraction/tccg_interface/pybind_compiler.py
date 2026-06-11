"""TCCG pybind11 wrapper compiler."""

import shutil
import subprocess
import tempfile
from pathlib import Path

# ruff: noqa: E501


class TCCGPyBind11Compiler:
    """Generates and compiles pybind11 wrapper for TCCG functions."""

    def __init__(self, cpp_path: str, fn_name: str, param_count: int, dtype_str: str):
        """Initialize the compiler.

        Args:
            cpp_path: Path to generated TCCG .cpp file.
            fn_name: Name of the generated TCCG function.
            param_count: Number of parameters in function signature.
            dtype_str: Either 'float' or 'double'.
        """
        self.cpp_path = Path(cpp_path)
        self.fn_name = fn_name
        self.param_count = param_count
        self.dtype_str = dtype_str
        self.has_work = param_count == 7

    def _generate_wrapper_source(self) -> str:
        """Generate pybind11 wrapper source code.

        Returns:
            C++ source code for pybind11 wrapper.
        """
        wrapper_source = f"""#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define py pybind11 

extern "C" int {self.fn_name}(
    const {self.dtype_str} *A,
    const {self.dtype_str} *B,
    {self.dtype_str} *C,
    {self.dtype_str} alpha,
    {self.dtype_str} beta"""

        if self.has_work:
            wrapper_source += f",\n    {self.dtype_str} *work_"
        wrapper_source += ");\n\n"

        wrapper_source += f"""using MatrixType = py::array_t<{self.dtype_str}, py::array::f_style | py::array::forcecast>;

int {self.fn_name}_wrapper(MatrixType A, MatrixType B, MatrixType C, {self.dtype_str} alpha, {self.dtype_str} beta"""

        if self.has_work:
            wrapper_source += ", py::array_t<char> work_buf"
        wrapper_source += f""") {{
    auto A_buf = A.request();
    auto B_buf = B.request();
    auto C_buf = C.request();

    {self.dtype_str} *A_ptr = static_cast<{self.dtype_str}*>(A_buf.ptr);
    {self.dtype_str} *B_ptr = static_cast<{self.dtype_str}*>(B_buf.ptr);
    {self.dtype_str} *C_ptr = static_cast<{self.dtype_str}*>(C_buf.ptr);

"""

        if self.has_work:
            wrapper_source += f"""    {self.dtype_str} *work_ptr = nullptr;
    if (work_buf.size() > 0) {{
        auto work_buf_req = work_buf.request();
        work_ptr = static_cast<{self.dtype_str}*>(work_buf_req.ptr);
    }}
    return {self.fn_name}(A_ptr, B_ptr, C_ptr, alpha, beta, work_ptr);"""
        else:
            wrapper_source += f"    return {self.fn_name}(A_ptr, B_ptr, C_ptr, alpha, beta);"

        wrapper_source += f"""
}}

PYBIND11_MODULE(tccg_kernel, m) {{
    m.def("{self.fn_name}", &{self.fn_name}_wrapper, "TCCG kernel contraction");
}}
"""
        return wrapper_source

    def compile(self) -> str:
        """Compile the wrapper and generate .so module.

        Returns:
            Path to compiled .so module.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wrapper_cpp = temp_path / "wrapper.cpp"
            wrapper_cpp.write_text(self._generate_wrapper_source())

            project_root = self.cpp_path.parent.parent.parent.parent
            hptt_include = project_root / "libs" / "hptt" / "include"
            blis_include = project_root / "libs" / "blis" / "include"
            hptt_lib = project_root / "libs" / "hptt" / "lib"
            blis_lib = project_root / "libs" / "blis" / "lib"

            so_path = self.cpp_path.parent / "tccg_kernel.so"

            pybind11_includes = subprocess.check_output(
                ["python", "-m", "pybind11", "--includes"],
                text=True,
            ).strip()

            python_ext_suffix = subprocess.check_output(
                ["python", "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
                text=True,
            ).strip()

            so_base_name = f"tccg_kernel{python_ext_suffix}"
            so_temp_path = temp_path / so_base_name

            compile_cmd = [
                "g++",
                "-O3",
                "-fPIC",
                "-static",
                str(wrapper_cpp),
                str(self.cpp_path),
                *pybind11_includes.split(" "),
                f"-I{hptt_include}",
                f"-I{blis_include}",
                f"-L{hptt_lib}",
                f"-L{blis_lib}",
                "-lhptt",
                "-lblis",
                "-o",
                str(so_temp_path),
            ]
            subprocess.run(compile_cmd, check=True)

            shutil.move(str(so_temp_path), str(so_path))

            return str(so_path)
