"""Evaluator for CUDA C++ kernels on GPU."""

import os
import time
import signal
import tempfile
import subprocess
import hashlib
import numpy as np
import torch
from typing import Tuple, Optional

from ..schemas import ProblemSpec, KernelCandidate
from .evaluator_utils import Evaluator, timeout_handler


class CudaGPUEvaluator(Evaluator):
    """Evaluates CUDA C++ kernel candidates on the GPU."""

    def __init__(self, problem_path: str, timeout_seconds: int = 10):
        """Initialize the evaluator.

        Args:
            problem_path: Path to the problem directory.
            timeout_seconds: Timeout for kernel execution in seconds.
        """
        super().__init__(problem_path, timeout_seconds)

    def _compile(self, code: str, func_name: str = "kernel") -> Optional[str]:
        """Compile CUDA C++ kernel code using pybind11 and NVCC.

        Args:
            code: The kernel code to compile.
            func_name: The name of the kernel function.

        Returns:
            Path to the compiled module, or None if compilation failed.
        """
        # This wrapper uses pybind11 to handle data type conversions and function calls.
        # It expects the kernel to accept raw data pointers.
        source = f"""
        #include <pybind11/pybind11.h>
        #include <pybind11/numpy.h>

        // The user-provided CUDA kernel code
        {code}

        // Pybind11 wrapper
        PYBIND11_MODULE(candidate, m) {{
            m.def("{func_name}", [](uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr) {{
                // Cast the integer pointers back to actual pointers
                float* a = reinterpret_cast<float*>(a_ptr);
                float* b = reinterpret_cast<float*>(b_ptr);
                float* c = reinterpret_cast<float*>(c_ptr);

                // Call the user-defined kernel
                kernel(a, b, c);
            }}, "A function that runs a CUDA kernel");
        }}
        """

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compiled_cuda")
        os.makedirs(output_dir, exist_ok=True)

        kernel_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        output_path = os.path.join(output_dir, f"candidate_{kernel_hash}.so")

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "candidate.cu")
            with open(source_path, "w") as f:
                f.write(source)

            try:
                # Note: This compilation command is for a specific GPU architecture (sm_80, e.g., A100).
                # You may need to adjust this for your specific GPU.
                cmd = [
                    "nvcc", "-Xcompiler", "-fPIC", "-shared", "-std=c++17", "-O3",
                    "-gencode=arch=compute_80,code=sm_80",
                    f"-I{self.pybind11_include}",
                    f"-I{self.python_include}",
                    source_path, "-o", output_path
                ]

                print(f"Compiling CUDA kernel with command: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )

                if result.returncode != 0:
                    print("-" * 80)
                    print(f"NVCC Compilation failed with return code: {result.returncode}")
                    print(f"Stdout: {result.stdout.decode()[:2000]}")
                    print(f"Stderr: {result.stderr.decode()[:2000]}")
                    print("-" * 80)
                    return None

                if os.path.exists(output_path):
                    print(f"Successfully compiled kernel to {output_path}")
                    return output_path
                else:
                    print(f"Compilation succeeded but output file not found at {output_path}")
                    return None
            except Exception as e:
                print(f"Compilation process error: {str(e)}")
                return None

    def evaluate(self, candidate: KernelCandidate) -> Tuple[bool, Optional[float]]:
        """Evaluate a CUDA C++ kernel candidate.

        Args:
            candidate: The kernel candidate to evaluate.

        Returns:
            Tuple of (is_correct, latency_ms).
        """
        try:
            if not torch.cuda.is_available():
                print("CUDA evaluator requires a CUDA-enabled GPU.")
                return False, None
            device = torch.device("cuda")

            print("Starting CUDA evaluation...")

            inputs = self._generate_inputs(self.problem_spec)
            ref_output = self._generate_reference_output(self.problem_spec, inputs)

            module_path = self._compile(candidate.code)
            if not module_path:
                return False, None

            import importlib.util
            spec = importlib.util.spec_from_file_location("candidate", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            torch_inputs = [torch.from_numpy(arr).to(device) for arr in inputs]
            output_shape = ref_output.shape
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)

            # Pass raw pointers to the CUDA kernel
            input_ptrs = [t.data_ptr() for t in torch_inputs]
            output_ptr = output_tensor.data_ptr()

            start_time = time.time()
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                module.kernel(*input_ptrs, output_ptr)
                torch.cuda.synchronize()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"Kernel execution failed: {e}")
                return False, None
            finally:
                signal.signal(signal.SIGALRM, original_handler)

            latency_ms = (time.time() - start_time) * 1000

            candidate_output = output_tensor.cpu().numpy()

            is_correct = np.allclose(candidate_output, ref_output, atol=1e-5, rtol=1e-5)

            if not is_correct:
                print("Output verification failed.")
            else:
                print("Output verification succeeded.")

            return is_correct, latency_ms

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return False, None
