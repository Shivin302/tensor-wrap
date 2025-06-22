import os
import time
import signal
import tempfile
import subprocess
import numpy as np
import sysconfig
from typing import Dict, Tuple, Optional, Any
import pybind11
from ..schemas import ProblemSpec, KernelCandidate
from .evaluator_utils import Evaluator


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""

    def __init__(self, problem_path: str, timeout_seconds: int = 2):
        """Initialize the evaluator.
        
        Args:
            problem_path: Path to the problem directory
            timeout_seconds: Timeout for kernel execution in seconds
        """
        super().__init__(problem_path, timeout_seconds)
    

    def evaluate(self, candidate: KernelCandidate) -> Tuple[bool, Optional[float]]:
        """Evaluate a kernel candidate.
        
        Args:
            candidate: The kernel candidate to evaluate
            
        Returns:
            Tuple of (is_correct, latency_ms)
            If evaluation fails, returns (False, None)
        """
        # Return good latency for ideas that sound promising
        latency_ms = 50.0
        if "shared memory" in candidate.idea.lower():
            latency_ms = 25.0
        elif "tiling" in candidate.idea.lower() or "blocking" in candidate.idea.lower():
            latency_ms = 30.0
        elif "vectorization" in candidate.idea.lower() or "simd" in candidate.idea.lower():
            latency_ms = 35.0
        elif "loop unrolling" in candidate.idea.lower():
            latency_ms = 40.0
        
        # Always return correctly functioning kernel in mock mode
        return True, latency_ms

            

class LocalCPUEvaluator(Evaluator):
    """Evaluates kernel candidates locally on CPU."""
    
    def __init__(self, problem_path: str, timeout_seconds: int = 2):
        """Initialize the evaluator.
        
        Args:
            problem_path: Path to the problem directory
            timeout_seconds: Timeout for kernel execution in seconds
        """
        super().__init__(problem_path, timeout_seconds)


    
    def _compile(self, code: str, func_name: str = "kernel") -> Optional[str]:
        """Compile kernel code using pybind11.
        
        Args:
            code: The kernel code to compile
            func_name: The name of the kernel function
            
        Returns:
            Path to the compiled module, or None if compilation failed
        """
        source = f"""
        #include <pybind11/pybind11.h>
        #include <pybind11/numpy.h>
        {code}
        PYBIND11_MODULE(candidate, m) {{
            m.def("{func_name}", &{func_name});
        }}
        """
        
        # Create a more persistent output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compiled")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename for this kernel
        import hashlib
        kernel_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        output_path = os.path.join(output_dir, f"candidate_{kernel_hash}.so")
        
        # Create a temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the source code to a file
            source_path = os.path.join(tmpdir, "candidate.cpp")
            with open(source_path, "w") as f:
                f.write(source)
            
            # Compile the code using g++ with pybind11
            try:
                cmd = [
                    "g++", "-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", "-march=native", "-mtune=native",
                    f"-I{self.pybind11_include}",
                    f"-I{self.python_include}",
                    f"-I{self.python_include_config}",
                    f"-L{self.python_lib}",
                    f"-lpython{self.python_version}",
                    source_path, "-o", output_path
                ]
                
                print(f"Compiling candidate kernel with command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False  # Don't raise exception so we can log more details
                )
                
                if result.returncode != 0:
                    print("*" * 80)
                    print(f"Compilation failed with return code: {result.returncode}")
                    print(f"Stdout: {result.stdout.decode()[:2000]}")
                    print(f"Stderr: {result.stderr.decode()[:2000]}")
                    print("*" * 80)
                    return None
                
                # Make sure the file exists and is accessible
                if os.path.exists(output_path):
                    print(f"Successfully compiled kernel to {output_path}")
                    return output_path
                else:
                    print(f"Compilation succeeded but output file not found at {output_path}")
                    return None
            except Exception as e:
                print(f"Compilation process error: {str(e)}")
                return None
    