from typing import Dict, Tuple, Optional, Any, List
from ..schemas import ProblemSpec, KernelCandidate
from .evaluator_utils import Evaluator, Compiler


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

            

class LocalCPUCompiler(Compiler):
    """Evaluates kernel candidates locally on CPU."""
    
    def get_compile_command(self, source_path: str, output_path: str) -> List[str]:
        """Get the compile command for a given kernel code."""
        compiler = "g++"
        flags = ["-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", "-march=native", "-mtune=native",]
        compile_cmd = [
            compiler,
            *flags,
            f"-I{self.pybind11_include}",
            f"-I{self.python_include}",
            f"-I{self.python_include_config}",
            f"-L{self.python_lib}",
            f"-lpython{self.python_version}",
            source_path, "-o", output_path
        ]
        return compile_cmd