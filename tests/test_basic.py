import os
import sys
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorwrap.schemas import ProblemSpec, KernelCandidate
from tensorwrap.evaluator.cpp_cpu import LocalCPUEvaluator

class TestBasicFunctionality(unittest.TestCase):
    """Basic tests for TensorWrap functionality."""
    
    def test_matmul_baseline(self):
        """Test that the baseline matmul kernel works correctly."""
        # Define problem spec
        problem_spec = ProblemSpec(name="matmul", shape_a=[32, 32], shape_b=[32, 32], dtype="float32")
        
        # Create a simple baseline kernel
        baseline_code = """
        #include <cstddef>
        #include <pybind11/numpy.h>
        namespace py = pybind11;
        
        py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b) {
            auto buf_a = a.request();
            auto buf_b = b.request();
            
            if (buf_a.ndim != 2 || buf_b.ndim != 2) {
                throw std::runtime_error("Number of dimensions must be 2");
            }
            
            if (buf_a.shape[1] != buf_b.shape[0]) {
                throw std::runtime_error("Incompatible shapes");
            }
            
            size_t M = buf_a.shape[0];
            size_t K = buf_a.shape[1];
            size_t N = buf_b.shape[1];
            
            py::array_t<float> result({M, N});
            auto buf_result = result.request();
            
            float* ptr_a = static_cast<float*>(buf_a.ptr);
            float* ptr_b = static_cast<float*>(buf_b.ptr);
            float* ptr_result = static_cast<float*>(buf_result.ptr);
            
            // Simple three-loop matmul implementation
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += ptr_a[i * K + k] * ptr_b[k * N + j];
                    }
                    ptr_result[i * N + j] = sum;
                }
            }
            
            return result;
        }
        """
        
        # Create kernel candidate
        candidate = KernelCandidate(
            problem="matmul",
            round=0,
            code=baseline_code,
            idea="Baseline implementation"
        )
        
        # Evaluate the kernel
        evaluator = LocalCPUEvaluator(problem_path="tensorwrap/problems/matmul", timeout_seconds=5)
        is_correct, latency_ms = evaluator.evaluate(candidate)
        
        # Check that the kernel is correctly evaluated
        self.assertTrue(is_correct, "Baseline kernel should be correct")
        self.assertIsNotNone(latency_ms, "Latency should be measured")

if __name__ == "__main__":
    unittest.main()
