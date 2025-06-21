import os
import re
import jinja2
from typing import List, Optional, Dict, Any, Union

# Try to import different LLM providers, with fallbacks
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from .optimizer import LLMSelfHealingOptimizer

class CodeGenerator:
    """Generates optimized kernel code from ideas."""
    
    def __init__(self, mock_mode=False, problem_path=None, self_healing=True, max_iterations=3):
        """Initialize the code generator.
        
        Args:
            mock_mode: If True, use mock responses for testing without API calls
            problem_path: Path to the problem specification directory
            self_healing: Whether to use the self-healing optimizer
            max_iterations: Maximum number of optimization iterations for self-healing
        """
        self.template_loader = jinja2.FileSystemLoader("tensorwrap/templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.implement_template = self.template_env.get_template("implement.j2")
        self.mock_mode = mock_mode
        self.problem_path = problem_path
        self.self_healing = self_healing and not mock_mode
        self.max_iterations = max_iterations
        
        # Setup LLM providers if available and not in mock mode
        if not mock_mode:
            if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                # Initialize the self-healing optimizer if needed
                if self.self_healing and self.problem_path:
                    self.optimizer = LLMSelfHealingOptimizer(
                        self.openai_client, 
                        self.problem_path,
                        max_iterations
                    )
            elif GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
                self.provider = "google"
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                self.self_healing = False  # Only support OpenAI for self-healing for now
            else:
                raise RuntimeError("No LLM provider available. Set OPENAI_API_KEY or GOOGLE_API_KEY.")
        else:
            self.provider = "mock"
    
    def generate_code(self, baseline_code: str, idea: str, problem_spec: Any = None) -> Union[str, Dict[str, Any]]:
        """Generate optimized kernel code from an idea.
        
        Args:
            baseline_code: The baseline kernel code to optimize
            idea: The optimization idea to implement
            problem_spec: The problem specification (for self-healing optimizer)
            
        Returns:
            Generated optimized code or a dict with code and metadata if self-healing is used
        """
        # Try using the self-healing optimizer if enabled and we have access to the problem spec
        if self.self_healing and problem_spec and hasattr(self, 'optimizer'):
            print(f"Using self-healing optimizer for idea: {idea[:50]}...")
            result = self.optimizer.optimize_kernel(idea, baseline_code, problem_spec)
            if result:
                return result
            else:
                print("Self-healing optimization failed, falling back to standard code generation")
        
        # Fall back to standard code generation
        prompt = self.implement_template.render(baseline=baseline_code, idea=idea)
        
        if self.mock_mode:
            return self._generate_mock_code(baseline_code, idea)
        elif self.provider == "openai":
            return self._generate_with_openai(prompt)
        else:  # google
            return self._generate_with_google(prompt)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate code using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated code
        """
        # Using the latest o3 model
        response = self.openai_client.chat.completions.create(
            model="o3-2025-04-16",  # latest o3 model
            messages=[
                {"role": "system", "content": "You are an expert CUDA engineer."},
                {"role": "user", "content": prompt}
            ]
            # o3-2025-04-16 only supports the default temperature (1.0)
        )
        
        # Extract code from response
        text = response.choices[0].message.content
        return self._extract_code(text)
    
    def _generate_with_google(self, prompt: str) -> str:
        """Generate code using Google Generative AI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated code
        """
        # TODO: Implement actual Google Generative AI API call
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Extract code from response
        text = response.text
        return self._extract_code(text)
    
    def _generate_mock_code(self, baseline_code: str, idea: str) -> str:
        """Generate mock optimized kernel code for testing without LLM API calls.
        
        Args:
            baseline_code: The baseline kernel code to optimize
            idea: The optimization idea to implement
            
        Returns:
            Mock optimized kernel code
        """
        # Create a simple optimization based on the idea
        if "shared memory" in idea.lower():
            return """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Optimized matmul kernel using shared memory
void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 16;
    float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // Initialize output block
            float block_C[BLOCK_SIZE][BLOCK_SIZE] = {0};
            
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                // Load blocks into shared memory
                for (int b_i = 0; b_i < BLOCK_SIZE && i + b_i < M; ++b_i) {
                    for (int b_k = 0; b_k < BLOCK_SIZE && k + b_k < K; ++b_k) {
                        shared_A[b_i][b_k] = A[(i + b_i) * K + (k + b_k)];
                    }
                }
                
                for (int b_k = 0; b_k < BLOCK_SIZE && k + b_k < K; ++b_k) {
                    for (int b_j = 0; b_j < BLOCK_SIZE && j + b_j < N; ++b_j) {
                        shared_B[b_k][b_j] = B[(k + b_k) * N + (j + b_j)];
                    }
                }
                
                // Compute block multiplication
                for (int b_i = 0; b_i < BLOCK_SIZE && i + b_i < M; ++b_i) {
                    for (int b_j = 0; b_j < BLOCK_SIZE && j + b_j < N; ++b_j) {
                        for (int b_k = 0; b_k < BLOCK_SIZE && k + b_k < K; ++b_k) {
                            block_C[b_i][b_j] += shared_A[b_i][b_k] * shared_B[b_k][b_j];
                        }
                    }
                }
            }
            
            // Write results back to C
            for (int b_i = 0; b_i < BLOCK_SIZE && i + b_i < M; ++b_i) {
                for (int b_j = 0; b_j < BLOCK_SIZE && j + b_j < N; ++b_j) {
                    C[(i + b_i) * N + (j + b_j)] = block_C[b_i][b_j];
                }
            }
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    m.def("matmul", [](py::array_t<float> A, py::array_t<float> B) {
        py::buffer_info A_info = A.request();
        py::buffer_info B_info = B.request();
        
        if (A_info.ndim != 2 || B_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }
        
        int M = A_info.shape[0];
        int K = A_info.shape[1];
        
        if (B_info.shape[0] != K) {
            throw std::runtime_error("Inner matrix dimensions must match");
        }
        
        int N = B_info.shape[1];
        
        py::array_t<float> C({M, N});
        py::buffer_info C_info = C.request();
        
        float* A_ptr = static_cast<float*>(A_info.ptr);
        float* B_ptr = static_cast<float*>(B_info.ptr);
        float* C_ptr = static_cast<float*>(C_info.ptr);
        
        matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K);
        
        return C;
    }, "Matrix multiplication using shared memory");
}
"""
        elif "tiling" in idea.lower() or "blocking" in idea.lower():
            return """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Optimized matmul kernel using tiling/blocking
void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 32; // block size for M
    const int BN = 32; // block size for N
    const int BK = 32; // block size for K
    
    // Zero the output matrix
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    // Loop over blocks of the matrices
    for (int i0 = 0; i0 < M; i0 += BM) {
        for (int j0 = 0; j0 < N; j0 += BN) {
            for (int k0 = 0; k0 < K; k0 += BK) {
                // Process one block
                for (int i = i0; i < std::min(i0 + BM, M); i++) {
                    for (int j = j0; j < std::min(j0 + BN, N); j++) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < std::min(k0 + BK, K); k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    m.def("matmul", [](py::array_t<float> A, py::array_t<float> B) {
        py::buffer_info A_info = A.request();
        py::buffer_info B_info = B.request();
        
        if (A_info.ndim != 2 || B_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }
        
        int M = A_info.shape[0];
        int K = A_info.shape[1];
        
        if (B_info.shape[0] != K) {
            throw std::runtime_error("Inner matrix dimensions must match");
        }
        
        int N = B_info.shape[1];
        
        py::array_t<float> C({M, N});
        py::buffer_info C_info = C.request();
        
        float* A_ptr = static_cast<float*>(A_info.ptr);
        float* B_ptr = static_cast<float*>(B_info.ptr);
        float* C_ptr = static_cast<float*>(C_info.ptr);
        
        matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K);
        
        return C;
    }, "Matrix multiplication using tiling/blocking");
}
"""
        elif "unroll" in idea.lower():
            return """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Optimized matmul kernel with loop unrolling
void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Zero the output matrix
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int k = 0;
            
            // Unrolled loop in chunks of 4
            for (; k + 3 < K; k += 4) {
                sum += A[i * K + k] * B[k * N + j];
                sum += A[i * K + k + 1] * B[(k + 1) * N + j];
                sum += A[i * K + k + 2] * B[(k + 2) * N + j];
                sum += A[i * K + k + 3] * B[(k + 3) * N + j];
            }
            
            // Handle remaining elements
            for (; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            
            C[i * N + j] = sum;
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    m.def("matmul", [](py::array_t<float> A, py::array_t<float> B) {
        py::buffer_info A_info = A.request();
        py::buffer_info B_info = B.request();
        
        if (A_info.ndim != 2 || B_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }
        
        int M = A_info.shape[0];
        int K = A_info.shape[1];
        
        if (B_info.shape[0] != K) {
            throw std::runtime_error("Inner matrix dimensions must match");
        }
        
        int N = B_info.shape[1];
        
        py::array_t<float> C({M, N});
        py::buffer_info C_info = C.request();
        
        float* A_ptr = static_cast<float*>(A_info.ptr);
        float* B_ptr = static_cast<float*>(B_info.ptr);
        float* C_ptr = static_cast<float*>(C_info.ptr);
        
        matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K);
        
        return C;
    }, "Matrix multiplication with loop unrolling");
}
"""
        else:
            # Default optimization with vectorization
            return """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Default optimized matmul kernel
void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Zero the output matrix
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    // Compute matrix multiplication with basic optimizations
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    m.def("matmul", [](py::array_t<float> A, py::array_t<float> B) {
        py::buffer_info A_info = A.request();
        py::buffer_info B_info = B.request();
        
        if (A_info.ndim != 2 || B_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }
        
        int M = A_info.shape[0];
        int K = A_info.shape[1];
        
        if (B_info.shape[0] != K) {
            throw std::runtime_error("Inner matrix dimensions must match");
        }
        
        int N = B_info.shape[1];
        
        py::array_t<float> C({M, N});
        py::buffer_info C_info = C.request();
        
        float* A_ptr = static_cast<float*>(A_info.ptr);
        float* B_ptr = static_cast<float*>(B_info.ptr);
        float* C_ptr = static_cast<float*>(C_info.ptr);
        
        matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K);
        
        return C;
    }, "Default optimized matrix multiplication");
}
"""
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            Extracted code or None if no code found
        """
        # Look for code blocks wrapped in triple backticks
        code_regex = r'```(?:cpp|c\+\+)?\s*\n([\s\S]*?)\n```'
        match = re.search(code_regex, text)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: if no code blocks found, return entire response
        # In practice, you'd want to make this more robust
        return text.strip()