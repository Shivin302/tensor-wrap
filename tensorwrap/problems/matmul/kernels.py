slow = """
#include <cstddef>
#include <cmath>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Helper function to access matrix element
float get_element(const float* ptr, size_t row, size_t col, size_t stride) {
    return ptr[row * stride + col];
}

// Helper function to set matrix element
void set_element(float* ptr, size_t row, size_t col, size_t stride, float value) {
    ptr[row * stride + col] = value;
}

// Very slow matrix multiplication function that mimics NumPy's behavior
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
    
    // Create the result array with the correct shape
    py::array_t<float> result = py::array_t<float>({M, N});
    auto buf_result = result.request();
    
    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    // Initialize the result array to zeros
    for (size_t i = 0; i < M * N; ++i) {
        ptr_result[i] = 0.0f;
    }
    
    // Use cache-unfriendly iteration order (i, k, j) for poor locality
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            // Redundant computation of A[i][k]
            float aik = get_element(ptr_a, i, k, K);
            for (size_t j = 0; j < N; ++j) {
                // Non-contiguous access to B[k][j] and C[i][j]
                float current = get_element(ptr_result, i, j, N);
                set_element(ptr_result, i, j, N, current + aik * get_element(ptr_b, k, j, N));
            }
        }
    }
    
    return result;
}
"""



basic = """
#include <cstddef>
#include <cmath>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Define a simple matrix multiplication function that attempts to mimic NumPy's behavior
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
    
    // Create the result array with the correct shape
    py::array_t<float> result = py::array_t<float>({M, N});
    auto buf_result = result.request();
    
    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    // Initialize the result array to zeros
    for (size_t i = 0; i < M * N; ++i) {
        ptr_result[i] = 0.0f;
    }
    
    // Use cache-friendly iteration order and accumulate in blocks
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




vectorize = """
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



shared_memory = """
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


tiling = """
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

unroll = """
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
