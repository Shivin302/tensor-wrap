#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Declare CUDA function
void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Synchronize CUDA device
void synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA synchronization failed: " + std::string(cudaGetErrorString(err)));
    }
}

pybind11::array_t<float> matmul(pybind11::array_t<float> a, pybind11::array_t<float> b) {
    // Check input dimensions and format
    auto a_info = a.request();
    auto b_info = b.request();
    
    if (a_info.ndim != 2 || b_info.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }
    if (a_info.shape[1] != b_info.shape[0]) {
        throw std::runtime_error("Inner dimensions must match");
    }
    if (a_info.format != "f" || b_info.format != "f") {
        throw std::runtime_error("Input arrays must be float32");
    }
    // Check for contiguity
    if (a_info.strides[0] != a_info.shape[1] * sizeof(float) ||
        a_info.strides[1] != sizeof(float) ||
        b_info.strides[0] != b_info.shape[1] * sizeof(float) ||
        b_info.strides[1] != sizeof(float)) {
        throw std::runtime_error("Input arrays must be contiguous in row-major order");
    }

    int M = a_info.shape[0];
    int K = a_info.shape[1];
    int N = b_info.shape[0];

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed for A: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error("CUDA malloc failed for B: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_C, size_C);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("CUDA malloc failed for C: " + std::string(cudaGetErrorString(err)));
    }

    // Copy input arrays to GPU
    err = cudaMemcpy(d_A, a_info.ptr, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("CUDA memcpy to device failed for A: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(d_B, b_info.ptr, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("CUDA memcpy to device failed for B: " + std::string(cudaGetErrorString(err)));
    }

    // Call CUDA kernel
    cuda_matmul(d_A, d_B, d_C, M, N, K);

    // Allocate output NumPy array
    pybind11::array_t<float> result({M, N});
    auto result_info = result.request();

    // Copy result back to CPU
    err = cudaMemcpy(result_info.ptr, d_C, size_C, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("CUDA memcpy to host failed for C: " + std::string(cudaGetErrorString(err)));
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

PYBIND11_MODULE(cuda_matmul, m) {
    m.def("matmul", &matmul, "CUDA matrix multiplication");
    m.def("synchronize", &synchronize, "Synchronize CUDA device");
}