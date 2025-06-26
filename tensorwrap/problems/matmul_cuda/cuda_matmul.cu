#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for matrix multiplication C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Set up block and grid dimensions
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
    }
}


// Compile Command

// export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH && \
// nvcc -c cuda_matmul.cu -o cuda_matmul.o -I/usr/local/cuda/include -Xcompiler -fPIC -Wno-deprecated-gpu-targets && \
// g++ -v -O3 -Wall -shared -std=c++17 -fPIC -Wl,--no-undefined \
// -I/home/shivin/repos/envs/default/lib/python3.12/site-packages/pybind11/include \
// -I/usr/include/python3.12 \
// -I/usr/local/cuda/include \
// bindings.cpp cuda_matmul.o \
// -L/usr/lib/x86_64-linux-gnu -lpython3.12 -lcrypt -ldl -lm \
// -L/usr/local/cuda/lib64 -lcudart \
// -Wl,-rpath,/usr/local/cuda/lib64 -o cuda_matmul.so