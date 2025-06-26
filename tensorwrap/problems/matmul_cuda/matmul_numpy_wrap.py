import numpy as np
import time
import cuda_matmul  # The compiled Pybind11 module
import torch  # For CUDA synchronization only

def numpy_matmul():
    # Create random 1024x1024 arrays on CPU
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    
    # Warm-up
    np.dot(A, B)
    
    # Time the operation
    start = time.time()
    C = np.dot(A, B)
    end = time.time()
    
    return C, end - start

def custom_cuda_matmul():
    # Create random 1024x1024 arrays on CPU
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    
    # Warm-up
    cuda_matmul.matmul(A, B)
    torch.cuda.synchronize()
    
    # Time the operation
    start = time.time()
    C = cuda_matmul.matmul(A, B)
    torch.cuda.synchronize()
    end = time.time()
    
    return C, end - start

def compare_results():
    # Run both implementations
    C_numpy, time_numpy = numpy_matmul()
    C_cuda, time_cuda = custom_cuda_matmul()
    
    # Compute precision (L2 norm of difference)
    diff = np.linalg.norm(C_numpy - C_cuda)
    
    # Print results
    print(f"NumPy matmul time: {time_numpy:.6f} seconds")
    print(f"Custom CUDA matmul time: {time_cuda:.6f} seconds")
    print(f"L2 norm of difference: {diff:.6e}")
    
    return time_numpy, time_cuda, diff

if __name__ == "__main__":
    compare_results()