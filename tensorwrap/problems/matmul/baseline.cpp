#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <cmath>
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

PYBIND11_MODULE(candidate, m) {{
    m.def("kernel", &kernel);
}}
