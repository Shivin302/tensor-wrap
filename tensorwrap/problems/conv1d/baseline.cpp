#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <cmath>
namespace py = pybind11;


// Mimics NumPy's 1D convolution with 'full' mode.
py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();

    if (buf_a.ndim != 1 || buf_b.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }

    size_t N = buf_a.shape[0];
    size_t K = buf_b.shape[0];

    if (N == 0 || K == 0) {
        return py::array_t<float>({0});
    }

    size_t result_size = N + K - 1;

    // Create the result array with the correct shape
    py::array_t<float> result = py::array_t<float>({result_size});
    auto buf_result = result.request();

    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    // Initialize the result array to zeros
    std::fill(ptr_result, ptr_result + result_size, 0.0f);

    // Perform the 1D convolution
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < K; ++j) {
            ptr_result[i + j] += ptr_a[i] * ptr_b[j];
        }
    }

    return result;
}

PYBIND11_MODULE(candidate, m) {{
    m.def("kernel", &kernel);
}}