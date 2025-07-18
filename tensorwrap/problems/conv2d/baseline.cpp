#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <cmath>
namespace py = pybind11;


// Mimics Scipy's 2D convolution with 'full' mode.
py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();

    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }

    size_t H_a = buf_a.shape[0];
    size_t W_a = buf_a.shape[1];
    size_t H_b = buf_b.shape[0];
    size_t W_b = buf_b.shape[1];

    size_t H_res = H_a + H_b - 1;
    size_t W_res = W_a + W_b - 1;

    py::array_t<float> result = py::array_t<float>({H_res, W_res});
    auto buf_result = result.request();

    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    std::fill(ptr_result, ptr_result + (H_res * W_res), 0.0f);

    for (size_t i = 0; i < H_a; ++i) {
        for (size_t j = 0; j < W_a; ++j) {
            for (size_t h = 0; h < H_b; ++h) {
                for (size_t w = 0; w < W_b; ++w) {
                    ptr_result[(i + h) * W_res + (j + w)] += ptr_a[i * W_a + j] * ptr_b[h * W_b + w];
                }
            }
        }
    }

    return result;
}

PYBIND11_MODULE(candidate, m) {{
    m.def("kernel", &kernel);
}}
