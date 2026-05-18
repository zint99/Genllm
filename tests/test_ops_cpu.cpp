#include "test_utils.h"
#include "core/tensor.hpp"
#include "utils/utils.hpp"
#include "utils/bfloat16.hpp"
#include "backend/cpu/arithmetic.h"
#include "backend/cpu/normalization.h"

#include <numeric>
#include <algorithm>
#include <cstring>

static Tensor make_tensor(const std::array<int64_t, 4>& dims, DataType dtype) {
    Tensor t;
    std::copy(dims.begin(), dims.end(), t.dims.begin());
    t.dtype = dtype;
    t.device = Device::CPU;
    t.op_type = OperationType::OP_TYPE_ADD;
    t.type = TensorType::TENSOR_TYPE_ACTIVATION;
    size_t nb = t.bytes();
    if (nb > 0) t.data = new char[nb];
    return t;
}

static void free_tensor(Tensor& t) {
    delete[] static_cast<char*>(t.data);
    t.data = nullptr;
}

template <typename T>
static void fill_sequence(Tensor& t, T start) {
    T* ptr = static_cast<T*>(t.data);
    size_t n = t.num_elements();
    for (size_t i = 0; i < n; i++) ptr[i] = start + static_cast<T>(i);
}

template <typename T>
static bool check_tensor(const Tensor& t, const std::vector<T>& expected) {
    T* ptr = static_cast<T*>(t.data);
    size_t n = t.num_elements();
    if (n != expected.size()) {
        std::println("    size mismatch: got {}, expected {}", n, expected.size());
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        if (std::abs(static_cast<double>(ptr[i]) - static_cast<double>(expected[i])) > 1e-3) {
            std::println("    mismatch at [{}]: got {}, expected {}", i,
                         static_cast<double>(ptr[i]), static_cast<double>(expected[i]));
            return false;
        }
    }
    return true;
}

// ========== Arithmetic ==========
// fill_sequence(a, 1) → a = [1, 2, 3, 4, 5, 6]
// fill_sequence(b, 10) → b = [10, 11, 12, 13, 14, 15]
// out = a ⊕ b → [11, 13, 15, 17, 19, 21]

static bool test_cpu_add_f32() {
    Tensor a = make_tensor({2, 3, 1, 0}, DataType::GGML_TYPE_F32);
    Tensor b = make_tensor({2, 3, 1, 0}, DataType::GGML_TYPE_F32);
    Tensor out = make_tensor({2, 3, 1, 0}, DataType::GGML_TYPE_F32);

    fill_sequence(a, 1.0f);
    fill_sequence(b, 10.0f);
    out.src[0] = &a;
    out.src[1] = &b;

    ops::AddImpl<Device::CPU>::execute(&out, 0);

    ASSERT(check_tensor<float>(out, {11.0f, 13.0f, 15.0f, 17.0f, 19.0f, 21.0f}), "add f32");

    free_tensor(a); free_tensor(b); free_tensor(out);
    return true;
}

static bool test_cpu_mul_f32() {
    Tensor a = make_tensor({4, 1, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor b = make_tensor({4, 1, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor out = make_tensor({4, 1, 0, 0}, DataType::GGML_TYPE_F32);

    fill_sequence(a, 2.0f);
    fill_sequence(b, 3.0f);
    out.src[0] = &a;
    out.src[1] = &b;

    ops::MulImpl<Device::CPU>::execute(&out, 0);

    ASSERT(check_tensor<float>(out, {6.0f, 12.0f, 20.0f, 30.0f}), "mul f32");

    free_tensor(a); free_tensor(b); free_tensor(out);
    return true;
}

static bool test_cpu_div_f32() {
    Tensor a = make_tensor({3, 1, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor b = make_tensor({3, 1, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor out = make_tensor({3, 1, 0, 0}, DataType::GGML_TYPE_F32);

    static_cast<float*>(a.data)[0] = 10.0f; static_cast<float*>(b.data)[0] = 2.0f;
    static_cast<float*>(a.data)[1] = 9.0f;  static_cast<float*>(b.data)[1] = 3.0f;
    static_cast<float*>(a.data)[2] = 7.0f;  static_cast<float*>(b.data)[2] = 0.5f;

    out.src[0] = &a;
    out.src[1] = &b;

    ops::DivImpl<Device::CPU>::execute(&out, 0);

    ASSERT(check_tensor<float>(out, {5.0f, 3.0f, 14.0f}), "div f32");

    free_tensor(a); free_tensor(b); free_tensor(out);
    return true;
}

// ========== Normalization ==========
// Note: CPU RmsNormImpl hard-casts to bfloat16_t*, so we use BF16 dtype

static bool test_cpu_rms_norm_bf16() {
    Tensor x = make_tensor({2, 4, 0, 0}, DataType::GGML_TYPE_BF16);
    Tensor w = make_tensor({4, 0, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor out = make_tensor({2, 4, 0, 0}, DataType::GGML_TYPE_BF16);

    bfloat16_t* xp = static_cast<bfloat16_t*>(x.data);
    xp[0] = bfloat16_t(1.0f); xp[1] = bfloat16_t(2.0f);
    xp[2] = bfloat16_t(3.0f); xp[3] = bfloat16_t(4.0f);
    xp[4] = bfloat16_t(2.0f); xp[5] = bfloat16_t(4.0f);
    xp[6] = bfloat16_t(6.0f); xp[7] = bfloat16_t(8.0f);

    float* wp = static_cast<float*>(w.data);
    wp[0] = 1.0f; wp[1] = 1.0f; wp[2] = 1.0f; wp[3] = 1.0f;

    out.src[0] = &x;
    out.src[1] = &w;
    out.op_params[0] = 1e-5f;

    ops::RmsNormImpl<Device::CPU>::execute(&out, 0);

    bfloat16_t* op = static_cast<bfloat16_t*>(out.data);
    float inv_rms0 = 1.0f / std::sqrt((1.0f+4.0f+9.0f+16.0f)/4.0f);
    ASSERT_NEAR(float(op[0]), 1.0f * inv_rms0, 1e-2f, "rms_norm[0]");
    ASSERT_NEAR(float(op[3]), 4.0f * inv_rms0, 1e-2f, "rms_norm[3]");

    float inv_rms1 = 1.0f / std::sqrt((4.0f+16.0f+36.0f+64.0f)/4.0f);
    ASSERT_NEAR(float(op[4]), 2.0f * inv_rms1, 1e-2f, "rms_norm[4]");

    free_tensor(x); free_tensor(w); free_tensor(out);
    return true;
}

static bool test_cpu_layer_norm_f32() {
    Tensor x = make_tensor({2, 3, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor w = make_tensor({3, 0, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor b = make_tensor({3, 0, 0, 0}, DataType::GGML_TYPE_F32);
    Tensor out = make_tensor({2, 3, 0, 0}, DataType::GGML_TYPE_F32);

    float* xp = static_cast<float*>(x.data);
    xp[0] = 1; xp[1] = 2; xp[2] = 3;
    xp[3] = 4; xp[4] = 5; xp[5] = 6;

    float* wp = static_cast<float*>(w.data);
    wp[0] = 1; wp[1] = 1; wp[2] = 1;
    float* bp = static_cast<float*>(b.data);
    bp[0] = 0; bp[1] = 0; bp[2] = 0;

    out.src[0] = &x;
    out.src[1] = &w;
    out.src[2] = &b;
    out.op_params[0] = 1e-5f;

    ops::LayerNormImpl<Device::CPU>::execute(&out, 0);

    float* op = static_cast<float*>(out.data);
    float mean0 = 2.0f;
    float std0 = std::sqrt(2.0f / 3.0f);
    ASSERT_NEAR(op[0], (1 - mean0) / std0, 1e-4, "layer_norm[0]");
    ASSERT_NEAR(op[1], (2 - mean0) / std0, 1e-4, "layer_norm[1]");

    free_tensor(x); free_tensor(w); free_tensor(b); free_tensor(out);
    return true;
}

int main() {
    return run_tests({
        {"add f32",          test_cpu_add_f32},
        {"mul f32",          test_cpu_mul_f32},
        {"div f32",          test_cpu_div_f32},
        {"rms_norm bf16",    test_cpu_rms_norm_bf16},
        {"layer_norm f32",   test_cpu_layer_norm_f32},
    });
}
