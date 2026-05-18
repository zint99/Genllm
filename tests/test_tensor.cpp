#include "test_utils.h"
#include "core/tensor.hpp"
#include "utils/utils.hpp"

static bool test_num_elements() {
    Tensor t;
    t.dims = {2, 3, 4, 0};
    ASSERT(t.num_elements() == 24, "2x3x4 = 24");
    return true;
}

static bool test_num_elements_single() {
    Tensor t;
    t.dims = {1, 0, 0, 0};
    ASSERT(t.num_elements() == 1, "single element");
    return true;
}

static bool test_num_elements_zero_dim() {
    Tensor t;
    t.dims = {0, 0, 0, 0};
    ASSERT(t.num_elements() == 1, "all zero dims = 1 (empty product)");
    return true;
}

static bool test_bytes_f32() {
    Tensor t;
    t.dims = {2, 3, 4, 0};
    t.dtype = DataType::GGML_TYPE_F32;
    ASSERT(t.bytes() == 24 * 4, "f32: 24 * 4 = 96");
    return true;
}

static bool test_bytes_f16() {
    Tensor t;
    t.dims = {4, 256, 0, 0};
    t.dtype = DataType::GGML_TYPE_F16;
    ASSERT(t.bytes() == 4 * 256 * 2, "f16: 1024 * 2 = 2048");
    return true;
}

static bool test_bytes_bf16() {
    Tensor t;
    t.dims = {128, 128, 0, 0};
    t.dtype = DataType::GGML_TYPE_BF16;
    ASSERT(t.bytes() == 128 * 128 * 2, "bf16: 16384 * 2 = 32768");
    return true;
}

static bool test_bytes_at_resolve() {
    Tensor t;
    t.dims = {1, -1, 64, 0};
    t.dtype = DataType::GGML_TYPE_F32;
    ASSERT(t.bytes_at(128) == 1 * 128 * 64 * 4, "bytes_at(128) with -1 dim");
    return true;
}

static bool test_is_computed() {
    Tensor t;
    t.op_type = OperationType::OP_TYPE_ADD;
    t.type = TensorType::TENSOR_TYPE_ACTIVATION;
    ASSERT(t.is_computed(), "activation with op is computed");

    t.type = TensorType::TENSOR_TYPE_WEIGHT;
    ASSERT(!t.is_computed(), "weight not computed");

    t.type = TensorType::TENSOR_TYPE_INPUT;
    ASSERT(!t.is_computed(), "input not computed");
    return true;
}

static bool test_device_enum() {
    ASSERT(static_cast<int>(Device::CPU) == 0, "cpu=0");
    ASSERT(static_cast<int>(Device::CUDA) == 1, "cuda=1");
    ASSERT(static_cast<int>(Device::VULKAN) == 4, "vulkan=4");
    return true;
}

int main() {
    return run_tests({
        {"num_elements 2x3x4",        test_num_elements},
        {"num_elements single",       test_num_elements_single},
        {"num_elements zero dim",     test_num_elements_zero_dim},
        {"bytes f32",                 test_bytes_f32},
        {"bytes f16",                 test_bytes_f16},
        {"bytes bf16",                test_bytes_bf16},
        {"bytes_at dynamic",          test_bytes_at_resolve},
        {"is_computed",               test_is_computed},
        {"device enum",               test_device_enum},
    });
}
