#include "test_utils.h"
#include "utils/bfloat16.hpp"
#include "utils/float16.hpp"
#include "utils/dtype_traits.hpp"
#include "utils/utils.hpp"

static bool test_bf16_roundtrip() {
    float vals[] = {0.0f, 1.0f, -1.0f, 3.140625f, 0.5f, 65504.0f, -65504.0f};
    for (float v : vals) {
        bfloat16_t b(v);
        float back = static_cast<float>(b);
        ASSERT_NEAR(back, v, std::abs(v) * 1e-2f + 1e-6f, "bf16 round-trip");
    }
    return true;
}

static bool test_bf16_special() {
    ASSERT(bfloat16_t::infinity().is_inf(), "bf16 inf");
    ASSERT(bfloat16_t::neg_infinity().is_inf(), "bf16 -inf");
    ASSERT(bfloat16_t::nan().is_nan(), "bf16 nan");
    ASSERT(bfloat16_t::zero().bits() == 0, "bf16 zero bits");
    ASSERT(!bfloat16_t(1.0f).is_nan(), "bf16 1.0 not nan");
    return true;
}

static bool test_f16_roundtrip() {
    float vals[] = {0.0f, 1.0f, -1.0f, 3.140625f, 0.5f, 65504.0f, -65504.0f};
    for (float v : vals) {
        float16_t f(v);
        float back = static_cast<float>(f);
        ASSERT_NEAR(back, v, 1e-3f, "f16 round-trip");
    }
    return true;
}

static bool test_f16_special() {
    ASSERT(float16_t::infinity().is_inf(), "f16 inf");
    ASSERT(float16_t::nan().is_nan(), "f16 nan");
    ASSERT(!float16_t(1.0f).is_nan(), "f16 1.0 not nan");
    return true;
}

static bool test_data_type_size() {
    ASSERT(data_type_size(DataType::GGML_TYPE_F32) == 4, "f32 size");
    ASSERT(data_type_size(DataType::GGML_TYPE_F16) == 2, "f16 size");
    ASSERT(data_type_size(DataType::GGML_TYPE_BF16) == 2, "bf16 size");
    ASSERT(data_type_size(DataType::GGML_TYPE_I32) == 4, "i32 size");
    ASSERT(data_type_size(DataType::GGML_TYPE_I8) == 1, "i8 size");
    return true;
}

static bool test_dtype_traits() {
    ASSERT((std::is_same_v<dtype::type_t<DataType::GGML_TYPE_F32>, float>), "f32 trait");
    ASSERT((std::is_same_v<dtype::type_t<DataType::GGML_TYPE_F16>, float16_t>), "f16 trait");
    ASSERT((std::is_same_v<dtype::type_t<DataType::GGML_TYPE_BF16>, bfloat16_t>), "bf16 trait");
    return true;
}

int main() {
    return run_tests({
        {"bf16 round-trip",            test_bf16_roundtrip},
        {"bf16 special values",        test_bf16_special},
        {"f16 round-trip",             test_f16_roundtrip},
        {"f16 special values",         test_f16_special},
        {"data_type_size",             test_data_type_size},
        {"dtype traits",               test_dtype_traits},
    });
}
