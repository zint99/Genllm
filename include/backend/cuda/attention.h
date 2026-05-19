#pragma once
#include "core/tensor.hpp"
#include <cstdint>

namespace ops {

    template <Device D> struct SoftmaxImpl;
    template <Device D> struct PagedAttentionImpl;
    template <Device D> struct FlashAttentionImpl;

    template <>
    struct SoftmaxImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct PagedAttentionImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct FlashAttentionImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct SoftmaxImpl<Device::CUDA>;
    extern template struct PagedAttentionImpl<Device::CUDA>;
    extern template struct FlashAttentionImpl<Device::CUDA>;

}
