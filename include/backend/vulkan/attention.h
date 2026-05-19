#pragma once
#include "core/tensor.hpp"


namespace ops {

    template <Device D> struct SoftmaxImpl;
    template <Device D> struct PagedAttentionImpl;
    template <Device D> struct FlashAttentionImpl;

    template <>
    struct SoftmaxImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct PagedAttentionImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct FlashAttentionImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct SoftmaxImpl<Device::VULKAN>;
    extern template struct PagedAttentionImpl<Device::VULKAN>;
    extern template struct FlashAttentionImpl<Device::VULKAN>;

}
