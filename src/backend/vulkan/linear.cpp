#include "backend/vulkan/linear.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/linear.h"

namespace ops {

struct LinearPushConstants {
    int32_t B;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t has_bias;
    int32_t transpose_w;
};

static void dispatch_linear(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 4, sizeof(LinearPushConstants));

    Tensor* x    = out->src[0];
    Tensor* w    = out->src[1];
    Tensor* bias = out->src[2];

    vk::Buffer buf_x = reinterpret_cast<VkBuffer>(x->device_handle);
    vk::Buffer buf_w = reinterpret_cast<VkBuffer>(w->device_handle);
    vk::Buffer buf_bias = bias ? reinterpret_cast<VkBuffer>(bias->device_handle) : nullptr;
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo x_info(buf_x, x->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo w_info(buf_w, w->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo bias_info(buf_bias, bias ? bias->offset : 0, bias ? VK_WHOLE_SIZE : 0);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {x_info, w_info, bias_info, dst_info});

    bool   transpose_w = out->op_params[0] == 1.0f;
    int32_t B = static_cast<int32_t>(x->dims[0]);
    int32_t M = static_cast<int32_t>(x->dims[1]);
    int32_t K = static_cast<int32_t>(x->dims[2]);
    int32_t N = static_cast<int32_t>(transpose_w ? w->dims[0] : w->dims[1]);

    LinearPushConstants pc{B, M, N, K, bias && bias->data ? 1 : 0, transpose_w ? 1 : 0};

    uint32_t tiles_m = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t tiles_n = (static_cast<uint32_t>(N) + 15) / 16;

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(LinearPushConstants), &pc);
    cmd.dispatch(tiles_n, tiles_m, static_cast<uint32_t>(B));
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void LinearImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_linear(ctx, dev_id, "linear_f16", vkspv::linear_f16_spv, vkspv::linear_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_linear(ctx, dev_id, "linear_bf16", vkspv::linear_bf16_spv, vkspv::linear_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_linear(ctx, dev_id, "linear_f32", vkspv::linear_f32_spv, vkspv::linear_f32_spv_len, out);
        }
    });
}

void MatmulImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan MatmulImpl: not implemented");
}
void TransposeImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan TransposeImpl: not implemented");
}

template struct MatmulImpl<Device::VULKAN>;
template struct LinearImpl<Device::VULKAN>;
template struct TransposeImpl<Device::VULKAN>;

}

#endif
