#include "backend/vulkan/rope.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/rope.h"

namespace ops {

struct RopePushConstants {
    int32_t head_dim;
    int32_t half_dim;
    int32_t seq_len;
    int32_t start_pos;
    int32_t n_heads;
    int32_t B;
};

static void dispatch_rope(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 4, sizeof(RopePushConstants));

    Tensor* src = out->src[0];
    Tensor* cos = out->src[1];
    Tensor* sin = out->src[2];
    vk::Buffer buf_src = reinterpret_cast<VkBuffer>(src->device_handle);
    vk::Buffer buf_cos = reinterpret_cast<VkBuffer>(cos->device_handle);
    vk::Buffer buf_sin = reinterpret_cast<VkBuffer>(sin->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src_info(buf_src, src->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo cos_info(buf_cos, cos->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo sin_info(buf_sin, sin->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src_info, cos_info, sin_info, dst_info});

    int32_t head_dim  = static_cast<int32_t>(out->op_params[0]);
    int32_t half_dim  = head_dim / 2;
    int32_t start_pos = static_cast<int32_t>(out->op_params[2]);
    int32_t B       = static_cast<int32_t>(src->dims[0]);
    int32_t n_heads = static_cast<int32_t>(src->dims[1]);
    int32_t seq_len = static_cast<int32_t>(src->dims[2]);

    RopePushConstants pc{head_dim, half_dim, seq_len, start_pos, n_heads, B};

    int64_t n_pairs = static_cast<int64_t>(B) * n_heads * seq_len * half_dim;
    uint32_t wg_x = (static_cast<uint32_t>(n_pairs) + 255) / 256;

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(RopePushConstants), &pc);
    cmd.dispatch(wg_x, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void ApplyRopeImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_rope(ctx, dev_id, "rope_f16", vkspv::rope_f16_spv, vkspv::rope_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_rope(ctx, dev_id, "rope_bf16", vkspv::rope_bf16_spv, vkspv::rope_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_rope(ctx, dev_id, "rope_f32", vkspv::rope_f32_spv, vkspv::rope_f32_spv_len, out);
        }
    });
}

template struct ApplyRopeImpl<Device::VULKAN>;

}

#endif
