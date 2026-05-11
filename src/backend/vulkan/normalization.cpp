#include "backend/vulkan/normalization.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/rms_norm.h"
#include "backend/vulkan/spv/layer_norm.h"

namespace ops {

struct NormPushConstants {
    int32_t rows;
    int32_t hidden_size;
    float eps;
};

struct LayerNormPushConstants {
    int32_t rows;
    int32_t hidden_size;
    float eps;
    int32_t has_bias;
};

static void dispatch_rms_norm(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 3, sizeof(NormPushConstants));

    Tensor* src = out->src[0];
    Tensor* weight = out->src[1];
    vk::Buffer buf_src = reinterpret_cast<VkBuffer>(src->device_handle);
    vk::Buffer buf_weight = reinterpret_cast<VkBuffer>(weight->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src_info(buf_src, src->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo weight_info(buf_weight, weight->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src_info, weight_info, dst_info});

    size_t hidden_size = weight->num_elements();
    int32_t rows = static_cast<int32_t>(src->num_elements() / hidden_size);
    float eps = out->op_params[0];

    NormPushConstants pc{rows, static_cast<int32_t>(hidden_size), eps};

    // 256 threads/WG, subgroupSize=32 → 8 subgroups/WG → 8 rows/WG
    uint32_t wg_x = (static_cast<uint32_t>(rows) + 7) / 8;

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(NormPushConstants), &pc);
    cmd.dispatch(wg_x, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

static void dispatch_layer_norm(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 4, sizeof(LayerNormPushConstants));

    Tensor* src = out->src[0];
    Tensor* weight = out->src[1];
    Tensor* bias = out->src[2];
    vk::Buffer buf_src = reinterpret_cast<VkBuffer>(src->device_handle);
    vk::Buffer buf_weight = reinterpret_cast<VkBuffer>(weight->device_handle);
    vk::Buffer buf_bias = bias ? vk::Buffer(reinterpret_cast<VkBuffer>(bias->device_handle)) : buf_weight;
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src_info(buf_src, src->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo weight_info(buf_weight, weight->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo bias_info(buf_bias, bias ? bias->offset : 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src_info, weight_info, bias_info, dst_info});

    size_t hidden_size = weight->num_elements();
    int32_t rows = static_cast<int32_t>(src->num_elements() / hidden_size);
    float eps = out->op_params[0];
    int32_t has_bias = bias ? 1 : 0;

    LayerNormPushConstants pc{rows, static_cast<int32_t>(hidden_size), eps, has_bias};

    uint32_t wg_x = (static_cast<uint32_t>(rows) + 7) / 8;

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(LayerNormPushConstants), &pc);
    cmd.dispatch(wg_x, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void RmsNormImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_f16", vkspv::rms_norm_f16_spv, vkspv::rms_norm_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_bf16", vkspv::rms_norm_bf16_spv, vkspv::rms_norm_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_f32", vkspv::rms_norm_f32_spv, vkspv::rms_norm_f32_spv_len, out);
        }
    });
}

void LayerNormImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_f16", vkspv::layer_norm_f16_spv, vkspv::layer_norm_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_bf16", vkspv::layer_norm_bf16_spv, vkspv::layer_norm_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_f32", vkspv::layer_norm_f32_spv, vkspv::layer_norm_f32_spv_len, out);
        }
    });
}

template struct RmsNormImpl<Device::VULKAN>;
template struct LayerNormImpl<Device::VULKAN>;

}

#endif
