#include "backend/vulkan/shape.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/permute.h"

namespace ops {

struct PermutePushConstants {
    int32_t ndim;
    int32_t total;
    int32_t out_strides[TENSOR_MAX_DIMS];
    int32_t src_strides[TENSOR_MAX_DIMS];
    int32_t perm[TENSOR_MAX_DIMS];
};

static void dispatch_permute(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out, const PermutePushConstants& pc)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 2, sizeof(PermutePushConstants));

    Tensor* x = out->src[0];
    vk::Buffer buf_src = reinterpret_cast<VkBuffer>(x->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);
    vk::DescriptorBufferInfo src_info(buf_src, x->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src_info, dst_info});

    uint32_t wg = (static_cast<uint32_t>(pc.total) + 255) / 256;

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PermutePushConstants), &pc);
    cmd.dispatch(wg, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void ReshapeImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    const Tensor* x = out->src[0];
    out->device_handle = x->device_handle;
    out->offset        = x->offset;

    size_t elem_sz = data_type_size(out->dtype);
    size_t stride  = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (out->dims[i] == 0) {
            out->strides[i] = 0;
        } else {
            out->strides[i] = stride * elem_sz;
            stride *= static_cast<size_t>(out->dims[i]);
        }
    }
}

void PermuteImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* x = out->src[0];

    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) ndim = i + 1;
    if (ndim <= 1) return;

    size_t elem_sz = data_type_size(out->dtype);

    PermutePushConstants pc{};
    pc.ndim  = ndim;
    pc.total = static_cast<int32_t>(out->num_elements());
    for (int i = 0; i < ndim; i++) {
        pc.out_strides[i] = static_cast<int32_t>(out->strides[i] / elem_sz);
        pc.src_strides[i] = static_cast<int32_t>(x->strides[i] / elem_sz);
        pc.perm[i]        = static_cast<int32_t>(out->op_params[i]);
    }

    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_permute(ctx, dev_id, "permute_f16", vkspv::permute_f16_spv, vkspv::permute_f16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_permute(ctx, dev_id, "permute_bf16", vkspv::permute_bf16_spv, vkspv::permute_bf16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_permute(ctx, dev_id, "permute_f32", vkspv::permute_f32_spv, vkspv::permute_f32_spv_len, out, pc);
        }
    });
}

void ConcatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan ConcatImpl: not implemented");
}
void RepeatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan RepeatImpl: not implemented");
}

template struct ReshapeImpl<Device::VULKAN>;
template struct PermuteImpl<Device::VULKAN>;
template struct ConcatImpl<Device::VULKAN>;
template struct RepeatImpl<Device::VULKAN>;

}

#endif
