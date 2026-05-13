#include "backend/vulkan/attention.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/attention.h"
#include "core/manager.h"
#include "core/page_attention.h"

namespace ops {

struct SdpaPushConstants {
    int32_t B;
    int32_t n_heads;
    int32_t Sq;
    int32_t n_kv_heads;
    int32_t Skv;
    int32_t head_dim;
    int32_t num_kv_groups;
    float   scale;
    int32_t causal;
};

struct PagedAttnPushConstants {
    int32_t B;
    int32_t n_heads;
    int32_t Sq;
    int32_t n_kv_heads;
    int32_t total_cached;
    int32_t num_blocks;
    int32_t num_kv_groups;
    int32_t head_dim;
    float   scale;
    int32_t causal;
    int32_t block_stride_elems;
    int32_t q_start;
};

static void dispatch_sdpa(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out, const SdpaPushConstants& pc)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 4, sizeof(SdpaPushConstants));

    Tensor* Q = out->src[0];
    Tensor* K = out->src[1];
    Tensor* V = out->src[2];

    vk::Buffer buf_q = reinterpret_cast<VkBuffer>(Q->device_handle);
    vk::Buffer buf_k = reinterpret_cast<VkBuffer>(K->device_handle);
    vk::Buffer buf_v = reinterpret_cast<VkBuffer>(V->device_handle);
    vk::Buffer buf_o = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);
    vk::DescriptorBufferInfo qi(buf_q, Q->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo ki(buf_k, K->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo vi(buf_v, V->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo oi(buf_o, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {qi, ki, vi, oi});

    uint32_t blocks_x = (static_cast<uint32_t>(pc.B * pc.n_heads) + 3) / 4;
    uint32_t blocks_y = static_cast<uint32_t>(pc.Sq);

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SdpaPushConstants), &pc);
    cmd.dispatch(blocks_x, blocks_y, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void AttentionImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* Q = out->src[0];
    const Tensor* K = out->src[1];
    const Tensor* V = out->src[2];

    int32_t B        = static_cast<int32_t>(Q->dims[0]);
    int32_t n_heads  = static_cast<int32_t>(Q->dims[1]);
    int32_t Sq       = static_cast<int32_t>(Q->dims[2]);
    int32_t head_dim = static_cast<int32_t>(Q->dims[3]);
    int32_t n_kv_heads = static_cast<int32_t>(K->dims[1]);
    int32_t Skv      = static_cast<int32_t>(K->dims[2]);

    if (head_dim != 128) throw std::runtime_error("Vulkan SDPA requires head_dim == 128");

    SdpaPushConstants pc{};
    pc.B     = B;
    pc.n_heads = n_heads;
    pc.Sq    = Sq;
    pc.n_kv_heads = n_kv_heads;
    pc.Skv   = Skv;
    pc.head_dim = head_dim;
    pc.num_kv_groups = static_cast<int32_t>(out->op_params[3]);
    pc.scale = out->op_params[1];
    pc.causal = static_cast<int32_t>(out->op_params[2]) != 0 ? 1 : 0;

    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_sdpa(ctx, dev_id, "sdpa_f16", vkspv::sdpa_f16_spv, vkspv::sdpa_f16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_sdpa(ctx, dev_id, "sdpa_bf16", vkspv::sdpa_bf16_spv, vkspv::sdpa_bf16_spv_len, out, pc);
        }
    });
}

void SdpaImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto* mgr = g_mem_manager->get_attention_manager(out->device, dev_id);
    if (mgr && mgr->is_active(out->layer_id)) {
        FlashAttentionImpl<Device::VULKAN>::execute(out, dev_id);
    } else {
        AttentionImpl<Device::VULKAN>::execute(out, dev_id);
    }
}

void FlashAttentionImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto* mgr = g_mem_manager->get_attention_manager(out->device, dev_id);
    int32_t layer_id = out->layer_id;
    if (!mgr || !mgr->is_active(layer_id)) return;

    Tensor* Q = out->src[0];
    Tensor* K = out->src[1];
    Tensor* V = out->src[2];

    int32_t B        = static_cast<int32_t>(Q->dims[0]);
    int32_t n_heads  = static_cast<int32_t>(Q->dims[1]);
    int32_t Sq       = static_cast<int32_t>(Q->dims[2]);
    int32_t head_dim = static_cast<int32_t>(Q->dims[3]);
    int32_t n_kv_heads = static_cast<int32_t>(K->dims[1]);
    int32_t Skv      = static_cast<int32_t>(K->dims[2]);

    if (head_dim != 128) throw std::runtime_error("Vulkan FlashAttention requires head_dim == 128");

    auto& ctx = VulkanContext::get();
    auto& layer = mgr->get_layer(layer_id);

    if (Skv > 0) {
        int32_t prev_cached = layer.num_cached;
        mgr->append_kv_pages(layer_id, Skv);

        size_t elem_sz = data_type_size(K->dtype);
        size_t hd_bytes = static_cast<size_t>(head_dim) * elem_sz;
        size_t head_stride = static_cast<size_t>(Skv) * hd_bytes;
        size_t pool_block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * n_kv_heads * hd_bytes;

        vk::Buffer buf_k = reinterpret_cast<VkBuffer>(K->device_handle);
        vk::Buffer buf_v = reinterpret_cast<VkBuffer>(V->device_handle);
        vk::Buffer buf_kp = reinterpret_cast<VkBuffer>(layer.k_pool.device_handle());
        vk::Buffer buf_vp = reinterpret_cast<VkBuffer>(layer.v_pool.device_handle());

        std::vector<vk::BufferCopy> k_regions, v_regions;

        for (int32_t p = 0; p < Skv; ++p) {
            int32_t global_pos = prev_cached + p;
            int32_t blk = global_pos / PAGE_BLOCK_SIZE;
            int32_t off = global_pos % PAGE_BLOCK_SIZE;

            const PageEntry& e = layer.page_table[blk];
            VkDeviceSize k_dst_base = static_cast<VkDeviceSize>(layer.k_pool.device_offset()) + e.k_block_id * pool_block_bytes + static_cast<VkDeviceSize>(off) * n_kv_heads * hd_bytes;
            VkDeviceSize v_dst_base = static_cast<VkDeviceSize>(layer.v_pool.device_offset()) + e.v_block_id * pool_block_bytes + static_cast<VkDeviceSize>(off) * n_kv_heads * hd_bytes;

            for (int32_t h = 0; h < n_kv_heads; ++h) {
                k_regions.push_back({K->offset + h * head_stride + static_cast<size_t>(p) * hd_bytes,
                                     k_dst_base + h * hd_bytes, hd_bytes});
                v_regions.push_back({V->offset + h * head_stride + static_cast<size_t>(p) * hd_bytes,
                                     v_dst_base + h * hd_bytes, hd_bytes});
            }
        }

        if (!k_regions.empty()) {
            auto cmd = ctx.beginCommandBuffer(dev_id);
            cmd.copyBuffer(buf_k, buf_kp, k_regions);
            cmd.copyBuffer(buf_v, buf_vp, v_regions);
            ctx.endSubmitAndWait(dev_id, cmd);
        }
    }

    int32_t total_cached = layer.num_cached;
    int32_t num_blocks = static_cast<int32_t>(layer.page_table.size());
    if (total_cached <= 0 || num_blocks <= 0) return;

    int32_t block_stride_elems = PAGE_BLOCK_SIZE * n_kv_heads * head_dim;

    std::vector<int32_t> h_page_k(num_blocks), h_page_v(num_blocks);
    for (int32_t i = 0; i < num_blocks; ++i) {
        h_page_k[i] = layer.page_table[i].k_block_id;
        h_page_v[i] = layer.page_table[i].v_block_id;
    }

    size_t pt_bytes = num_blocks * sizeof(int32_t);
    vk::DeviceMemory ptk_mem, ptv_mem;
    void *ptk_ptr = nullptr, *ptv_ptr = nullptr;
    vk::Buffer buf_ptk = ctx.createSmallSSBO(dev_id, pt_bytes, &ptk_mem, &ptk_ptr);
    vk::Buffer buf_ptv = ctx.createSmallSSBO(dev_id, pt_bytes, &ptv_mem, &ptv_ptr);
    std::memcpy(ptk_ptr, h_page_k.data(), pt_bytes);
    std::memcpy(ptv_ptr, h_page_v.data(), pt_bytes);

    size_t k_handle = layer.k_pool.device_handle();
    size_t v_handle = layer.v_pool.device_handle();
    vk::Buffer buf_kp = reinterpret_cast<VkBuffer>(k_handle);
    vk::Buffer buf_vp = reinterpret_cast<VkBuffer>(v_handle);

    PagedAttnPushConstants pc{};
    pc.B           = B;
    pc.n_heads     = n_heads;
    pc.Sq          = Sq;
    pc.n_kv_heads  = n_kv_heads;
    pc.total_cached = total_cached;
    pc.num_blocks  = num_blocks;
    pc.num_kv_groups = static_cast<int32_t>(out->op_params[3]);
    pc.head_dim    = head_dim;
    pc.scale       = out->op_params[1];
    pc.causal      = static_cast<int32_t>(out->op_params[2]) != 0 ? 1 : 0;
    pc.block_stride_elems = block_stride_elems;
    pc.q_start     = total_cached - Sq;

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            const char*   name = std::is_same_v<T, float16_t> ? "page_attn_f16" : "page_attn_bf16";
            const uint32_t* spv = std::is_same_v<T, float16_t> ? vkspv::page_attn_f16_spv : vkspv::page_attn_bf16_spv;
            size_t       spv_len = std::is_same_v<T, float16_t> ? vkspv::page_attn_f16_spv_len : vkspv::page_attn_bf16_spv_len;

            auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 6, sizeof(PagedAttnPushConstants));

            vk::Buffer buf_q = reinterpret_cast<VkBuffer>(Q->device_handle);
            vk::Buffer buf_o = reinterpret_cast<VkBuffer>(out->device_handle);

            auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);
            vk::DescriptorBufferInfo qi(buf_q, Q->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo oi(buf_o, out->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo pki(buf_ptk, 0, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo pvi(buf_ptv, 0, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo kpi(buf_kp, layer.k_pool.device_offset(), VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo vpi(buf_vp, layer.v_pool.device_offset(), VK_WHOLE_SIZE);
            ctx.updateDescriptorSets(dev_id, ds, {qi, oi, pki, pvi, kpi, vpi});

            uint32_t bx = (static_cast<uint32_t>(B * n_heads) + 3) / 4;
            uint32_t by = static_cast<uint32_t>(Sq);

            auto cmd = ctx.beginCommandBuffer(dev_id);
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
            cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PagedAttnPushConstants), &pc);
            cmd.dispatch(bx, by, 1);
            ctx.endSubmitAndWait(dev_id, cmd);

            ctx.freeDescriptorSet(dev_id, ds);
        }
    });

    ctx.destroySmallBuffer(buf_ptk, ptk_mem, ptk_ptr);
    ctx.destroySmallBuffer(buf_ptv, ptv_mem, ptv_ptr);
}

void SoftmaxImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan SoftmaxImpl: not implemented");
}
void DiagMaskInfImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan DiagMaskInfImpl: not implemented");
}

template struct SoftmaxImpl<Device::VULKAN>;
template struct DiagMaskInfImpl<Device::VULKAN>;
template struct SdpaImpl<Device::VULKAN>;
template struct AttentionImpl<Device::VULKAN>;
template struct FlashAttentionImpl<Device::VULKAN>;

}

#endif
