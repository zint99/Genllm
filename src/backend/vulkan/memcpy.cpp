#include "backend/vulkan/memcpy.h"
#include "utils/utils.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ops {

void MemcpyImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    Tensor* src = out->src[0];

    if (!src) {
        throw std::runtime_error("MemcpyImpl<VULKAN>: source tensor is null");
    }

    size_t nbytes = out->bytes();
    Device src_dev = src->device;
    Device dst_dev = out->device;

    // ── CPU → Vulkan ──
    if (src_dev == Device::CPU && dst_dev == Device::VULKAN) {
        vk::DeviceMemory staging_mem;
        void* staging_mapped;
        vk::Buffer staging_buf = ctx.createStagingBuffer(dev_id, nbytes, &staging_mem, &staging_mapped);

        std::memcpy(staging_mapped, src->data, nbytes);

        auto cmd = ctx.beginCommandBuffer(dev_id);
        vk::BufferCopy region(0, out->offset, nbytes);
        cmd.copyBuffer(staging_buf,
                       reinterpret_cast<VkBuffer>(out->device_handle),
                       region);
        ctx.endSubmitAndWait(dev_id, cmd);

        ctx.destroyStagingBuffer(dev_id, staging_buf, staging_mem, staging_mapped);
        return;
    }

    // ── Vulkan → CPU ──
    if (src_dev == Device::VULKAN && dst_dev == Device::CPU) {
        vk::DeviceMemory staging_mem;
        void* staging_mapped;
        vk::Buffer staging_buf = ctx.createStagingBuffer(dev_id, nbytes, &staging_mem, &staging_mapped);

        auto cmd = ctx.beginCommandBuffer(dev_id);
        vk::BufferCopy region(src->offset, 0, nbytes);
        cmd.copyBuffer(reinterpret_cast<VkBuffer>(src->device_handle),
                       staging_buf,
                       region);
        ctx.endSubmitAndWait(dev_id, cmd);

        std::memcpy(out->data, staging_mapped, nbytes);

        ctx.destroyStagingBuffer(dev_id, staging_buf, staging_mem, staging_mapped);
        return;
    }

    // ── Vulkan → Vulkan (同设备或跨设备) ──
    if (src_dev == Device::VULKAN && dst_dev == Device::VULKAN) {
        int src_dev_id = ctx.bufferDeviceId(src->device_handle);

        if (src_dev_id == dev_id) {
            // 同设备：直接 vkCmdCopyBuffer
            auto cmd = ctx.beginCommandBuffer(dev_id);
            vk::BufferCopy region(src->offset, out->offset, nbytes);
            cmd.copyBuffer(reinterpret_cast<VkBuffer>(src->device_handle),
                           reinterpret_cast<VkBuffer>(out->device_handle),
                           region);
            ctx.endSubmitAndWait(dev_id, cmd);
        } else {
            // 跨设备：Vulkan:0 → host → Vulkan:1
            std::vector<std::byte> staging(nbytes);

            // D2H from source device
            {
                vk::DeviceMemory s_mem;
                void* s_mapped;
                vk::Buffer s_buf = ctx.createStagingBuffer(src_dev_id, nbytes, &s_mem, &s_mapped);

                auto cmd = ctx.beginCommandBuffer(src_dev_id);
                vk::BufferCopy region(src->offset, 0, nbytes);
                cmd.copyBuffer(reinterpret_cast<VkBuffer>(src->device_handle),
                               s_buf, region);
                ctx.endSubmitAndWait(src_dev_id, cmd);

                std::memcpy(staging.data(), s_mapped, nbytes);
                ctx.destroyStagingBuffer(src_dev_id, s_buf, s_mem, s_mapped);
            }

            // H2D to destination device
            {
                vk::DeviceMemory d_mem;
                void* d_mapped;
                vk::Buffer d_buf = ctx.createStagingBuffer(dev_id, nbytes, &d_mem, &d_mapped);

                std::memcpy(d_mapped, staging.data(), nbytes);

                auto cmd = ctx.beginCommandBuffer(dev_id);
                vk::BufferCopy region(0, out->offset, nbytes);
                cmd.copyBuffer(d_buf,
                               reinterpret_cast<VkBuffer>(out->device_handle),
                               region);
                ctx.endSubmitAndWait(dev_id, cmd);

                ctx.destroyStagingBuffer(dev_id, d_buf, d_mem, d_mapped);
            }
        }
        return;
    }

    throw std::runtime_error(
        "MemcpyImpl<VULKAN>: unsupported copy (" +
        device_to_string(src_dev) + " -> " + device_to_string(dst_dev) + ")");
}

template struct MemcpyImpl<Device::VULKAN>;

}

#endif
