#pragma once

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <format>
#include <print>
#include <cstddef>
#include <memory>

class VulkanContext {
public:
    // ==================== Pipeline 信息 ====================
    struct PipelineInfo {
        vk::Pipeline pipeline;
        vk::PipelineLayout layout;
        vk::DescriptorSetLayout ds_layout;
        uint32_t binding_count;
    };

    // ==================== 每设备资源 ====================
    struct DeviceSlot {
        uint32_t subgroup_size = 32;
        uint32_t compute_queue_family = 0;
        vk::PhysicalDevice physical_device;
        vk::Device device;
        vk::Queue compute_queue;
        vk::CommandPool command_pool;
        vk::DescriptorPool descriptor_pool;
        std::unique_ptr<std::mutex> pipeline_mutex = std::make_unique<std::mutex>();
        std::unordered_map<std::string, PipelineInfo> pipeline_infos;

        ~DeviceSlot() {
            if (!device) return;
            device.waitIdle();
            for (auto& [name, info] : pipeline_infos) {
                device.destroyPipeline(info.pipeline);
                device.destroyPipelineLayout(info.layout);
                device.destroyDescriptorSetLayout(info.ds_layout);
            }
            device.destroyDescriptorPool(descriptor_pool);
            device.destroyCommandPool(command_pool);
            device.destroy();
        }
        DeviceSlot() = default;
        DeviceSlot(DeviceSlot&& o) noexcept
            : physical_device(o.physical_device)
            , device(std::exchange(o.device, nullptr))
            , compute_queue(o.compute_queue)
            , compute_queue_family(o.compute_queue_family)
            , command_pool(std::exchange(o.command_pool, nullptr))
            , descriptor_pool(std::exchange(o.descriptor_pool, nullptr))
            , subgroup_size(o.subgroup_size)
            , pipeline_mutex(std::move(o.pipeline_mutex))
            , pipeline_infos(std::move(o.pipeline_infos))
        {}
        DeviceSlot(const DeviceSlot&) = delete;
        DeviceSlot& operator=(const DeviceSlot&) = delete;
        DeviceSlot& operator=(DeviceSlot&&) = delete;
    };

    // ==================== 单例 ====================

    static VulkanContext& get() {
        static VulkanContext instance;
        return instance;
    }

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    ~VulkanContext() {
        devices_.clear();
        if (instance_) instance_.destroy();
    }

    // ==================== 设备访问 ====================

    int device_count() const { return static_cast<int>(devices_.size()); }
    DeviceSlot& slot(int device_id) { return devices_.at(device_id); }
    const DeviceSlot& slot(int device_id) const { return devices_.at(device_id); }

    vk::Instance instance() const { return instance_; }
    vk::Device device(int device_id) const { return slot(device_id).device; }
    vk::PhysicalDevice physical_device(int device_id) const { return slot(device_id).physical_device; }
    uint32_t subgroup_size(int device_id) const { return slot(device_id).subgroup_size; }

    // ==================== Pipeline 管理（懒加载） ====================

    const PipelineInfo& getOrCreatePipeline(
        int device_id,
        const std::string& name,
        const uint32_t* spv_data,
        size_t spv_size_words,
        uint32_t binding_count,
        uint32_t push_constant_size)
    {
        auto& s = slot(device_id);
        std::lock_guard<std::mutex> lock(*s.pipeline_mutex);
        auto it = s.pipeline_infos.find(name);
        if (it != s.pipeline_infos.end()) return it->second;

        auto dev = s.device;

        std::vector<vk::DescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            bindings[i] = vk::DescriptorSetLayoutBinding(
                i, vk::DescriptorType::eStorageBuffer, 1,
                vk::ShaderStageFlagBits::eCompute);
        }
        vk::DescriptorSetLayoutCreateInfo ds_info({}, bindings);
        auto ds_layout = dev.createDescriptorSetLayout(ds_info);

        vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0, push_constant_size);
        vk::PipelineLayoutCreateInfo layout_info({}, ds_layout, pc_range);
        auto layout = dev.createPipelineLayout(layout_info);

        vk::ShaderModuleCreateInfo shader_info({}, spv_size_words * sizeof(uint32_t), spv_data);
        auto shader_module = dev.createShaderModule(shader_info);

        vk::PipelineShaderStageCreateInfo stage_info({}, vk::ShaderStageFlagBits::eCompute, shader_module, "main");
        vk::ComputePipelineCreateInfo pipe_info({}, stage_info, layout);
        auto [result, pipeline] = dev.createComputePipeline(nullptr, pipe_info);
        dev.destroyShaderModule(shader_module);

        if (result != vk::Result::eSuccess) {
            dev.destroyDescriptorSetLayout(ds_layout);
            dev.destroyPipelineLayout(layout);
            throw std::runtime_error(std::format(
                "VulkanContext: createComputePipeline '{}' failed ({})", name, vk::to_string(result)));
        }

        auto [ins_it, ok] = s.pipeline_infos.emplace(name, PipelineInfo{
            pipeline, layout, ds_layout, binding_count});
        return ins_it->second;
    }

    // ==================== Descriptor Set ====================

    vk::DescriptorSet allocateDescriptorSet(int device_id, vk::DescriptorSetLayout layout) {
        auto& s = slot(device_id);
        vk::DescriptorSetAllocateInfo alloc_info(s.descriptor_pool, 1, &layout);
        auto sets = s.device.allocateDescriptorSets(alloc_info);
        if (sets.empty()) throw std::runtime_error("VulkanContext: allocateDescriptorSet failed");
        return sets[0];
    }

    void freeDescriptorSet(int device_id, vk::DescriptorSet ds) {
        auto& s = slot(device_id);
        s.device.freeDescriptorSets(s.descriptor_pool, ds);
    }

    void updateDescriptorSets(int device_id,
        vk::DescriptorSet ds,
        const std::vector<vk::DescriptorBufferInfo>& buffer_infos)
    {
        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(buffer_infos.size());
        for (uint32_t i = 0; i < buffer_infos.size(); ++i) {
            writes.emplace_back(ds, i, 0, 1, vk::DescriptorType::eStorageBuffer,
                nullptr, &buffer_infos[i]);
        }
        slot(device_id).device.updateDescriptorSets(writes, {});
    }

    // ==================== 命令提交 ====================

    vk::CommandBuffer beginCommandBuffer(int device_id) {
        auto& s = slot(device_id);
        vk::CommandBufferAllocateInfo alloc_info(s.command_pool, vk::CommandBufferLevel::ePrimary, 1);
        auto cmds = s.device.allocateCommandBuffers(alloc_info);
        if (cmds.empty()) throw std::runtime_error("VulkanContext: allocateCommandBuffer failed");
        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmds[0].begin(begin_info);
        return cmds[0];
    }

    void endSubmitAndWait(int device_id, vk::CommandBuffer cmd) {
        auto& s = slot(device_id);
        cmd.end();
        vk::SubmitInfo submit_info({}, {}, cmd);
        vk::Fence fence = s.device.createFence({});
        s.compute_queue.submit(submit_info, fence);
        VkFence vk_fence = static_cast<VkFence>(fence);
        vkWaitForFences(static_cast<VkDevice>(s.device), 1, &vk_fence, VK_TRUE, UINT64_MAX);
        s.device.destroyFence(fence);
        s.device.freeCommandBuffers(s.command_pool, cmd);
    }

    // ==================== Buffer 辅助 ====================

    vk::Buffer createStagingBuffer(int device_id, size_t size, vk::DeviceMemory* out_memory, void** out_mapped) {
        auto& s = slot(device_id);
        auto dev = s.device;
        vk::BufferCreateInfo buf_info({}, size,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive);
        auto buffer = dev.createBuffer(buf_info);

        auto mem_reqs = dev.getBufferMemoryRequirements(buffer);
        auto mem_props = s.physical_device.getMemoryProperties();
        uint32_t mem_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_reqs.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags &
                 (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
                mem_type = i;
                break;
            }
        }
        if (mem_type == UINT32_MAX) {
            dev.destroyBuffer(buffer);
            throw std::runtime_error("VulkanContext: no host-visible memory for staging");
        }

        vk::MemoryAllocateInfo alloc_info(mem_reqs.size, mem_type);
        auto memory = dev.allocateMemory(alloc_info);
        dev.bindBufferMemory(buffer, memory, 0);
        *out_memory = memory;
        *out_mapped = dev.mapMemory(memory, 0, size);
        return buffer;
    }

    void destroyStagingBuffer(int device_id, vk::Buffer buffer, vk::DeviceMemory memory, void* mapped) {
        auto dev = slot(device_id).device;
        if (mapped) dev.unmapMemory(memory);
        if (buffer) dev.destroyBuffer(buffer);
        if (memory) dev.freeMemory(memory);
    }

    vk::Buffer createSmallSSBO(int device_id, size_t size, vk::DeviceMemory* out_mem, void** out_mapped) {
        auto& s = slot(device_id);
        auto dev = s.device;
        vk::BufferCreateInfo bi({}, size, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive);
        auto buf = dev.createBuffer(bi);
        auto mr = dev.getBufferMemoryRequirements(buf);
        auto mp = s.physical_device.getMemoryProperties();
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1 << i)) &&
                (mp.memoryTypes[i].propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
                mt = i; break;
            }
        }
        vk::MemoryAllocateInfo ai(mr.size, mt);
        auto mem = dev.allocateMemory(ai);
        dev.bindBufferMemory(buf, mem, 0);
        *out_mem = mem;
        *out_mapped = dev.mapMemory(mem, 0, size);
        return buf;
    }

    void destroySmallBuffer(vk::Buffer buf, vk::DeviceMemory mem, void* mapped) {
        auto& s = slot(0);
        if (mapped) s.device.unmapMemory(mem);
        if (buf) s.device.destroyBuffer(buf);
        if (mem) s.device.freeMemory(mem);
    }

    // ==================== Buffer → device_id 查找 ====================

    void registerBuffer(size_t buffer_handle, int device_id) {
        buffer_device_[buffer_handle] = device_id;
    }

    int bufferDeviceId(size_t buffer_handle) const {
        auto it = buffer_device_.find(buffer_handle);
        if (it == buffer_device_.end())
            throw std::runtime_error("VulkanContext: buffer handle not registered");
        return it->second;
    }

private:
    VulkanContext() { init(); }

    void init() {
        this->createInstance();
        this->enumerateAndCreateDevices();
    }

    void createInstance() {
        vk::ApplicationInfo app_info;
        app_info.setPApplicationName("Genllm")
                .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
                .setPEngineName("GenllmEngine")
                .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
                .setApiVersion(VK_API_VERSION_1_4);

        vk::InstanceCreateInfo create_info({}, &app_info);

        auto result = vk::createInstance(&create_info, nullptr, &instance_);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("VulkanContext: createInstance failed ({})", vk::to_string(result)));
        }
    }

    void enumerateAndCreateDevices() {
        auto phys_devices = instance_.enumeratePhysicalDevices();
        if (phys_devices.empty()) {
            throw std::runtime_error("VulkanContext: no Vulkan-capable devices");
        }

        for (auto& phy : phys_devices) {
            auto props = phy.getProperties();
            auto families = phy.getQueueFamilyProperties();

            bool has_compute = false;
            uint32_t queue_family = 0;
            for (uint32_t i = 0; i < families.size(); ++i) {
                if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                    has_compute = true;
                    queue_family = i;
                    break;
                }
            }
            if (!has_compute) continue;

            vk::PhysicalDeviceSubgroupProperties subgroup_props;
            vk::PhysicalDeviceProperties2 props2({}, &subgroup_props);
            phy.getProperties2(&props2);
            uint32_t sg_size = subgroup_props.subgroupSize;

            float priority = 1.0f;
            vk::DeviceQueueCreateInfo queue_info({}, queue_family, 1, &priority);

            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR coop_feat;
            coop_feat.cooperativeMatrix = VK_TRUE;
            vk::DeviceCreateInfo device_info({}, queue_info);
            device_info.pNext = &coop_feat;

            vk::Device dev;
            try {
                dev = phy.createDevice(device_info);
            } catch (const vk::SystemError& e) {
                vk::DeviceCreateInfo device_info_fallback({}, queue_info);
                try {
                    dev = phy.createDevice(device_info_fallback);
                } catch (const vk::SystemError& e2) {
                    std::println("Vulkan: skipping device {} ({})", std::string_view(props.deviceName), e2.what());
                    continue;
                }
            }

            auto queue = dev.getQueue(queue_family, 0);

            vk::CommandPoolCreateInfo cmd_pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family);
            auto cmd_pool = dev.createCommandPool(cmd_pool_info);

            vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 4096);
            vk::DescriptorPoolCreateInfo desc_pool_info(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1024, pool_size);
            auto desc_pool = dev.createDescriptorPool(desc_pool_info);

            DeviceSlot slot;
            slot.physical_device = phy;
            slot.device = dev;
            slot.compute_queue = queue;
            slot.compute_queue_family = queue_family;
            slot.command_pool = cmd_pool;
            slot.descriptor_pool = desc_pool;
            slot.subgroup_size = sg_size;
            devices_.push_back(std::move(slot));

            std::println("Vulkan[{}]: {}", devices_.size() - 1, std::string_view(props.deviceName));
        }

        if (devices_.empty()) {
            throw std::runtime_error("VulkanContext: no suitable GPU found");
        }
    }

    vk::Instance instance_;
    std::vector<DeviceSlot> devices_;
    std::unordered_map<size_t, int> buffer_device_;
};

#endif // BACKEND_VULKAN
