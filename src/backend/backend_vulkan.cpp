#include "backend/backend.h"

#ifdef BACKEND_VULKAN

#include "backend/vulkan/vulkan_context.h"
#include <vulkan/vulkan.hpp>
#include <print>

VulkanBackendProvider::VulkanBackendProvider() {
    try {
        ctx_ = &VulkanContext::get();
        available_ = ctx_->device_count() > 0;
    } catch (const std::exception& e) {
        std::println("VulkanBackendProvider: initialization failed: {}", e.what());
        available_ = false;
    }
}

bool VulkanBackendProvider::is_available() const { 
    return available_; 
}
int VulkanBackendProvider::get_device_count() const { 
    return available_ ? ctx_->device_count() : 0;
 }

BackendInfo VulkanBackendProvider::get_backend_info(int device_id) const {
    BackendInfo info;
    info.id = static_cast<size_t>(device_id);
    info.device = Device::VULKAN;

    if (!available_) return info;

    auto phy = ctx_->physical_device(device_id);
    auto mem_props = phy.getMemoryProperties();

    size_t total = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            total += mem_props.memoryHeaps[i].size;
        }
    }

    info.total_memory = total;
    info.used_memory = 0;
    info.compute_power = 1.0;
    info.bandwidth = 32.0;
    return info;
}

void VulkanBackendProvider::print_device_info(int device_id) const {
    if (!available_) return;

    auto phy = ctx_->physical_device(device_id);
    auto props = phy.getProperties();
    auto mem_props = phy.getMemoryProperties();

    std::println("  Device Name:       {}", std::string_view(props.deviceName));
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        auto heap = mem_props.memoryHeaps[i];
        bool is_device_local = (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
        std::println("  Memory Heap[{}]:    {} MB ({})",
            i, heap.size / (1024 * 1024),
            is_device_local ? "DeviceLocal (VRAM)" : "HostVisible (System RAM)");
    }
    std::println("  Max Work Group Size:    [{}, {}, {}]",
        props.limits.maxComputeWorkGroupSize[0],
        props.limits.maxComputeWorkGroupSize[1],
        props.limits.maxComputeWorkGroupSize[2]);
    std::println("  Max Work Group Count:    [{}, {}, {}]",
        props.limits.maxComputeWorkGroupCount[0],
        props.limits.maxComputeWorkGroupCount[1],
        props.limits.maxComputeWorkGroupCount[2]);
    std::println("  Max Work Group Invocations:    {}", props.limits.maxComputeWorkGroupInvocations);

    auto print_cooperative_matrix_info = [&]() {
        auto inst = ctx_->instance();
        auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            inst.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (!fn) {
            std::println("  Cooperative Matrix: not supported");
            return;
        }

        uint32_t count = 0;
        VkResult result = fn(static_cast<VkPhysicalDevice>(phy), &count, nullptr);
        if (result != VK_SUCCESS || count == 0) {
            std::println("  Cooperative Matrix: no configurations");
            return;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> cm_props(count);
        for (auto& p : cm_props) {
            p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            p.pNext = nullptr;
        }
        fn(static_cast<VkPhysicalDevice>(phy), &count, cm_props.data());

        std::println("  --- Cooperative Matrix (KHR) ---");
        for (size_t i = 0; i < cm_props.size(); ++i) {
            auto& p = cm_props[i];
            std::println("    [{0}]: M={1} N={2} K={3} A={4} B={5} C={6} Result={7} scope={8} satAcc={9}",
                i, p.MSize, p.NSize, p.KSize,
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.AType)),
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.BType)),
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.CType)),
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.ResultType)),
                vk::to_string(static_cast<vk::ScopeKHR>(p.scope)),
                static_cast<bool>(p.saturatingAccumulation));
        }
    };
    print_cooperative_matrix_info();
}

static struct VulkanBackendProviderRegistrar {
    VulkanBackendProviderRegistrar() {
        auto provider = std::make_unique<VulkanBackendProvider>();
        if (provider->is_available()) {
            BackendRegistry::instance().register_provider(std::move(provider));
        }
    }
} g_vulkan_backend_registrar;

#endif // BACKEND_VULKAN
