#include "test_utils.h"
#include "core/page_attention.h"

static bool test_block_pool_alloc_free() {
    std::byte buf[65536];
    BlockPool pool(buf, 65536, 100, 8, 128, DataType::GGML_TYPE_BF16);
    ASSERT(pool.num_blocks() == 0, "initially 0 blocks");
    ASSERT(pool.num_free() == 0, "initially 0 free");

    int32_t id = pool.alloc();
    ASSERT(id == 0, "first alloc returns 0");
    ASSERT(pool.num_blocks() == 1, "1 block after alloc");
    ASSERT(pool.num_free() == 0, "0 free after alloc");

    int32_t id2 = pool.alloc();
    ASSERT(id2 == 1, "second alloc returns 1");

    pool.free(id);
    ASSERT(pool.num_free() == 1, "1 free after free");

    int32_t id3 = pool.alloc();
    ASSERT(id3 == id, "reuse freed block");
    ASSERT(pool.num_free() == 0, "0 free after re-alloc");
    return true;
}

static bool test_block_pool_many_allocs() {
    std::byte buf[65536];
    BlockPool pool(buf, 65536, 10, 8, 128, DataType::GGML_TYPE_BF16);
    for (int i = 0; i < 10; i++) {
        ASSERT(pool.alloc() >= 0, "alloc within capacity");
    }
    ASSERT(pool.alloc() == -1, "beyond capacity returns -1");
    return true;
}

static bool test_block_pool_reset() {
    std::byte buf[65536];
    BlockPool pool(buf, 65536, 10, 8, 128, DataType::GGML_TYPE_BF16);
    (void)pool.alloc(); (void)pool.alloc(); (void)pool.alloc();
    pool.reset();
    ASSERT(pool.num_blocks() == 3, "reset keeps num_blocks");
    ASSERT(pool.num_free() == 3, "reset adds all back to free");
    // reset 后 free_list = [0,1,2] (push_back), alloc 从末尾 pop → 返回 2
    ASSERT(pool.alloc() >= 0, "reset allows re-alloc");
    return true;
}

static bool test_page_table_append_kv() {
    PagedAttentionManager mgr;
    mgr.init_layer(0, 8, 128, DataType::GGML_TYPE_BF16);
    // Without a real pool, we only test the page table logic
    auto& layer = mgr.get_layer(0);
    ASSERT(layer.active, "layer active");
    ASSERT(layer.num_cached == 0, "initially 0 cached");

    mgr.append_kv_pages(0, 1); // 1 token
    ASSERT(layer.num_cached == 1, "1 token cached");

    mgr.append_kv_pages(0, 31); // 1 more page
    ASSERT(layer.num_cached == 32, "32 tokens cached");
    ASSERT(layer.page_table.size() == 2, "2 pages (16+16)");
    return true;
}

static bool test_page_table_multi_layer() {
    PagedAttentionManager mgr;
    mgr.init_layer(0, 8, 128, DataType::GGML_TYPE_F16);
    mgr.init_layer(1, 4, 64, DataType::GGML_TYPE_F32);

    ASSERT(mgr.is_active(0), "layer 0 active");
    ASSERT(mgr.is_active(1), "layer 1 active");
    ASSERT(!mgr.is_active(2), "layer 2 not active");

    mgr.append_kv_pages(0, 5);
    mgr.append_kv_pages(1, 20);

    ASSERT(mgr.get_layer(0).num_cached == 5, "layer0 cached 5");
    ASSERT(mgr.get_layer(1).num_cached == 20, "layer1 cached 20");

    mgr.reset();
    ASSERT(mgr.get_layer(0).num_cached == 0, "after reset layer0 = 0");
    ASSERT(mgr.get_layer(1).page_table.empty(), "after reset page_table empty");
    return true;
}

int main() {
    return run_tests({
        {"block_pool alloc/free",             test_block_pool_alloc_free},
        {"block_pool many allocs",            test_block_pool_many_allocs},
        {"block_pool reset",                  test_block_pool_reset},
        {"page_table append_kv",              test_page_table_append_kv},
        {"page_table multi layer",            test_page_table_multi_layer},
    });
}
