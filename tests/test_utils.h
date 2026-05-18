#pragma once
#include <print>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

struct TestCase {
    std::string name;
    std::function<bool()> fn;
};

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::println("    ✗ at {}:{} — {}: {}", __FILE__, __LINE__, msg, #cond); \
            return false; \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, eps, msg) \
    do { \
        auto _a = (a); auto _b = (b); \
        if (std::abs(static_cast<double>(_a) - static_cast<double>(_b)) > static_cast<double>(eps)) { \
            std::println("    ✗ at {}:{} — {}: {} ≈ {} (eps={})", __FILE__, __LINE__, msg, \
                         static_cast<double>(_a), static_cast<double>(_b), static_cast<double>(eps)); \
            return false; \
        } \
    } while(0)

inline int run_tests(const std::vector<TestCase>& tests) {
    int passed = 0, failed = 0;
    for (auto& t : tests) {
        bool ok = t.fn();
        std::println("  {}  {}", ok ? "PASS" : "FAIL", t.name);
        if (ok) ++passed; else ++failed;
    }
    std::println("\n{}/{} passed", passed, passed + failed);
    return failed > 0 ? 1 : 0;
}
