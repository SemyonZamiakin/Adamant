// Minimal test helpers for Adamant.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define ADM_TEST_PID() static_cast<long>(_getpid())
#else
#include <unistd.h>
#define ADM_TEST_PID() static_cast<long>(getpid())
#endif

inline std::atomic<int> g_failures{0};

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_failures.fetch_add(1);                                                  \
        }                                                                             \
    } while (0)

#define CHECK_MSG(cond, ...)                                                          \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                        \
            std::fprintf(stderr, "\n");                                               \
            g_failures.fetch_add(1);                                                  \
        }                                                                             \
    } while (0)

inline int test_result(const char* name) {
    const int f = g_failures.load();
    if (f) std::fprintf(stderr, "%s: %d failure(s)\n", name, f);
    else std::printf("%s: OK\n", name);
    return f ? 1 : 0;
}

inline bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

inline std::vector<float> gaussian(std::size_t n, std::mt19937& rng, float sigma = 1.f) {
    std::normal_distribution<float> nd(0.f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

inline std::string temp_name(const char* stem) {
    return std::string(stem) + "-" + std::to_string(ADM_TEST_PID()) + ".gguf";
}
