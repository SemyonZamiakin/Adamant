// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/common.hpp"

#include <chrono>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace adamant {

void* aligned_malloc(std::size_t bytes, std::size_t align) {
    if (bytes == 0) bytes = align;
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

void aligned_free(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

namespace {

float half_bits_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | static_cast<std::uint32_t>(127 - 15 - e) << 23 | mant << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | mant << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | mant << 13;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct HalfTable {
    std::vector<float> t;
    HalfTable() : t(65536) {
        for (std::uint32_t i = 0; i < 65536; ++i) t[i] = half_bits_to_float(static_cast<std::uint16_t>(i));
    }
};

const HalfTable& half_table() {
    static const HalfTable table;
    return table;
}

}  // namespace

float fp16_to_fp32(std::uint16_t h) noexcept { return half_table().t[h]; }

std::uint16_t fp32_to_fp16(float f) noexcept {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0u));
    if (absx >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (absx < 0x38800000u) {                                                    // subnormal or zero
        if (absx < 0x33000000u) return static_cast<std::uint16_t>(sign);
        const std::uint32_t e = absx >> 23;
        const std::uint32_t m = (absx & 0x7FFFFFu) | 0x800000u;
        // value = m * 2^(e-150); in units of the smallest half subnormal 2^-24: m >> (126 - e)
        const std::uint32_t drop = 126u - e;  // 14..24
        std::uint32_t r = m >> drop;
        const std::uint32_t rem = m & ((1u << drop) - 1u);
        const std::uint32_t half = 1u << (drop - 1u);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return static_cast<std::uint16_t>(sign | r);
    }
    std::uint32_t r = ((absx >> 13) - (112u << 10));
    const std::uint32_t rem = absx & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return static_cast<std::uint16_t>(sign | r);
}

std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace adamant
