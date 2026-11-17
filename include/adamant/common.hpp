// Adamant SIMD-VIC by Euclase: common definitions.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#define ADAMANT_VERSION_MAJOR 0
#define ADAMANT_VERSION_MINOR 1
#define ADAMANT_VERSION_PATCH 0
#define ADAMANT_VERSION_STRING "0.1.0"

namespace adamant {

// One x86-64 cache line. Every buffer Adamant allocates (activations, KV cache,
// scratch) starts on this boundary, so a 32-byte AVX2 load never splits a line.
inline constexpr std::size_t kCacheLine = 64;

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void* aligned_malloc(std::size_t bytes, std::size_t align = kCacheLine);
void aligned_free(void* p) noexcept;

// Owning, cache-line aligned, zero-initialised array of trivially copyable T.
// The byte size is rounded up to a whole number of cache lines, so the tail of
// the last line belongs to this buffer and no other object shares it.
template <class T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n) { resize(n); }
    ~AlignedBuffer() { aligned_free(p_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            aligned_free(p_);
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }

    void resize(std::size_t n) {
        aligned_free(p_);
        p_ = nullptr;
        n_ = 0;
        if (n == 0) return;
        const std::size_t bytes = (n * sizeof(T) + kCacheLine - 1) / kCacheLine * kCacheLine;
        p_ = static_cast<T*>(aligned_malloc(bytes));
        std::memset(static_cast<void*>(p_), 0, bytes);
        n_ = n;
    }

    T* data() noexcept { return p_; }
    const T* data() const noexcept { return p_; }
    std::size_t size() const noexcept { return n_; }
    T& operator[](std::size_t i) noexcept { return p_[i]; }
    const T& operator[](std::size_t i) const noexcept { return p_[i]; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

// IEEE half <-> float. fp16_to_fp32 is exact (table lookup, 256 KiB, built once).
float fp16_to_fp32(std::uint16_t h) noexcept;
std::uint16_t fp32_to_fp16(float f) noexcept;  // round to nearest even

// Monotonic clock in nanoseconds.
std::uint64_t now_ns() noexcept;

}  // namespace adamant
