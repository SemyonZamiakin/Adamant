// Adamant SIMD-VIC by Euclase: quantized block formats and compute kernels.
//
// Block layouts are byte-compatible with ggml / GGUF, so weights are used
// straight from the mapped file. Each kernel exists twice: a portable scalar
// reference and a hand-written AVX2 version. Both compute the integer part of
// every block dot product exactly and then apply the floating-point scales in
// the same order (and both are compiled with -ffp-contract=off), so the AVX2
// results are bit-identical to the scalar reference. tests/test_kernels.cpp
// checks exactly that.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <cstddef>
#include <cstdint>

namespace adamant {

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
    Q8_K = 15,  // activations only
};

inline constexpr int QK8_0 = 32;
inline constexpr int QK_K = 256;
inline constexpr int K_SCALE_SIZE = 12;

// 32 weights: w = d * q, q in [-127, 127]. 34 bytes = 8.5 bits per weight.
struct BlockQ8_0 {
    std::uint16_t d;  // fp16 scale
    std::int8_t qs[QK8_0];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 layout");

// 256 weights in 8 sub-blocks of 32. w = d*sc[j]*q - dmin*m[j], q in [0, 15],
// sc and m are 6-bit, packed into scales[12]. 144 bytes = 4.5 bits per weight.
struct BlockQ4K {
    std::uint16_t d;     // fp16 super-block scale for the sub-block scales
    std::uint16_t dmin;  // fp16 super-block scale for the sub-block mins
    std::uint8_t scales[K_SCALE_SIZE];
    std::uint8_t qs[QK_K / 2];  // nibbles: low nibble = first 32 of a 64-chunk, high = next 32
};
static_assert(sizeof(BlockQ4K) == 144, "BlockQ4K layout");

// 256 weights in 16 sub-blocks of 16. w = d * sc[j] * (q - 32), q in [0, 63]
// (low 4 bits in ql, high 2 bits in qh). 210 bytes = 6.5625 bits per weight.
struct BlockQ6K {
    std::uint8_t ql[QK_K / 2];
    std::uint8_t qh[QK_K / 4];
    std::int8_t scales[QK_K / 16];
    std::uint16_t d;
};
static_assert(sizeof(BlockQ6K) == 210, "BlockQ6K layout");

// Activation block for K-quant dot products: 256 values, float scale, and the
// sum of every 16 quants (lets the Q4_K kernel apply the mins without touching
// the quants a second time).
struct BlockQ8K {
    float d;
    std::int8_t qs[QK_K];
    std::int16_t bsums[QK_K / 16];
};
static_assert(sizeof(BlockQ8K) == 292, "BlockQ8K layout");

bool ggml_type_known(GgmlType t);
const char* ggml_type_name(GgmlType t);
std::int64_t ggml_block_size(GgmlType t);   // elements per block
std::size_t ggml_type_size(GgmlType t);     // bytes per block
std::size_t ggml_row_size(GgmlType t, std::int64_t n);
double ggml_bits_per_weight(GgmlType t);

// Unpacks sub-block j (0..7) scale and min of a Q4_K block.
inline void q4k_scale_min(int j, const std::uint8_t* q, std::uint8_t* sc, std::uint8_t* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// A set of kernels for one instruction set. `n` is always a multiple of the
// block size of the format involved. Pointers need only 2-byte alignment (GGUF only
// guarantees 32-byte tensor alignment and blocks are 34 / 144 / 210 bytes).
struct Kernels {
    const char* name;

    // activations: float -> Q8_0 / Q8_K
    void (*quantize_q8_0)(const float* x, BlockQ8_0* y, std::int64_t n);
    void (*quantize_q8_K)(const float* x, BlockQ8K* y, std::int64_t n);

    // dot products of one weight row with quantized activations
    float (*dot_q8_0)(const BlockQ8_0* w, const BlockQ8_0* x, std::int64_t n);
    float (*dot_q4_K)(const BlockQ4K* w, const BlockQ8K* x, std::int64_t n);
    float (*dot_q6_K)(const BlockQ6K* w, const BlockQ8K* x, std::int64_t n);
    float (*dot_f16)(const std::uint16_t* w, const float* x, std::int64_t n);
    float (*dot_f32)(const float* w, const float* x, std::int64_t n);

    // weight row -> float
    void (*dequant_q8_0)(const BlockQ8_0* w, float* y, std::int64_t n);
    void (*dequant_q4_K)(const BlockQ4K* w, float* y, std::int64_t n);
    void (*dequant_q6_K)(const BlockQ6K* w, float* y, std::int64_t n);
    void (*dequant_f16)(const std::uint16_t* w, float* y, std::int64_t n);
};

const Kernels& kernels_scalar();
// nullptr when Adamant was built without AVX2 support or the CPU lacks AVX2/FMA/F16C.
const Kernels* kernels_avx2();
// Best kernels for this CPU; ADAMANT_ISA=scalar|avx2 overrides.
const Kernels& kernels_default();
// By name ("scalar", "avx2"); nullptr if unavailable.
const Kernels* kernels_by_name(const char* name);
bool cpu_has_avx2_fma_f16c();

// Dequantizes n elements of any supported weight type.
void dequantize_row(const Kernels& k, GgmlType t, const void* src, float* dst, std::int64_t n);

// Reference quantizers (scalar). Used to write test models and for the
// round-trip tests; not on the inference hot path.
void quantize_row_q8_0_ref(const float* x, BlockQ8_0* y, std::int64_t n);
void quantize_row_q4_K_ref(const float* x, BlockQ4K* y, std::int64_t n);
void quantize_row_q6_K_ref(const float* x, BlockQ6K* y, std::int64_t n);
void quantize_row(GgmlType t, const float* x, void* dst, std::int64_t n);

namespace detail {
// Defined in quant_avx2.cpp when ADAMANT_HAVE_AVX2 is set.
const Kernels& kernels_avx2_impl();
}  // namespace detail

}  // namespace adamant
