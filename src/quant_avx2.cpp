// Hand-written AVX2 kernels. Compiled with -mavx2 -mfma -mf16c -ffp-contract=off.
//
// Every block dot product is computed in two stages:
//   1. integer stage, fully vectorised: unpack the 4- / 6- / 8-bit weights with
//      shifts and masks, multiply with the int8 activations (vpmaddubsw), widen
//      with the per-sub-block scale (vpmaddwd) and sum into int32 lanes;
//   2. one horizontal integer sum per block, then the block's float scales are
//      applied with exactly the same operations, in the same order, as the
//      scalar reference.
// Integer addition is associative, so stage 1 is exact in any order, and stage
// 2 is identical by construction: the results match the scalar reference bit
// for bit, which tests/test_kernels.cpp checks on millions of random blocks.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/common.hpp"
#include "adamant/quant.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace adamant {
namespace {

inline int hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// Four horizontal sums at once: returns {sum(a), sum(b), sum(c), sum(d)}.
inline __m128i hsum4_i32(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab = _mm256_hadd_epi32(a, b);
    const __m256i cd = _mm256_hadd_epi32(c, d);
    const __m256i abcd = _mm256_hadd_epi32(ab, cd);
    return _mm_add_epi32(_mm256_castsi256_si128(abcd), _mm256_extracti128_si256(abcd, 1));
}

inline float hmax_ps(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline float h2f(std::uint16_t h) { return _cvtsh_ss(h); }  // exact, same value as fp16_to_fp32

// int8 x int8 products of 32 lanes, summed in pairs and then in quads: 8 x int32.
inline __m256i dot32_i8(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);  // |x| as unsigned (|-128| = 128 still fits u8)
    const __m256i sy = _mm256_sign_epi8(y, x);  // y with x's sign; y is an activation, never -128
    return _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1));
}

// ---------------------------------------------------------------------------
// activation quantization
// ---------------------------------------------------------------------------

// A function, not a global: a global __m256 would run AVX code during static init, even on CPUs without AVX.
inline __m256 abs_mask() { return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); }

inline __m256i pack32_i8(__m256i i0, __m256i i1, __m256i i2, __m256i i3) {
    const __m256i a = _mm256_packs_epi32(i0, i1);
    const __m256i b = _mm256_packs_epi32(i2, i3);
    const __m256i c = _mm256_packs_epi16(a, b);
    return _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

void v_quantize_q8_0(const float* x, BlockQ8_0* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK8_0; ++i, x += QK8_0) {
        const __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
        const __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        __m256 m = _mm256_max_ps(_mm256_and_ps(v0, abs_mask()), _mm256_and_ps(v1, abs_mask()));
        m = _mm256_max_ps(m, _mm256_max_ps(_mm256_and_ps(v2, abs_mask()), _mm256_and_ps(v3, abs_mask())));
        const float d = hmax_ps(m) / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        y[i].d = fp32_to_fp16(d);
        const __m256 vid = _mm256_set1_ps(id);
        const __m256i q = pack32_i8(_mm256_cvtps_epi32(_mm256_mul_ps(v0, vid)), _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid)),
                                    _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid)), _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y[i].qs), q);
    }
}

void v_quantize_q8_K(const float* x, BlockQ8K* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        __m256 m = _mm256_setzero_ps();
        for (int l = 0; l < QK_K; l += 8) m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(x + l), abs_mask()));
        const float d = hmax_ps(m) / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        y[i].d = d;
        const __m256 vid = _mm256_set1_ps(id);
        for (int g = 0; g < QK_K / 32; ++g) {
            const float* v = x + 32 * g;
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v), vid));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + 8), vid));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + 16), vid));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + 24), vid));
            y[i].bsums[2 * g] = static_cast<std::int16_t>(hsum_i32(_mm256_add_epi32(i0, i1)));
            y[i].bsums[2 * g + 1] = static_cast<std::int16_t>(hsum_i32(_mm256_add_epi32(i2, i3)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(y[i].qs + 32 * g), pack32_i8(i0, i1, i2, i3));
        }
    }
}

// ---------------------------------------------------------------------------
// dot products
// ---------------------------------------------------------------------------

float v_dot_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, std::int64_t n) {
    const std::int64_t nb = n / QK8_0;
    float sumf = 0.f;
    std::int64_t i = 0;
    alignas(16) int s[4];
    for (; i + 4 <= nb; i += 4) {
        const __m256i p0 = dot32_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i + 0].qs)),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i + 0].qs)));
        const __m256i p1 = dot32_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i + 1].qs)),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i + 1].qs)));
        const __m256i p2 = dot32_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i + 2].qs)),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i + 2].qs)));
        const __m256i p3 = dot32_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i + 3].qs)),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i + 3].qs)));
        _mm_store_si128(reinterpret_cast<__m128i*>(s), hsum4_i32(p0, p1, p2, p3));
        for (int k = 0; k < 4; ++k) sumf += h2f(w[i + k].d) * h2f(x[i + k].d) * static_cast<float>(s[k]);
    }
    for (; i < nb; ++i) {
        const int si = hsum_i32(dot32_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i].qs)),
                                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qs))));
        sumf += h2f(w[i].d) * h2f(x[i].d) * static_cast<float>(si);
    }
    return sumf;
}

float v_dot_q4_K(const BlockQ4K* w, const BlockQ8K* x, std::int64_t n) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float sumf = 0.f;
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        std::uint8_t sc[8], m[8];
        for (int j = 0; j < 8; ++j) q4k_scale_min(j, w[i].scales, &sc[j], &m[j]);
        int summ = 0;
        for (int j = 0; j < 8; ++j) summ += m[j] * (x[i].bsums[2 * j] + x[i].bsums[2 * j + 1]);

        const std::uint8_t* q4 = w[i].qs;
        const std::int8_t* q8 = x[i].qs;
        __m256i acc = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 32 * j));
            const __m256i lo = _mm256_and_si256(bits, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j + 32));
            const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(sc[2 * j]));
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(sc[2 * j + 1]));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(p0, p1));
        }
        const int sumi = hsum_i32(acc);
        const float d = h2f(w[i].d) * x[i].d;
        const float dm = h2f(w[i].dmin) * x[i].d;
        sumf += d * static_cast<float>(sumi) - dm * static_cast<float>(summ);
    }
    return sumf;
}

float v_dot_q6_K(const BlockQ6K* w, const BlockQ8K* x, std::int64_t n) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    const __m256i m32 = _mm256_set1_epi8(32);
    float sumf = 0.f;
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const std::uint8_t* ql = w[i].ql;
        const std::uint8_t* qh = w[i].qh;
        const std::int8_t* sc = w[i].scales;
        const std::int8_t* q8 = x[i].qs;
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i bh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            __m256i q6[4];
            q6[0] = _mm256_or_si256(_mm256_and_si256(b1, m4), _mm256_slli_epi16(_mm256_and_si256(bh, m2), 4));
            q6[1] = _mm256_or_si256(_mm256_and_si256(b2, m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 2), m2), 4));
            q6[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b1, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 4), m2), 4));
            q6[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b2, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 6), m2), 4));
            for (int k = 0; k < 4; ++k) {
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32 * k));
                // (q - 32) * y  ==  q*y - 32*y, both halves in vpmaddubsw range
                const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(q6[k], y), _mm256_maddubs_epi16(m32, y));
                const __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * k + 1]), _mm_set1_epi16(sc[2 * k]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, scv));
            }
            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }
        sumf += h2f(w[i].d) * x[i].d * static_cast<float>(hsum_i32(acc));
    }
    return sumf;
}

float v_dot_f16(const std::uint16_t* w, const float* x, std::int64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    std::int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i))), _mm256_loadu_ps(x + i)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8))), _mm256_loadu_ps(x + i + 8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 16))), _mm256_loadu_ps(x + i + 16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 24))), _mm256_loadu_ps(x + i + 24)));
    }
    float s = hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; ++i) s += h2f(w[i]) * x[i];
    return s;
}

float v_dot_f32(const float* w, const float* x, std::int64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    std::int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24)));
    }
    float s = hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; ++i) s += w[i] * x[i];
    return s;
}

// ---------------------------------------------------------------------------
// dequantization
// ---------------------------------------------------------------------------

// Stores 8 floats d * q[k] (q: 8 signed bytes in the low half of v).
inline void store8_i8(float* y, __m128i v, __m256 d) {
    _mm256_storeu_ps(y, _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v))));
}

void v_dequant_q8_0(const BlockQ8_0* w, float* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK8_0; ++i, y += QK8_0) {
        const __m256 d = _mm256_set1_ps(h2f(w[i].d));
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[i].qs));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[i].qs + 16));
        store8_i8(y, a, d);
        store8_i8(y + 8, _mm_srli_si128(a, 8), d);
        store8_i8(y + 16, b, d);
        store8_i8(y + 24, _mm_srli_si128(b, 8), d);
    }
}

// 32 unsigned nibble values (one per byte) -> d*q - m, in the scalar order.
inline void store32_u8_dm(float* y, __m256i q, float d, float m) {
    const __m256 vd = _mm256_set1_ps(d), vm = _mm256_set1_ps(m);
    const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
    const __m128i parts[4] = {lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)};
    for (int k = 0; k < 4; ++k) {
        const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(parts[k]));
        _mm256_storeu_ps(y + 8 * k, _mm256_sub_ps(_mm256_mul_ps(vd, f), vm));
    }
}

void v_dequant_q4_K(const BlockQ4K* w, float* y, std::int64_t n) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(w[i].d);
        const float dmin = h2f(w[i].dmin);
        for (int j = 0; j < 4; ++j) {
            std::uint8_t sc, m;
            q4k_scale_min(2 * j, w[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            q4k_scale_min(2 * j + 1, w[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[i].qs + 32 * j));
            store32_u8_dm(y, _mm256_and_si256(bits, m4), d1, m1);
            store32_u8_dm(y + 32, _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), d2, m2);
            y += 64;
        }
    }
}

void v_dequant_q6_K(const BlockQ6K* w, float* y, std::int64_t n) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    const __m256i m32 = _mm256_set1_epi8(32);
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(w[i].d);
        const std::uint8_t* ql = w[i].ql;
        const std::uint8_t* qh = w[i].qh;
        const std::int8_t* sc = w[i].scales;
        for (int h = 0; h < 2; ++h) {
            const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i bh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            __m256i q6[4];
            q6[0] = _mm256_or_si256(_mm256_and_si256(b1, m4), _mm256_slli_epi16(_mm256_and_si256(bh, m2), 4));
            q6[1] = _mm256_or_si256(_mm256_and_si256(b2, m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 2), m2), 4));
            q6[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b1, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 4), m2), 4));
            q6[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b2, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(bh, 6), m2), 4));
            for (int k = 0; k < 4; ++k) {
                const __m256i q = _mm256_sub_epi8(q6[k], m32);
                const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
                const __m256 dA = _mm256_set1_ps(d * static_cast<float>(sc[2 * k]));
                const __m256 dB = _mm256_set1_ps(d * static_cast<float>(sc[2 * k + 1]));
                float* o = y + 32 * k;
                store8_i8(o, lo, dA);
                store8_i8(o + 8, _mm_srli_si128(lo, 8), dA);
                store8_i8(o + 16, hi, dB);
                store8_i8(o + 24, _mm_srli_si128(hi, 8), dB);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void v_dequant_f16(const std::uint16_t* w, float* y, std::int64_t n) {
    std::int64_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i))));
    for (; i < n; ++i) y[i] = h2f(w[i]);
}

const Kernels kAvx2 = {
    "avx2",
    v_quantize_q8_0, v_quantize_q8_K,
    v_dot_q8_0, v_dot_q4_K, v_dot_q6_K, v_dot_f16, v_dot_f32,
    v_dequant_q8_0, v_dequant_q4_K, v_dequant_q6_K, v_dequant_f16,
};

}  // namespace

namespace detail {
const Kernels& kernels_avx2_impl() { return kAvx2; }
}  // namespace detail

}  // namespace adamant
