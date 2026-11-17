// Scalar reference kernels, reference quantizers and ISA dispatch.
// Compiled with -ffp-contract=off: see quant.hpp.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/common.hpp"
#include "adamant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace adamant {

// ---------------------------------------------------------------------------
// type table
// ---------------------------------------------------------------------------

bool ggml_type_known(GgmlType t) {
    switch (t) {
        case GgmlType::F32: case GgmlType::F16: case GgmlType::Q8_0:
        case GgmlType::Q4_K: case GgmlType::Q6_K: return true;
        default: return false;
    }
}

const char* ggml_type_name(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return "F32";
        case GgmlType::F16: return "F16";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::Q8_K: return "Q8_K";
    }
    return "?";
}

std::int64_t ggml_block_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32: case GgmlType::F16: return 1;
        case GgmlType::Q8_0: return QK8_0;
        case GgmlType::Q4_K: case GgmlType::Q6_K: case GgmlType::Q8_K: return QK_K;
    }
    return 1;
}

std::size_t ggml_type_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::Q8_0: return sizeof(BlockQ8_0);
        case GgmlType::Q4_K: return sizeof(BlockQ4K);
        case GgmlType::Q6_K: return sizeof(BlockQ6K);
        case GgmlType::Q8_K: return sizeof(BlockQ8K);
    }
    return 0;
}

std::size_t ggml_row_size(GgmlType t, std::int64_t n) {
    return ggml_type_size(t) * static_cast<std::size_t>(n / ggml_block_size(t));
}

double ggml_bits_per_weight(GgmlType t) {
    return 8.0 * static_cast<double>(ggml_type_size(t)) / static_cast<double>(ggml_block_size(t));
}

// ---------------------------------------------------------------------------
// scalar kernels
// ---------------------------------------------------------------------------

namespace {

inline int nearest_int(float x) { return static_cast<int>(std::nearbyint(x)); }

void s_quantize_q8_0(const float* x, BlockQ8_0* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK8_0; ++i, x += QK8_0) {
        float amax = 0.f;
        for (int l = 0; l < QK8_0; ++l) amax = std::max(amax, std::fabs(x[l]));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        y[i].d = fp32_to_fp16(d);
        for (int l = 0; l < QK8_0; ++l) y[i].qs[l] = static_cast<std::int8_t>(std::clamp(nearest_int(x[l] * id), -127, 127));
    }
}

void s_quantize_q8_K(const float* x, BlockQ8K* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        float amax = 0.f;
        for (int l = 0; l < QK_K; ++l) amax = std::max(amax, std::fabs(x[l]));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        y[i].d = d;
        for (int l = 0; l < QK_K; ++l) y[i].qs[l] = static_cast<std::int8_t>(std::clamp(nearest_int(x[l] * id), -127, 127));
        for (int g = 0; g < QK_K / 16; ++g) {
            int s = 0;
            for (int l = 0; l < 16; ++l) s += y[i].qs[g * 16 + l];
            y[i].bsums[g] = static_cast<std::int16_t>(s);
        }
    }
}

float s_dot_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, std::int64_t n) {
    float sumf = 0.f;
    for (std::int64_t i = 0; i < n / QK8_0; ++i) {
        int s = 0;
        for (int l = 0; l < QK8_0; ++l) s += w[i].qs[l] * x[i].qs[l];
        sumf += fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d) * static_cast<float>(s);
    }
    return sumf;
}

float s_dot_q4_K(const BlockQ4K* w, const BlockQ8K* x, std::int64_t n) {
    float sumf = 0.f;
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        std::uint8_t sc[8], m[8];
        for (int j = 0; j < 8; ++j) q4k_scale_min(j, w[i].scales, &sc[j], &m[j]);
        const std::uint8_t* q4 = w[i].qs;
        const std::int8_t* q8 = x[i].qs;
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            int lo = 0, hi = 0;
            for (int l = 0; l < 32; ++l) {
                lo += (q4[l] & 0xF) * q8[l];
                hi += (q4[l] >> 4) * q8[l + 32];
            }
            sumi += sc[2 * j] * lo + sc[2 * j + 1] * hi;
            q4 += 32;
            q8 += 64;
        }
        int summ = 0;
        for (int j = 0; j < 8; ++j) summ += m[j] * (x[i].bsums[2 * j] + x[i].bsums[2 * j + 1]);
        const float d = fp16_to_fp32(w[i].d) * x[i].d;
        const float dm = fp16_to_fp32(w[i].dmin) * x[i].d;
        sumf += d * static_cast<float>(sumi) - dm * static_cast<float>(summ);
    }
    return sumf;
}

float s_dot_q6_K(const BlockQ6K* w, const BlockQ8K* x, std::int64_t n) {
    float sumf = 0.f;
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const std::uint8_t* ql = w[i].ql;
        const std::uint8_t* qh = w[i].qh;
        const std::int8_t* sc = w[i].scales;
        const std::int8_t* q8 = x[i].qs;
        int sumi = 0;
        for (int h = 0; h < 2; ++h) {
            int s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s[is + 0] += q1 * q8[l];
                s[is + 2] += q2 * q8[l + 32];
                s[is + 4] += q3 * q8[l + 64];
                s[is + 6] += q4 * q8[l + 96];
            }
            for (int j = 0; j < 8; ++j) sumi += sc[j] * s[j];
            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }
        sumf += fp16_to_fp32(w[i].d) * x[i].d * static_cast<float>(sumi);
    }
    return sumf;
}

// Float dot products: 32 partial sums (the 4 x 8 lanes of the AVX2 kernel), the
// same reduction tree as hsum_ps, then the tail in order. Mul and add are
// separate roundings in both versions, so the results are bit-identical.
template <class Load>
float dot_lanes32(Load w, const float* x, std::int64_t n) {
    float acc[32] = {};
    std::int64_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (int k = 0; k < 32; ++k) acc[k] = acc[k] + w(i + k) * x[i + k];
    float t[8], q[4];
    for (int j = 0; j < 8; ++j) t[j] = (acc[j] + acc[8 + j]) + (acc[16 + j] + acc[24 + j]);
    for (int j = 0; j < 4; ++j) q[j] = t[j] + t[j + 4];
    float s = (q[0] + q[2]) + (q[1] + q[3]);
    for (; i < n; ++i) s += w(i) * x[i];
    return s;
}

float s_dot_f16(const std::uint16_t* w, const float* x, std::int64_t n) {
    return dot_lanes32([w](std::int64_t i) { return fp16_to_fp32(w[i]); }, x, n);
}

float s_dot_f32(const float* w, const float* x, std::int64_t n) {
    return dot_lanes32([w](std::int64_t i) { return w[i]; }, x, n);
}

void s_dequant_q8_0(const BlockQ8_0* w, float* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK8_0; ++i) {
        const float d = fp16_to_fp32(w[i].d);
        for (int l = 0; l < QK8_0; ++l) *y++ = d * static_cast<float>(w[i].qs[l]);
    }
}

void s_dequant_q4_K(const BlockQ4K* w, float* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const float d = fp16_to_fp32(w[i].d);
        const float dmin = fp16_to_fp32(w[i].dmin);
        const std::uint8_t* q = w[i].qs;
        for (int j = 0; j < 4; ++j) {
            std::uint8_t sc, m;
            q4k_scale_min(2 * j, w[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            q4k_scale_min(2 * j + 1, w[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
            q += 32;
        }
    }
}

void s_dequant_q6_K(const BlockQ6K* w, float* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i) {
        const float d = fp16_to_fp32(w[i].d);
        const std::uint8_t* ql = w[i].ql;
        const std::uint8_t* qh = w[i].qh;
        const std::int8_t* sc = w[i].scales;
        for (int h = 0; h < 2; ++h) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = (d * static_cast<float>(sc[is + 0])) * static_cast<float>(q1);
                y[l + 32] = (d * static_cast<float>(sc[is + 2])) * static_cast<float>(q2);
                y[l + 64] = (d * static_cast<float>(sc[is + 4])) * static_cast<float>(q3);
                y[l + 96] = (d * static_cast<float>(sc[is + 6])) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void s_dequant_f16(const std::uint16_t* w, float* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(w[i]);
}

const Kernels kScalar = {
    "scalar",
    s_quantize_q8_0, s_quantize_q8_K,
    s_dot_q8_0, s_dot_q4_K, s_dot_q6_K, s_dot_f16, s_dot_f32,
    s_dequant_q8_0, s_dequant_q4_K, s_dequant_q6_K, s_dequant_f16,
};

}  // namespace

const Kernels& kernels_scalar() { return kScalar; }

bool cpu_has_avx2_fma_f16c() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}

const Kernels* kernels_avx2() {
#ifdef ADAMANT_HAVE_AVX2
    static const bool ok = cpu_has_avx2_fma_f16c();
    return ok ? &detail::kernels_avx2_impl() : nullptr;
#else
    return nullptr;
#endif
}

const Kernels* kernels_by_name(const char* name) {
    if (!name) return nullptr;
    if (std::strcmp(name, "scalar") == 0) return &kScalar;
    if (std::strcmp(name, "avx2") == 0) return kernels_avx2();
    return nullptr;
}

const Kernels& kernels_default() {
    if (const char* env = std::getenv("ADAMANT_ISA")) {
        if (const Kernels* k = kernels_by_name(env)) return *k;
    }
    if (const Kernels* k = kernels_avx2()) return *k;
    return kScalar;
}

void dequantize_row(const Kernels& k, GgmlType t, const void* src, float* dst, std::int64_t n) {
    switch (t) {
        case GgmlType::F32: std::memcpy(dst, src, static_cast<std::size_t>(n) * 4); break;
        case GgmlType::F16: k.dequant_f16(static_cast<const std::uint16_t*>(src), dst, n); break;
        case GgmlType::Q8_0: k.dequant_q8_0(static_cast<const BlockQ8_0*>(src), dst, n); break;
        case GgmlType::Q4_K: k.dequant_q4_K(static_cast<const BlockQ4K*>(src), dst, n); break;
        case GgmlType::Q6_K: k.dequant_q6_K(static_cast<const BlockQ6K*>(src), dst, n); break;
        default: throw Error(std::string("dequantize_row: unsupported type ") + ggml_type_name(t));
    }
}

// ---------------------------------------------------------------------------
// reference quantizers
// ---------------------------------------------------------------------------

void quantize_row_q8_0_ref(const float* x, BlockQ8_0* y, std::int64_t n) { s_quantize_q8_0(x, y, n); }

void quantize_row_q4_K_ref(const float* x, BlockQ4K* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        float scales[8], mins[8];
        for (int j = 0; j < 8; ++j) {
            const float* v = x + 32 * j;
            float mn = v[0], mx = v[0];
            for (int l = 1; l < 32; ++l) {
                mn = std::min(mn, v[l]);
                mx = std::max(mx, v[l]);
            }
            if (mn > 0.f) mn = 0.f;
            scales[j] = (mx - mn) / 15.f;
            mins[j] = -mn;
        }
        float max_scale = 0.f, max_min = 0.f;
        for (int j = 0; j < 8; ++j) {
            max_scale = std::max(max_scale, scales[j]);
            max_min = std::max(max_min, mins[j]);
        }
        const float inv_scale = max_scale > 0.f ? 63.f / max_scale : 0.f;
        const float inv_min = max_min > 0.f ? 63.f / max_min : 0.f;
        std::uint8_t ls[8], lm[8];
        for (int j = 0; j < 8; ++j) {
            ls[j] = static_cast<std::uint8_t>(std::min(63, nearest_int(inv_scale * scales[j])));
            lm[j] = static_cast<std::uint8_t>(std::min(63, nearest_int(inv_min * mins[j])));
        }
        BlockQ4K& b = y[i];
        std::memset(b.scales, 0, sizeof(b.scales));
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                b.scales[j] = ls[j];
                b.scales[j + 4] = lm[j];
            } else {
                b.scales[j + 4] = static_cast<std::uint8_t>((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
                b.scales[j - 4] |= static_cast<std::uint8_t>((ls[j] >> 4) << 6);
                b.scales[j] |= static_cast<std::uint8_t>((lm[j] >> 4) << 6);
            }
        }
        b.d = fp32_to_fp16(max_scale / 63.f);
        b.dmin = fp32_to_fp16(max_min / 63.f);
        const float d = fp16_to_fp32(b.d), dmin = fp16_to_fp32(b.dmin);
        std::uint8_t L[QK_K];
        for (int j = 0; j < 8; ++j) {
            std::uint8_t sc, m;
            q4k_scale_min(j, b.scales, &sc, &m);
            const float dd = d * sc;
            const float dm = dmin * m;
            for (int l = 0; l < 32; ++l) {
                const int q = dd != 0.f ? nearest_int((x[32 * j + l] + dm) / dd) : 0;
                L[32 * j + l] = static_cast<std::uint8_t>(std::clamp(q, 0, 15));
            }
        }
        for (int j = 0; j < QK_K; j += 64)
            for (int l = 0; l < 32; ++l)
                b.qs[j / 2 + l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 4));
    }
}

void quantize_row_q6_K_ref(const float* x, BlockQ6K* y, std::int64_t n) {
    for (std::int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        float scales[16];
        float max_abs_scale = 0.f, max_scale = 0.f;
        for (int j = 0; j < 16; ++j) {
            const float* v = x + 16 * j;
            float vmax = 0.f, amax = 0.f;
            for (int l = 0; l < 16; ++l) {
                if (std::fabs(v[l]) > amax) {
                    amax = std::fabs(v[l]);
                    vmax = v[l];
                }
            }
            scales[j] = vmax / -32.f;  // maps the largest-magnitude value to q = -32
            if (std::fabs(scales[j]) > max_abs_scale) {
                max_abs_scale = std::fabs(scales[j]);
                max_scale = scales[j];
            }
        }
        BlockQ6K& b = y[i];
        const float iscale = max_abs_scale > 0.f ? -128.f / max_scale : 0.f;
        b.d = fp32_to_fp16(max_abs_scale > 0.f ? 1.f / iscale : 0.f);
        const float d = fp16_to_fp32(b.d);
        std::int8_t L[QK_K];
        for (int j = 0; j < 16; ++j) {
            b.scales[j] = static_cast<std::int8_t>(std::clamp(nearest_int(iscale * scales[j]), -128, 127));
            const float dd = d * b.scales[j];
            for (int l = 0; l < 16; ++l) {
                const int q = dd != 0.f ? nearest_int(x[16 * j + l] / dd) : 0;
                L[16 * j + l] = static_cast<std::int8_t>(std::clamp(q, -32, 31) + 32);
            }
        }
        for (int h = 0; h < 2; ++h) {
            for (int l = 0; l < 32; ++l) {
                const std::uint8_t q1 = static_cast<std::uint8_t>(L[128 * h + l]);
                const std::uint8_t q2 = static_cast<std::uint8_t>(L[128 * h + l + 32]);
                const std::uint8_t q3 = static_cast<std::uint8_t>(L[128 * h + l + 64]);
                const std::uint8_t q4 = static_cast<std::uint8_t>(L[128 * h + l + 96]);
                b.ql[64 * h + l] = static_cast<std::uint8_t>((q1 & 0xF) | ((q3 & 0xF) << 4));
                b.ql[64 * h + l + 32] = static_cast<std::uint8_t>((q2 & 0xF) | ((q4 & 0xF) << 4));
                b.qh[32 * h + l] = static_cast<std::uint8_t>((q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6));
            }
        }
    }
}

void quantize_row(GgmlType t, const float* x, void* dst, std::int64_t n) {
    switch (t) {
        case GgmlType::F32: std::memcpy(dst, x, static_cast<std::size_t>(n) * 4); break;
        case GgmlType::F16: {
            auto* h = static_cast<std::uint16_t*>(dst);
            for (std::int64_t i = 0; i < n; ++i) h[i] = fp32_to_fp16(x[i]);
            break;
        }
        case GgmlType::Q8_0: quantize_row_q8_0_ref(x, static_cast<BlockQ8_0*>(dst), n); break;
        case GgmlType::Q4_K: quantize_row_q4_K_ref(x, static_cast<BlockQ4K*>(dst), n); break;
        case GgmlType::Q6_K: quantize_row_q6_K_ref(x, static_cast<BlockQ6K*>(dst), n); break;
        default: throw Error(std::string("quantize_row: unsupported type ") + ggml_type_name(t));
    }
}

}  // namespace adamant
