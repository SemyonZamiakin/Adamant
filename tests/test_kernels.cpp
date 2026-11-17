// AVX2 kernels must be bit-identical to the scalar reference.
// Weights are random raw bytes (every bit pattern the format allows, with
// finite fp16 scales), activations come from both quantizers.
// SPDX-License-Identifier: MIT
#include "adamant/common.hpp"
#include "adamant/quant.hpp"
#include "check.hpp"

#include <cmath>

using namespace adamant;

namespace {

std::uint16_t random_half(std::mt19937& rng) {
    // finite, not subnormal-heavy: exponent 1..29
    std::uniform_int_distribution<int> e(1, 29), m(0, 1023), s(0, 1);
    return static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
}

template <class B>
std::vector<B> random_blocks(std::size_t n, std::mt19937& rng) {
    std::vector<B> v(n);
    auto* p = reinterpret_cast<unsigned char*>(v.data());
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t i = 0; i < n * sizeof(B); ++i) p[i] = static_cast<unsigned char>(byte(rng));
    return v;
}

}  // namespace

int main() {
    const Kernels* v = kernels_avx2();
    if (!v) {
        std::printf("test_kernels: SKIP (AVX2 kernels not built or CPU lacks AVX2/FMA/F16C)\n");
        return 77;
    }
    const Kernels& s = kernels_scalar();
    std::mt19937 rng(42);
    const int trials = 400;
    long long cmp = 0;

    for (int t = 0; t < trials; ++t) {
        const std::int64_t n = 256 * (1 + t % 24);  // 256 .. 6144 elements
        // activations, including scale extremes
        const float sigma = t % 7 == 0 ? 1e-6f : (t % 11 == 0 ? 1e4f : 1.f);
        auto x = gaussian(static_cast<std::size_t>(n), rng, sigma);
        if (t % 13 == 0) std::fill(x.begin(), x.begin() + 256, 0.f);  // an all-zero block

        std::vector<BlockQ8_0> a80(static_cast<std::size_t>(n / QK8_0)), b80(a80.size());
        std::vector<BlockQ8K> a8k(static_cast<std::size_t>(n / QK_K)), b8k(a8k.size());
        s.quantize_q8_0(x.data(), a80.data(), n);
        v->quantize_q8_0(x.data(), b80.data(), n);
        s.quantize_q8_K(x.data(), a8k.data(), n);
        v->quantize_q8_K(x.data(), b8k.data(), n);
        CHECK_MSG(std::memcmp(a80.data(), b80.data(), a80.size() * sizeof(BlockQ8_0)) == 0, "quantize_q8_0 differs (trial %d)", t);
        CHECK_MSG(std::memcmp(a8k.data(), b8k.data(), a8k.size() * sizeof(BlockQ8K)) == 0, "quantize_q8_K differs (trial %d)", t);

        auto w80 = random_blocks<BlockQ8_0>(static_cast<std::size_t>(n / QK8_0), rng);
        auto w4 = random_blocks<BlockQ4K>(static_cast<std::size_t>(n / QK_K), rng);
        auto w6 = random_blocks<BlockQ6K>(static_cast<std::size_t>(n / QK_K), rng);
        for (auto& b : w80) b.d = random_half(rng);
        for (auto& b : w4) {
            b.d = random_half(rng);
            b.dmin = random_half(rng);
        }
        for (auto& b : w6) b.d = random_half(rng);

        const float d1 = s.dot_q8_0(w80.data(), a80.data(), n), d2 = v->dot_q8_0(w80.data(), a80.data(), n);
        CHECK_MSG(same_bits(d1, d2), "dot_q8_0 %.9g vs %.9g (trial %d)", d1, d2, t);
        const float e1 = s.dot_q4_K(w4.data(), a8k.data(), n), e2 = v->dot_q4_K(w4.data(), a8k.data(), n);
        CHECK_MSG(same_bits(e1, e2), "dot_q4_K %.9g vs %.9g (trial %d)", e1, e2, t);
        const float f1 = s.dot_q6_K(w6.data(), a8k.data(), n), f2 = v->dot_q6_K(w6.data(), a8k.data(), n);
        CHECK_MSG(same_bits(f1, f2), "dot_q6_K %.9g vs %.9g (trial %d)", f1, f2, t);
        cmp += 3;

        std::vector<float> y1(static_cast<std::size_t>(n)), y2(static_cast<std::size_t>(n));
        s.dequant_q8_0(w80.data(), y1.data(), n);
        v->dequant_q8_0(w80.data(), y2.data(), n);
        CHECK_MSG(std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0, "dequant_q8_0 differs (trial %d)", t);
        s.dequant_q4_K(w4.data(), y1.data(), n);
        v->dequant_q4_K(w4.data(), y2.data(), n);
        CHECK_MSG(std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0, "dequant_q4_K differs (trial %d)", t);
        s.dequant_q6_K(w6.data(), y1.data(), n);
        v->dequant_q6_K(w6.data(), y2.data(), n);
        CHECK_MSG(std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0, "dequant_q6_K differs (trial %d)", t);
        cmp += 3 * n;

        // Float dot products: same 32 partial sums and reduction tree -> identical too.
        std::vector<std::uint16_t> h(static_cast<std::size_t>(n + 5));
        for (auto& e : h) e = fp32_to_fp16(std::normal_distribution<float>(0.f, 1.f)(rng));
        std::vector<float> xf(h.size());
        for (auto& e : xf) e = std::normal_distribution<float>(0.f, 1.f)(rng);
        const auto nn = static_cast<std::int64_t>(h.size());
        double ref = 0, mag = 0;
        for (std::int64_t i = 0; i < nn; ++i) {
            ref += static_cast<double>(fp16_to_fp32(h[static_cast<std::size_t>(i)])) * xf[static_cast<std::size_t>(i)];
            mag += std::fabs(static_cast<double>(fp16_to_fp32(h[static_cast<std::size_t>(i)])) * xf[static_cast<std::size_t>(i)]);
        }
        CHECK(std::fabs(v->dot_f16(h.data(), xf.data(), nn) - ref) <= 1e-5 * mag + 1e-6);
        CHECK(std::fabs(s.dot_f16(h.data(), xf.data(), nn) - ref) <= 1e-5 * mag + 1e-6);
        std::vector<float> wf(xf.size());
        for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = fp16_to_fp32(h[i]);
        CHECK(std::fabs(v->dot_f32(wf.data(), xf.data(), nn) - ref) <= 1e-5 * mag + 1e-6);
        CHECK_MSG(same_bits(s.dot_f16(h.data(), xf.data(), nn), v->dot_f16(h.data(), xf.data(), nn)), "dot_f16 differs (trial %d)", t);
        CHECK_MSG(same_bits(s.dot_f32(wf.data(), xf.data(), nn), v->dot_f32(wf.data(), xf.data(), nn)), "dot_f32 differs (trial %d)", t);
        for (std::int64_t len : {1, 7, 31, 32, 33, 64, 95}) {
            CHECK(same_bits(s.dot_f32(wf.data(), xf.data(), len), v->dot_f32(wf.data(), xf.data(), len)));
        }
        cmp += 9;
        s.dequant_f16(h.data(), y1.data(), std::min<std::int64_t>(n, nn));
        v->dequant_f16(h.data(), y2.data(), std::min<std::int64_t>(n, nn));
        CHECK(std::memcmp(y1.data(), y2.data(), static_cast<std::size_t>(std::min<std::int64_t>(n, nn)) * 4) == 0);
    }

    // Pointers that are not 32-byte aligned (GGUF tensors are 32-byte aligned, but rows of 34/144/210-byte
    // blocks start anywhere that is 2-byte aligned).
    {
        const std::int64_t n = 1024;
        std::vector<unsigned char> raw(ggml_row_size(GgmlType::Q4_K, n) + 64);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto& c : raw) c = static_cast<unsigned char>(byte(rng));
        auto x = gaussian(static_cast<std::size_t>(n), rng);
        std::vector<BlockQ8K> xq(static_cast<std::size_t>(n / QK_K));
        s.quantize_q8_K(x.data(), xq.data(), n);
        for (int off = 2; off < 32; off += 2) {  // blocks hold uint16 fields: 2-byte alignment is the contract
            auto* w = reinterpret_cast<BlockQ4K*>(raw.data() + off);
            for (int b = 0; b < n / QK_K; ++b) {
                const std::uint16_t d = random_half(rng), dm = random_half(rng);
                std::memcpy(raw.data() + off + b * 144, &d, 2);
                std::memcpy(raw.data() + off + b * 144 + 2, &dm, 2);
            }
            const float a = s.dot_q4_K(w, xq.data(), n), b = v->dot_q4_K(w, xq.data(), n);
            CHECK_MSG(same_bits(a, b), "unaligned dot_q4_K offset %d", off);
        }
    }
    std::printf("  %lld results compared bit for bit\n", cmp);
    return test_result("test_kernels");
}
