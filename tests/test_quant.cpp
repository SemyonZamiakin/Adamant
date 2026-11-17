// Quantization round-trips and the accuracy of quantized dot products.
// SPDX-License-Identifier: MIT
#include "adamant/common.hpp"
#include "adamant/quant.hpp"
#include "check.hpp"

#include <cmath>

using namespace adamant;

namespace {

double rel_rmse(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        s += static_cast<double>(a[i]) * a[i];
    }
    return std::sqrt(e / s);
}

// Measured round-trip error of a format on Gaussian data must stay under `limit`.
void round_trip(GgmlType t, double limit, std::mt19937& rng) {
    const std::int64_t n = 256 * 64;
    const auto x = gaussian(static_cast<std::size_t>(n), rng, 0.02f);
    std::vector<std::uint8_t> q(ggml_row_size(t, n));
    quantize_row(t, x.data(), q.data(), n);
    std::vector<float> y(static_cast<std::size_t>(n));
    dequantize_row(kernels_scalar(), t, q.data(), y.data(), n);
    const double e = rel_rmse(x, y);
    std::printf("  %-5s %.3f bits/weight, round-trip relative RMSE %.4f (limit %.3f)\n", ggml_type_name(t), ggml_bits_per_weight(t), e, limit);
    CHECK_MSG(e < limit, "%s round-trip error %.4f", ggml_type_name(t), e);

    // Quantizing the dequantized values again must not move them (idempotence,
    // up to the rounding of the fp16 scales).
    std::vector<std::uint8_t> q2(q.size());
    quantize_row(t, y.data(), q2.data(), n);
    std::vector<float> y2(y.size());
    dequantize_row(kernels_scalar(), t, q2.data(), y2.data(), n);
    CHECK_MSG(rel_rmse(y, y2) < limit / 4, "%s requantization drift %.4f", ggml_type_name(t), rel_rmse(y, y2));
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    round_trip(GgmlType::Q8_0, 0.0065, rng);
    round_trip(GgmlType::Q6_K, 0.025, rng);
    round_trip(GgmlType::Q4_K, 0.11, rng);
    round_trip(GgmlType::F16, 0.0005, rng);

    // Known values: a Q8_0 block of 0..31 scaled.
    {
        std::vector<float> x(32);
        for (int i = 0; i < 32; ++i) x[static_cast<std::size_t>(i)] = (i - 16) * 0.5f;
        BlockQ8_0 b;
        quantize_row_q8_0_ref(x.data(), &b, 32);
        CHECK(b.qs[0] == -127);  // -8.0 is the largest magnitude
        float y[32];
        kernels_scalar().dequant_q8_0(&b, y, 32);
        for (int i = 0; i < 32; ++i) CHECK(std::fabs(y[i] - x[static_cast<std::size_t>(i)]) < 0.04f);
    }
    // All-zero input: scales 0, output exactly 0, no NaN.
    for (GgmlType t : {GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K}) {
        std::vector<float> z(256, 0.f), y(256, 1.f);
        std::vector<std::uint8_t> q(ggml_row_size(t, 256));
        quantize_row(t, z.data(), q.data(), 256);
        dequantize_row(kernels_scalar(), t, q.data(), y.data(), 256);
        bool ok = true;
        for (float v : y) ok &= v == 0.f;
        CHECK_MSG(ok, "%s: zeros do not stay zero", ggml_type_name(t));
    }

    // Quantized dot (weights in format t, activations in Q8_0 / Q8_K) against the
    // float dot of the dequantized weights with the float activations.
    const Kernels& k = kernels_scalar();
    for (GgmlType t : {GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K}) {
        const std::int64_t n = 4096;
        double worst = 0;
        for (int trial = 0; trial < 50; ++trial) {
            const auto w = gaussian(static_cast<std::size_t>(n), rng, 0.02f);
            const auto x = gaussian(static_cast<std::size_t>(n), rng, 1.f);
            std::vector<std::uint8_t> q(ggml_row_size(t, n));
            quantize_row(t, w.data(), q.data(), n);
            std::vector<float> wd(static_cast<std::size_t>(n));
            dequantize_row(k, t, q.data(), wd.data(), n);
            double ref = 0, norm = 0;
            for (std::int64_t i = 0; i < n; ++i) {
                ref += static_cast<double>(wd[static_cast<std::size_t>(i)]) * x[static_cast<std::size_t>(i)];
                norm += std::fabs(static_cast<double>(wd[static_cast<std::size_t>(i)]) * x[static_cast<std::size_t>(i)]);
            }
            float got;
            if (t == GgmlType::Q8_0) {
                std::vector<BlockQ8_0> xq(static_cast<std::size_t>(n / QK8_0));
                k.quantize_q8_0(x.data(), xq.data(), n);
                got = k.dot_q8_0(reinterpret_cast<const BlockQ8_0*>(q.data()), xq.data(), n);
            } else {
                std::vector<BlockQ8K> xq(static_cast<std::size_t>(n / QK_K));
                k.quantize_q8_K(x.data(), xq.data(), n);
                got = t == GgmlType::Q4_K ? k.dot_q4_K(reinterpret_cast<const BlockQ4K*>(q.data()), xq.data(), n)
                                          : k.dot_q6_K(reinterpret_cast<const BlockQ6K*>(q.data()), xq.data(), n);
            }
            worst = std::max(worst, std::fabs(got - ref) / norm);
        }
        std::printf("  %-5s x Q8 dot: worst error %.5f of sum|w*x| (activation quantization only)\n", ggml_type_name(t), worst);
        CHECK_MSG(worst < 0.01, "%s dot error %.5f", ggml_type_name(t), worst);
    }
    return test_result("test_quant");
}
