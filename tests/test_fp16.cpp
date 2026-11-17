// fp16 <-> fp32: every half round-trips; rounding is to nearest even.
// SPDX-License-Identifier: MIT
#include "adamant/common.hpp"
#include "check.hpp"

#include <cmath>

using namespace adamant;

int main() {
    int bad = 0;
    for (std::uint32_t h = 0; h < 65536; ++h) {
        const auto hh = static_cast<std::uint16_t>(h);
        const float f = fp16_to_fp32(hh);
        if (std::isnan(f)) {
            if (!std::isnan(fp16_to_fp32(fp32_to_fp16(f)))) ++bad;
            continue;
        }
        if (fp32_to_fp16(f) != hh) ++bad;
    }
    CHECK_MSG(bad == 0, "%d halves do not round-trip", bad);

    CHECK(fp16_to_fp32(0x3C00) == 1.0f);
    CHECK(fp16_to_fp32(0xC000) == -2.0f);
    CHECK(fp16_to_fp32(0x0001) == std::ldexp(1.0f, -24));  // smallest subnormal
    CHECK(fp16_to_fp32(0x7BFF) == 65504.0f);
    CHECK(fp32_to_fp16(1.0f + std::ldexp(1.0f, -11)) == 0x3C00);  // tie -> even (down)
    CHECK(fp32_to_fp16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3C02);  // tie -> even (up)
    CHECK(fp32_to_fp16(70000.f) == 0x7C00);  // overflow -> inf
    CHECK(fp32_to_fp16(std::ldexp(1.0f, -26)) == 0x0000);  // below half the smallest subnormal
    CHECK(fp32_to_fp16(std::ldexp(3.0f, -26)) == 0x0001);

    // Midpoints between consecutive halves round to the even neighbour.
    int mid_bad = 0;
    for (std::uint32_t h = 0; h < 0x7BFF; ++h) {
        const double a = fp16_to_fp32(static_cast<std::uint16_t>(h)), b = fp16_to_fp32(static_cast<std::uint16_t>(h + 1));
        const float mid = static_cast<float>((a + b) / 2);
        if (static_cast<double>(mid) != (a + b) / 2) continue;  // not exactly representable
        const std::uint16_t want = (h & 1) ? static_cast<std::uint16_t>(h + 1) : static_cast<std::uint16_t>(h);
        if (fp32_to_fp16(mid) != want) ++mid_bad;
    }
    CHECK_MSG(mid_bad == 0, "%d midpoints rounded wrong", mid_bad);
    return test_result("test_fp16");
}
