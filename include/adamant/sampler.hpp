// Adamant SIMD-VIC by Euclase: token sampling.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <cstdint>
#include <vector>

namespace adamant {

struct SamplerParams {
    float temperature = 0.f;  // 0: greedy (deterministic)
    int top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.0f;  // 1: off
    int repeat_last_n = 64;
    std::uint64_t seed = 0x5EED;
};

class Sampler {
public:
    explicit Sampler(const SamplerParams& p) : p_(p), rng_(p.seed ? p.seed : 0x9E3779B97F4A7C15ull) {}
    // logits may be modified (repeat penalty). history = tokens seen so far.
    int sample(float* logits, int n_vocab, const std::vector<int>& history);

private:
    double uniform();
    SamplerParams p_;
    std::uint64_t rng_;
    std::vector<std::pair<float, int>> cand_;
};

int argmax(const float* v, int n);

}  // namespace adamant
