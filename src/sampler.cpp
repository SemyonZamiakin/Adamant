// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/sampler.hpp"

#include <algorithm>
#include <cmath>

namespace adamant {

int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

double Sampler::uniform() {
    // xorshift64*
    rng_ ^= rng_ >> 12;
    rng_ ^= rng_ << 25;
    rng_ ^= rng_ >> 27;
    return static_cast<double>((rng_ * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0);
}

int Sampler::sample(float* logits, int n_vocab, const std::vector<int>& history) {
    if (p_.repeat_penalty != 1.f && p_.repeat_last_n > 0) {
        const std::size_t start = history.size() > static_cast<std::size_t>(p_.repeat_last_n)
                                      ? history.size() - static_cast<std::size_t>(p_.repeat_last_n) : 0;
        for (std::size_t i = start; i < history.size(); ++i) {
            const int t = history[i];
            if (t < 0 || t >= n_vocab) continue;
            logits[t] = logits[t] > 0 ? logits[t] / p_.repeat_penalty : logits[t] * p_.repeat_penalty;
        }
    }
    if (p_.temperature <= 0.f) return argmax(logits, n_vocab);

    cand_.resize(static_cast<std::size_t>(n_vocab));
    for (int i = 0; i < n_vocab; ++i) cand_[static_cast<std::size_t>(i)] = {logits[i] / p_.temperature, i};
    const int k = p_.top_k > 0 ? std::min(p_.top_k, n_vocab) : n_vocab;
    std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(), [](auto& a, auto& b) { return a.first > b.first; });
    cand_.resize(static_cast<std::size_t>(k));
    const float mx = cand_[0].first;
    double sum = 0;
    for (auto& c : cand_) {
        c.first = std::exp(c.first - mx);
        sum += c.first;
    }
    double cum = 0;
    std::size_t keep = cand_.size();
    for (std::size_t i = 0; i < cand_.size(); ++i) {
        cum += cand_[i].first / sum;
        if (cum >= p_.top_p) {
            keep = i + 1;
            break;
        }
    }
    cand_.resize(keep);
    double total = 0;
    for (auto& c : cand_) total += c.first;
    double r = uniform() * total;
    for (auto& c : cand_) {
        r -= c.first;
        if (r <= 0) return c.second;
    }
    return cand_.back().second;
}

}  // namespace adamant
