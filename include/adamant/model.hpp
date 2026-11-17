// Adamant SIMD-VIC by Euclase: llama-architecture transformer inference.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include "adamant/common.hpp"
#include "adamant/gguf.hpp"
#include "adamant/quant.hpp"
#include "adamant/thread_pool.hpp"

#include <memory>
#include <string>
#include <vector>

namespace adamant {

struct ModelConfig {
    std::string arch, name;
    int n_layer = 0, dim = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0;
    int n_vocab = 0, n_ctx_train = 0;
    float rms_eps = 1e-5f, rope_base = 10000.f;
};

struct Layer {
    const Tensor *attn_norm, *wq, *wk, *wv, *wo;
    const Tensor *ffn_norm, *w_gate, *w_up, *w_down;
};

// A float vector together with its Q8_0 / Q8_K quantizations, made once and
// shared by every matrix that multiplies it.
struct QVec {
    const float* f = nullptr;
    std::int64_t n = 0;
    AlignedBuffer<BlockQ8_0> q8_0;
    AlignedBuffer<BlockQ8K> q8_k;
    bool has_q8_0 = false, has_q8_k = false;
};

struct ForwardStats {
    std::uint64_t tokens = 0;
    std::uint64_t ns = 0;
    std::uint64_t weight_bytes = 0;  // weight bytes streamed by the matrix-vector products
};

class Model {
public:
    // n_ctx = 0: the model's training context (capped at 4096).
    Model(const std::string& path, int n_ctx = 0, int n_threads = 0, const Kernels* kernels = nullptr);

    const ModelConfig& config() const { return cfg_; }
    const GgufFile& gguf() const { return file_; }
    int n_ctx() const { return n_ctx_; }
    int n_threads() const { return pool_.size(); }
    const Kernels& kernels() const { return *k_; }
    void set_kernels(const Kernels& k) { k_ = &k; }

    // Runs one token at position pos (0-based, < n_ctx) and returns the logits
    // (n_vocab floats, valid until the next call). Positions must be fed in
    // order; feeding pos again overwrites that KV-cache slot.
    const float* forward(int token, int pos);

    std::size_t weight_bytes_per_token() const { return weight_bytes_per_token_; }
    std::size_t kv_cache_bytes() const { return (key_cache_.size() + value_cache_.size()) * sizeof(float); }
    std::size_t scratch_bytes() const;
    ForwardStats& stats() { return stats_; }
    std::string weight_type_summary() const;

    // Matrix-vector product out[r] = W[r] . x for r in [0, rows); public for tests and benchmarks.
    void matvec(const Tensor& w, QVec& x, float* out);
    void prepare(QVec& v, const float* x, std::int64_t n, bool need_q8_0, bool need_q8_k) const;

private:
    struct MatJob {
        const Tensor* w;
        QVec* x;
        float* out;
    };
    void matvec_multi(const MatJob* jobs, int n_jobs);
    void prepare_for(QVec& v, const float* x, std::int64_t n, std::initializer_list<const Tensor*> ws) const;
    float dot_row(const Tensor& w, std::int64_t row, const QVec& x) const;

    GgufFile file_;
    ModelConfig cfg_;
    const Tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr;
    std::vector<Layer> layers_;
    int n_ctx_ = 0;
    ThreadPool pool_;
    const Kernels* k_;
    std::size_t weight_bytes_per_token_ = 0;
    ForwardStats stats_;

    // activations, all cache-line aligned
    AlignedBuffer<float> x_, xb_, xb2_, q_, k_buf_, v_buf_, hb_, hb2_, att_, logits_;
    AlignedBuffer<float> key_cache_, value_cache_;  // [layer][pos][kv_dim]
    AlignedBuffer<float> rope_cos_, rope_sin_;      // [pos][rope_dim/2]
    QVec qv_xb_, qv_att_, qv_hb_;
};

}  // namespace adamant
