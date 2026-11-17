// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <iomanip>
#include <sstream>

namespace adamant {

namespace {

constexpr std::int64_t kRowsPerTask = 32;

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float scale = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

const float* f32_data(const Tensor& t, std::int64_t n, const char* what) {
    if (t.type != GgmlType::F32 || t.ne[0] != n)
        throw Error(std::string("tensor ") + t.name + " (" + what + ") must be F32 with " + std::to_string(n) + " elements");
    return reinterpret_cast<const float*>(t.data);
}

void check_matrix(const Tensor* t, std::int64_t cols, std::int64_t rows) {
    if (t->ne[0] != cols || t->rows() != rows)
        throw Error("tensor " + t->name + " has shape [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->rows()) +
                    "], expected [" + std::to_string(cols) + ", " + std::to_string(rows) + "]");
    if (t->type == GgmlType::Q8_K) throw Error("tensor " + t->name + ": Q8_K is not a weight format");
}

}  // namespace

Model::Model(const std::string& path, int n_ctx, int n_threads, const Kernels* kernels)
    : file_(path), pool_(n_threads > 0 ? n_threads : default_thread_count()), k_(kernels ? kernels : &kernels_default()) {
    cfg_.arch = file_.get_string("general.architecture").value_or("");
    if (cfg_.arch != "llama")
        throw Error("architecture '" + cfg_.arch + "' is not supported: Adamant 0.1 runs llama-architecture models");
    cfg_.name = file_.get_string("general.name").value_or("unnamed");
    auto u = [&](const char* key, bool required, std::uint64_t def = 0) -> int {
        const auto v = file_.get_uint(std::string("llama.") + key);
        if (!v && required) throw Error(std::string("GGUF metadata llama.") + key + " is missing");
        const std::uint64_t r = v ? *v : def;
        if (r > (1u << 24)) throw Error(std::string("GGUF metadata llama.") + key + " is implausible");
        return static_cast<int>(r);
    };
    cfg_.n_layer = u("block_count", true);
    cfg_.dim = u("embedding_length", true);
    cfg_.n_ff = u("feed_forward_length", true);
    cfg_.n_head = u("attention.head_count", true);
    cfg_.n_head_kv = u("attention.head_count_kv", false, static_cast<std::uint64_t>(cfg_.n_head));
    cfg_.n_ctx_train = u("context_length", false, 2048);
    if (auto e = file_.get_float("llama.attention.layer_norm_rms_epsilon")) cfg_.rms_eps = static_cast<float>(*e);
    if (auto b = file_.get_float("llama.rope.freq_base")) cfg_.rope_base = static_cast<float>(*b);
    if (cfg_.n_layer < 1 || cfg_.dim < 1 || cfg_.n_head < 1 || cfg_.n_head_kv < 1 || cfg_.dim % cfg_.n_head != 0 ||
        cfg_.n_head % cfg_.n_head_kv != 0)
        throw Error("inconsistent llama hyper-parameters in GGUF metadata");
    cfg_.head_dim = cfg_.dim / cfg_.n_head;
    cfg_.rope_dim = u("rope.dimension_count", false, static_cast<std::uint64_t>(cfg_.head_dim));
    if (cfg_.rope_dim > cfg_.head_dim || cfg_.rope_dim % 2 != 0) throw Error("bad llama.rope.dimension_count");
    if (file_.get_string("llama.rope.scaling.type").value_or("none") != "none" &&
        file_.get_string("llama.rope.scaling.type").value_or("none") != "")
        throw Error("RoPE scaling is not supported by Adamant 0.1");

    tok_embd_ = &file_.get("token_embd.weight");
    cfg_.n_vocab = static_cast<int>(tok_embd_->rows());
    check_matrix(tok_embd_, cfg_.dim, cfg_.n_vocab);
    out_norm_ = &file_.get("output_norm.weight");
    output_ = file_.find("output.weight");
    if (!output_) output_ = tok_embd_;  // tied embeddings
    check_matrix(output_, cfg_.dim, cfg_.n_vocab);

    const int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    layers_.resize(static_cast<std::size_t>(cfg_.n_layer));
    weight_bytes_per_token_ = output_->nbytes + tok_embd_->row_bytes();
    for (int l = 0; l < cfg_.n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer& L = layers_[static_cast<std::size_t>(l)];
        L.attn_norm = &file_.get(p + "attn_norm.weight");
        L.wq = &file_.get(p + "attn_q.weight");
        L.wk = &file_.get(p + "attn_k.weight");
        L.wv = &file_.get(p + "attn_v.weight");
        L.wo = &file_.get(p + "attn_output.weight");
        L.ffn_norm = &file_.get(p + "ffn_norm.weight");
        L.w_gate = &file_.get(p + "ffn_gate.weight");
        L.w_up = &file_.get(p + "ffn_up.weight");
        L.w_down = &file_.get(p + "ffn_down.weight");
        f32_data(*L.attn_norm, cfg_.dim, "attn_norm");
        f32_data(*L.ffn_norm, cfg_.dim, "ffn_norm");
        check_matrix(L.wq, cfg_.dim, cfg_.dim);
        check_matrix(L.wk, cfg_.dim, kv_dim);
        check_matrix(L.wv, cfg_.dim, kv_dim);
        check_matrix(L.wo, cfg_.dim, cfg_.dim);
        check_matrix(L.w_gate, cfg_.dim, cfg_.n_ff);
        check_matrix(L.w_up, cfg_.dim, cfg_.n_ff);
        check_matrix(L.w_down, cfg_.n_ff, cfg_.dim);
        for (const Tensor* t : {L.wq, L.wk, L.wv, L.wo, L.w_gate, L.w_up, L.w_down}) weight_bytes_per_token_ += t->nbytes;
    }
    f32_data(*out_norm_, cfg_.dim, "output_norm");

    n_ctx_ = n_ctx > 0 ? n_ctx : std::min(cfg_.n_ctx_train > 0 ? cfg_.n_ctx_train : 2048, 4096);

    x_.resize(static_cast<std::size_t>(cfg_.dim));
    xb_.resize(static_cast<std::size_t>(std::max(cfg_.dim, cfg_.n_ff)));
    xb2_.resize(static_cast<std::size_t>(cfg_.dim));
    q_.resize(static_cast<std::size_t>(cfg_.dim));
    k_buf_.resize(static_cast<std::size_t>(kv_dim));
    v_buf_.resize(static_cast<std::size_t>(kv_dim));
    hb_.resize(static_cast<std::size_t>(cfg_.n_ff));
    hb2_.resize(static_cast<std::size_t>(cfg_.n_ff));
    att_.resize(static_cast<std::size_t>(cfg_.n_head) * static_cast<std::size_t>(n_ctx_));
    logits_.resize(static_cast<std::size_t>(cfg_.n_vocab));
    const std::size_t kv = static_cast<std::size_t>(cfg_.n_layer) * static_cast<std::size_t>(n_ctx_) * static_cast<std::size_t>(kv_dim);
    key_cache_.resize(kv);
    value_cache_.resize(kv);

    const int half = cfg_.rope_dim / 2;
    rope_cos_.resize(static_cast<std::size_t>(n_ctx_) * static_cast<std::size_t>(half));
    rope_sin_.resize(static_cast<std::size_t>(n_ctx_) * static_cast<std::size_t>(half));
    for (int p = 0; p < n_ctx_; ++p)
        for (int i = 0; i < half; ++i) {
            const double theta = std::pow(static_cast<double>(cfg_.rope_base), -2.0 * i / cfg_.rope_dim);
            const double a = p * theta;
            rope_cos_[static_cast<std::size_t>(p * half + i)] = static_cast<float>(std::cos(a));
            rope_sin_[static_cast<std::size_t>(p * half + i)] = static_cast<float>(std::sin(a));
        }
}

std::size_t Model::scratch_bytes() const {
    std::size_t b = 0;
    for (const AlignedBuffer<float>* p : {&x_, &xb_, &xb2_, &q_, &k_buf_, &v_buf_, &hb_, &hb2_, &att_, &logits_, &rope_cos_, &rope_sin_})
        b += p->size() * sizeof(float);
    return b;
}

std::string Model::weight_type_summary() const {
    std::map<std::string, std::size_t> bytes;
    for (const Tensor& t : file_.tensors()) bytes[ggml_type_name(t.type)] += t.nbytes;
    std::ostringstream os;
    bool first = true;
    for (auto& [k, v] : bytes) {
        os << (first ? "" : ", ") << k << " " << std::fixed << std::setprecision(1) << (static_cast<double>(v) / (1 << 20)) << " MiB";
        first = false;
    }
    return os.str();
}

void Model::prepare(QVec& v, const float* x, std::int64_t n, bool need_q8_0, bool need_q8_k) const {
    v.f = x;
    v.n = n;
    v.has_q8_0 = v.has_q8_k = false;
    if (need_q8_0) {
        if (v.q8_0.size() < static_cast<std::size_t>(n / QK8_0)) v.q8_0.resize(static_cast<std::size_t>(n / QK8_0));
        k_->quantize_q8_0(x, v.q8_0.data(), n);
        v.has_q8_0 = true;
    }
    if (need_q8_k) {
        if (v.q8_k.size() < static_cast<std::size_t>(n / QK_K)) v.q8_k.resize(static_cast<std::size_t>(n / QK_K));
        k_->quantize_q8_K(x, v.q8_k.data(), n);
        v.has_q8_k = true;
    }
}

void Model::prepare_for(QVec& v, const float* x, std::int64_t n, std::initializer_list<const Tensor*> ws) const {
    bool q80 = false, q8k = false;
    for (const Tensor* t : ws) {
        q80 |= t->type == GgmlType::Q8_0;
        q8k |= t->type == GgmlType::Q4_K || t->type == GgmlType::Q6_K;
    }
    prepare(v, x, n, q80, q8k);
}

float Model::dot_row(const Tensor& w, std::int64_t row, const QVec& x) const {
    const std::uint8_t* r = w.data + static_cast<std::size_t>(row) * w.row_bytes();
    switch (w.type) {
        case GgmlType::Q4_K: return k_->dot_q4_K(reinterpret_cast<const BlockQ4K*>(r), x.q8_k.data(), x.n);
        case GgmlType::Q6_K: return k_->dot_q6_K(reinterpret_cast<const BlockQ6K*>(r), x.q8_k.data(), x.n);
        case GgmlType::Q8_0: return k_->dot_q8_0(reinterpret_cast<const BlockQ8_0*>(r), x.q8_0.data(), x.n);
        case GgmlType::F16: return k_->dot_f16(reinterpret_cast<const std::uint16_t*>(r), x.f, x.n);
        case GgmlType::F32: return k_->dot_f32(reinterpret_cast<const float*>(r), x.f, x.n);
        default: return 0.f;
    }
}

void Model::matvec_multi(const MatJob* jobs, int n_jobs) {
    int first_task[8];
    int total = 0;
    for (int j = 0; j < n_jobs; ++j) {
        first_task[j] = total;
        total += static_cast<int>((jobs[j].w->rows() + kRowsPerTask - 1) / kRowsPerTask);
    }
    pool_.run(total, [&](int task) {
        int j = n_jobs - 1;
        while (first_task[j] > task) --j;
        const MatJob& m = jobs[j];
        const std::int64_t r0 = static_cast<std::int64_t>(task - first_task[j]) * kRowsPerTask;
        const std::int64_t r1 = std::min(r0 + kRowsPerTask, m.w->rows());
        for (std::int64_t r = r0; r < r1; ++r) m.out[r] = dot_row(*m.w, r, *m.x);
    });
}

void Model::matvec(const Tensor& w, QVec& x, float* out) {
    const MatJob j{&w, &x, out};
    matvec_multi(&j, 1);
}

const float* Model::forward(int token, int pos) {
    if (token < 0 || token >= cfg_.n_vocab) throw Error("token id out of range");
    if (pos < 0 || pos >= n_ctx_) throw Error("context is full (" + std::to_string(n_ctx_) + " tokens)");
    const std::uint64_t t0 = now_ns();
    const int dim = cfg_.dim, hd = cfg_.head_dim, kv_dim = cfg_.n_head_kv * hd;
    const int kv_mul = cfg_.n_head / cfg_.n_head_kv, half = cfg_.rope_dim / 2;
    float* x = x_.data();

    dequantize_row(*k_, tok_embd_->type, tok_embd_->data + static_cast<std::size_t>(token) * tok_embd_->row_bytes(), x, dim);

    for (int l = 0; l < cfg_.n_layer; ++l) {
        const Layer& L = layers_[static_cast<std::size_t>(l)];
        rmsnorm(xb_.data(), x, reinterpret_cast<const float*>(L.attn_norm->data), dim, cfg_.rms_eps);
        prepare_for(qv_xb_, xb_.data(), dim, {L.wq, L.wk, L.wv});
        const MatJob qkv[3] = {{L.wq, &qv_xb_, q_.data()}, {L.wk, &qv_xb_, k_buf_.data()}, {L.wv, &qv_xb_, v_buf_.data()}};
        matvec_multi(qkv, 3);

        // RoPE (llama.cpp GGUF convention: rotate adjacent pairs)
        const float* rc = rope_cos_.data() + static_cast<std::size_t>(pos) * static_cast<std::size_t>(half);
        const float* rs = rope_sin_.data() + static_cast<std::size_t>(pos) * static_cast<std::size_t>(half);
        auto rope = [&](float* v, int n_heads) {
            for (int h = 0; h < n_heads; ++h) {
                float* p = v + h * hd;
                for (int i = 0; i < half; ++i) {
                    const float a = p[2 * i], b = p[2 * i + 1];
                    p[2 * i] = a * rc[i] - b * rs[i];
                    p[2 * i + 1] = a * rs[i] + b * rc[i];
                }
            }
        };
        rope(q_.data(), cfg_.n_head);
        rope(k_buf_.data(), cfg_.n_head_kv);

        const std::size_t layer_off = static_cast<std::size_t>(l) * static_cast<std::size_t>(n_ctx_) * static_cast<std::size_t>(kv_dim);
        float* kc = key_cache_.data() + layer_off;
        float* vc = value_cache_.data() + layer_off;
        std::memcpy(kc + static_cast<std::size_t>(pos) * kv_dim, k_buf_.data(), static_cast<std::size_t>(kv_dim) * 4);
        std::memcpy(vc + static_cast<std::size_t>(pos) * kv_dim, v_buf_.data(), static_cast<std::size_t>(kv_dim) * 4);

        const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(hd));
        pool_.run(cfg_.n_head, [&](int h) {
            const float* q = q_.data() + h * hd;
            float* att = att_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(n_ctx_);
            const int kvh = h / kv_mul;
            float mx = -INFINITY;
            for (int t = 0; t <= pos; ++t) {
                const float s = k_->dot_f32(q, kc + static_cast<std::size_t>(t) * kv_dim + kvh * hd, hd) * inv_sqrt;
                att[t] = s;
                mx = std::max(mx, s);
            }
            float sum = 0.f;
            for (int t = 0; t <= pos; ++t) {
                att[t] = std::exp(att[t] - mx);
                sum += att[t];
            }
            const float inv = 1.f / sum;
            float* out = xb2_.data() + h * hd;
            std::fill(out, out + hd, 0.f);
            for (int t = 0; t <= pos; ++t) {
                const float a = att[t] * inv;
                const float* v = vc + static_cast<std::size_t>(t) * kv_dim + kvh * hd;
                for (int i = 0; i < hd; ++i) out[i] += a * v[i];
            }
        });

        prepare_for(qv_att_, xb2_.data(), dim, {L.wo});
        matvec(*L.wo, qv_att_, xb_.data());
        for (int i = 0; i < dim; ++i) x[i] += xb_[static_cast<std::size_t>(i)];

        rmsnorm(xb_.data(), x, reinterpret_cast<const float*>(L.ffn_norm->data), dim, cfg_.rms_eps);
        prepare_for(qv_xb_, xb_.data(), dim, {L.w_gate, L.w_up});
        const MatJob ffn[2] = {{L.w_gate, &qv_xb_, hb_.data()}, {L.w_up, &qv_xb_, hb2_.data()}};
        matvec_multi(ffn, 2);
        for (int i = 0; i < cfg_.n_ff; ++i) {
            const float g = hb_[static_cast<std::size_t>(i)];
            hb_[static_cast<std::size_t>(i)] = g / (1.f + std::exp(-g)) * hb2_[static_cast<std::size_t>(i)];
        }
        prepare_for(qv_hb_, hb_.data(), cfg_.n_ff, {L.w_down});
        matvec(*L.w_down, qv_hb_, xb_.data());
        for (int i = 0; i < dim; ++i) x[i] += xb_[static_cast<std::size_t>(i)];
    }

    rmsnorm(xb_.data(), x, reinterpret_cast<const float*>(out_norm_->data), dim, cfg_.rms_eps);
    prepare_for(qv_xb_, xb_.data(), dim, {output_});
    matvec(*output_, qv_xb_, logits_.data());

    stats_.tokens += 1;
    stats_.ns += now_ns() - t0;
    stats_.weight_bytes += weight_bytes_per_token_;
    return logits_.data();
}

}  // namespace adamant
