// End-to-end forward pass on a synthetic llama model (random weights, every
// supported weight format), against an independent double-precision reference
// implementation written here from the llama equations.
//
// The engine quantizes activations to 8 bits before every matrix product; the
// reference does not. The difference is the activation-quantization error,
// which the test bounds and prints.
// SPDX-License-Identifier: MIT
#include "adamant/common.hpp"
#include "adamant/gguf_writer.hpp"
#include "adamant/model.hpp"
#include "adamant/tokenizer.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <map>

using namespace adamant;

namespace {

constexpr int kDim = 256, kHeads = 4, kKvHeads = 2, kFF = 512, kLayers = 2, kVocab = 300;
constexpr int kHeadDim = kDim / kHeads, kKvDim = kKvHeads * kHeadDim;
constexpr float kEps = 1e-5f, kRopeBase = 10000.f;

struct RefWeights {
    std::map<std::string, std::vector<double>> w;  // dequantized, row-major [rows][cols]
};

std::vector<float> make(std::mt19937& rng, std::size_t n, float sigma, float mean = 0.f) {
    auto v = gaussian(n, rng, sigma);
    for (auto& x : v) x += mean;
    return v;
}

void add(GgufWriter& g, RefWeights& ref, const std::string& name, GgmlType t, int cols, int rows, const std::vector<float>& v) {
    g.add_tensor(name, t, cols, rows, v);
    // what the file holds, decoded by the scalar reference dequantizer
    std::vector<std::uint8_t> q(ggml_row_size(t, cols) * static_cast<std::size_t>(rows));
    for (int r = 0; r < rows; ++r)
        quantize_row(t, v.data() + static_cast<std::size_t>(r) * cols, q.data() + static_cast<std::size_t>(r) * ggml_row_size(t, cols), cols);
    std::vector<float> d(static_cast<std::size_t>(cols));
    auto& out = ref.w[name];
    for (int r = 0; r < rows; ++r) {
        dequantize_row(kernels_scalar(), t, q.data() + static_cast<std::size_t>(r) * ggml_row_size(t, cols), d.data(), cols);
        out.insert(out.end(), d.begin(), d.end());
    }
}

std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x, int rows) {
    const std::size_t cols = x.size();
    std::vector<double> y(static_cast<std::size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) y[static_cast<std::size_t>(r)] += W[static_cast<std::size_t>(r) * cols + c] * x[c];
    return y;
}

std::vector<double> rmsnorm(const std::vector<double>& x, const std::vector<double>& w) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double s = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + kEps);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s * w[i];
    return y;
}

void rope(std::vector<double>& v, int n_heads, int pos) {
    for (int h = 0; h < n_heads; ++h)
        for (int i = 0; i < kHeadDim / 2; ++i) {
            const double th = pos * std::pow(static_cast<double>(kRopeBase), -2.0 * i / kHeadDim);
            double& a = v[static_cast<std::size_t>(h * kHeadDim + 2 * i)];
            double& b = v[static_cast<std::size_t>(h * kHeadDim + 2 * i + 1)];
            const double a0 = a, b0 = b;
            a = a0 * std::cos(th) - b0 * std::sin(th);
            b = a0 * std::sin(th) + b0 * std::cos(th);
        }
}

struct RefModel {
    RefWeights& R;
    std::vector<std::vector<double>> kc, vc;  // per layer, appended per position
    explicit RefModel(RefWeights& r) : R(r), kc(kLayers), vc(kLayers) {}

    std::vector<double> forward(int token, int pos) {
        const auto& E = R.w["token_embd.weight"];
        std::vector<double> x(E.begin() + static_cast<std::ptrdiff_t>(token) * kDim, E.begin() + static_cast<std::ptrdiff_t>(token + 1) * kDim);
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            auto xb = rmsnorm(x, R.w[p + "attn_norm.weight"]);
            auto q = matvec(R.w[p + "attn_q.weight"], xb, kDim);
            auto k = matvec(R.w[p + "attn_k.weight"], xb, kKvDim);
            auto v = matvec(R.w[p + "attn_v.weight"], xb, kKvDim);
            rope(q, kHeads, pos);
            rope(k, kKvHeads, pos);
            kc[static_cast<std::size_t>(l)].insert(kc[static_cast<std::size_t>(l)].end(), k.begin(), k.end());
            vc[static_cast<std::size_t>(l)].insert(vc[static_cast<std::size_t>(l)].end(), v.begin(), v.end());
            std::vector<double> att_out(kDim, 0.0);
            for (int h = 0; h < kHeads; ++h) {
                const int kvh = h / (kHeads / kKvHeads);
                std::vector<double> sc(static_cast<std::size_t>(pos + 1));
                double mx = -1e300;
                for (int t = 0; t <= pos; ++t) {
                    double s = 0;
                    for (int i = 0; i < kHeadDim; ++i)
                        s += q[static_cast<std::size_t>(h * kHeadDim + i)] * kc[static_cast<std::size_t>(l)][static_cast<std::size_t>(t * kKvDim + kvh * kHeadDim + i)];
                    sc[static_cast<std::size_t>(t)] = s / std::sqrt(static_cast<double>(kHeadDim));
                    mx = std::max(mx, sc[static_cast<std::size_t>(t)]);
                }
                double sum = 0;
                for (auto& s : sc) sum += (s = std::exp(s - mx));
                for (int t = 0; t <= pos; ++t)
                    for (int i = 0; i < kHeadDim; ++i)
                        att_out[static_cast<std::size_t>(h * kHeadDim + i)] +=
                            sc[static_cast<std::size_t>(t)] / sum * vc[static_cast<std::size_t>(l)][static_cast<std::size_t>(t * kKvDim + kvh * kHeadDim + i)];
            }
            auto o = matvec(R.w[p + "attn_output.weight"], att_out, kDim);
            for (int i = 0; i < kDim; ++i) x[static_cast<std::size_t>(i)] += o[static_cast<std::size_t>(i)];
            xb = rmsnorm(x, R.w[p + "ffn_norm.weight"]);
            auto g = matvec(R.w[p + "ffn_gate.weight"], xb, kFF);
            auto u = matvec(R.w[p + "ffn_up.weight"], xb, kFF);
            for (int i = 0; i < kFF; ++i) {
                const double gi = g[static_cast<std::size_t>(i)];
                g[static_cast<std::size_t>(i)] = gi / (1.0 + std::exp(-gi)) * u[static_cast<std::size_t>(i)];
            }
            auto d = matvec(R.w[p + "ffn_down.weight"], g, kDim);
            for (int i = 0; i < kDim; ++i) x[static_cast<std::size_t>(i)] += d[static_cast<std::size_t>(i)];
        }
        return matvec(R.w["output.weight"], rmsnorm(x, R.w["output_norm.weight"]), kVocab);
    }
};

std::string build(RefWeights& ref) {
    std::mt19937 rng(2026);
    GgufWriter g;
    g.set_string("general.architecture", "llama");
    g.set_string("general.name", "synthetic-test");
    g.set_u32("llama.block_count", kLayers);
    g.set_u32("llama.embedding_length", kDim);
    g.set_u32("llama.feed_forward_length", kFF);
    g.set_u32("llama.attention.head_count", kHeads);
    g.set_u32("llama.attention.head_count_kv", kKvHeads);
    g.set_u32("llama.context_length", 64);
    g.set_f32("llama.attention.layer_norm_rms_epsilon", kEps);
    g.set_f32("llama.rope.freq_base", kRopeBase);
    g.set_string("tokenizer.ggml.model", "llama");
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::vector<float> scores = {0, 0, 0};
    std::vector<std::int32_t> types = {2, 3, 3};
    for (int i = 3; i < kVocab; ++i) {
        toks.push_back("\xE2\x96\x81w" + std::to_string(i));
        scores.push_back(-static_cast<float>(i));
        types.push_back(1);
    }
    g.set_strings("tokenizer.ggml.tokens", toks);
    g.set_f32s("tokenizer.ggml.scores", scores);
    g.set_i32s("tokenizer.ggml.token_type", types);
    g.set_u32("tokenizer.ggml.bos_token_id", 1);
    g.set_u32("tokenizer.ggml.eos_token_id", 2);

    add(g, ref, "token_embd.weight", GgmlType::Q6_K, kDim, kVocab, make(rng, static_cast<std::size_t>(kDim) * kVocab, 1.0f));
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(g, ref, p + "attn_norm.weight", GgmlType::F32, kDim, 1, make(rng, kDim, 0.1f, 1.f));
        add(g, ref, p + "attn_q.weight", GgmlType::Q4_K, kDim, kDim, make(rng, kDim * kDim, 0.08f));
        add(g, ref, p + "attn_k.weight", GgmlType::Q6_K, kDim, kKvDim, make(rng, kDim * kKvDim, 0.08f));
        add(g, ref, p + "attn_v.weight", GgmlType::Q8_0, kDim, kKvDim, make(rng, kDim * kKvDim, 0.08f));
        add(g, ref, p + "attn_output.weight", l == 0 ? GgmlType::Q4_K : GgmlType::F16, kDim, kDim, make(rng, kDim * kDim, 0.05f));
        add(g, ref, p + "ffn_norm.weight", GgmlType::F32, kDim, 1, make(rng, kDim, 0.1f, 1.f));
        add(g, ref, p + "ffn_gate.weight", GgmlType::Q4_K, kDim, kFF, make(rng, kDim * kFF, 0.08f));
        add(g, ref, p + "ffn_up.weight", GgmlType::Q8_0, kDim, kFF, make(rng, kDim * kFF, 0.08f));
        add(g, ref, p + "ffn_down.weight", GgmlType::Q6_K, kFF, kDim, make(rng, kFF * kDim, 0.05f));
    }
    add(g, ref, "output_norm.weight", GgmlType::F32, kDim, 1, make(rng, kDim, 0.1f, 1.f));
    add(g, ref, "output.weight", GgmlType::Q8_0, kDim, kVocab, make(rng, static_cast<std::size_t>(kDim) * kVocab, 0.08f));
    const std::string path = temp_name("test_model");
    g.write(path);
    return path;
}

}  // namespace

int main() {
    RefWeights ref;
    const std::string path = build(ref);
    const std::vector<int> seq = {1, 17, 99, 250, 4, 4, 123, 7, 299, 42, 3, 200};

    Model m1(path, 64, 1, &kernels_scalar());
    Model m8(path, 64, 8, &kernels_scalar());
    CHECK(m1.config().n_layer == kLayers && m1.config().dim == kDim && m1.config().n_head_kv == kKvHeads);
    CHECK(m1.config().n_vocab == kVocab && m1.config().head_dim == kHeadDim);
    Tokenizer tok(m1.gguf());
    CHECK(tok.n_vocab() == kVocab && tok.piece(17) == " w17");

    const Kernels* avx = kernels_avx2();
    std::unique_ptr<Model> mv;
    if (avx) mv = std::make_unique<Model>(path, 64, 4, avx);

    RefModel rm(ref);
    double worst_rel = 0, worst_cos = 1, worst_simd = 0;
    int argmax_agree = 0;
    for (std::size_t p = 0; p < seq.size(); ++p) {
        const auto want = rm.forward(seq[p], static_cast<int>(p));
        const float* a = m1.forward(seq[p], static_cast<int>(p));
        std::vector<float> got(a, a + kVocab);
        const float* b = m8.forward(seq[p], static_cast<int>(p));
        CHECK_MSG(std::memcmp(got.data(), b, kVocab * 4) == 0, "1 thread vs 8 threads differ at pos %zu", p);

        double dot = 0, na = 0, nb = 0, maxdiff = 0, maxref = 0;
        int am_ref = 0, am_got = 0;
        for (int i = 0; i < kVocab; ++i) {
            const double r = want[static_cast<std::size_t>(i)], g = got[static_cast<std::size_t>(i)];
            dot += r * g;
            na += r * r;
            nb += g * g;
            maxdiff = std::max(maxdiff, std::fabs(r - g));
            maxref = std::max(maxref, std::fabs(r));
            if (r > want[static_cast<std::size_t>(am_ref)]) am_ref = i;
            if (g > got[static_cast<std::size_t>(am_got)]) am_got = i;
        }
        const double cosv = dot / std::sqrt(na * nb);
        worst_cos = std::min(worst_cos, cosv);
        worst_rel = std::max(worst_rel, maxdiff / maxref);
        argmax_agree += am_ref == am_got;

        if (mv) {
            const float* c = mv->forward(seq[p], static_cast<int>(p));
            double md = 0;
            for (int i = 0; i < kVocab; ++i) md = std::max(md, std::fabs(static_cast<double>(c[i]) - got[static_cast<std::size_t>(i)]));
            worst_simd = std::max(worst_simd, md / maxref);
            CHECK_MSG(std::memcmp(c, got.data(), kVocab * 4) == 0, "avx2 engine differs from scalar engine at pos %zu", p);
        }
    }
    std::printf("  vs double reference over %zu positions: worst cosine %.6f, worst max|diff|/max|logit| %.4f, argmax agrees %d/%zu\n",
                seq.size(), worst_cos, worst_rel, argmax_agree, seq.size());
    CHECK_MSG(worst_cos > 0.999, "cosine %.6f", worst_cos);
    CHECK_MSG(worst_rel < 0.05, "relative logit error %.4f", worst_rel);
    CHECK(argmax_agree >= static_cast<int>(seq.size()) - 1);
    if (mv) {
        std::printf("  avx2 engine (4 threads) vs scalar engine (1 thread): logits bit-identical at every position (max diff %.1e)\n", worst_simd);
        CHECK_MSG(worst_simd < 1e-4, "avx2 vs scalar %.3e", worst_simd);
    }

    // errors: out-of-range token, full context
    bool threw = false;
    try { m1.forward(kVocab, 0); } catch (const Error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { m1.forward(1, 64); } catch (const Error&) { threw = true; }
    CHECK(threw);

    mv.reset();
    std::remove(path.c_str());
    return test_result("test_model");
}
