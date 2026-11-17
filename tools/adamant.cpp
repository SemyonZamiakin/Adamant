// adamant: command-line front end of Adamant SIMD-VIC by Euclase.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/common.hpp"
#include "adamant/gguf.hpp"
#include "adamant/model.hpp"
#include "adamant/quant.hpp"
#include "adamant/sampler.hpp"
#include "adamant/stream.hpp"
#include "adamant/token_wire.h"
#include "adamant/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using namespace adamant;

namespace {

std::vector<std::string> utf8_args(int argc, char** argv) {
    std::vector<std::string> a;
#ifdef _WIN32
    int n = 0;
    if (LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n)) {
        for (int i = 0; i < n; ++i) {
            const int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(static_cast<std::size_t>(len > 0 ? len - 1 : 0), '\0');
            if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s.data(), len, nullptr, nullptr);
            a.push_back(std::move(s));
        }
        LocalFree(w);
        return a;
    }
#endif
    for (int i = 0; i < argc; ++i) a.emplace_back(argv[i]);
    return a;
}

struct Args {
    std::map<std::string, std::string> opt;
    std::vector<std::string> pos;
    bool has(const std::string& k) const { return opt.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opt.find(k);
        return it == opt.end() ? def : it->second;
    }
    int geti(const std::string& k, int def) const { return has(k) ? std::stoi(get(k)) : def; }
    double getf(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
};

// Options that take no value.
bool is_flag(const std::string& k) {
    static const char* flags[] = {"--chat", "--stream", "--warm", "--raw", "--quiet", "--no-cache-bench", "--help", "--tensors"};
    for (const char* f : flags)
        if (k == f) return true;
    return false;
}

Args parse(const std::vector<std::string>& a, std::size_t from) {
    static const std::map<std::string, std::string> alias = {
        {"-m", "--model"}, {"-p", "--prompt"}, {"-n", "--n-predict"}, {"-t", "--threads"}, {"-c", "--ctx"}, {"-s", "--seed"}};
    Args r;
    for (std::size_t i = from; i < a.size(); ++i) {
        std::string k = a[i];
        if (auto it = alias.find(k); it != alias.end()) k = it->second;
        if (k.rfind("--", 0) == 0) {
            if (is_flag(k)) r.opt[k] = "1";
            else if (i + 1 < a.size()) r.opt[k] = a[++i];
            else throw Error("option " + k + " needs a value");
        } else {
            r.pos.push_back(a[i]);
        }
    }
    return r;
}

const Kernels& pick_isa(const Args& a) {
    if (!a.has("--isa")) return kernels_default();
    const Kernels* k = kernels_by_name(a.get("--isa").c_str());
    if (!k) throw Error("ISA '" + a.get("--isa") + "' is not available on this CPU / build (try scalar or avx2)");
    return *k;
}

std::string human_bytes(double b) {
    char buf[64];
    if (b >= 1 << 30) std::snprintf(buf, sizeof buf, "%.2f GiB", b / (1 << 30));
    else if (b >= 1 << 20) std::snprintf(buf, sizeof buf, "%.1f MiB", b / (1 << 20));
    else std::snprintf(buf, sizeof buf, "%.1f KiB", b / 1024);
    return buf;
}

std::string cpu_name() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int regs[12];
    for (unsigned i = 0; i < 3; ++i) {
        unsigned a, b, c, d;
        __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0x80000002u + i), "c"(0));
        regs[i * 4 + 0] = a; regs[i * 4 + 1] = b; regs[i * 4 + 2] = c; regs[i * 4 + 3] = d;
    }
    char s[49];
    std::memcpy(s, regs, 48);
    s[48] = 0;
    std::string r(s);
    r.erase(0, r.find_first_not_of(' '));
    return r;
#else
    return "unknown";
#endif
}

void print_banner() {
    std::fprintf(stderr, "\033[38;2;45;212;191m\xE2\x97\x86 Adamant SIMD-VIC\033[0m %s \xC2\xB7 by Euclase\n", ADAMANT_VERSION_STRING);
}

// ---------------------------------------------------------------------------

int cmd_info(const Args& a) {
    const std::string path = a.has("--model") ? a.get("--model") : (a.pos.empty() ? "" : a.pos[0]);
    if (path.empty()) throw Error("usage: adamant info <model.gguf>");
    GgufFile f(path);
    std::printf("file      %s (%s, GGUF v%u)\n", path.c_str(), human_bytes(static_cast<double>(f.mapping().size())).c_str(), f.version());
    std::printf("metadata  %zu keys\n", f.metadata().size());
    std::vector<std::string> keys;
    for (auto& kv : f.metadata()) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) std::printf("  %-44s %s\n", k.c_str(), f.describe_value(f.metadata().at(k)).c_str());
    std::map<std::string, std::pair<int, std::size_t>> by_type;
    for (auto& t : f.tensors()) {
        auto& e = by_type[ggml_type_name(t.type)];
        e.first++;
        e.second += t.nbytes;
    }
    std::printf("tensors   %zu\n", f.tensors().size());
    for (auto& [k, v] : by_type) std::printf("  %-6s %4d tensors  %s\n", k.c_str(), v.first, human_bytes(static_cast<double>(v.second)).c_str());
    if (a.has("--tensors"))
        for (auto& t : f.tensors())
            std::printf("  %-32s %-5s [%lld, %lld]\n", t.name.c_str(), ggml_type_name(t.type), static_cast<long long>(t.ne[0]),
                        static_cast<long long>(t.rows()));
    return 0;
}

int cmd_run(const Args& a) {
    const std::string path = a.get("--model");
    if (path.empty()) throw Error("usage: adamant run -m model.gguf -p \"prompt\" [-n 128] [--chat] [--stream]");
    const bool quiet = a.has("--quiet");
    if (!quiet) print_banner();
    const Kernels& k = pick_isa(a);
    const std::uint64_t t_load = now_ns();
    Model model(path, a.geti("--ctx", 0), a.geti("--threads", 0), &k);
    Tokenizer tok(model.gguf());
    if (a.has("--warm")) (void)model.gguf().mapping().touch_all();
    const auto& c = model.config();
    if (!quiet)
        std::fprintf(stderr, "model     %s \xC2\xB7 %d layers \xC2\xB7 dim %d \xC2\xB7 %d heads (%d kv) \xC2\xB7 vocab %d\n"
                             "weights   %s, mapped zero-copy (%s)\n"
                             "runtime   %s kernels \xC2\xB7 %d threads \xC2\xB7 ctx %d \xC2\xB7 KV cache %s \xC2\xB7 loaded in %.0f ms\n",
                     c.name.c_str(), c.n_layer, c.dim, c.n_head, c.n_head_kv, c.n_vocab,
                     human_bytes(static_cast<double>(model.weight_bytes_per_token())).c_str(), model.weight_type_summary().c_str(),
                     k.name, model.n_threads(), model.n_ctx(), human_bytes(static_cast<double>(model.kv_cache_bytes())).c_str(),
                     static_cast<double>(now_ns() - t_load) / 1e6);

    std::string prompt = a.get("--prompt", "Once upon a time");
    if (a.has("--prompt-file")) {
        std::ifstream in(a.get("--prompt-file"), std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        prompt = ss.str();
    }
    const bool chat = a.has("--chat");
    const std::string text = chat ? apply_chat_template(tok, a.get("--system"), prompt) : prompt;
    std::vector<int> ids = tok.encode(text, true, chat);
    if (static_cast<int>(ids.size()) >= model.n_ctx()) throw Error("prompt is longer than the context");
    const std::size_t n_prompt = ids.size();

    TokenStream stream;
    if (a.has("--stream")) {
        std::string err;
        if (!stream.open(a.get("--ring", ADM_RING_NAME).c_str(), &err)) throw Error(err);
        stream.session(c.name, k.name, static_cast<std::uint32_t>(model.n_ctx()), static_cast<std::uint32_t>(model.n_threads()),
                       static_cast<std::uint32_t>(c.n_layer), model.weight_bytes_per_token());
        if (!quiet) std::fprintf(stderr, "stream    Corundum ring %s (watch it with adamant-view)\n", a.get("--ring", ADM_RING_NAME).c_str());
    }

    SamplerParams sp;
    sp.temperature = static_cast<float>(a.getf("--temp", 0.0));
    sp.top_k = a.geti("--top-k", 40);
    sp.top_p = static_cast<float>(a.getf("--top-p", 0.95));
    sp.repeat_penalty = static_cast<float>(a.getf("--repeat-penalty", 1.0));
    sp.seed = static_cast<std::uint64_t>(a.geti("--seed", 42));
    Sampler sampler(sp);

    if (!quiet) std::fprintf(stderr, "\n");
    if (!chat && !a.has("--raw")) {
        std::fputs(prompt.c_str(), stdout);
        std::fflush(stdout);
    }

    const std::uint64_t t0 = now_ns();
    const float* logits = nullptr;
    for (std::size_t i = 0; i < ids.size(); ++i) logits = model.forward(ids[i], static_cast<int>(i));
    const std::uint64_t prefill_ns = now_ns() - t0;
    stream.prompt(static_cast<std::uint32_t>(ids.size()), prefill_ns);

    const int n_predict = a.geti("--n-predict", 128);
    std::vector<float> lg(static_cast<std::size_t>(c.n_vocab));
    std::uint64_t gen_ns = 0;
    int n_gen = 0;
    std::uint32_t reason = ADM_END_LIMIT;
    std::uint64_t last_forward = prefill_ns / std::max<std::size_t>(ids.size(), 1);
    bool first = true;
    for (int pos = static_cast<int>(ids.size()); n_gen < n_predict; ++pos) {
        std::memcpy(lg.data(), logits, lg.size() * sizeof(float));
        const int next = sampler.sample(lg.data(), c.n_vocab, ids);
        ids.push_back(next);
        if (tok.is_eog(next)) {
            reason = ADM_END_EOS;
            break;
        }
        std::string piece = tok.piece(next);
        if (first && chat) {  // the reply starts at its first visible character
            piece.erase(0, piece.find_first_not_of(" \n\r\t") == std::string::npos ? piece.size() : piece.find_first_not_of(" \n\r\t"));
            first = piece.empty();
        }
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        stream.token(static_cast<std::uint64_t>(n_gen), next, piece, last_forward);
        ++n_gen;
        if (n_gen >= n_predict) break;
        if (pos >= model.n_ctx()) {
            reason = ADM_END_CONTEXT;
            break;
        }
        const std::uint64_t f0 = now_ns();
        logits = model.forward(next, pos);
        last_forward = now_ns() - f0;
        gen_ns += last_forward;
    }
    std::fputs("\n", stdout);
    std::fflush(stdout);
    stream.end(static_cast<std::uint64_t>(n_gen), gen_ns, reason);
    stream.close();

    if (!quiet) {
        const double pre_s = static_cast<double>(prefill_ns) / 1e9;
        const int n_eval = std::max(n_gen - 1, 0);  // forward passes after the first sampled token
        const double gen_s = static_cast<double>(gen_ns) / 1e9;
        std::fprintf(stderr, "\n\033[38;2;45;212;191mprompt\033[0m   %zu tokens in %.2f s (%.1f tok/s, one token at a time)\n", n_prompt,
                     pre_s, static_cast<double>(n_prompt) / pre_s);
        if (n_eval > 0)
            std::fprintf(stderr, "\033[38;2;45;212;191mgenerate\033[0m %d tokens, %.1f tok/s, %.1f ms/token, weights streamed at %.1f GB/s\n",
                         n_gen, n_eval / gen_s, gen_s * 1e3 / n_eval,
                         static_cast<double>(model.weight_bytes_per_token()) * n_eval / gen_s / 1e9);
        if (stream.is_open() || a.has("--stream"))
            std::fprintf(stderr, "stream    %llu records published, %llu dropped\n", static_cast<unsigned long long>(stream.published()),
                         static_cast<unsigned long long>(stream.dropped()));
    }
    return 0;
}

// End-to-end generation benchmark: fixed prompt, greedy, per ISA and thread count.
int cmd_bench(const Args& a) {
    const std::string path = a.get("--model");
    if (path.empty()) throw Error("usage: adamant bench -m model.gguf [--isa scalar,avx2] [-t 1,4,8] [-n 64]");
    print_banner();
    std::vector<std::string> isas;
    {
        std::stringstream ss(a.get("--isa", "scalar,avx2"));
        for (std::string s; std::getline(ss, s, ',');) isas.push_back(s);
    }
    std::vector<int> threads;
    {
        std::stringstream ss(a.get("--threads", std::to_string(default_thread_count())));
        for (std::string s; std::getline(ss, s, ',');) threads.push_back(std::stoi(s));
    }
    const int n_gen = a.geti("--n-predict", 64);
    const int reps = a.geti("--reps", 3);
    const std::string prompt = a.get("--prompt", "The history of the transistor begins");
    std::FILE* csv = a.has("--csv") ? std::fopen(a.get("--csv").c_str(), "w") : nullptr;
    if (csv) std::fprintf(csv, "model,isa,threads,rep,prompt_tokens,prefill_tok_s,gen_tokens,gen_tok_s,ms_per_token,weight_gb_s,weight_bytes_per_token\n");

    std::printf("cpu       %s\n", cpu_name().c_str());
    for (int nt : threads) {
        Model model(path, 512, nt, &kernels_scalar());
        Tokenizer tok(model.gguf());
        (void)model.gguf().mapping().touch_all();
        const std::vector<int> pids = tok.encode(prompt, true);
        for (const std::string& isa : isas) {
            const Kernels* k = kernels_by_name(isa.c_str());
            if (!k) {
                std::printf("skip      %s (not available)\n", isa.c_str());
                continue;
            }
            model.set_kernels(*k);
            const int r_count = isa == "scalar" ? std::max(1, reps / 3) : reps;
            const int n_this = isa == "scalar" ? std::max(8, n_gen / 4) : n_gen;
            for (int rep = 0; rep < r_count; ++rep) {
                std::uint64_t t0 = now_ns();
                const float* lg = nullptr;
                for (std::size_t i = 0; i < pids.size(); ++i) lg = model.forward(pids[i], static_cast<int>(i));
                const double pre = static_cast<double>(now_ns() - t0) / 1e9;
                int next = argmax(lg, model.config().n_vocab);
                t0 = now_ns();
                for (int g = 0; g < n_this; ++g) {
                    lg = model.forward(next, static_cast<int>(pids.size()) + g);
                    next = argmax(lg, model.config().n_vocab);
                }
                const double gen = static_cast<double>(now_ns() - t0) / 1e9;
                const double tps = n_this / gen;
                const double gbs = static_cast<double>(model.weight_bytes_per_token()) * tps / 1e9;
                std::printf("%-7s t=%-2d rep %d  prefill %6.1f tok/s  generate %6.2f tok/s  %7.1f ms/tok  %6.2f GB/s\n", isa.c_str(), nt,
                            rep, static_cast<double>(pids.size()) / pre, tps, 1e3 / tps, gbs);
                std::fflush(stdout);
                if (csv)
                    std::fprintf(csv, "%s,%s,%d,%d,%zu,%.3f,%d,%.4f,%.3f,%.4f,%zu\n", model.config().name.c_str(), isa.c_str(), nt, rep,
                                 pids.size(), static_cast<double>(pids.size()) / pre, n_this, tps, 1e3 / tps, gbs,
                                 model.weight_bytes_per_token());
            }
        }
    }
    if (csv) std::fclose(csv);
    return 0;
}

// Sums n 64-bit words; only the loads matter. With AVX2: four independent 256-bit
// accumulators, 128 bytes per iteration, so the loop is limited by memory, not by ALUs.
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2"))) std::uint64_t read_sum_avx2(const std::uint64_t* p, std::size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_add_epi64(a0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)));
        a1 = _mm256_add_epi64(a1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 4)));
        a2 = _mm256_add_epi64(a2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 8)));
        a3 = _mm256_add_epi64(a3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 12)));
    }
    std::uint64_t t[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(t), _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3)));
    std::uint64_t s = t[0] + t[1] + t[2] + t[3];
    for (; i < n; ++i) s += p[i];
    return s;
}
#endif

std::uint64_t read_sum(const std::uint64_t* p, std::size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (kernels_avx2()) return read_sum_avx2(p, n);
#endif
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
    }
    for (; i < n; ++i) s0 += p[i];
    return s0 + s1 + s2 + s3;
}

// Kernel micro-benchmark on synthetic weights: matrix-vector product per format.
int cmd_kbench(const Args& a) {
    print_banner();
    std::printf("cpu       %s\n", cpu_name().c_str());
    std::FILE* csv = a.has("--csv") ? std::fopen(a.get("--csv").c_str(), "w") : nullptr;
    if (csv) std::fprintf(csv, "type,isa,threads,rows,cols,weight_bytes,ns_per_matvec,gb_s,gflops\n");
    std::vector<int> threads;
    {
        std::stringstream ss(a.get("--threads", "1," + std::to_string(default_thread_count())));
        for (std::string s; std::getline(ss, s, ',');) threads.push_back(std::stoi(s));
    }
    struct Shape {
        std::int64_t rows, cols;
        const char* label;
    };
    const bool custom = a.has("--rows") || a.has("--cols");
    std::vector<Shape> shapes = {{a.geti("--rows", 16384), a.geti("--cols", 8192),
                                  custom ? "custom shape" : "large (weights far larger than a 16 MiB L3: DRAM-bound)"}};
    if (shapes[0].cols % QK_K != 0 || shapes[0].rows < 1) throw Error("--cols must be a multiple of 256");
    if (!a.has("--no-cache-bench")) shapes.push_back({1024, 2048, "cache-resident"});
    const double min_time = a.getf("--min-time", 0.5);

    // Reference: plain sequential read bandwidth of this machine (sum of 64-bit words over a
    // buffer far larger than the caches, AVX2 loads when available). Weight streaming sits near it.
    {
        const std::size_t words = static_cast<std::size_t>(a.geti("--stream-mib", 1024)) * (1 << 20) / 8;
        AlignedBuffer<std::uint64_t> buf(words);
        for (std::size_t i = 0; i < words; ++i) buf[i] = i;
        std::printf("\nmemory read bandwidth (%s buffer, sequential, %s loads)\n", human_bytes(static_cast<double>(words) * 8).c_str(),
                    kernels_avx2() ? "256-bit AVX2" : "64-bit");
        for (int nt : threads) {
            ThreadPool pool(nt);
            std::vector<std::uint64_t> sums(static_cast<std::size_t>(nt) * 16 * 8);  // one slot per task, 64 bytes apart
            const int tasks = nt * 16;
            auto pass = [&] {
                pool.run(tasks, [&](int task) {
                    const std::size_t b = words / static_cast<std::size_t>(tasks) * static_cast<std::size_t>(task);
                    const std::size_t e = task == tasks - 1 ? words : b + words / static_cast<std::size_t>(tasks);
                    sums[static_cast<std::size_t>(task) * 8] += read_sum(buf.data() + b, e - b);
                });
            };
            pass();
            int iters = 0;
            const std::uint64_t t0 = now_ns();
            std::uint64_t el = 0;
            do {
                pass();
                ++iters;
                el = now_ns() - t0;
            } while (static_cast<double>(el) < min_time * 1e9 * 2);
            const double gbs = static_cast<double>(words) * 8.0 * iters / static_cast<double>(el);
            std::printf("  read   t=%-2d  %7.2f GB/s\n", nt, gbs);
            if (csv) std::fprintf(csv, "stream,read,%d,0,0,%.0f,%.1f,%.4f,0\n", nt, static_cast<double>(words) * 8.0,
                                  static_cast<double>(el) / iters, gbs);
        }
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const Shape& sh : shapes) {
        std::vector<float> xs(static_cast<std::size_t>(sh.cols));
        for (auto& v : xs) v = nd(rng);
        std::vector<float> row(static_cast<std::size_t>(sh.cols));
        std::printf("\n%s: %lld x %lld\n", sh.label, static_cast<long long>(sh.rows), static_cast<long long>(sh.cols));
        for (GgmlType t : {GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::F16}) {
            const std::size_t rb = ggml_row_size(t, sh.cols);
            AlignedBuffer<std::uint8_t> w(rb * static_cast<std::size_t>(sh.rows));
            for (std::int64_t r = 0; r < sh.rows; ++r) {
                for (auto& v : row) v = nd(rng) * 0.05f;
                quantize_row(t, row.data(), w.data() + static_cast<std::size_t>(r) * rb, sh.cols);
            }
            const double wbytes = static_cast<double>(w.size());
            for (const char* isa : {"scalar", "avx2"}) {
                const Kernels* k = kernels_by_name(isa);
                if (!k) continue;
                AlignedBuffer<BlockQ8_0> x80(static_cast<std::size_t>(sh.cols / QK8_0));
                AlignedBuffer<BlockQ8K> x8k(static_cast<std::size_t>(sh.cols / QK_K));
                AlignedBuffer<float> out(static_cast<std::size_t>(sh.rows));
                for (int nt : threads) {
                    ThreadPool pool(nt);
                    auto run_once = [&] {
                        k->quantize_q8_0(xs.data(), x80.data(), sh.cols);
                        k->quantize_q8_K(xs.data(), x8k.data(), sh.cols);
                        const int tasks = static_cast<int>((sh.rows + 31) / 32);
                        pool.run(tasks, [&](int task) {
                            const std::int64_t r1 = std::min<std::int64_t>((task + 1) * 32, sh.rows);
                            for (std::int64_t r = task * 32; r < r1; ++r) {
                                const std::uint8_t* p = w.data() + static_cast<std::size_t>(r) * rb;
                                float v = 0;
                                switch (t) {
                                    case GgmlType::Q4_K: v = k->dot_q4_K(reinterpret_cast<const BlockQ4K*>(p), x8k.data(), sh.cols); break;
                                    case GgmlType::Q6_K: v = k->dot_q6_K(reinterpret_cast<const BlockQ6K*>(p), x8k.data(), sh.cols); break;
                                    case GgmlType::Q8_0: v = k->dot_q8_0(reinterpret_cast<const BlockQ8_0*>(p), x80.data(), sh.cols); break;
                                    default: v = k->dot_f16(reinterpret_cast<const std::uint16_t*>(p), xs.data(), sh.cols); break;
                                }
                                out[static_cast<std::size_t>(r)] = v;
                            }
                        });
                    };
                    run_once();
                    run_once();
                    int iters = 0;
                    const std::uint64_t t0 = now_ns();
                    std::uint64_t el = 0;
                    do {
                        run_once();
                        ++iters;
                        el = now_ns() - t0;
                    } while (static_cast<double>(el) < min_time * 1e9);
                    const double ns = static_cast<double>(el) / iters;
                    const double gbs = wbytes / ns;
                    const double gflops = 2.0 * static_cast<double>(sh.rows) * static_cast<double>(sh.cols) / ns;
                    std::printf("  %-5s %-6s t=%-2d  %9.1f us/matvec  %7.2f GB/s  %7.2f GFLOP/s\n", ggml_type_name(t), isa, nt, ns / 1e3, gbs, gflops);
                    std::fflush(stdout);
                    if (csv)
                        std::fprintf(csv, "%s,%s,%d,%lld,%lld,%.0f,%.1f,%.4f,%.4f\n", ggml_type_name(t), isa, nt, static_cast<long long>(sh.rows),
                                     static_cast<long long>(sh.cols), wbytes, ns, gbs, gflops);
                }
            }
        }
    }
    if (csv) std::fclose(csv);
    return 0;
}

// Writes the logits after feeding a token sequence: one float per line.
int cmd_logits(const Args& a) {
    const std::string path = a.get("--model");
    Model model(path, a.geti("--ctx", 512), a.geti("--threads", 0), &pick_isa(a));
    Tokenizer tok(model.gguf());
    std::vector<int> ids;
    if (a.has("--tokens")) {
        std::stringstream ss(a.get("--tokens"));
        for (std::string s; std::getline(ss, s, ',');) ids.push_back(std::stoi(s));
    } else {
        ids = tok.encode(a.get("--prompt", "Hello"), true);
    }
    const float* lg = nullptr;
    for (std::size_t i = 0; i < ids.size(); ++i) lg = model.forward(ids[i], static_cast<int>(i));
    std::FILE* f = a.has("--out") ? std::fopen(a.get("--out").c_str(), "w") : stdout;
    if (!f) throw Error("cannot write " + a.get("--out"));
    std::fprintf(f, "# tokens");
    for (int id : ids) std::fprintf(f, " %d", id);
    std::fprintf(f, "\n");
    for (int i = 0; i < model.config().n_vocab; ++i) std::fprintf(f, "%.6g\n", static_cast<double>(lg[i]));
    if (f != stdout) std::fclose(f);
    return 0;
}

int cmd_tokenize(const Args& a) {
    GgufFile f(a.get("--model"));
    Tokenizer tok(f);
    const auto ids = tok.encode(a.get("--prompt"), !a.has("--raw"), a.has("--chat"));
    for (int id : ids) std::printf("%6d  '%s'\n", id, tok.token_text(id).c_str());
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "Adamant SIMD-VIC %s by Euclase: SIMD-vectorized inference core for quantized LLMs\n\n"
                 "usage: adamant <command> [options]\n\n"
                 "  run       -m model.gguf -p \"prompt\" [-n 128] [--chat] [--system S] [--temp 0.7] [--seed N]\n"
                 "            [-t threads] [-c ctx] [--isa scalar|avx2] [--stream [--ring NAME]] [--warm]\n"
                 "  bench     -m model.gguf [--isa scalar,avx2] [-t 1,4,8] [-n 64] [--reps 3] [--csv out.csv]\n"
                 "  kbench    [-t 1,8] [--min-time 0.5] [--rows R --cols C] [--csv out.csv]   kernel micro-benchmark, no model\n"
                 "  info      model.gguf [--tensors]\n"
                 "  logits    -m model.gguf (--tokens 1,2,3 | -p text) [--out file]\n"
                 "  tokenize  -m model.gguf -p text [--chat] [--raw]\n\n"
                 "kernels available: scalar%s (default: %s; override with --isa or ADAMANT_ISA)\n",
                 ADAMANT_VERSION_STRING, kernels_avx2() ? ", avx2" : "", kernels_default().name);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode)) SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    try {
        const auto args = utf8_args(argc, argv);
        if (args.size() < 2 || args[1] == "--help" || args[1] == "-h" || args[1] == "help") {
            usage();
            return args.size() < 2 ? 1 : 0;
        }
        const Args a = parse(args, 2);
        const std::string& cmd = args[1];
        if (cmd == "run") return cmd_run(a);
        if (cmd == "bench") return cmd_bench(a);
        if (cmd == "kbench") return cmd_kbench(a);
        if (cmd == "info") return cmd_info(a);
        if (cmd == "logits") return cmd_logits(a);
        if (cmd == "tokenize") return cmd_tokenize(a);
        if (cmd == "--version" || cmd == "version") {
            std::printf("adamant %s\n", ADAMANT_VERSION_STRING);
            return 0;
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "adamant: %s\n", e.what());
        return 1;
    }
}
