// GGUF reader: round-trip through the writer, and hostile input never escapes
// the mapping (truncation at every length, random byte corruption).
// SPDX-License-Identifier: MIT
#include "adamant/common.hpp"
#include "adamant/gguf.hpp"
#include "adamant/gguf_writer.hpp"
#include "check.hpp"

#include <cstdio>
#include <fstream>

using namespace adamant;

namespace {

std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& p, const std::vector<char>& b, std::size_t n) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(b.data(), static_cast<std::streamsize>(n));
}

// Parses and touches every tensor; returns true if it parsed.
bool try_parse(const std::string& p) {
    try {
        GgufFile f(p);
        volatile unsigned sum = 0;
        for (auto& t : f.tensors())
            for (std::size_t i = 0; i < t.nbytes; i += 61) sum = sum + t.data[i];
        for (auto& kv : f.metadata()) (void)f.describe_value(kv.second);
        (void)f.get_string_array("tok");
        return true;
    } catch (const Error&) {
        return false;
    }
}

}  // namespace

int main() {
    std::mt19937 rng(9);
    const std::string path = temp_name("test_gguf");
    {
        GgufWriter w;
        w.set_string("general.architecture", "llama");
        w.set_u32("answer", 42);
        w.set_f32("pi", 3.25f);
        w.set_bool("flag", true);
        w.set_strings("tok", {"<s>", "a", "\xE2\x96\x81" "b"});
        w.set_f32s("scores", {0.f, -1.f, -2.5f});
        w.set_i32s("types", {3, 1, 1});
        w.add_tensor("w.q4", GgmlType::Q4_K, 512, 3, gaussian(512 * 3, rng));
        w.add_tensor("w.q6", GgmlType::Q6_K, 256, 2, gaussian(256 * 2, rng));
        w.add_tensor("w.q8", GgmlType::Q8_0, 64, 5, gaussian(64 * 5, rng));
        w.add_tensor("w.f16", GgmlType::F16, 10, 2, gaussian(20, rng));
        w.add_tensor("w.f32", GgmlType::F32, 7, 1, gaussian(7, rng));
        w.write(path);
    }
    {
        GgufFile f(path);
        CHECK(f.version() == 3);
        CHECK(f.get_uint("answer") == 42u);
        CHECK(f.get_float("pi") == 3.25);
        CHECK(f.get_uint("flag") == 1u);
        CHECK(f.get_string("general.architecture") == std::string("llama"));
        const auto tok = f.get_string_array("tok");
        CHECK(tok.size() == 3 && tok[2] == "\xE2\x96\x81" "b");
        CHECK(f.get_f32_array("scores").size() == 3 && f.get_f32_array("scores")[2] == -2.5f);
        CHECK(f.get_i32_array("types").size() == 3 && f.get_i32_array("types")[0] == 3);
        CHECK(f.tensors().size() == 5);
        const Tensor& q4 = f.get("w.q4");
        CHECK(q4.type == GgmlType::Q4_K && q4.ne[0] == 512 && q4.rows() == 3 && q4.nbytes == 3 * 2 * 144);
        CHECK(reinterpret_cast<std::uintptr_t>(q4.data) % 32 == 0);
        CHECK(f.find("missing") == nullptr);
        CHECK(f.get("w.f32").nbytes == 28);
        // zero-copy: tensor data points inside the mapping
        CHECK(q4.data >= f.mapping().data() && q4.data + q4.nbytes <= f.mapping().data() + f.mapping().size());
    }

    // Truncation at every length must be rejected cleanly (or parse, if only padding was cut).
    const auto bytes = read_file(path);
    const std::string bad = temp_name("test_gguf_bad");
    int parsed = 0;
    for (std::size_t n = 0; n < bytes.size(); n += (n < 400 ? 1 : 13)) {
        write_file(bad, bytes, n);
        // the last tensor (7 floats = 28 bytes) is followed by 4 bytes of padding
        if (try_parse(bad) && n < bytes.size() - 4) ++parsed;
    }
    CHECK_MSG(parsed == 0, "%d truncated files parsed", parsed);

    // Random corruption: must either parse or throw adamant::Error, never crash.
    std::uniform_int_distribution<std::size_t> pos(0, 400);
    std::uniform_int_distribution<int> byte(0, 255);
    int ok = 0, rejected = 0;
    for (int i = 0; i < 1500; ++i) {
        auto b = bytes;
        const int flips = 1 + i % 4;
        for (int k = 0; k < flips; ++k) b[pos(rng) % b.size()] = static_cast<char>(byte(rng));
        write_file(bad, b, b.size());
        (try_parse(bad) ? ok : rejected)++;
    }
    std::printf("  corruption fuzz: %d parsed, %d rejected, 0 crashes\n", ok, rejected);

    // Not a GGUF at all
    write_file(bad, std::vector<char>{'G', 'G', 'M', 'L', 0, 0, 0, 0}, 8);
    CHECK(!try_parse(bad));
    std::remove(bad.c_str());
    std::remove(path.c_str());
    return test_result("test_gguf");
}
