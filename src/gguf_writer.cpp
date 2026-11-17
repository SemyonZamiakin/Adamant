// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/gguf_writer.hpp"

#include "adamant/common.hpp"

#include <cstdio>
#include <cstring>

namespace adamant {

namespace {
template <class T>
void put(std::vector<std::uint8_t>& b, T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<std::uint8_t>& b, const std::string& s) {
    put<std::uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
}  // namespace

void GgufWriter::set_u32(const std::string& key, std::uint32_t v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 4);
    put(kv.bytes, v);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_f32(const std::string& key, float v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 6);
    put(kv.bytes, v);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_bool(const std::string& key, bool v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 7);
    put<std::uint8_t>(kv.bytes, v ? 1 : 0);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_string(const std::string& key, const std::string& v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 8);
    put_str(kv.bytes, v);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_strings(const std::string& key, const std::vector<std::string>& v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 9);
    put<std::uint32_t>(kv.bytes, 8);
    put<std::uint64_t>(kv.bytes, v.size());
    for (auto& s : v) put_str(kv.bytes, s);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_f32s(const std::string& key, const std::vector<float>& v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 9);
    put<std::uint32_t>(kv.bytes, 6);
    put<std::uint64_t>(kv.bytes, v.size());
    for (float f : v) put(kv.bytes, f);
    kv_.push_back(std::move(kv));
}
void GgufWriter::set_i32s(const std::string& key, const std::vector<std::int32_t>& v) {
    KV kv{key, {}};
    put<std::uint32_t>(kv.bytes, 9);
    put<std::uint32_t>(kv.bytes, 5);
    put<std::uint64_t>(kv.bytes, v.size());
    for (auto i : v) put(kv.bytes, i);
    kv_.push_back(std::move(kv));
}

void GgufWriter::add_tensor(const std::string& name, GgmlType type, std::int64_t cols, std::int64_t rows,
                            const std::vector<float>& values) {
    if (static_cast<std::int64_t>(values.size()) != cols * rows) throw Error("add_tensor: size mismatch for " + name);
    if (cols % ggml_block_size(type) != 0) throw Error("add_tensor: row length not a multiple of the block size");
    T t{name, type, cols, rows, {}};
    const std::size_t rb = ggml_row_size(type, cols);
    t.data.resize(rb * static_cast<std::size_t>(rows));
    for (std::int64_t r = 0; r < rows; ++r)
        quantize_row(type, values.data() + r * cols, t.data.data() + static_cast<std::size_t>(r) * rb, cols);
    tensors_.push_back(std::move(t));
}

void GgufWriter::write(const std::string& path) const {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<std::uint32_t>(b, 3);
    put<std::uint64_t>(b, tensors_.size());
    put<std::uint64_t>(b, kv_.size());
    for (auto& kv : kv_) {
        put_str(b, kv.key);
        b.insert(b.end(), kv.bytes.begin(), kv.bytes.end());
    }
    constexpr std::size_t align = 32;
    std::uint64_t off = 0;
    for (auto& t : tensors_) {
        put_str(b, t.name);
        put<std::uint32_t>(b, 2);
        put<std::uint64_t>(b, static_cast<std::uint64_t>(t.ne0));
        put<std::uint64_t>(b, static_cast<std::uint64_t>(t.ne1));
        put<std::uint32_t>(b, static_cast<std::uint32_t>(t.type));
        put<std::uint64_t>(b, off);
        off += (t.data.size() + align - 1) / align * align;
    }
    b.resize((b.size() + align - 1) / align * align, 0);
    for (auto& t : tensors_) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        b.resize((b.size() + align - 1) / align * align, 0);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw Error("cannot write " + path);
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    if (std::fclose(f) != 0 || !ok) throw Error("write failed: " + path);
}

}  // namespace adamant
