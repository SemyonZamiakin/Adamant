// Adamant SIMD-VIC by Euclase: minimal GGUF v3 writer (tests, synthetic models).
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include "adamant/quant.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace adamant {

class GgufWriter {
public:
    void set_u32(const std::string& key, std::uint32_t v);
    void set_f32(const std::string& key, float v);
    void set_bool(const std::string& key, bool v);
    void set_string(const std::string& key, const std::string& v);
    void set_strings(const std::string& key, const std::vector<std::string>& v);
    void set_f32s(const std::string& key, const std::vector<float>& v);
    void set_i32s(const std::string& key, const std::vector<std::int32_t>& v);
    // Quantizes `values` (rows x cols, row-major) to `type` and adds it.
    void add_tensor(const std::string& name, GgmlType type, std::int64_t cols, std::int64_t rows,
                    const std::vector<float>& values);
    void write(const std::string& path) const;  // throws adamant::Error

private:
    struct KV {
        std::string key;
        std::vector<std::uint8_t> bytes;  // type + value, already encoded
    };
    struct T {
        std::string name;
        GgmlType type;
        std::int64_t ne0, ne1;
        std::vector<std::uint8_t> data;
    };
    std::vector<KV> kv_;
    std::vector<T> tensors_;
};

}  // namespace adamant
