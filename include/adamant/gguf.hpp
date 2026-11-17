// Adamant SIMD-VIC by Euclase: GGUF reader (versions 2 and 3).
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include "adamant/mmap_file.hpp"
#include "adamant/quant.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace adamant {

enum class GgufType : std::uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

struct GgufKV {
    GgufType type{};
    std::size_t value_off = 0;      // file offset of the value (array: of the first element)
    GgufType elem_type{};           // arrays only
    std::uint64_t count = 0;        // arrays only
};

struct Tensor {
    std::string name;
    GgmlType type{};
    int n_dims = 0;
    std::int64_t ne[4] = {1, 1, 1, 1};  // ne[0] is the contiguous dimension (row length)
    const std::uint8_t* data = nullptr; // points into the mapping: zero-copy
    std::size_t nbytes = 0;

    std::int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
    std::int64_t cols() const { return ne[0]; }
    std::size_t row_bytes() const;
};

// Parses the header, metadata and tensor table of a GGUF file mapped in
// memory. Every length, count, offset and tensor extent is bounds-checked
// against the file size, so a truncated or malicious file raises adamant::Error
// instead of reading outside the mapping.
class GgufFile {
public:
    explicit GgufFile(const std::string& path);

    const MappedFile& mapping() const { return file_; }
    std::uint32_t version() const { return version_; }
    std::size_t data_offset() const { return data_off_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }
    const Tensor* find(std::string_view name) const;
    const Tensor& get(std::string_view name) const;  // throws if absent

    bool has(std::string_view key) const { return kv_.count(std::string(key)) != 0; }
    const std::unordered_map<std::string, GgufKV>& metadata() const { return kv_; }
    std::optional<std::uint64_t> get_uint(std::string_view key) const;  // any integer type
    std::optional<double> get_float(std::string_view key) const;        // f32 / f64 / integer
    std::optional<std::string> get_string(std::string_view key) const;
    std::vector<std::string> get_string_array(std::string_view key) const;
    std::vector<float> get_f32_array(std::string_view key) const;
    std::vector<std::int32_t> get_i32_array(std::string_view key) const;

    std::string describe_value(const GgufKV& kv, std::size_t max_items = 8) const;

private:
    MappedFile file_;
    std::uint32_t version_ = 0;
    std::size_t data_off_ = 0;
    std::unordered_map<std::string, GgufKV> kv_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, std::size_t> by_name_;
};

const char* gguf_type_name(GgufType t);

}  // namespace adamant
