// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/gguf.hpp"

#include "adamant/common.hpp"

#include <cstring>
#include <limits>
#include <sstream>

namespace adamant {

namespace {

constexpr std::uint64_t kMaxString = 1ull << 24;  // 16 MiB: far above any real key or token
constexpr std::uint64_t kMaxCount = 1ull << 28;

std::size_t scalar_size(GgufType t) {
    switch (t) {
        case GgufType::U8: case GgufType::I8: case GgufType::BOOL: return 1;
        case GgufType::U16: case GgufType::I16: return 2;
        case GgufType::U32: case GgufType::I32: case GgufType::F32: return 4;
        case GgufType::U64: case GgufType::I64: case GgufType::F64: return 8;
        default: return 0;
    }
}

class Reader {
public:
    Reader(const std::uint8_t* base, std::size_t size) : base_(base), size_(size) {}
    std::size_t pos() const { return pos_; }
    void need(std::uint64_t n) const {
        if (n > size_ - pos_) throw Error("GGUF: unexpected end of file at offset " + std::to_string(pos_));
    }
    template <class T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string_view read_string() {
        const auto n = read<std::uint64_t>();
        if (n > kMaxString) throw Error("GGUF: string too long at offset " + std::to_string(pos_));
        need(n);
        std::string_view s(reinterpret_cast<const char*>(base_ + pos_), static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return s;
    }
    void skip(std::uint64_t n) {
        need(n);
        pos_ += static_cast<std::size_t>(n);
    }
    void seek(std::size_t p) {
        if (p > size_) throw Error("GGUF: seek past end");
        pos_ = p;
    }

private:
    const std::uint8_t* base_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

void skip_value(Reader& r, GgufType t) {
    if (t == GgufType::STRING) {
        r.read_string();
        return;
    }
    const std::size_t s = scalar_size(t);
    if (s == 0) throw Error("GGUF: unknown value type " + std::to_string(static_cast<unsigned>(t)));
    r.skip(s);
}

}  // namespace

const char* gguf_type_name(GgufType t) {
    switch (t) {
        case GgufType::U8: return "u8";
        case GgufType::I8: return "i8";
        case GgufType::U16: return "u16";
        case GgufType::I16: return "i16";
        case GgufType::U32: return "u32";
        case GgufType::I32: return "i32";
        case GgufType::F32: return "f32";
        case GgufType::BOOL: return "bool";
        case GgufType::STRING: return "str";
        case GgufType::ARRAY: return "arr";
        case GgufType::U64: return "u64";
        case GgufType::I64: return "i64";
        case GgufType::F64: return "f64";
    }
    return "?";
}

std::size_t Tensor::row_bytes() const { return ggml_row_size(type, ne[0]); }

GgufFile::GgufFile(const std::string& path) : file_(path) {
    Reader r(file_.data(), file_.size());
    r.need(4);
    if (std::memcmp(file_.data(), "GGUF", 4) != 0) throw Error(path + ": not a GGUF file (bad magic)");
    r.skip(4);
    version_ = r.read<std::uint32_t>();
    if (version_ != 2 && version_ != 3) throw Error(path + ": unsupported GGUF version " + std::to_string(version_));
    const auto n_tensors = r.read<std::uint64_t>();
    const auto n_kv = r.read<std::uint64_t>();
    if (n_tensors > kMaxCount || n_kv > kMaxCount) throw Error("GGUF: implausible tensor or key count");

    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key(r.read_string());
        GgufKV kv;
        kv.type = static_cast<GgufType>(r.read<std::uint32_t>());
        if (kv.type == GgufType::ARRAY) {
            kv.elem_type = static_cast<GgufType>(r.read<std::uint32_t>());
            kv.count = r.read<std::uint64_t>();
            if (kv.count > kMaxCount) throw Error("GGUF: array too long: " + key);
            kv.value_off = r.pos();
            if (kv.elem_type == GgufType::STRING) {
                for (std::uint64_t j = 0; j < kv.count; ++j) r.read_string();
            } else if (kv.elem_type == GgufType::ARRAY) {
                throw Error("GGUF: nested arrays are not supported: " + key);
            } else {
                const std::size_t s = scalar_size(kv.elem_type);
                if (s == 0) throw Error("GGUF: unknown array element type in " + key);
                r.skip(kv.count * s);
            }
        } else {
            kv.value_off = r.pos();
            skip_value(r, kv.type);
        }
        kv_[std::move(key)] = kv;
    }

    tensors_.resize(static_cast<std::size_t>(n_tensors));
    std::vector<std::uint64_t> offsets(tensors_.size());
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        Tensor& t = tensors_[i];
        t.name = std::string(r.read_string());
        t.n_dims = static_cast<int>(r.read<std::uint32_t>());
        if (t.n_dims < 1 || t.n_dims > 4) throw Error("GGUF: tensor " + t.name + " has bad rank");
        for (int d = 0; d < t.n_dims; ++d) {
            const auto n = r.read<std::uint64_t>();
            if (n == 0 || n > (1ull << 40)) throw Error("GGUF: tensor " + t.name + " has bad shape");
            t.ne[d] = static_cast<std::int64_t>(n);
        }
        const auto ty = r.read<std::uint32_t>();
        t.type = static_cast<GgmlType>(ty);
        if (!ggml_type_known(t.type))
            throw Error("GGUF: tensor " + t.name + " uses ggml type " + std::to_string(ty) +
                        ", which Adamant does not implement (supported: F32, F16, Q8_0, Q4_K, Q6_K)");
        if (t.ne[0] % ggml_block_size(t.type) != 0)
            throw Error("GGUF: tensor " + t.name + " row length is not a multiple of its block size");
        offsets[i] = r.read<std::uint64_t>();
        const long double elems = static_cast<long double>(t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
        if (elems > static_cast<long double>(1ull << 42)) throw Error("GGUF: tensor " + t.name + " too large");
        t.nbytes = ggml_row_size(t.type, t.ne[0]) * static_cast<std::size_t>(t.rows());
        if (!by_name_.emplace(t.name, i).second) throw Error("GGUF: duplicate tensor " + t.name);
    }

    std::uint64_t align = 32;
    if (auto a = get_uint("general.alignment")) {
        align = *a;
        if (align == 0 || (align & (align - 1)) != 0 || align > 65536) throw Error("GGUF: bad general.alignment");
    }
    data_off_ = (r.pos() + align - 1) / align * align;
    if (data_off_ > file_.size()) throw Error("GGUF: data section starts past end of file");
    const std::size_t data_size = file_.size() - data_off_;
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        Tensor& t = tensors_[i];
        if (offsets[i] % align != 0) throw Error("GGUF: tensor " + t.name + " is misaligned");
        if (offsets[i] > data_size || t.nbytes > data_size - offsets[i])
            throw Error("GGUF: tensor " + t.name + " extends past end of file (truncated download?)");
        t.data = file_.data() + data_off_ + offsets[i];
        if (reinterpret_cast<std::uintptr_t>(t.data) % 2 != 0) throw Error("GGUF: tensor " + t.name + " is not 2-byte aligned");
    }
}

const Tensor* GgufFile::find(std::string_view name) const {
    auto it = by_name_.find(std::string(name));
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const Tensor& GgufFile::get(std::string_view name) const {
    const Tensor* t = find(name);
    if (!t) throw Error("GGUF: missing tensor " + std::string(name));
    return *t;
}

std::optional<std::uint64_t> GgufFile::get_uint(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    if (it == kv_.end()) return std::nullopt;
    const std::uint8_t* p = file_.data() + it->second.value_off;
    switch (it->second.type) {
        case GgufType::U8: case GgufType::BOOL: return *p;
        case GgufType::I8: { std::int8_t v; std::memcpy(&v, p, 1); if (v < 0) return std::nullopt; return static_cast<std::uint64_t>(v); }
        case GgufType::U16: { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
        case GgufType::I16: { std::int16_t v; std::memcpy(&v, p, 2); if (v < 0) return std::nullopt; return static_cast<std::uint64_t>(v); }
        case GgufType::U32: { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
        case GgufType::I32: { std::int32_t v; std::memcpy(&v, p, 4); if (v < 0) return std::nullopt; return static_cast<std::uint64_t>(v); }
        case GgufType::U64: { std::uint64_t v; std::memcpy(&v, p, 8); return v; }
        case GgufType::I64: { std::int64_t v; std::memcpy(&v, p, 8); if (v < 0) return std::nullopt; return static_cast<std::uint64_t>(v); }
        default: return std::nullopt;
    }
}

std::optional<double> GgufFile::get_float(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    if (it == kv_.end()) return std::nullopt;
    const std::uint8_t* p = file_.data() + it->second.value_off;
    if (it->second.type == GgufType::F32) { float v; std::memcpy(&v, p, 4); return v; }
    if (it->second.type == GgufType::F64) { double v; std::memcpy(&v, p, 8); return v; }
    if (auto u = get_uint(key)) return static_cast<double>(*u);
    return std::nullopt;
}

std::optional<std::string> GgufFile::get_string(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    if (it == kv_.end() || it->second.type != GgufType::STRING) return std::nullopt;
    Reader r(file_.data(), file_.size());
    r.seek(it->second.value_off);
    return std::string(r.read_string());
}

std::vector<std::string> GgufFile::get_string_array(std::string_view key) const {
    std::vector<std::string> out;
    auto it = kv_.find(std::string(key));
    if (it == kv_.end() || it->second.type != GgufType::ARRAY || it->second.elem_type != GgufType::STRING) return out;
    Reader r(file_.data(), file_.size());
    r.seek(it->second.value_off);
    out.reserve(static_cast<std::size_t>(it->second.count));
    for (std::uint64_t i = 0; i < it->second.count; ++i) out.emplace_back(r.read_string());
    return out;
}

std::vector<float> GgufFile::get_f32_array(std::string_view key) const {
    std::vector<float> out;
    auto it = kv_.find(std::string(key));
    if (it == kv_.end() || it->second.type != GgufType::ARRAY || it->second.elem_type != GgufType::F32) return out;
    out.resize(static_cast<std::size_t>(it->second.count));
    std::memcpy(out.data(), file_.data() + it->second.value_off, out.size() * 4);
    return out;
}

std::vector<std::int32_t> GgufFile::get_i32_array(std::string_view key) const {
    std::vector<std::int32_t> out;
    auto it = kv_.find(std::string(key));
    if (it == kv_.end() || it->second.type != GgufType::ARRAY) return out;
    if (it->second.elem_type != GgufType::I32 && it->second.elem_type != GgufType::U32) return out;
    out.resize(static_cast<std::size_t>(it->second.count));
    std::memcpy(out.data(), file_.data() + it->second.value_off, out.size() * 4);
    return out;
}

std::string GgufFile::describe_value(const GgufKV& kv, std::size_t max_items) const {
    std::ostringstream os;
    Reader r(file_.data(), file_.size());
    r.seek(kv.value_off);
    auto scalar = [&](GgufType t) {
        switch (t) {
            case GgufType::U8: os << +r.read<std::uint8_t>(); break;
            case GgufType::I8: os << +r.read<std::int8_t>(); break;
            case GgufType::BOOL: os << (r.read<std::uint8_t>() ? "true" : "false"); break;
            case GgufType::U16: os << r.read<std::uint16_t>(); break;
            case GgufType::I16: os << r.read<std::int16_t>(); break;
            case GgufType::U32: os << r.read<std::uint32_t>(); break;
            case GgufType::I32: os << r.read<std::int32_t>(); break;
            case GgufType::F32: os << r.read<float>(); break;
            case GgufType::U64: os << r.read<std::uint64_t>(); break;
            case GgufType::I64: os << r.read<std::int64_t>(); break;
            case GgufType::F64: os << r.read<double>(); break;
            case GgufType::STRING: {
                auto s = r.read_string();
                if (s.size() > 60) os << '"' << s.substr(0, 57) << "...\"";
                else os << '"' << s << '"';
                break;
            }
            default: os << '?';
        }
    };
    if (kv.type != GgufType::ARRAY) {
        scalar(kv.type);
    } else {
        os << '[' << gguf_type_name(kv.elem_type) << " x " << kv.count << "] ";
        for (std::uint64_t i = 0; i < kv.count && i < max_items; ++i) {
            if (i) os << ", ";
            scalar(kv.elem_type);
        }
        if (kv.count > max_items) os << ", ...";
    }
    return os.str();
}

}  // namespace adamant
