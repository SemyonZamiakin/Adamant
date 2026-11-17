// Adamant SIMD-VIC by Euclase: Corundum HFT token streamer.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace adamant {

// Publishes generation events on a Corundum ring (see token_wire.h). Writes
// go straight into shared memory (reserve -> fill -> commit), never block and
// never allocate: a full ring drops the record and counts it.
class TokenStream {
public:
    TokenStream();
    ~TokenStream();
    TokenStream(const TokenStream&) = delete;
    TokenStream& operator=(const TokenStream&) = delete;

    // Creates (replacing a stale one) and opens the ring. Returns false and
    // fills *err on failure.
    bool open(const char* ring_name, std::string* err);
    bool is_open() const { return impl_ != nullptr; }

    void session(std::string_view model_name, const char* isa, std::uint32_t n_ctx, std::uint32_t n_threads,
                 std::uint32_t n_layer, std::uint64_t weight_bytes_per_token);
    void prompt(std::uint32_t n_prompt, std::uint64_t prefill_ns);
    void token(std::uint64_t seq, int id, std::string_view text, std::uint64_t forward_ns);
    void end(std::uint64_t n_generated, std::uint64_t gen_ns, std::uint32_t reason);
    void close();

    std::uint64_t dropped() const { return dropped_; }
    std::uint64_t published() const { return published_; }

private:
    void* reserve(std::uint32_t size);
    void commit(std::uint32_t size, std::uint32_t type);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string name_;
    std::uint64_t dropped_ = 0, published_ = 0;
};

}  // namespace adamant
