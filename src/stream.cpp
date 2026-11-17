// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/stream.hpp"

#include "adamant/common.hpp"
#include "adamant/token_wire.h"
#include "corundum.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <process.h>
#define ADM_GETPID() static_cast<std::uint32_t>(_getpid())
#else
#include <unistd.h>
#define ADM_GETPID() static_cast<std::uint32_t>(getpid())
#endif

namespace adamant {

struct TokenStream::Impl {
    crd_writer w;
};

TokenStream::TokenStream() = default;
TokenStream::~TokenStream() { close(); }

bool TokenStream::open(const char* ring_name, std::string* err) {
    close();
    crd_status s = crd_create(ring_name, ADM_RING_CAPACITY, 0600, CRD_CREATE_REPLACE);
    auto impl = std::make_unique<Impl>();
    // EEXIST (Windows): a viewer still holds the ring from a previous run; reuse it.
    if (s == CRD_OK || s == CRD_EEXIST) s = crd_writer_open(&impl->w, ring_name, CRD_OPEN_PREFAULT | CRD_OPEN_TAKEOVER);
    if (s != CRD_OK) {
        if (err) *err = std::string("cannot open Corundum ring ") + ring_name + ": " + crd_strerror(s);
        return false;
    }
    impl_ = std::move(impl);
    name_ = ring_name;
    return true;
}

void* TokenStream::reserve(std::uint32_t size) {
    if (!impl_) return nullptr;
    void* dst = nullptr;
    if (size > crd_max_payload(&impl_->w) || crd_reserve(&impl_->w, size, &dst) != CRD_OK) {
        ++dropped_;
        return nullptr;
    }
    return dst;
}

void TokenStream::commit(std::uint32_t size, std::uint32_t type) {
    crd_commit(&impl_->w, size, type);
    crd_notify(&impl_->w);
    ++published_;
}

void TokenStream::session(std::string_view model_name, const char* isa, std::uint32_t n_ctx, std::uint32_t n_threads,
                          std::uint32_t n_layer, std::uint64_t weight_bytes_per_token) {
    if (model_name.size() > 1024) model_name = model_name.substr(0, 1024);
    const auto size = static_cast<std::uint32_t>(sizeof(adm_session) + model_name.size());
    auto* p = static_cast<unsigned char*>(reserve(size));
    if (!p) return;
    adm_session h{};
    h.t_ns = now_ns();
    h.weight_bytes_per_token = weight_bytes_per_token;
    h.wire_version = ADM_WIRE_VERSION;
    h.n_ctx = n_ctx;
    h.n_threads = n_threads;
    h.n_layer = n_layer;
    std::memcpy(h.isa, isa, std::min(std::strlen(isa), sizeof h.isa));  // NUL-padded, not terminated
    h.pid = ADM_GETPID();
    h.text_len = static_cast<std::uint32_t>(model_name.size());
    std::memcpy(p, &h, sizeof h);
    std::memcpy(p + sizeof h, model_name.data(), model_name.size());
    commit(size, ADM_REC_SESSION);
}

void TokenStream::prompt(std::uint32_t n_prompt, std::uint64_t prefill_ns) {
    auto* p = static_cast<adm_prompt*>(reserve(sizeof(adm_prompt)));
    if (!p) return;
    adm_prompt h{};
    h.t_ns = now_ns();
    h.prefill_ns = prefill_ns;
    h.n_prompt = n_prompt;
    std::memcpy(p, &h, sizeof h);
    commit(sizeof h, ADM_REC_PROMPT);
}

void TokenStream::token(std::uint64_t seq, int id, std::string_view text, std::uint64_t forward_ns) {
    if (text.size() > 4096) text = text.substr(0, 4096);
    const auto size = static_cast<std::uint32_t>(sizeof(adm_token) + text.size());
    auto* p = static_cast<unsigned char*>(reserve(size));
    if (!p) return;
    adm_token h{};
    h.seq = seq;
    h.t_ns = now_ns();
    h.forward_ns = forward_ns;
    h.id = id;
    h.text_len = static_cast<std::uint32_t>(text.size());
    std::memcpy(p, &h, sizeof h);
    std::memcpy(p + sizeof h, text.data(), text.size());
    commit(size, ADM_REC_TOKEN);
}

void TokenStream::end(std::uint64_t n_generated, std::uint64_t gen_ns, std::uint32_t reason) {
    auto* p = static_cast<adm_end*>(reserve(sizeof(adm_end)));
    if (!p) return;
    adm_end h{};
    h.t_ns = now_ns();
    h.n_generated = n_generated;
    h.gen_ns = gen_ns;
    h.dropped = dropped_;
    h.reason = reason;
    std::memcpy(p, &h, sizeof h);
    commit(sizeof h, ADM_REC_END);
}

void TokenStream::close() {
    if (!impl_) return;
    // Closing marks the stream finished; the segment stays until the viewer
    // detaches (Windows) or is replaced by the next run (Linux).
    crd_writer_close(&impl_->w);
    impl_.reset();
}

}  // namespace adamant
