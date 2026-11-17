// Adamant SIMD-VIC by Euclase: SentencePiece (llama) tokenizer from GGUF metadata.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include "adamant/gguf.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace adamant {

enum class TokenAttr : std::int32_t { Undefined = 0, Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 };

class Tokenizer {
public:
    explicit Tokenizer(const GgufFile& f);
    // For tests: build from explicit tables (model "llama").
    Tokenizer(std::vector<std::string> tokens, std::vector<float> scores, std::vector<std::int32_t> types,
              int bos, int eos, bool add_space_prefix = true);

    // SentencePiece BPE as in llama.cpp: spaces become U+2581, a space is
    // prepended to text that starts the input or follows a special token,
    // adjacent pieces are merged by highest vocabulary score, and characters
    // with no piece fall back to <0xXX> byte tokens. With parse_special,
    // control-token strings in the text (e.g. "</s>") become those tokens.
    std::vector<int> encode(std::string_view text, bool add_bos, bool parse_special = false) const;
    // Text of one token as it should be printed (U+2581 -> ' ', byte tokens -> raw byte).
    std::string piece(int id) const;
    std::string decode(const std::vector<int>& ids) const;

    int n_vocab() const { return static_cast<int>(tokens_.size()); }
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    const std::string& token_text(int id) const { return tokens_.at(static_cast<std::size_t>(id)); }
    TokenAttr attr(int id) const { return static_cast<TokenAttr>(types_.at(static_cast<std::size_t>(id))); }
    int find(std::string_view text) const;  // -1 if absent
    bool is_eog(int id) const { return id == eos_ || id == eot_; }
    const std::string& chat_template() const { return chat_template_; }

private:
    void init();
    void encode_fragment(std::string text, std::vector<int>& out) const;
    void resegment(std::string_view text, std::vector<int>& out) const;

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<std::int32_t> types_;
    std::unordered_map<std::string, int> index_;
    std::vector<int> specials_;  // control / user-defined tokens, longest first
    int bos_ = 1, eos_ = 2, eot_ = -1;
    int byte_tokens_[256];
    bool add_space_prefix_ = true;
    std::string chat_template_;
};

// Formats one user turn with the model's chat template (zephyr <|user|>,
// chatml <|im_start|>, llama-2 [INST]); detected from tokenizer.chat_template.
std::string apply_chat_template(const Tokenizer& tok, const std::string& system, const std::string& user);

}  // namespace adamant
