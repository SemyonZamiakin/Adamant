// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/tokenizer.hpp"

#include "adamant/common.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>

namespace adamant {

namespace {

const std::string kSpace = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK

std::size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid lead byte: treat as one byte (falls back to a byte token)
}

struct Symbol {
    int prev, next;
    const char* text;
    std::size_t n;
};

struct Bigram {
    int left, right;
    float score;
    std::size_t size;
};

struct BigramLess {
    bool operator()(const Bigram& a, const Bigram& b) const {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
    }
};

}  // namespace

Tokenizer::Tokenizer(const GgufFile& f) {
    const auto model = f.get_string("tokenizer.ggml.model").value_or("");
    if (model != "llama")
        throw Error("tokenizer model '" + model +
                    "' is not supported: Adamant 0.1 implements the SentencePiece (llama) tokenizer only");
    tokens_ = f.get_string_array("tokenizer.ggml.tokens");
    scores_ = f.get_f32_array("tokenizer.ggml.scores");
    types_ = f.get_i32_array("tokenizer.ggml.token_type");
    if (tokens_.empty()) throw Error("GGUF has no tokenizer.ggml.tokens");
    if (auto v = f.get_uint("tokenizer.ggml.bos_token_id")) bos_ = static_cast<int>(*v);
    if (auto v = f.get_uint("tokenizer.ggml.eos_token_id")) eos_ = static_cast<int>(*v);
    if (auto v = f.get_uint("tokenizer.ggml.add_space_prefix")) add_space_prefix_ = *v != 0;
    chat_template_ = f.get_string("tokenizer.chat_template").value_or("");
    init();
}

Tokenizer::Tokenizer(std::vector<std::string> tokens, std::vector<float> scores, std::vector<std::int32_t> types,
                     int bos, int eos, bool add_space_prefix)
    : tokens_(std::move(tokens)), scores_(std::move(scores)), types_(std::move(types)), bos_(bos), eos_(eos),
      add_space_prefix_(add_space_prefix) {
    init();
}

void Tokenizer::init() {
    const std::size_t n = tokens_.size();
    if (scores_.size() != n) scores_.assign(n, 0.f);
    if (types_.size() != n) types_.assign(n, static_cast<std::int32_t>(TokenAttr::Normal));
    if (bos_ < 0 || static_cast<std::size_t>(bos_) >= n || eos_ < 0 || static_cast<std::size_t>(eos_) >= n)
        throw Error("tokenizer: BOS/EOS id out of range");
    index_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) index_.emplace(tokens_[i], static_cast<int>(i));
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        byte_tokens_[b] = find(buf);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto a = static_cast<TokenAttr>(types_[i]);
        if ((a == TokenAttr::Control || a == TokenAttr::UserDefined) && !tokens_[i].empty())
            specials_.push_back(static_cast<int>(i));
    }
    std::sort(specials_.begin(), specials_.end(),
              [&](int a, int b) { return tokens_[static_cast<std::size_t>(a)].size() > tokens_[static_cast<std::size_t>(b)].size(); });
    for (const char* t : {"<|eot_id|>", "<|im_end|>", "<|end|>"}) {
        const int id = find(t);
        if (id >= 0) {
            eot_ = id;
            break;
        }
    }
}

int Tokenizer::find(std::string_view text) const {
    auto it = index_.find(std::string(text));
    return it == index_.end() ? -1 : it->second;
}

void Tokenizer::resegment(std::string_view text, std::vector<int>& out) const {
    const int id = find(text);
    if (id >= 0) {
        out.push_back(id);
        return;
    }
    for (unsigned char c : text) {
        const int b = byte_tokens_[c];
        if (b >= 0) out.push_back(b);
        else if (const int unk = find("<unk>"); unk >= 0) out.push_back(unk);
    }
}

void Tokenizer::encode_fragment(std::string text, std::vector<int>& out) const {
    // escape whitespace
    std::string s;
    s.reserve(text.size() * 2);
    for (char c : text) {
        if (c == ' ') s += kSpace;
        else s += c;
    }
    if (s.empty()) return;

    std::vector<Symbol> sym;
    for (std::size_t off = 0; off < s.size();) {
        const std::size_t len = std::min(utf8_len(static_cast<unsigned char>(s[off])), s.size() - off);
        sym.push_back({static_cast<int>(sym.size()) - 1, static_cast<int>(sym.size()) + 1, s.data() + off, len});
        off += len;
    }
    sym.back().next = -1;

    std::priority_queue<Bigram, std::vector<Bigram>, BigramLess> pq;
    std::unordered_map<std::string, std::pair<int, int>> rev_merge;
    auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const std::string_view piece(sym[static_cast<std::size_t>(left)].text,
                                     sym[static_cast<std::size_t>(left)].n + sym[static_cast<std::size_t>(right)].n);
        const int id = find(piece);
        if (id < 0) return;
        pq.push({left, right, scores_[static_cast<std::size_t>(id)], piece.size()});
        rev_merge[std::string(piece)] = {left, right};
    };
    for (std::size_t i = 1; i < sym.size(); ++i) try_add(static_cast<int>(i) - 1, static_cast<int>(i));

    while (!pq.empty()) {
        const Bigram b = pq.top();
        pq.pop();
        Symbol& l = sym[static_cast<std::size_t>(b.left)];
        Symbol& r = sym[static_cast<std::size_t>(b.right)];
        if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) continue;  // stale
        l.n += r.n;
        r.n = 0;
        l.next = r.next;
        if (r.next >= 0) sym[static_cast<std::size_t>(r.next)].prev = b.left;
        try_add(l.prev, b.left);
        try_add(b.left, l.next);
    }

    // Emit; pieces not in the vocabulary are split back along their merges.
    std::vector<std::pair<std::string_view, int>> stack;
    for (int i = 0; i >= 0; i = sym[static_cast<std::size_t>(i)].next) {
        const Symbol& x = sym[static_cast<std::size_t>(i)];
        std::string_view piece(x.text, x.n);
        if (find(piece) >= 0) {
            out.push_back(find(piece));
            continue;
        }
        // iterative version of llama.cpp's resegment()
        stack.clear();
        stack.emplace_back(piece, 0);
        while (!stack.empty()) {
            auto [p, depth] = stack.back();
            stack.pop_back();
            if (find(p) >= 0) {
                out.push_back(find(p));
                continue;
            }
            auto it = rev_merge.find(std::string(p));
            if (it == rev_merge.end() || depth > 64) {
                resegment(p, out);
                continue;
            }
            const Symbol& a = sym[static_cast<std::size_t>(it->second.first)];
            const std::string_view left(a.text, static_cast<std::size_t>(sym[static_cast<std::size_t>(it->second.second)].text - a.text));
            const std::string_view right(p.data() + left.size(), p.size() - left.size());
            stack.emplace_back(right, depth + 1);
            stack.emplace_back(left, depth + 1);
        }
    }
}

std::vector<int> Tokenizer::encode(std::string_view text, bool add_bos, bool parse_special) const {
    std::vector<int> out;
    if (add_bos) out.push_back(bos_);
    bool prev_special = true;
    std::size_t start = 0;
    auto flush = [&](std::size_t end) {
        if (end <= start) return;
        std::string frag(text.substr(start, end - start));
        if (add_space_prefix_ && prev_special) frag.insert(frag.begin(), ' ');
        encode_fragment(std::move(frag), out);
        prev_special = false;
    };
    if (parse_special && !specials_.empty()) {
        for (std::size_t i = 0; i < text.size();) {
            int hit = -1;
            for (int id : specials_) {
                const std::string& t = tokens_[static_cast<std::size_t>(id)];
                if (text.compare(i, t.size(), t) == 0) {
                    hit = id;
                    break;
                }
            }
            if (hit < 0) {
                ++i;
                continue;
            }
            flush(i);
            out.push_back(hit);
            prev_special = true;
            i += tokens_[static_cast<std::size_t>(hit)].size();
            start = i;
        }
    }
    flush(text.size());
    return out;
}

std::string Tokenizer::piece(int id) const {
    if (id < 0 || id >= n_vocab()) return {};
    const std::string& t = tokens_[static_cast<std::size_t>(id)];
    switch (attr(id)) {
        case TokenAttr::Control: case TokenAttr::Unknown: case TokenAttr::Unused: return {};
        case TokenAttr::Byte: {
            unsigned v = 0;
            if (t.size() == 6 && std::sscanf(t.c_str(), "<0x%02X>", &v) == 1) return std::string(1, static_cast<char>(v));
            return {};
        }
        default: break;
    }
    std::string out;
    out.reserve(t.size());
    for (std::size_t i = 0; i < t.size();) {
        if (t.compare(i, 3, kSpace) == 0) {
            out += ' ';
            i += 3;
        } else {
            out += t[i++];
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += piece(id);
    return s;
}

std::string apply_chat_template(const Tokenizer& tok, const std::string& system, const std::string& user) {
    const std::string& t = tok.chat_template();
    const std::string eos = tok.token_text(tok.eos());
    if (t.find("<|im_start|>") != std::string::npos) {
        std::string s;
        if (!system.empty()) s += "<|im_start|>system\n" + system + "<|im_end|>\n";
        return s + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    }
    if (t.find("[INST]") != std::string::npos) {
        if (!system.empty()) return "[INST] <<SYS>>\n" + system + "\n<</SYS>>\n\n" + user + " [/INST]";
        return "[INST] " + user + " [/INST]";
    }
    // zephyr (TinyLlama-Chat) and the default
    std::string s;
    if (!system.empty()) s += "<|system|>\n" + system + eos + "\n";
    return s + "<|user|>\n" + user + eos + "\n<|assistant|>\n";
}

}  // namespace adamant
