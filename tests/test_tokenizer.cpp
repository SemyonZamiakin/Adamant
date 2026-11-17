// SentencePiece tokenizer: score-ordered merges, byte fallback, special tokens, decode.
// SPDX-License-Identifier: MIT
#include "adamant/tokenizer.hpp"
#include "check.hpp"

#include <cstdio>

using namespace adamant;

int main() {
    std::vector<std::string> tok = {"<unk>", "<s>", "</s>"};
    std::vector<float> sc = {0, 0, 0};
    std::vector<std::int32_t> ty = {2, 3, 3};
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        tok.push_back(buf);
        sc.push_back(0);
        ty.push_back(6);
    }
    const std::string S = "\xE2\x96\x81";
    auto add = [&](const std::string& t, float s) {
        tok.push_back(t);
        sc.push_back(s);
        ty.push_back(1);
        return static_cast<int>(tok.size()) - 1;
    };
    const int id_sp = add(S, -5), id_h = add("h", -6), id_e = add("e", -7), id_l = add("l", -8), id_o = add("o", -20);
    const int id_sph = add(S + "h", -10), id_sphe = add(S + "he", -5), id_ll = add("ll", -6);
    const int id_sphell = add(S + "hell", -3), id_sphello = add(S + "hello", -1), id_i = add("i", -9), id_sphi = add(S + "hi", -2);
    (void)id_i; (void)id_h; (void)id_e; (void)id_l; (void)id_sph; (void)id_sphe; (void)id_ll; (void)id_sphell;
    Tokenizer t(tok, sc, ty, 1, 2);

    // "hello" -> "▁hello" through ll, ▁h, ▁he, ▁hell, ▁hello
    auto ids = t.encode("hello", true);
    CHECK(ids.size() == 2 && ids[0] == 1 && ids[1] == id_sphello);

    // a word break: "hello hello" -> ▁hello ▁hello
    ids = t.encode("hello hello", false);
    CHECK(ids.size() == 2 && ids[0] == id_sphello && ids[1] == id_sphello);

    // "helo" cannot reach ▁hello: ▁he + l + o
    ids = t.encode("helo", false);
    CHECK(ids.size() == 3 && ids[0] == id_sphe && ids[1] == id_l && ids[2] == id_o);

    // byte fallback for characters with no piece: é = C3 A9
    ids = t.encode("h\xC3\xA9", false);
    CHECK(ids.size() == 3 && ids[0] == id_sph && ids[1] == 3 + 0xC3 && ids[2] == 3 + 0xA9);
    CHECK(t.decode(ids) == " h\xC3\xA9");

    // special tokens: only with parse_special; text after a special gets a space prefix
    ids = t.encode("hi</s>hi", false, true);
    CHECK(ids.size() == 3 && ids[0] == id_sphi && ids[1] == 2 && ids[2] == id_sphi);
    ids = t.encode("hi</s>", false, false);
    CHECK(ids.size() > 3);  // "</s>" spelled out as bytes

    // decode: U+2581 -> space, control tokens vanish
    CHECK(t.piece(id_sphello) == " hello");
    CHECK(t.piece(1).empty() && t.piece(2).empty());
    CHECK(t.piece(3 + 'A') == "A");
    CHECK(t.piece(id_sp) == " ");
    CHECK(t.is_eog(2) && !t.is_eog(1));
    CHECK(t.encode("", false).empty());
    return test_result("test_tokenizer");
}
