// TokenStream -> Corundum ring -> reader: records arrive intact and in order.
// SPDX-License-Identifier: MIT
#include "adamant/stream.hpp"
#include "adamant/token_wire.h"
#include "check.hpp"
#include "corundum.h"

using namespace adamant;

int main() {
    const std::string ring = "/adamant-test-" + std::to_string(ADM_TEST_PID());
    TokenStream s;
    std::string err;
    if (!s.open(ring.c_str(), &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    crd_reader r;
    CHECK(crd_reader_open(&r, ring.c_str(), 0) == CRD_OK);

    s.session("TestModel", "avx2", 512, 8, 22, 123456789);
    s.prompt(17, 5000000);
    const char* words[] = {" Hello", ",", " world", "", "\xE2\x9C\x93"};
    for (int i = 0; i < 5; ++i) s.token(static_cast<std::uint64_t>(i), 100 + i, words[i], 1000u + static_cast<unsigned>(i));
    s.end(5, 4242, ADM_END_LIMIT);

    crd_msg m;
    int n = 0;
    while (crd_read(&r, &m) == CRD_OK) {
        if (n == 0) {
            CHECK(m.type == ADM_REC_SESSION && m.size == sizeof(adm_session) + 9);
            adm_session h;
            std::memcpy(&h, m.data, sizeof h);
            CHECK(h.wire_version == ADM_WIRE_VERSION && h.n_ctx == 512 && h.n_threads == 8 && h.n_layer == 22);
            CHECK(h.weight_bytes_per_token == 123456789 && std::strcmp(h.isa, "avx2") == 0 && h.text_len == 9);
            CHECK(std::memcmp(static_cast<const char*>(m.data) + sizeof h, "TestModel", 9) == 0);
        } else if (n == 1) {
            CHECK(m.type == ADM_REC_PROMPT);
            const adm_prompt* p = CRD_MSG_AS(adm_prompt, &m);
            CHECK(p && p->n_prompt == 17 && p->prefill_ns == 5000000);
        } else if (n >= 2 && n <= 6) {
            const int i = n - 2;
            CHECK(m.type == ADM_REC_TOKEN);
            adm_token h;
            std::memcpy(&h, m.data, sizeof h);
            CHECK(h.seq == static_cast<std::uint64_t>(i) && h.id == 100 + i && h.forward_ns == 1000u + static_cast<unsigned>(i));
            CHECK(h.text_len == std::strlen(words[i]) && m.size == sizeof h + h.text_len);
            CHECK(std::memcmp(static_cast<const char*>(m.data) + sizeof h, words[i], h.text_len) == 0);
        } else if (n == 7) {
            CHECK(m.type == ADM_REC_END);
            const adm_end* e = CRD_MSG_AS(adm_end, &m);
            CHECK(e && e->n_generated == 5 && e->gen_ns == 4242 && e->reason == ADM_END_LIMIT && e->dropped == 0);
        }
        ++n;
    }
    crd_release(&r);
    CHECK(n == 8);
    CHECK(s.dropped() == 0 && s.published() == 8);

    // A full ring drops instead of blocking.
    for (int i = 0; i < 100000; ++i) s.token(static_cast<std::uint64_t>(i), 1, "0123456789abcdef0123456789abcdef", 1);
    CHECK(s.dropped() > 0);
    s.close();
    CHECK(crd_writer_closed(&r));
    crd_reader_close(&r);
    crd_unlink(ring.c_str());
    return test_result("test_stream");
}
