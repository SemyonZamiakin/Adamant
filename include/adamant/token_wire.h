/*
 * Adamant token stream wire format, version 1 ("ATS1").
 *
 * While it generates, `adamant run --stream` publishes on the Corundum ring
 * ADM_RING_NAME (Windows: "Local\corundum.adamant-tokens"). Every record starts
 * with a fixed, checked struct; records that carry text append UTF-8 bytes
 * (no NUL) right after the struct, `text_len` of them.
 *
 *   ADM_REC_SESSION  adm_session + model name     once, before the prompt is processed
 *   ADM_REC_PROMPT   adm_prompt                   prompt done (prefill timing)
 *   ADM_REC_TOKEN    adm_token + token text       one per generated token
 *   ADM_REC_END      adm_end                      generation finished
 *
 * The producer never blocks: if no reader drains the ring and it fills up,
 * records are dropped and counted (adm_end.dropped). All integers are
 * little-endian. Header-only, C99 / C++11. MIT, Copyright (c) 2026 Euclase.
 */
#ifndef ADAMANT_TOKEN_WIRE_H
#define ADAMANT_TOKEN_WIRE_H

#include <stdint.h>

#include "wire.h" /* Corundum HFT: CRD_WIRE_SIZE / CRD_WIRE_PACKED */

#define ADM_RING_NAME "/adamant-tokens"
#define ADM_RING_CAPACITY (1u << 20)
#define ADM_WIRE_VERSION 1u

#define ADM_REC_SESSION 1u
#define ADM_REC_PROMPT 2u
#define ADM_REC_TOKEN 3u
#define ADM_REC_END 4u

typedef struct adm_session {
    uint64_t t_ns;        /* producer's monotonic clock */
    uint64_t weight_bytes_per_token;
    uint32_t wire_version;
    uint32_t n_ctx;
    uint32_t n_threads;
    uint32_t n_layer;
    char isa[8];          /* "avx2" / "scalar", NUL-padded */
    uint32_t pid;
    uint32_t text_len;    /* model name follows */
} adm_session;
CRD_WIRE_SIZE(adm_session, 48);
CRD_WIRE_PACKED(adm_session, t_ns, weight_bytes_per_token, wire_version, n_ctx, n_threads, n_layer, isa, pid, text_len);

typedef struct adm_prompt {
    uint64_t t_ns;
    uint64_t prefill_ns;
    uint32_t n_prompt;
    uint32_t reserved;
} adm_prompt;
CRD_WIRE_SIZE(adm_prompt, 24);
CRD_WIRE_PACKED(adm_prompt, t_ns, prefill_ns, n_prompt, reserved);

typedef struct adm_token {
    uint64_t seq;         /* 0, 1, 2, ... per session */
    uint64_t t_ns;        /* when the token was sampled */
    uint64_t forward_ns;  /* time of the forward pass that produced it */
    int32_t id;
    uint32_t text_len;    /* token text follows (may be 0, e.g. control tokens) */
} adm_token;
CRD_WIRE_SIZE(adm_token, 32);
CRD_WIRE_PACKED(adm_token, seq, t_ns, forward_ns, id, text_len);

#define ADM_END_EOS 0u
#define ADM_END_LIMIT 1u
#define ADM_END_CONTEXT 2u

typedef struct adm_end {
    uint64_t t_ns;
    uint64_t n_generated;
    uint64_t gen_ns;      /* sum of forward passes for generated tokens */
    uint64_t dropped;     /* records lost because the ring was full */
    uint32_t reason;      /* ADM_END_* */
    uint32_t reserved;
} adm_end;
CRD_WIRE_SIZE(adm_end, 40);
CRD_WIRE_PACKED(adm_end, t_ns, n_generated, gen_ns, dropped, reason, reserved);

#endif
