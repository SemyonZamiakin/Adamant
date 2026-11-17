# Changelog

## 0.1.0 (2026-11-17)

First version.

- **GGUF v2/v3 loader** over a read-only memory mapping (`mmap` on Linux, `CreateFileMapping` / `MapViewOfFile`
  on Windows): weights are used in place, never copied. Every length, count, shape and offset is bounds-checked.
- **Kernels for Q4_K, Q6_K, Q8_0, F16, F32**: a scalar reference (compiled without auto-vectorization) and
  hand-written AVX2 (`vpmaddubsw` / `vpmaddwd` integer pipelines, F16C), selected at run time with `cpuid`.
  AVX2 results are **bit-identical** to the scalar reference: quantizers, dot products, dequantizers and the
  float dot products used in attention. `--isa scalar` and `--isa avx2` generate the same tokens.
- **llama-architecture inference**: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, float32 KV cache,
  tied or separate output matrix; SentencePiece tokenizer with byte fallback; zephyr, ChatML and llama-2 chat
  templates; greedy, temperature, top-k, top-p and repetition penalty sampling.
- **Cache-aware layout**: 64-byte aligned activation, KV-cache and scratch buffers; per-cache-line atomics in a
  fork-join thread pool that spins between jobs and sleeps when idle.
- **Euclase integration**: tokens and timing streamed over a Corundum HFT ring (`/adamant-tokens`, wire format
  `ATS1` with compile-time layout checks); `adamant-view` console dashboard with live per-core CPU load and RAM
  from Euclase Monitor's EMS1 ring, or its own sampler.
- **Tools**: `adamant run | bench | kbench | info | logits | tokenize`, `adamant-view`.
- **Verification**: logits of TinyLlama-1.1B-Chat (Q4_K_M and Q8_0) compared with an independent numpy
  reference that dequantizes with the `gguf` package; 8 test suites (fp16, quantization round-trips, AVX2 =
  scalar, GGUF fuzzing, tokenizer, thread pool, synthetic model vs a double-precision reference, token stream).
  CI: Linux GCC / Clang, Debug, ASan + UBSan, TSan, a build without AVX2; Windows MinGW-w64 MINGW64 and UCRT64.
