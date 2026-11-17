# Security policy

## Supported versions

Only the latest release of Adamant SIMD-VIC receives security fixes.

## Reporting a vulnerability

Please do not publish exploit details in a public issue.

1. Use GitHub private vulnerability reporting: the **Security** tab of this repository, then **Report a vulnerability**.
2. If that option is not available, open an issue titled "Security contact request" without technical details, and a private channel will be arranged.

Expect an acknowledgement within a few days. This is a small project maintained by one person; there is no bug bounty.

## Threat model

Model files are downloaded from the internet and must be treated as untrusted input.

| Concern | Status |
|---|---|
| Malformed or truncated GGUF file | Every string length, array count, tensor shape, type and offset is checked against the file size before use; tensors that extend past the end of the file or are misaligned are refused. Covered by a truncation and corruption fuzz loop in `test_gguf` and by ASan/UBSan in CI |
| Huge allocations from metadata | Counts and lengths are capped (strings 16 MiB, arrays 2^28 entries); hyper-parameters are bounded before buffers are sized |
| Weights with extreme values (NaN, inf) | Not a memory-safety issue: they produce garbage logits, not out-of-bounds access |
| Token ring `/adamant-tokens` | Created with mode 0600 on Linux (current user only); on Windows a named mapping in the session's `Local\` namespace. Another process of the same user can read the generated text |
| Reading `/euclase-monitor` | Records are copied out of shared memory and parsed with bounds-checked readers, so a misbehaving writer cannot make the viewer read out of bounds |

## In scope

- Memory-safety bugs in the GGUF parser, tokenizer, kernels or forward pass reachable from a model file.
- Out-of-bounds reads in `adamant-view` reachable from ring contents.

## Out of scope

- The content a model generates.
- Attacks that require the same user account or administrator rights.
