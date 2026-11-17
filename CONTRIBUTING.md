# Contributing to Adamant SIMD-VIC

Thanks for your interest. Bug reports (a GGUF file that loads wrong or generates nonsense), benchmark results
from other CPUs, and focused pull requests are welcome.

## Before you start

- For anything larger than a small fix, open an issue first so the approach can be agreed.
- Keep changes focused: one topic per pull request.

## Build and test

Linux (GCC and Clang, Release, AddressSanitizer + UndefinedBehaviorSanitizer, ThreadSanitizer):

```bash
scripts/check-linux.sh
```

Windows (MSYS2 MinGW-w64, from Git Bash or the MSYS2 shell with `C:\msys64\mingw64\bin` on `PATH`):

```bash
cmake -S . -B build -G "MinGW Makefiles" -DADAMANT_WERROR=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Rules for the code

- **Every kernel has a scalar reference.** A new AVX2 (or other ISA) kernel goes in its own file, is added to
  the `Kernels` table, and must be bit-identical to the scalar version: integer work exact, float scales applied
  in the same order, no FMA contraction. Extend `test_kernels` for it.
- **Weights are never copied.** They are read in place from the mapping; anything that would allocate memory
  proportional to the model size needs a very good reason.
- **The GGUF parser trusts nothing.** Every length and offset read from a file is checked before use, and
  `test_gguf`'s truncation and corruption loops must keep passing.
- **Only measured numbers.** A performance claim in the README or docs comes with the command that produced it
  and the raw log in `docs/bench/`.
- C++17, no dependencies besides the vendored Corundum HFT, warning-free with GCC and Clang
  (`-Wall -Wextra -Wpedantic -Wshadow`).

## Commit messages

Short imperative subject line, a blank line, then what changed and why.

## License

By contributing you agree that your contribution is licensed under the MIT License of this repository.
