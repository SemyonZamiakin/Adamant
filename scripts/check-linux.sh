#!/usr/bin/env bash
# Builds and tests Adamant on Linux with GCC and Clang, Release and ASan+UBSan.
# Usage: scripts/check-linux.sh [build-root]   (default: ~/adamant-build)
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$HOME/adamant-build}"
gcc_c="$(command -v gcc-15 || command -v gcc)"; gcc_cxx="$(command -v g++-15 || command -v g++)"
clang_c="$(command -v clang-21 || command -v clang)"; clang_cxx="$(command -v clang++-21 || command -v clang++)"
jobs="$(nproc)"
export ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

run() {  # name cc cxx build-type sanitize
  local dir="$root/$1"
  echo "=== $1: $("$2" --version | head -1) | $4 ${5:+| sanitize=$5}"
  CC="$2" CXX="$3" cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE="$4" -DADAMANT_WERROR=ON -DADAMANT_SANITIZE="$5" >/dev/null
  cmake --build "$dir" -j"$jobs" >/dev/null
  (cd "$dir" && ctest --output-on-failure 2>&1 | tail -n 3)
}

run gcc-release "$gcc_c" "$gcc_cxx" Release ""
run clang-release "$clang_c" "$clang_cxx" Release ""
run gcc-asan-ubsan "$gcc_c" "$gcc_cxx" Debug "address,undefined"
run clang-asan-ubsan "$clang_c" "$clang_cxx" Debug "address,undefined"

# ThreadSanitizer on the threaded parts (thread pool, model forward)
echo "=== clang-tsan: thread pool + model"
CC="$clang_c" CXX="$clang_cxx" cmake -S "$src" -B "$root/clang-tsan" -DCMAKE_BUILD_TYPE=Debug -DADAMANT_SANITIZE=thread >/dev/null
cmake --build "$root/clang-tsan" -j"$jobs" --target test_thread_pool test_model >/dev/null
(cd "$root/clang-tsan" && TSAN_OPTIONS=halt_on_error=1 ./test_thread_pool && TSAN_OPTIONS=halt_on_error=1 ./test_model)
