#!/usr/bin/env bash
# Linux benchmark run used for docs/bench/*-wsl2-*.txt.
# Usage: scripts/bench-linux.sh MODEL_DIR   (copies the models to ~/adamant-models first:
# mmap over a Windows drive (/mnt/c, /mnt/d) in WSL is far slower than a native file system)
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
models="${1:?model directory}"
build="$HOME/adamant-build/bench-release"
local_models="$HOME/adamant-models"
mkdir -p "$local_models"
for f in "$models"/*.gguf; do
  [ -f "$local_models/$(basename "$f")" ] || cp "$f" "$local_models/"
done
CC="$(command -v gcc-15 || command -v gcc)" CXX="$(command -v g++-15 || command -v g++)" \
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DADAMANT_BUILD_TESTS=OFF >/dev/null
cmake --build "$build" -j"$(nproc)" >/dev/null
out="$src/docs/bench"
"$build/adamant" kbench -t 1,8 --min-time 0.5 --csv "$out/kbench-wsl2-i9-9900k.csv" > "$out/kbench-wsl2-i9-9900k.txt" 2>&1
for q in Q4_K_M Q8_0; do
  l="$(echo "$q" | tr 'A-Z' 'a-z')"
  "$build/adamant" bench -m "$local_models/tinyllama-1.1b-chat-v1.0.$q.gguf" --isa scalar,avx2 -t 1,4,8,12 -n 64 --reps 3 \
    --csv "$out/bench-wsl2-$l.csv" > "$out/bench-wsl2-$l.txt" 2>&1
done
uname -sr > "$out/wsl2-system.txt"; "$(command -v gcc-15 || command -v gcc)" --version | head -1 >> "$out/wsl2-system.txt"
echo done
