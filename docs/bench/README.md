# Raw benchmark logs

Everything the main README quotes comes from these files. Machine: Intel Core i9-9900K (8 cores / 16 threads,
3.6 GHz base), dual-channel DDR4, 32 GB. Model files: TinyLlama-1.1B-Chat v1.0 from
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (SHA-256 in the main README).

| File | Command | System |
|---|---|---|
| `kbench-windows-i9-9900k.txt` / `.csv` | `adamant kbench -t 1,8 --min-time 0.5` | Windows 11, MSYS2 GCC 15.2 |
| `bench-windows-q4_k_m.txt` / `.csv` | `adamant bench -m ...Q4_K_M.gguf --isa scalar,avx2 -t 1,2,4,6,8,12,16 -n 64 --reps 3` | Windows 11 |
| `bench-windows-q8_0.txt` / `.csv` | same, Q8_0 | Windows 11 |
| `kbench-wsl2-i9-9900k.*`, `bench-wsl2-*.*` | `scripts/bench-linux.sh` (threads 1,4,8,12) | WSL2 Ubuntu, GCC 15.2, models on the Linux file system (`wsl2-system.txt`) |
| `reference-logits-q4_k_m.txt`, `reference-logits-q8_0.txt` | `adamant logits` + `scripts/reference_logits.py` | Windows 11, Python 3.14, numpy, gguf |
| `stream-session-windows.jsonl` | `adamant-view --record` during `adamant run --stream` with Euclase Monitor 1.1 running | Windows 11; source of `docs/img/terminal.gif` |

Notes:

- GB/s = weight bytes read per token (or per matrix-vector product) × rate. Activations, KV cache and scratch are
  not counted.
- `bench` generates greedily after the prompt "The history of the transistor begins" (8 tokens) and times the
  forward passes of the generated tokens only. Scalar runs are shorter (16 tokens, one repetition) because they
  are slow.
- The desktop was in normal use (Euclase Monitor running, other applications open); runs vary by about ±10 %, and every run is kept.
