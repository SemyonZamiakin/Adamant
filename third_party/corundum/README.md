# Corundum HFT (vendored)

Copy of [Corundum HFT](https://github.com/SemyonZamiakin/CorundumHFT) 1.2.0 (commit `06774be`):
`include/corundum/corundum.h`, `include/corundum/wire.h`, `src/corundum.c`. MIT, same authors (see `LICENSE`).

Adamant uses it to stream generated tokens to other local programs through shared memory
(ring `/adamant-tokens`) and to read live CPU and RAM metrics published by
[Euclase Monitor](https://github.com/SemyonZamiakin/EuclaseMonitor) (ring `/euclase-monitor`).
The only local change is the include path in `corundum.c` (`"corundum.h"`). Update all three files together.
