# TwoCrypto – Vyper Reference, C++ Port, and Benchmarks

High-performance C++ implementation of Curve’s TwoCrypto AMM with exact parity to the Vyper reference, plus math/pool benchmarks and reproducible datasets.

## Overview

This repository contains:
- Vyper reference contracts (submodule) used for validation.
- A clean, parity-accurate C++ implementation of the TwoCrypto pool and math (uint256 variant).
- A fast double-precision variant for performance exploration.
- Harnesses and Python tooling to benchmark and compare C++ vs Vyper.

Parity goals met:
- Integer/uint256 variant mirrors `Twocrypto.vy` (function order, rounding, fees) with exact parity.
- Donation logic: burns only unlocked shares; protection window; cap respected.
- Oracle EMA matches Vyper (moving average with 2× price_scale cap).
- Time semantics for testing: relative time travel supported; synchronized start timestamp.

Performance note:
- Double-precision (`_d`) harness runs about 2× faster but loses roughly ~1% accuracy over 1,000,000 trades. Use `_i` for parity validation and `_d` for speed.

## Repository Layout

```
cpp-twocrypto/
├── twocrypto-ng/                        # Vyper reference (submodule)
│   └── contracts/main/*.vy              # Twocrypto, Math, Views, Factory
│
├── cpp/                                 # C++ implementation + harness (templated)
│   ├── include/
│   │   ├── stableswap_math.hpp          # Templated math (uint256 and double specializations)
│   │   └── twocrypto.hpp                # Templated pool (TwoCryptoPoolT<T>)
│   ├── src/
│   │   └── benchmark_harness.cpp        # Unified JSON harness (mode i|d)
│   └── CMakeLists.txt                   # Builds benchmark_harness
│
└── python/                              # Benchmarks and runners
    ├── benchmark_math/                  # Math-only benchmarks (C++ vs Vyper)
    ├── benchmark_pool/                  # Pool benchmark orchestration
    │   ├── generate_data.py             # Create pools + sequences (deterministic options)
    │   ├── run_full_benchmark.py        # Run both C++ variants and Vyper
    │   └── data/                        # Generated configs + per-run results
    ├── cpp_pool/
    │   └── cpp_pool_runner.py           # Build + run unified C++ harness (mode i|d)
    └── vyper_pool/vyper_pool_runner.py  # Deploy and run Vyper via titanoboa
```

## Requirements

- C++17 compiler + CMake
- Boost (with `json` component)
- Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- Vyper benchmarking uses [titanoboa](https://github.com/vyperlang/titanoboa) (installed via uv)
- Initialize submodule:
  ```bash
  git submodule update --init --recursive
  ```

## Build C++ and Run Harness

```bash
# Build (Release recommended)
cmake -B cpp/build cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target benchmark_harness -j  # Unified harness (mode i|d)

# Run harness directly (JSON I/O handled by Python runners normally)
./cpp/build/benchmark_harness i <pools.json> <sequences.json> <output.json>  # Integer/uint256
./cpp/build/benchmark_harness d <pools.json> <sequences.json> <output.json>  # Double-precision

# Control internal parallelism
CPP_THREADS=8 ./cpp/build/benchmark_harness i ...
```

Notes:
- The harness uses a thread pool internally (default = hardware threads). Override with env `CPP_THREADS`.
- One templated implementation is provided via `benchmark_harness`:
  - Mode `i`: Integer/uint256 (exact parity)
  - Mode `d`: Double-precision (approximate, usually faster)
- The double-precision mode converts 1e18-scaled JSON inputs to doubles (1.0 == 1e18) and back for outputs

## Math Benchmarks (C++ vs Vyper)

```bash
cd python
uv sync

# Basic math benchmark
uv run benchmark_math/main.py

# (Optional) alternate script
uv run benchmark_math/v2.py
```

## Pool Benchmarks (C++ vs Vyper)

Generate datasets (single sequence). Aggregated files are written to `python/benchmark_pool/data/pools.json` and `python/benchmark_pool/data/sequences.json`:
```bash
uv run python/benchmark_pool/generate_data.py --pools 3 --trades 20 --seed 42
```

Run full benchmark (runs both C++ variants and Vyper):
```bash
# C++ threads per process; Vyper worker processes
uv run python/benchmark_pool/run_full_benchmark.py --n-cpp 8 --n-py 1
```

What this does:
- C++ phase: processes all pools with internal threads (`--n-cpp` → env `CPP_THREADS`) for both `_i` and `_d` harnesses.
- Vyper phase: validation runs across all pools with `--n-py` worker processes (1 = sequential). Each worker handles a shard of pools in its own process for isolation.
- Writes results to a timestamped folder under `python/benchmark_pool/data/results/`.

Advanced:
- Run C++ only for a dataset (mode i|d):
  ```bash
  uv run python/cpp_pool/cpp_pool_runner.py i \
    python/benchmark_pool/data/pools.json \
    python/benchmark_pool/data/sequences.json \
    python/benchmark_pool/data/results/cpp_i_only.json

  uv run python/cpp_pool/cpp_pool_runner.py d \
    python/benchmark_pool/data/pools.json \
    python/benchmark_pool/data/sequences.json \
    python/benchmark_pool/data/results/cpp_d_only.json
  ```
- Run Vyper only:
  ```bash
  uv run python/vyper_pool/vyper_pool_runner.py \
    python/benchmark_pool/data/pools.json \
    python/benchmark_pool/data/sequences.json \
    python/benchmark_pool/data/results/vyper_only.json
  ```

## C++ Variants Benchmark (integer vs double)

The unified harness supports both variants; the full benchmark runs both and emits:
- `cpp_i_combined.json` (integer/uint256)
- `cpp_d_combined.json` (double)
- `vyper_combined.json` (reference)

Use debug helpers to compare variants: `uv run python/benchmark_pool/debug/variants_diff.py`

## Parity & Testing Notes

- Donation logic: only unlocked donation shares are burnable; protection window damping and cap enforced identically.
- Oracle EMA: moving average via `wad_exp`; state `last_prices` is capped to `2 * price_scale` when updating EMA.
- Time-travel: relative seconds preferred; absolute timestamp supported for legacy. `last_timestamp` updates only in EMA path (tweak_price) for parity.
- Debug helpers: see `python/benchmark_pool/debug/` for optional diff/inspect utilities.
- Integer (`_i`) is parity-accurate; double (`_d`) is approximate (≈1% over long runs) but faster (≈2×).
- Vyper parallelism: `--n-py > 1` enables multiple worker processes; depending on environment, titanoboa stability may prefer `--n-py 1–2`.
- Inspect runs (debug helpers):
  - Diff summary: `uv run python/benchmark_pool/debug/parse_and_diff.py [run_dir]`
  - Context around first divergence: `uv run python/benchmark_pool/debug/inspect_context.py <run_dir>`

## Outputs & Cleanup

- Results are written under `python/benchmark_pool/data/results/run_<UTCstamp>/`.
- The results directory is gitignored. To clean old runs:
  ```bash
  rm -rf python/benchmark_pool/data/results/*
  ```

## License / Credits

- Vyper reference contracts from Curve’s `twocrypto-ng` (see submodule).
- C++ port and benchmarking harness in this repository aim for exact numerical compatibility for research and performance testing.
