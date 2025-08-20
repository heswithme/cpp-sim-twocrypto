# TwoCrypto – Vyper Reference, C++ Port, and Benchmarks

High-performance C++ implementation of Curve’s TwoCrypto AMM with exact parity to the Vyper reference, plus math/pool benchmarks and reproducible datasets.

## Overview

This repository contains:
- Vyper reference contracts (submodule) used for validation.
- A clean, parity-accurate C++ implementation of the TwoCrypto pool and math.
- Harnesses and Python tooling to benchmark and compare C++ vs Vyper.

Parity goals met:
- Functionality mirrors `Twocrypto.vy` (function order, rounding, fees).
- Donation logic: burns only unlocked shares; protection window; cap respected.
- Oracle EMA matches Vyper (moving average with 2× price_scale cap).
- Time semantics for testing: absolute `time_travel`, synchronized start timestamp.

## Repository Layout

```
cpp-twocrypto/
├── twocrypto-ng/                    # Vyper reference (submodule)
│   └── contracts/main/*.vy          # Twocrypto, Math, Views, Factory
│
├── cpp/                             # C++ implementation + harness
│   ├── include/
│   │   └── twocrypto.hpp            # Pool class (reads like Vyper)
│   ├── src/
│   │   ├── stableswap_math.cpp      # Math: newton_D / get_y / get_p / wad_exp
│   │   ├── twocrypto.cpp            # Pool logic: add/exchange/remove/tweak_price
│   │   └── benchmark_harness.cpp    # JSON harness (internally parallel)
│   └── CMakeLists.txt
│
└── python/                          # Benchmarks and runners
    ├── benchmark_math/              # Math-only benchmarks (C++ vs Vyper)
    ├── benchmark_pool/              # Pool benchmark orchestration
    │   ├── generate_data.py         # Create pools + sequences (deterministic options)
    │   ├── run_full_benchmark.py    # Run C++ then Vyper (per-pool parallelism)
    │   └── data/                    # Generated configs + per-run results
    ├── cpp_pool/cpp_pool_runner.py  # Build + run C++ harness for a dataset
    └── vyper_pool/vyper_pool_runner.py # Deploy and run Vyper via titanoboa
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
cmake --build cpp/build --target benchmark_harness -j

# Run harness directly (JSON I/O handled by Python runners normally)
./cpp/build/benchmark_harness <pools.json> <sequences.json> <output.json>

# Control internal parallelism
CPP_THREADS=8 ./cpp/build/benchmark_harness ...
```

Notes:
- The harness uses a thread pool internally (default = hardware threads). Override with env `CPP_THREADS`.
- `twocrypto.cpp` mirrors Vyper logic and names; `_calc_token_fee` and `_fee` follow the same order and rounding.

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

Generate datasets (defaults: 3 pools × 4 sequences × 20 trades):
```bash
uv run python/benchmark_pool/generate_data.py \
  --pools 3 --sequences 4 --trades 20
```

Run full benchmark (C++ first, then Vyper):
```bash
# Python workers per phase (per-pool), C++ threads per process
uv run python/benchmark_pool/run_full_benchmark.py --n-py 1 --n-cpp 8
```

What this does:
- C++ phase: processes pools with internal threads (`--n-cpp` → env `CPP_THREADS`).
- Vyper phase: validation runs; sequential by default (`--n-py 1`).
- Logs progress per pool and writes results to a timestamped folder under `python/benchmark_pool/data/results/`.

Advanced:
- Run C++ only for a dataset:
  ```bash
  uv run python/cpp_pool/cpp_pool_runner.py \
    python/benchmark_pool/data/benchmark_pools.json \
    python/benchmark_pool/data/benchmark_sequences.json \
    python/benchmark_pool/data/results/cpp_only.json
  ```
- Run Vyper only:
  ```bash
  uv run python/vyper_pool/vyper_pool_runner.py \
    python/benchmark_pool/data/benchmark_pools.json \
    python/benchmark_pool/data/benchmark_sequences.json \
    python/benchmark_pool/data/results/vyper_only.json
  ```

## Parity & Testing Notes

- Donation logic: only unlocked donation shares are burnable; protection window damping and cap enforced identically.
- Oracle EMA: moving average via `wad_exp`; state `last_prices` is capped to `2 * price_scale` when updating EMA.
- Time-travel: absolute timestamp jumps affect `block_timestamp` only; `last_timestamp` is updated in EMA path (tweak_price) for parity.
- Debug trace: set `TRACE=1` to get concise internal logs from the C++ tweak_price path; filter with `TRACE_ONLY_POOL` and `TRACE_ONLY_SEQUENCE`.

## Outputs & Cleanup

- Results are written under `python/benchmark_pool/data/results/run_<UTCstamp>/`.
- The results directory is gitignored. To clean old runs:
  ```bash
  rm -rf python/benchmark_pool/data/results/*
  ```

## License / Credits

- Vyper reference contracts from Curve’s `twocrypto-ng` (see submodule).
- C++ port and benchmarking harness in this repository aim for exact numerical compatibility for research and performance testing.
