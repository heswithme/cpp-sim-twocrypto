# TwoCrypto – Vyper Reference, C++ Port, Benchmarks, and Arbitrage Sim

High‑performance C++ implementation of Curve’s TwoCrypto AMM with exact parity to the Vyper reference, plus math/pool benchmarks, a candle‑driven arbitrage simulator, and reproducible datasets.

## Overview

This repository contains:
- Vyper reference contracts (submodule) used for validation.
- A clean, parity‑accurate C++ implementation of the TwoCrypto pool and math (uint256 variant) and a fast double variant.
- Harnesses and Python tooling to benchmark and compare C++ vs Vyper.
- A fast, candle‑driven arbitrage simulator (C++ core with a Python launcher) designed for large datasets (100MB+) and parameter scans.

Parity goals met:
- Integer/uint256 variant mirrors `Twocrypto.vy` (function order, rounding, fees) with exact parity.
- Donation logic: burns only unlocked shares; protection window; cap respected.
- Oracle EMA matches Vyper (moving average with 2× price_scale cap).
- Time semantics for testing: relative time travel supported; synchronized start timestamp.

Performance notes:
- Double‑precision (`_d`) pool harness runs about 2× faster but loses roughly ~1% accuracy over long runs. Use `_i` for parity validation and `_d` for speed exploration.
- Arbitrage simulator uses mmap + byte‑scan for candles and is optimized for quick slicing and grid scans.

## Repository Layout

```
cpp-twocrypto/
├── twocrypto-ng/                        # Vyper reference (submodule)
│   └── contracts/main/*.vy              # Twocrypto, Math, Views, Factory
│
├── cpp/                                 # C++ implementation + harnesses
│   ├── include/
│   │   ├── stableswap_math.hpp          # Templated math (uint256 and double)
│   │   └── twocrypto.hpp                # Templated pool (TwoCryptoPoolT<T>)
│   ├── src/
│   │   ├── benchmark_harness.cpp        # Unified JSON pool harness (mode i|d)
│   │   └── arb_harness.cpp              # Candle-driven arbitrage harness (C++ core)
│   └── CMakeLists.txt                   # Builds benchmark_harness and arb_harness
│
└── python/                              # Benchmarks and runners
    ├── benchmark_math/                  # Math-only benchmarks (C++ vs Vyper)
    ├── benchmark_pool/                  # Pool benchmark orchestration
    │   ├── generate_data.py             # Create pools + sequences
    │   ├── run_full_benchmark.py        # Run both C++ variants and Vyper
    │   └── data/                        # Generated configs + per-run results
    ├── cpp_pool/
    │   └── cpp_pool_runner.py           # Build + run unified C++ pool harness (mode i|d)
    ├── vyper_pool/                      # Vyper via titanoboa
    └── arb_sim/                         # Arbitrage simulator launcher + data
        ├── arb_sim.py                   # Unified arb runner (single + grid)
        ├── generate_pools.py            # Two-parameter grid pool generator
        ├── plot_candles.py              # Optional visualization
        ├── run_data/                    # Outputs (pretty JSON)
        └── trade_data/                  # Input candles (e.g., brlusd/brlusd-1m.json)
```

## Requirements

- C++17 compiler + CMake
- Boost (with `json` component)
- Python 3.10+ (uv optional for Vyper tooling)
- For Vyper parity: [titanoboa](https://github.com/vyperlang/titanoboa)
- Initialize submodule:
  ```bash
  git submodule update --init --recursive
  ```

## Build C++ and Run Pool Harness

```bash
# Build (Release recommended)
cmake -B cpp/build cpp -DCMAKE_BUILD_TYPE=Release
# Build unified pool harness (accepts mode i|d at runtime)
cmake --build cpp/build --target benchmark_harness -j

# Run harness directly (JSON I/O normally driven by Python)
./cpp/build/benchmark_harness i <pools.json> <sequences.json> <output.json>  # Unified, mode-select
./cpp/build/benchmark_harness d <pools.json> <sequences.json> <output.json>  # Unified, double

# Control internal parallelism
CPP_THREADS=8 ./cpp/build/benchmark_harness i ...
```

Notes:
- The harness uses a thread pool internally (default = hardware threads). Override with env `CPP_THREADS`.
- Mode `i`: Integer/uint256 (exact parity). Mode `d`: Double (approximate, usually faster).
- The double mode interprets 1e18-scaled JSON inputs as 1.0 and converts back for outputs.

## Arbitrage Simulator

The arb simulator replays a candles file and simulates an arbitrageur against the pool (CEX with infinite liquidity at per-event price). It is optimized for large files via mmap + byte scanning and provides timing metrics.

Build:
```bash
cmake -B cpp/build cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target arb_harness -j
```

Python launcher (single or grid):
```bash
# Single run on first 1000 candles, with balance-based sizing and optional action capture
python3 python/arb_sim/arb_sim.py \
  python/arb_sim/trade_data/brlusd/brlusd-1m.json \
  --n-candles 1000 \
  --min-swap 1e-6 \
  --max-swap 1.0 \
  --save-actions

# Grid over A and fees (pretty prints all outputs)
python3 python/arb_sim/arb_sim.py \
  python/arb_sim/trade_data/brlusd/brlusd-1m.json \
  --A 50000,100000,150000 \
  --mid-fee-bps 2.5,3.0 \
  --out-fee-bps 5.0,6.0 \
  --jobs 4 \
  --n-candles 20000 \
  --min-swap 1e-6 --max-swap 1.0
```

Key flags:
- `--n-candles N`: slice the first N candles for rapid iteration (two events per candle).
- `--min-swap`, `--max-swap`: min/max fraction of from-side balance for binary sizing (default 1e-6 and 1.0). Also capped by `max_trade_frac` and per-event volume cap if enabled.
- `--save-actions`: include executed trades in result JSON for slower Vyper replay later.

Inputs:
- Candles JSON is an array of `[timestamp, open, high, low, close, volume]`. If `timestamp > 1e10`, it is interpreted as ms and divided by 1000.

Outputs:
- Written to `python/arb_sim/run_data/`:
  - Single: `arb_run_<UTC>.json` with:
    - `result`: events/trades/notional, LP fees, arb PnL, and timing (`candles_read_ms`, `exec_ms`).
    - `params`: the `pool` and `costs` used (1e18/1e10 scaling where applicable).
    - `final_state`: complete pool snapshot (1e18 strings for numeric fields).
    - `actions` (optional): executed trades with timestamps, sizes, fees, and profits.
  - Grid: per-combo `result_*.json` + `summary.json` (sorted by arb_pnl).

Pool grid generator (optional):
```bash
python3 python/arb_sim/generate_pools.py \
  --out python/arb_sim/run_data/pools.json \
  --param-x A --x-values 50000,100000,150000 \
  --param-y mid_fee_bps --y-values 2.5,3.0
```

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

- Pool benchmark results are written under `python/benchmark_pool/data/results/run_<UTCstamp>/`.
- Arbitrage simulator results are written under `python/arb_sim/run_data/`.
- These directories are gitignored. To clean old runs:
  ```bash
  rm -rf python/benchmark_pool/data/results/*
  rm -rf python/arb_sim/run_data/*
  ```

## License / Credits

- Vyper reference contracts from Curve’s `twocrypto-ng` (see submodule).
- C++ port and benchmarking harness aim for exact numerical compatibility for research and performance testing.
- Arbitrage simulator designed for high‑throughput parameter scans on large candle datasets.
