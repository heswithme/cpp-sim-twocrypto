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
        ├── arb_sim.py                   # Multi-pool arb runner (threaded C++)
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

The arb simulator replays a candles file and simulates an arbitrageur against the pool (CEX with infinite liquidity at per‑event price). It is optimized for large files via mmap + byte scanning and provides timing metrics. The C++ harness runs multiple pools in one process with internal threading.

Build:
```bash
cmake -B cpp/build cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target arb_harness -j
```

Python launcher (multi‑pool via pool_config.json):
```bash
# Generate a pool grid (writes python/arb_sim/run_data/pool_config.json)
python3 python/arb_sim/generate_pools.py

# Run all pools with 4 threads on first 20k candles
python3 python/arb_sim/arb_sim.py \
  python/arb_sim/trade_data/brlusd/brlusd-1m.json \
  -n 4 \
  --n-candles 20000 \
  --min-swap 1e-6 --max-swap 1.0
```

Key flags:
- `--n-candles N`: slice the first N candles for rapid iteration (two events per candle).
- `--min-swap`, `--max-swap`: min/max fraction of from-side balance for binary sizing (default 1e-6 and 1.0). Also capped by `max_trade_frac` and per-event volume cap if enabled.
- `--save-actions`: include executed trades in result JSON for slower Vyper replay later.
 - `-n/--threads N`: number of C++ worker threads (default: hardware threads).

Inputs:
- Candles JSON is an array of `[timestamp, open, high, low, close, volume]`. If `timestamp > 1e10`, it is interpreted as ms and divided by 1000.

Outputs:
- Written to `python/arb_sim/run_data/`:
  - Aggregated: `arb_run_<UTC>.json` with:
    - `metadata`: `candles_file`, `threads`, `candles_read_ms`, `exec_ms`, and `base_pool` + `grid` (if present).
    - `runs[]`: one entry per pool with:
      - `x_key`, `x_val`, `y_key`, `y_val` (from the grid or `None`).
      - `result`: events, trades, total_notional_coin0, lp_fee_coin0, arb_pnl_coin0, n_rebalances, pool_exec_ms.
      - `final_state`: complete pool snapshot (1e18 strings for numeric fields).
      - `actions` (optional): per-trade details if `--save-actions` was used.

Pool grid generator:
- Edit `python/arb_sim/generate_pools.py` to set `X_name`, `Y_name` and their ranges (uses `numpy.logspace`).
- All pool parameters are specified as integers in their native units (fees 1e10, WAD‑like fields 1e18, balances 1e18). The script only stringifies when saving.
- Runs with: `python3 python/arb_sim/generate_pools.py` and writes `python/arb_sim/run_data/pool_config.json` including `meta.base_pool` and `meta.grid`.

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

- Convert arb_sim actions to a benchmark sequence:
  ```bash
  # Produce an arb_run_*.json with --save-actions first
  python3 python/arb_sim/arb_sim.py python/arb_sim/trade_data/brlusd/brlusd-1m.json --save-actions

  # Convert latest arb_sim run’s extended actions to sequences.json and pools.json
  uv run python/benchmark_pool/arb_actions_to_sequence.py \
    --output-seq python/benchmark_pool/data/sequences.json \
    --output-pools python/benchmark_pool/data/pools.json
  # Or select by grid position if present in arb_run: --x-val <X> [--y-val <Y>]
  # Optionally point to the pool grid used by arb_sim (defaults to run_data/pool_config.json):
  #   --pool-config python/arb_sim/run_data/pool_config.json
  # Note: by default it expects a single actions-carrying run and picks the latest arb_run_* file.
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

### arb_sim Parity Replay (no wrapper)

Replay arb_sim trades and validate final-state parity with the cpp-double variant using three explicit steps:

1) Save actions from arb_sim:
   ```bash
   python3 python/arb_sim/arb_sim.py \
     python/arb_sim/trade_data/brlusd/brlusd-1m.json \
     --save-actions
   ```

2) Convert actions to benchmark inputs and run C++ variants (i and d):
   ```bash
   # Convert latest arb_run_* => python/benchmark_pool/data/{pools,sequences}.json
   uv run python/benchmark_pool/arb_actions_to_sequence.py

   # Run integer and double harnesses; saves to run_cpp_variants_<UTC>/
   uv run python/benchmark_pool/run_cpp_variants.py --n-cpp 1
   ```

3) Compare arb_sim final state vs cpp-double final state:
   ```bash
   # By default picks latest arb_run_* and latest run_* unless --run-dir is passed
   uv run python/benchmark_pool/debug/arb_vs_double.py --run-dir python/benchmark_pool/data/results/run_cpp_variants_YYYYMMDDTHHMMSSZ
   ```

Notes:
- For step-wise diffs against Vyper you still need a full run (includes Vyper):
  ```bash
  uv run python/benchmark_pool/run_full_benchmark.py --n-cpp 1 --n-py 1
  uv run python/benchmark_pool/debug/double_vs_vyper.py  # latest run_*
  ```

- Converter semantics (for exact replay):
  - Sequence `start_timestamp` is set to the pool’s `start_timestamp` from arb_sim run params when present; otherwise it is omitted to mirror arb_harness initialization (so EMA baseline matches).
  - Each trade is preceded by an absolute `time_travel { "timestamp": ts }` action to align time precisely.
  - `dx` is read with `Decimal` and scaled exactly to 1e18 using half-up rounding, minimizing any quantization drift; the double harness reads this back into `double`.

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
