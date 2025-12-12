# Modular arb_harness Execution Plan

## Goal
Refactor `cpp/src/arb_harness.cpp` (1608 lines, monolithic) into a modular structure that:
- Separates concerns into distinct files
- Supports multiple numeric types (double, float, long double)
- Enables future pool types via folder-based isolation
- Maintains exact output parity with the current `arb_harness`

## Reference Implementation
- **Source**: `cpp/src/arb_harness.cpp`
- **Pool math**: `cpp/include/stableswap_math.hpp`
- **Pool impl**: `cpp/include/twocrypto.hpp`
- **Python runner**: `python/arb_sim/arb_sim.py`

---

## Directory Structure

```
cpp_modular/
├── include/
│   ├── core/
│   │   ├── numeric_types.hpp     # NumTraits<T>, type selection
│   │   ├── json_utils.hpp        # parse_scaled_1e18<T>, to_str_1e18<T>, etc.
│   │   └── common.hpp            # differs_rel<T>, io_mutex
│   │
│   ├── pools/
│   │   └── twocrypto_fx/         # Current pool type (twocrypto + stableswap math + donations)
│   │       ├── math.hpp          # MathOps<T>, MathTraits<T>, wad_exp (from stableswap_math.hpp)
│   │       ├── pool.hpp          # TwoCryptoPoolT<T> (from twocrypto.hpp)
│   │       ├── helpers.hpp       # pool_xp_from, dyn_fee, simulate_exchange_once, instantaneous_dr
│   │       └── donation.hpp      # DonationCfg, make_donation<T>
│   │
│   ├── trading/
│   │   ├── costs.hpp             # Costs struct (pool-agnostic)
│   │   ├── decision.hpp          # Decision struct
│   │   ├── arbitrageur.hpp       # decide_trade<T>, decide_trade_size<T>, toms748_root
│   │   └── user_trader.hpp       # Synthetic user swap logic
│   │
│   ├── events/
│   │   ├── types.hpp             # Candle, Event structs (use double internally)
│   │   └── loader.hpp            # load_candles, load_events, gen_events declarations
│   │
│   ├── metrics/
│   │   └── tracker.hpp           # Metrics struct, all time-weighted tracking, APY windows
│   │
│   └── harness/
│       ├── cli.hpp               # CLI arg parsing structs/functions
│       ├── pool_job.hpp          # PoolJob<T>: per-pool simulation context
│       ├── event_loop.hpp        # run_event_loop<T>(): main per-pool processing
│       └── runner.hpp            # run_harness<T>(): top-level orchestration with thread pool
│
├── src/
│   ├── events.cpp                # Non-templated: load_candles, load_events, gen_events
│   ├── cli.cpp                   # CLI parsing implementation
│   └── main.cpp                  # Entry point with type dispatch
│
└── CMakeLists.txt
```

---

## Numeric Type Strategy

**Approach**: Three separate binaries (compile-time type selection via preprocessor).

| Binary | Macro | Type |
|--------|-------|------|
| `arb_harness` | (default) | `double` |
| `arb_harness_f` | `ARB_MODE_F` | `float` |
| `arb_harness_ld` | `ARB_MODE_LD` | `long double` |

In `main.cpp`:
```cpp
#if defined(ARB_MODE_F)
using RealT = float;
#elif defined(ARB_MODE_LD)
using RealT = long double;
#else
using RealT = double;
#endif
```

All templated code uses `RealT`. Non-templated code (events, CLI) uses `double`.

---

## Incremental Implementation Steps

Each step ends with a verification checkpoint. Human reviews before proceeding.

---

### Step 1: Skeleton + Build System

**Agent actions:**
1. Create `CMakeLists.txt` with all three targets
2. Create minimal `src/main.cpp` that:
   - Includes boost/json
   - Prints "arb_harness_mod: <type>" based on compile mode
   - Returns 0
3. Create empty placeholder headers to satisfy includes

**Build command:**
```bash
cd cpp_modular && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target arb_harness
```

**Human verification:**
```bash
./build/arb_harness
# Expected output: "arb_harness_mod: double"
./build/arb_harness_f
# Expected output: "arb_harness_mod: float"
```

**Checkpoint**: All three binaries build and print correct type.

---

### Step 2: Events Module (Non-templated)

**Agent actions:**
1. Create `include/events/types.hpp` with `Candle`, `Event` structs
2. Create `include/events/loader.hpp` with declarations
3. Create `src/events.cpp` with implementations from `arb_harness.cpp:626-769`
4. Update `main.cpp` to:
   - Accept CLI args for candles file
   - Load candles, print count
   - Load events, print count

**Build command:**
```bash
cd cpp_modular/build && cmake --build . --target arb_harness
```

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: "Loaded X candles -> Y events"
```

**Checkpoint**: Event loading works with real data.

---

### Step 3: Core Utilities

**Agent actions:**
1. Create `include/core/numeric_types.hpp` with `NumTraits<T>` for all 3 floating types
2. Create `include/core/common.hpp` with `differs_rel<T>`, `io_mutex`
3. Create `include/core/json_utils.hpp` with parsing helpers
4. Update `main.cpp` to test:
   - `differs_rel<double>(1.0, 1.0 + 1e-13)` → false
   - `differs_rel<double>(1.0, 1.1)` → true
   - Print results

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: "differs_rel tests passed" + event loading
```

**Checkpoint**: Core utilities work for all floating numeric types.

---

### Step 4: Pool + Math (twocrypto_fx)

**Agent actions:**
1. Create `include/pools/twocrypto_fx/math.hpp` (copy `stableswap_math.hpp`)
2. Create `include/pools/twocrypto_fx/pool.hpp` (copy `twocrypto.hpp`, update includes)
3. Update `main.cpp` to:
   - Create a pool with hardcoded params
   - Add initial liquidity
   - Print balances, D, price_scale

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: Pool created, balances printed, D > 0
```

**Checkpoint**: Pool math works, can create and initialize pool.

---

### Step 5: Pool Helpers

**Agent actions:**
1. Create `include/pools/twocrypto_fx/helpers.hpp` with:
   - `pool_xp_from`, `pool_xp_current`, `xp_to_tokens_j`
   - `dyn_fee`, `simulate_exchange_once`
   - `instantaneous_dr`, `balance_indicator`, `true_growth`
   - `coin0_equiv`
2. Update `main.cpp` to:
   - Simulate a small exchange (no state change)
   - Print dy, fee

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: "Simulated exchange: dy=X, fee=Y"
```

**Checkpoint**: Pool helpers work correctly.

---

### Step 6: Trading Logic

**Agent actions:**
1. Create `include/trading/costs.hpp` with `Costs` struct
2. Create `include/trading/decision.hpp` with `Decision` struct
3. Create `include/trading/arbitrageur.hpp` with:
   - `toms748_root`
   - `decide_trade`
   - `decide_trade_size`
4. Update `main.cpp` to:
   - Load first event, use its price as CEX price
   - Call `decide_trade` on the pool
   - Print decision (do_trade, i, j, dx, profit)

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: Decision printed (may or may not trade depending on price)
```

**Checkpoint**: Trading logic compiles and runs.

---

### Step 7: Single Event Processing

**Agent actions:**
1. Update `main.cpp` to:
   - Process first 10 events in a loop
   - For each: set timestamp, decide trade, execute if profitable
   - Print trade count, total profit

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json
# Expected: "Processed 10 events, trades=X, profit=Y"
```

**Checkpoint**: Basic event loop works.

---

### Step 8: CLI Module

**Agent actions:**
1. Create `include/harness/cli.hpp` with `CliArgs` struct
2. Create `src/cli.cpp` with `parse_cli()` implementation
3. Update `main.cpp` to use CLI parsing
4. Support: positional args, `--n-candles`, `--threads`, `--min-swap`, `--max-swap`

**Human verification:**
```bash
./build/arb_harness python/arb_sim/trade_data/eurusd/eurusd-1m.json --n-candles 100
# Expected: Only 100 candles loaded
```

**Checkpoint**: CLI parsing works.

---

### Step 9: Pool Config Parsing

**Agent actions:**
1. Create `include/harness/pool_job.hpp` with:
   - `PoolInit` struct
   - `parse_pool_entry<T>()` function
2. Update `main.cpp` to:
   - Accept pools.json as first arg
   - Parse pool config, create pool from it
   - Print pool params

**Human verification:**
```bash
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 100
# Expected: Pool params from config printed
```

**Checkpoint**: Pool config parsing works with real configs.

---

### Step 10: Metrics Tracker

**Agent actions:**
1. Create `include/metrics/tracker.hpp` with:
   - `Metrics` struct
   - `MetricsTracker<T>` class with all time-weighted tracking
   - Methods: `sample_pre_event()`, `record_trade()`, `finalize()`, `to_json()`
2. Update `main.cpp` to:
   - Create MetricsTracker
   - Process events, record trades
   - Print final metrics

**Human verification:**
```bash
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000
# Expected: Metrics printed (trades, notional, fees, etc.)
```

**Checkpoint**: Metrics tracking works.

---

### Step 11: Donation Logic

**Agent actions:**
1. Create `include/pools/twocrypto_fx/donation.hpp` with:
   - `DonationCfg` struct
   - `make_donation<T>()` function
2. Integrate into event loop in `main.cpp`
3. Print donation count if any

**Human verification:**
```bash
# Use a pool config with donation_apy > 0
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000
# Expected: "Donations: X" if config has donations
```

**Checkpoint**: Donations work.

---

### Step 12: User Trader

**Agent actions:**
1. Create `include/trading/user_trader.hpp` with synthetic user swap logic
2. Integrate into event loop
3. Add CLI flags: `--userswapfreq`, `--userswapsize`, `--userswapthresh`

**Human verification:**
```bash
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000 --userswapfreq 3600 --userswapsize 0.01
# Expected: User swaps executed (visible in metrics or logs)
```

**Checkpoint**: User swaps work.

---

### Step 13: Event Loop Extraction

**Agent actions:**
1. Create `include/harness/event_loop.hpp` with `run_event_loop<T>()`
2. Move all per-event logic from `main.cpp` into this function
3. `main.cpp` now just calls `run_event_loop()`

**Human verification:**
```bash
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000
# Expected: Same output as before
```

**Checkpoint**: Event loop extracted, behavior unchanged.

---

### Step 14: Thread Pool + Multi-Pool

**Agent actions:**
1. Create `include/harness/runner.hpp` with:
   - `run_harness<T>(cli_args)` function
   - Thread pool logic for parallel pool processing
2. Update `main.cpp` to call `run_harness<RealT>()`
3. Support multiple pools in config

**Human verification:**
```bash
# Use a config with multiple pools
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000 --threads 4
# Expected: Multiple pools processed in parallel
```

**Checkpoint**: Multi-pool threading works.

---

### Step 15: JSON Output

**Agent actions:**
1. Add JSON output building to `runner.hpp`
2. Output format: `metadata`, `runs[]` with `result`, `params`, `final_state`
3. Support `--save-actions` flag

**Human verification:**
```bash
./build/arb_harness python/arb_sim/run_data/pool_config.json python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/out.json --n-candles 1000
cat /tmp/out.json | python3 -m json.tool | head -50
# Expected: Valid JSON with expected structure
```

**Checkpoint**: JSON output matches expected format.

---

### Step 16: Parity Test

**Agent actions:**
1. Create `scripts/parity_test.py` that:
   - Runs old `cpp/build/arb_harness` and new `cpp_modular/build/arb_harness`
   - Compares outputs field-by-field with tolerance
   - Reports any differences

**Human verification:**
```bash
# Build old harness first
cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target arb_harness && cd ../..

# Run parity test
uv run python cpp_modular/scripts/parity_test.py \
    python/arb_sim/run_data/pool_config.json \
    python/arb_sim/trade_data/eurusd/eurusd-1m.json \
    --n-candles 1000
# Expected: "PASS: All fields match within tolerance"
```

**Checkpoint**: Exact parity with old implementation.

---

### Step 17: Python Runner Compatibility

**Agent actions:**
1. Update `arb_sim.py` to optionally use modular harness (env var or flag)
2. Test full pipeline

**Human verification:**
```bash
ARB_HARNESS_PATH=cpp_modular/build/arb_harness uv run python/arb_sim/arb_sim.py \
    python/arb_sim/trade_data/eurusd/eurusd-1m.json --n-candles 1000
# Expected: Same behavior as with old harness
```

**Checkpoint**: Python runner works with modular harness.

---

### Step 18: All Floating Types

**Agent actions:**
1. Build and test all three binaries
2. Run parity test for each

**Human verification:**
```bash
for bin in arb_harness arb_harness_f arb_harness_ld; do
    echo "Testing $bin..."
    ./cpp_modular/build/$bin python/arb_sim/run_data/pool_config.json \
        python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/${bin}_out.json \
        --n-candles 100 --threads 1
done
# Expected: All binaries produce valid output
```

**Checkpoint**: All floating types work.

---

### Step 19: Cleanup

**Agent actions:**
1. Remove debug prints from `main.cpp`
2. Ensure all headers have proper include guards
3. Final code review for style consistency

**Human verification:**
```bash
# Full run with real data
./cpp_modular/build/arb_harness python/arb_sim/run_data/pool_config.json \
    python/arb_sim/trade_data/eurusd/eurusd-1m.json /tmp/final.json
# Expected: Clean run, no debug output, valid JSON
```

**Checkpoint**: Production-ready code.

---

## Key Invariants to Preserve

1. **CLI interface**: Exact same arguments as current `arb_harness`
2. **JSON input format**: `pools.json` with `pool` and `costs` objects
3. **JSON output format**: `metadata`, `runs[]` with `result`, `params`, `final_state`, optional `actions`
4. **Threading**: Parallel pool processing via thread pool
5. **Error handling**: Silent catch on trade/donation failures (no crash)
6. **Action recording**: Runtime `--save-actions` flag

---

## Future Extensions (Out of Scope)

1. **New pool types**: Add folder `pools/<new_type>/` with math.hpp, pool.hpp, helpers.hpp
2. **Organic volume**: Add to `trading/organic_volume.hpp` - JSON stream + routing decision
3. **Pool type enum**: Add `"type": "twocrypto_fx"` to JSON config, dispatch in runner
