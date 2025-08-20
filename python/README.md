# TwoCrypto Benchmarking Suite

Comprehensive benchmarking infrastructure for the TwoCrypto AMM implementation, comparing C++ and Vyper versions.

## Structure

```
python/
├── benchmark_math/     # Mathematical function benchmarks
├── benchmark_pool/     # Full pool benchmarks
├── cpp_pool/          # C++ pool runner (single module)
└── vyper_pool/        # Vyper pool runner (single module)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run math benchmarks
uv run benchmark_math/main.py

# Or run the cleaner v2 math benchmark
uv run benchmark_math/v2.py

# Generate pool test data
uv run benchmark_pool/generate_data.py

# Run full pool benchmarks
uv run benchmark_pool/run_full_benchmark.py
```

## Components

### Math Benchmarks (`benchmark_math/`)
- Tests individual mathematical functions (newton_D, get_y, etc.)
- Compares C++ implementations against Vyper reference
- Validates precision across different input ranges
- v2 script: `benchmark_math/v2.py` (modular, typed, CLI flags)

### Pool Benchmarks (`benchmark_pool/`)
- Complete pool simulation with trading sequences
- Generates diverse test scenarios
- Compares state evolution between implementations

### C++ Pool Runner (`cpp_pool/`)
- Single module: `cpp_pool_runner.py`
- Handles CMake build and harness execution
- Processes JSON input/output

### Vyper Pool Runner (`vyper_pool/`)
- Single module: `vyper_pool_runner.py`
- Deploys actual Vyper contracts via titanoboa
- Executes on-chain transactions in simulated environment

## Results

All benchmarks output detailed JSON results for analysis:
- State snapshots after each action
- Performance metrics (execution time)
- Comparison statistics between implementations
