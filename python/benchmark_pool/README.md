# TwoCrypto Pool Benchmark System

Comprehensive benchmark system for comparing C++ and Vyper implementations of the TwoCrypto AMM pool.

## Structure

- `generate_data.py` - Generate test pool configurations and trading sequences
- `run_full_benchmark.py` - Main benchmark runner that executes both C++ and Vyper tests
- `data/` - Test data and results
- `cpp_pool/` - C++ pool runner and builder
- `vyper_pool/` - Vyper pool deployment and runner (all-in-one)

## Usage

1. Generate test data:
```bash
uv run benchmark_pool/generate_data.py
```

2. Run full benchmark:
```bash
uv run benchmark_pool/run_full_benchmark.py
```

This will:
- Generate 10 pool configurations with varying parameters
- Create 5 trading sequences with 20 trades each
- Run all 50 combinations through both C++ and Vyper implementations
- Compare and report results

## Components

### C++ Pool
- Uses Boost.Multiprecision for 256-bit arithmetic
- Mirrors exact Vyper logic including newton methods, price oracles, and fee calculations
- Compiled with CMake and executed via harness

### Vyper Pool
- Deploys actual Vyper contracts via titanoboa
- Uses snekmate for ERC20 mock tokens
- Executes on-chain transactions in simulated environment

## Results

Results are saved to `data/results/` including:
- Individual C++ and Vyper benchmark results
- Comparison metrics between implementations
- Execution time statistics