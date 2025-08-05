# C++ TwoCrypto Implementation

High-performance C++ implementation of the TwoCrypto AMM pool, mirroring exact Vyper logic.

## Structure

```
cpp/
├── include/
│   ├── stableswap_math.hpp  # Math functions (newton_D, get_y, etc.)
│   └── twocrypto.hpp        # Pool implementation
├── src/
│   ├── stableswap_math.cpp  # Math implementation
│   ├── twocrypto.cpp        # Pool logic
│   └── benchmark_harness.cpp # Benchmark runner
└── CMakeLists.txt           # Build configuration
```

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --target benchmark_harness
```

## Usage

The benchmark harness processes pool configurations and action sequences:

```bash
./benchmark_harness <pool_configs.json> <action_sequences.json> <output.json>
```

## Features

- **256-bit arithmetic** using Boost.Multiprecision
- **Exact Vyper logic** including newton methods, price oracles, and fee calculations
- **Parameter packing** matching Vyper's factory patterns
- **JSON I/O** for easy integration with testing infrastructure

## Integration

This implementation is used by the Python benchmark suite for performance comparison with Vyper contracts.