// Detailed per-candle output for compatibility with old simulator format
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "events/types.hpp"

namespace arb {
namespace harness {

// Per-candle state entry matching old simulator's detailed-output.json format
template <typename T>
struct DetailedEntry {
    uint64_t t;           // candle timestamp
    T token0;             // pool balance[0]
    T token1;             // pool balance[1]
    T price_oracle;       // EMA oracle price
    T price_scale;        // price scale
    T profit;             // virtual_price - 1.0
    T xcp;                // raw xcp_profit value
    T open;               // candle OHLC
    T high;
    T low;
    T close;
    T fee;                // dynamic fee at this point
};

// Write detailed entries to JSON file
// Format: array of objects matching old simulator's detailed-output.json
template <typename T>
bool write_detailed_log(const std::string& path, const std::vector<DetailedEntry<T>>& entries) {
    std::ofstream out(path);
    if (!out) return false;
    
    out << std::setprecision(8);
    out << "[\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        out << "{\"t\": " << e.t
            << ", \"token0\": " << e.token0
            << ", \"token1\": " << e.token1
            << ", \"price_oracle\": " << e.price_oracle
            << ", \"price_scale\": " << e.price_scale
            << ", \"profit\": " << e.profit
            << ", \"xcp\": " << e.xcp
            << ", \"open\": " << e.open
            << ", \"high\": " << e.high
            << ", \"low\": " << e.low
            << ", \"close\": " << e.close
            << ", \"fee\": " << e.fee
            << "}";
        if (i + 1 < entries.size()) out << ",";
        out << "\n";
    }
    out << "]\n";
    
    return out.good();
}

} // namespace harness
} // namespace arb
