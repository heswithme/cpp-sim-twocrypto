// Events module - data types for candles and price events
#pragma once

#include <cstdint>

namespace arb {

// CEX candle data (OHLCV)
struct Candle {
    uint64_t ts;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Simplified price event (timestamp + price + volume + source candle)
struct Event {
    uint64_t ts;
    double p_cex;
    double volume;
    Candle candle;  // source candle (for detailed logging)
};

} // namespace arb
