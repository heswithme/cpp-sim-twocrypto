// Trading costs configuration (templated)
#pragma once

#include <type_traits>

namespace arb {
namespace trading {

template <typename T>
struct Costs {
    T arb_fee_bps{static_cast<T>(10.0)};   // exchange taker fee in bps (default: 10 bps)
    T gas_coin0{static_cast<T>(0.0)};      // fixed gas cost denominated in coin0
    bool use_volume_cap{false};
    T volume_cap_mult{static_cast<T>(1.0)}; // multiplier over base notional cap
};

} // namespace trading
} // namespace arb
