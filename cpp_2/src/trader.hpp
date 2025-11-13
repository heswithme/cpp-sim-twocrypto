#pragma once

#include <cstdint>

#include "real_type.hpp"
#include "../twocrypto-fx/include/twocrypto.hpp"

namespace sim {

struct PriceEvent {
    uint64_t timestamp{0};
    RealT cex_price{0};
    RealT volume{0};
};

struct Costs {
    RealT arb_fee_bps{static_cast<RealT>(10.0)};
    RealT gas_coin0{static_cast<RealT>(0.0)};
    bool  use_volume_cap{false};
    RealT volume_cap_mult{static_cast<RealT>(1.0)};
};

struct TradeStats {
    size_t trades{0};
    RealT notional_coin0{0};
    RealT profit_coin0{0};
};

class Trader {
public:
    virtual ~Trader() = default;
    virtual void on_event(twocrypto::TwoCryptoPoolT<RealT>& pool,
                          const PriceEvent& event,
                          TradeStats& stats) = 0;
};

} // namespace sim
