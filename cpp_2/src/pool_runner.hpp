#pragma once

#include <array>
#include <vector>

#include "real_type.hpp"
#include "trader.hpp"

namespace sim {

struct RunResult {
    size_t trades{0};
    RealT notional_coin0{0};
    RealT profit_coin0{0};
    RealT final_price{0};
    std::array<RealT, 2> final_balances{RealT(0), RealT(0)};
};

class PoolRunner {
public:
    explicit PoolRunner(twocrypto::TwoCryptoPoolT<RealT> pool)
        : pool_(std::move(pool)) {}

    RunResult run(const std::vector<PriceEvent>& events, Trader& trader) {
        TradeStats stats;
        for (const auto& ev : events) {
            pool_.set_block_timestamp(ev.timestamp);
            trader.on_event(pool_, ev, stats);
        }
        RunResult result;
        result.trades = stats.trades;
        result.notional_coin0 = stats.notional_coin0;
        result.profit_coin0 = stats.profit_coin0;
        result.final_price = pool_.get_p();
        result.final_balances = pool_.balances;
        return result;
    }

    const twocrypto::TwoCryptoPoolT<RealT>& pool() const { return pool_; }

private:
    twocrypto::TwoCryptoPoolT<RealT> pool_;
};

} // namespace sim
