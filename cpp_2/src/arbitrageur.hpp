#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "real_type.hpp"
#include "trader.hpp"
#include "pool_runner.hpp"

namespace sim {

class SimpleArbitrageur final : public Trader {
public:
    struct Config {
        RealT threshold{static_cast<RealT>(0.001)};
        RealT trade_fraction{static_cast<RealT>(0.01)};
        RealT min_trade_frac{static_cast<RealT>(1e-6)};
        RealT max_trade_frac{static_cast<RealT>(0.1)};
    };

    SimpleArbitrageur(Config cfg, Costs costs)
        : cfg_(cfg), costs_(costs) {}

    void on_event(twocrypto::TwoCryptoPoolT<RealT>& pool,
                  const PriceEvent& event,
                  TradeStats& stats) override {
        if (!(event.cex_price > RealT(0))) {
            return;
        }
        const RealT pool_price = pool.get_p();
        if (!(pool_price > RealT(0))) {
            return;
        }
        const RealT rel_diff = (event.cex_price - pool_price) / pool_price;
        if (rel_diff > cfg_.threshold) {
            trade(pool, event, 0, 1, stats);
        } else if (rel_diff < -cfg_.threshold) {
            trade(pool, event, 1, 0, stats);
        }
    }

private:
    void trade(twocrypto::TwoCryptoPoolT<RealT>& pool,
               const PriceEvent& event,
               size_t i,
               size_t j,
               TradeStats& stats) {
        const RealT balance_from = pool.balances[i];
        if (!(balance_from > RealT(0))) {
            return;
        }
        const RealT min_dx = std::max(cfg_.min_trade_frac * balance_from, static_cast<RealT>(1e-18));
        RealT max_dx = cfg_.max_trade_frac * balance_from;
        if (!(max_dx > min_dx)) {
            return;
        }
        RealT dx = balance_from * cfg_.trade_fraction;
        if (dx < min_dx) dx = min_dx;
        if (dx > max_dx) dx = max_dx;
        if (costs_.use_volume_cap && event.volume > RealT(0)) {
            RealT cap = event.volume * costs_.volume_cap_mult;
            if (cap > RealT(0)) {
                if (i == 0) {
                    if (dx > cap) dx = cap;
                } else {
                    const RealT denom = std::max(event.cex_price, static_cast<RealT>(1e-12));
                    const RealT max_dx = cap / denom;
                    if (dx > max_dx) dx = max_dx;
                }
            }
        }
        if (!(dx > RealT(0))) {
            return;
        }
        try {
            auto res = pool.exchange(static_cast<RealT>(i), static_cast<RealT>(j), dx, RealT(0));
            const RealT dy_after_fee = res[0];
            const RealT fee_tokens   = res[1];
            const RealT fee_cex = costs_.arb_fee_bps / static_cast<RealT>(1e4);
            const RealT sell_factor = (RealT(1) - fee_cex);
            const RealT buy_factor  = (RealT(1) + fee_cex);
            const RealT price = std::max(event.cex_price, static_cast<RealT>(1e-12));
            RealT profit_coin0 = RealT(0);
            RealT notional_coin0 = (i == 0) ? dx : dx * price;
            if (i == 0) {
                profit_coin0 = dy_after_fee * price * sell_factor - dx - costs_.gas_coin0;
            } else {
                profit_coin0 = dy_after_fee - dx * price * buy_factor - costs_.gas_coin0;
            }
            stats.trades += 1;
            stats.notional_coin0 += notional_coin0;
            stats.profit_coin0 += profit_coin0;
        } catch (...) {
            // ignore failed trade
        }
    }

    Config cfg_{};
    Costs costs_{};
};

namespace detail {

inline std::array<RealT, 2> pool_xp_from(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                                         const std::array<RealT, 2>& balances,
                                         RealT price_scale) {
    return {
        balances[0] * pool.precisions[0],
        balances[1] * pool.precisions[1] * price_scale
    };
}

inline RealT xp_to_tokens_j(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                            size_t j,
                            RealT amount_xp,
                            RealT price_scale) {
    RealT v = amount_xp;
    if (j == 1) v /= price_scale;
    return v / pool.precisions[j];
}

inline RealT dyn_fee(const std::array<RealT, 2>& xp,
                     RealT mid_fee,
                     RealT out_fee,
                     RealT fee_gamma) {
    const RealT sum = xp[0] + xp[1];
    if (!(sum > RealT(0))) return mid_fee;
    RealT B = RealT(4) * (xp[0] / sum) * (xp[1] / sum);
    const RealT denom = fee_gamma * B + RealT(1) - B;
    if (denom > RealT(0)) {
        B = fee_gamma * B / denom;
    }
    return mid_fee * B + out_fee * (RealT(1) - B);
}

inline std::pair<RealT, RealT> simulate_exchange_once(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                                                      size_t i,
                                                      size_t j,
                                                      RealT dx) {
    using Ops = stableswap::MathOps<RealT>;
    const RealT ps = pool.cached_price_scale;

    auto balances = pool.balances;
    balances[i] += dx;
    auto xp = pool_xp_from(pool, balances, ps);
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, j);
    RealT dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    RealT dy_tokens = xp_to_tokens_j(pool, j, dy_xp, ps);
    RealT fee_pool = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    RealT fee_tokens = fee_pool * dy_tokens;
    return {dy_tokens - fee_tokens, fee_tokens};
}

} // namespace detail

class OptimalArbitrageur final : public Trader {
public:
    struct Config {
        RealT min_trade_frac{static_cast<RealT>(1e-6)};
        RealT max_trade_frac{static_cast<RealT>(0.1)};
    };

    OptimalArbitrageur(Config cfg, Costs costs)
        : cfg_(cfg), costs_(costs) {}

    void on_event(twocrypto::TwoCryptoPoolT<RealT>& pool,
                  const PriceEvent& event,
                  TradeStats& stats) override {
        if (!(event.cex_price > RealT(0))) return;
        const bool has_cap = costs_.use_volume_cap && (event.volume > RealT(0));
        const RealT cap_value = has_cap ? event.volume * costs_.volume_cap_mult : RealT(0);
        Decision d = decide_trade(pool, event.cex_price, has_cap, cap_value);
        if (!d.do_trade) return;
        try {
            auto res = pool.exchange(static_cast<RealT>(d.i), static_cast<RealT>(d.j), d.dx, RealT(0));
            const RealT dy_after_fee = res[0];
            const RealT fee_cex = costs_.arb_fee_bps / static_cast<RealT>(1e4);
            const RealT price = std::max(event.cex_price, static_cast<RealT>(1e-12));
            RealT profit_coin0 = RealT(0);
            if (d.i == 0) {
                profit_coin0 = dy_after_fee * price * (RealT(1) - fee_cex) - d.dx - costs_.gas_coin0;
            } else {
                profit_coin0 = dy_after_fee - d.dx * price * (RealT(1) + fee_cex) - costs_.gas_coin0;
            }
            if (!(profit_coin0 > RealT(0))) return;
            stats.trades += 1;
            stats.notional_coin0 += (d.i == 0) ? d.dx : d.dx * price;
            stats.profit_coin0 += profit_coin0;
        } catch (...) {
            // ignore failure
        }
    }

private:
    struct Decision {
        bool do_trade{false};
        int i{0};
        int j{1};
        RealT dx{0};
        RealT profit{0};
    };

    Decision decide_trade(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                          RealT cex_price,
                          bool has_cap,
                          RealT notional_cap_coin0) const {
        Decision d01 = maximize_direction(pool, cex_price, has_cap, notional_cap_coin0, 0, 1);
        Decision d10 = maximize_direction(pool, cex_price, has_cap, notional_cap_coin0, 1, 0);
        if (d01.do_trade && (!d10.do_trade || d01.profit >= d10.profit)) return d01;
        if (d10.do_trade) return d10;
        return {};
    }

    Decision maximize_direction(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                                RealT cex_price,
                                bool has_cap,
                                RealT notional_cap_coin0,
                                int i,
                                int j) const {
        Decision d{};
        const RealT avail = pool.balances[static_cast<size_t>(i)];
        if (!(avail > RealT(0))) return d;

        RealT min_frac = std::max(cfg_.min_trade_frac, static_cast<RealT>(1e-9));
        RealT max_frac = std::max(cfg_.max_trade_frac, min_frac * RealT(2));
        if (max_frac > RealT(1)) max_frac = RealT(1);

        RealT dx_lo = std::max(static_cast<RealT>(1e-18), avail * std::max(min_frac, static_cast<RealT>(1e-12)));
        RealT dx_hi = avail * max_frac;

        if (has_cap && (notional_cap_coin0 > RealT(0))) {
            if (i == 0) {
                dx_hi = std::min(dx_hi, notional_cap_coin0);
            } else {
                const RealT price = std::max(cex_price, static_cast<RealT>(1e-12));
                dx_hi = std::min(dx_hi, notional_cap_coin0 / price);
            }
        }
        if (!(dx_hi > dx_lo)) return d;

        const int samples = 12;
        const RealT log_lo = std::log(dx_lo);
        const RealT log_hi = std::log(dx_hi);
        RealT best_x = dx_lo;
        RealT best_profit = RealT(0);
        bool seeded = false;

        for (int k = 0; k < samples; ++k) {
            const RealT t = (samples == 1) ? RealT(0) : static_cast<RealT>(k) / static_cast<RealT>(samples - 1);
            const RealT x = std::exp(log_lo + t * (log_hi - log_lo));
            auto [p, _] = profit_for_dx(pool, cex_price, i, j, x);
            if (!seeded || p > best_profit) {
                seeded = true;
                best_profit = p;
                best_x = x;
            }
        }

        if (!seeded || !(best_profit > RealT(0))) return d;

        auto scalar_profit = [&](RealT x) {
            return profit_for_dx(pool, cex_price, i, j, x).first;
        };
        RealT refined_x = golden_section_max(dx_lo, dx_hi, scalar_profit);
        auto [refined_profit, _fee] = profit_for_dx(pool, cex_price, i, j, refined_x);
        if (!(refined_profit > RealT(0))) {
            refined_x = best_x;
            refined_profit = best_profit;
        }
        if (!(refined_profit > RealT(0))) return d;

        d.do_trade = true;
        d.i = i;
        d.j = j;
        d.dx = refined_x;
        d.profit = refined_profit;
        return d;
    }

    std::pair<RealT, RealT> profit_for_dx(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                                          RealT cex_price,
                                          int i,
                                          int j,
                                          RealT dx) const {
        if (!(dx > RealT(0))) return {RealT(0), RealT(0)};
        auto [dy_after_fee, fee_tokens] = detail::simulate_exchange_once(pool,
                                                                         static_cast<size_t>(i),
                                                                         static_cast<size_t>(j),
                                                                         dx);
        const RealT fee_cex = costs_.arb_fee_bps / static_cast<RealT>(1e4);
        const RealT price = std::max(cex_price, static_cast<RealT>(1e-12));
        RealT profit = RealT(0);
        if (i == 0) {
            profit = dy_after_fee * price * (RealT(1) - fee_cex) - dx - costs_.gas_coin0;
        } else {
            profit = dy_after_fee - dx * price * (RealT(1) + fee_cex) - costs_.gas_coin0;
        }
        return {profit, fee_tokens};
    }

    template <typename Fn>
    RealT golden_section_max(RealT a,
                             RealT b,
                             Fn&& f) const {
        if (!(b > a)) return a;
        const RealT invphi = static_cast<RealT>(0.6180339887498948482L);
        const RealT invphi2 = RealT(1) - invphi;
        RealT x1 = b - invphi * (b - a);
        RealT x2 = a + invphi * (b - a);
        RealT f1 = f(x1);
        RealT f2 = f(x2);
        for (int iter = 0; iter < 48; ++iter) {
            const RealT width = b - a;
            const RealT scale = std::max(std::abs(b), RealT(1));
            if (std::abs(width) <= scale * static_cast<RealT>(1e-9)) break;
            if (f1 < f2) {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + invphi * (b - a);
                f2 = f(x2);
            } else {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - invphi * (b - a);
                f1 = f(x1);
            }
        }
        return (f1 > f2) ? x1 : x2;
    }

    Config cfg_{};
    Costs costs_{};
};

} // namespace sim


