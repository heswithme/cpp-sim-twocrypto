// Action recording for --save-actions mode
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <boost/json.hpp>

namespace json = boost::json;

namespace arb {
namespace harness {

// Forward: to_str_1e18 for wei conversion
template <typename T>
std::string to_str_1e18(T val);

// Donation action (matches old harness schema exactly)
template <typename T>
struct DonationAction {
    uint64_t ts{0};
    uint64_t ts_due{0};
    std::array<T, 2> amounts{T(0), T(0)};
    T price_scale{0};
    T donation_ratio1{0};
    T apy_per_year{0};
    uint64_t freq_s{0};
    
    json::object to_json() const {
        json::object o;
        o["type"] = "donation";
        o["ts"] = ts;
        o["ts_due"] = ts_due;
        o["amounts"] = json::array{static_cast<double>(amounts[0]), static_cast<double>(amounts[1])};
        o["amounts_wei"] = json::array{to_str_1e18(amounts[0]), to_str_1e18(amounts[1])};
        o["price_scale"] = static_cast<double>(price_scale);
        o["donation_ratio1"] = static_cast<double>(donation_ratio1);
        o["apy_per_year"] = static_cast<double>(apy_per_year);
        o["freq_s"] = freq_s;
        return o;
    }
};

// Tick action (matches old harness schema - note "psafter" not "ps_after")
template <typename T>
struct TickAction {
    uint64_t ts{0};
    T p_cex{0};
    T ps_before{0};
    T ps_after{0};  // Note: output key is "psafter" (old harness typo preserved)
    T oracle_before{0};
    T oracle_after{0};
    T xcp_profit_before{0};
    T xcp_profit_after{0};
    T vp_before{0};
    T vp_after{0};
    
    json::object to_json() const {
        json::object o;
        o["type"] = "tick";
        o["ts"] = ts;
        o["p_cex"] = static_cast<double>(p_cex);
        o["ps_before"] = static_cast<double>(ps_before);
        o["psafter"] = static_cast<double>(ps_after);  // Note: typo preserved for parity
        o["oracle_before"] = static_cast<double>(oracle_before);
        o["oracle_after"] = static_cast<double>(oracle_after);
        o["xcp_profit_before"] = static_cast<double>(xcp_profit_before);
        o["xcp_profit_after"] = static_cast<double>(xcp_profit_after);
        o["vp_before"] = static_cast<double>(vp_before);
        o["vp_after"] = static_cast<double>(vp_after);
        return o;
    }
};

// Exchange action (matches old harness schema exactly)
template <typename T>
struct ExchangeAction {
    uint64_t ts{0};
    int i{0};
    int j{0};
    T dx{0};
    T dy_after_fee{0};
    T fee_tokens{0};
    T profit_coin0{0};
    T p_cex{0};
    T p_pool_before{0};
    T p_pool_after{0};
    T oracle_before{0};
    T oracle_after{0};
    T ps_before{0};
    T ps_after{0};
    uint64_t last_ts_before{0};
    uint64_t last_ts_after{0};
    T lp_before{0};
    T lp_after{0};
    T xcp_profit_before{0};
    T xcp_profit_after{0};
    T vp_before{0};
    T vp_after{0};
    T slippage{0};
    T liq_density{0};
    T balance_indicator{0};
    
    json::object to_json() const {
        json::object o;
        o["type"] = "exchange";
        o["ts"] = ts;
        o["i"] = i;
        o["j"] = j;
        o["dx"] = static_cast<double>(dx);
        o["dx_wei"] = to_str_1e18(dx);
        o["dy_after_fee"] = static_cast<double>(dy_after_fee);
        o["fee_tokens"] = static_cast<double>(fee_tokens);
        o["profit_coin0"] = static_cast<double>(profit_coin0);
        o["p_cex"] = static_cast<double>(p_cex);
        o["p_pool_before"] = static_cast<double>(p_pool_before);
        o["p_pool_after"] = static_cast<double>(p_pool_after);
        o["oracle_before"] = static_cast<double>(oracle_before);
        o["oracle_after"] = static_cast<double>(oracle_after);
        o["ps_before"] = static_cast<double>(ps_before);
        o["ps_after"] = static_cast<double>(ps_after);
        o["last_ts_before"] = last_ts_before;
        o["last_ts_after"] = last_ts_after;
        o["lp_before"] = static_cast<double>(lp_before);
        o["lp_after"] = static_cast<double>(lp_after);
        o["xcp_profit_before"] = static_cast<double>(xcp_profit_before);
        o["xcp_profit_after"] = static_cast<double>(xcp_profit_after);
        o["vp_before"] = static_cast<double>(vp_before);
        o["vp_after"] = static_cast<double>(vp_after);
        o["slippage"] = static_cast<double>(slippage);
        o["liq_density"] = static_cast<double>(liq_density);
        o["balance_indicator"] = static_cast<double>(balance_indicator);
        return o;
    }
};

// Variant for all action types
template <typename T>
using Action = std::variant<DonationAction<T>, TickAction<T>, ExchangeAction<T>>;

// Convert action variant to JSON
template <typename T>
json::object action_to_json(const Action<T>& action) {
    return std::visit([](const auto& a) { return a.to_json(); }, action);
}

// Convert actions vector to JSON array
template <typename T>
json::array actions_to_json(const std::vector<Action<T>>& actions) {
    json::array arr;
    arr.reserve(actions.size());
    for (const auto& a : actions) {
        arr.push_back(action_to_json(a));
    }
    return arr;
}

} // namespace harness
} // namespace arb
