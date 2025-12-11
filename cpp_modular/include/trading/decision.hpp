// Trading decision result (templated)
#pragma once

namespace arb {
namespace trading {

template <typename T>
struct Decision {
    bool do_trade{false};
    int i{0};
    int j{1};
    T dx{};
    T profit{};
    T fee_tokens{};
    T notional_coin0{};
};

} // namespace trading
} // namespace arb
