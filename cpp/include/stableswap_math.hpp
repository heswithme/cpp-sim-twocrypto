#ifndef STABLESWAP_MATH_HPP
#define STABLESWAP_MATH_HPP

#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <stdexcept>
#include <algorithm>

namespace stableswap {

using uint256 = boost::multiprecision::uint256_t;
using int256 = boost::multiprecision::int256_t;

constexpr size_t N_COINS = 2;
constexpr uint256 A_MULTIPLIER = 10000;

struct MathResult {
    uint256 value;
    uint256 unused;  // For compatibility with twocrypto
};

class StableswapMath {
public:
    static MathResult get_y(
        const uint256& _amp,
        const uint256& _gamma,  // unused, for compatibility
        const std::array<uint256, N_COINS>& xp,
        const uint256& D,
        size_t i
    );
    
    static uint256 newton_D(
        const uint256& _amp,
        const uint256& _gamma,  // unused, for compatibility
        const std::array<uint256, N_COINS>& _xp,
        const uint256& K0_prev = 0  // unused, for compatibility
    );
    
    static uint256 get_p(
        const std::array<uint256, N_COINS>& _xp,
        const uint256& _D,
        const std::array<uint256, N_COINS>& _A_gamma
    );
    
    static uint256 wad_exp(const int256& x);
};

} // namespace stableswap

#endif // STABLESWAP_MATH_HPP