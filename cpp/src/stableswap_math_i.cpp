#include "stableswap_math_i.hpp"
#include <cmath>
#include <cstring>

namespace stableswap {

MathResult StableswapMath::get_y(
    const uint256& _amp,
    const uint256& _gamma,
    const std::array<uint256, N_COINS>& xp,
    const uint256& D,
    size_t i
) {
    if (i >= N_COINS) {
        throw std::invalid_argument("i above N_COINS");
    }
    
    uint256 S_ = 0;
    uint256 _x = 0;
    uint256 y_prev = 0;
    uint256 c = D;
    uint256 Ann = _amp * N_COINS;
    
    for (size_t _i = 0; _i < N_COINS; ++_i) {
        if (_i != i) {
            _x = xp[_i];
            S_ += _x;
            c = c * D / (_x * N_COINS);
        }
    }
    
    c = c * D * A_MULTIPLIER / (Ann * N_COINS);
    uint256 b = S_ + D * A_MULTIPLIER / Ann;
    uint256 y = D;
    
    for (size_t _i = 0; _i < 255; ++_i) {
        y_prev = y;
        y = (y * y + c) / (2 * y + b - D);
        
        // Equality with the precision of 1
        if (y > y_prev) {
            if (y - y_prev <= 1) {
                return {y, 0};
            }
        } else {
            if (y_prev - y <= 1) {
                return {y, 0};
            }
        }
    }
    
    throw std::runtime_error("Did not converge");
}

uint256 StableswapMath::newton_D(
    const uint256& _amp,
    const uint256& _gamma,
    const std::array<uint256, N_COINS>& _xp,
    const uint256& K0_prev
) {
    // gamma and K0_prev are ignored
    // _amp is already multiplied by A_MULTIPLIER
    
    uint256 S = 0;
    for (const auto& x : _xp) {
        S += x;
    }
    
    if (S == 0) {
        return 0;
    }
    
    uint256 D = S;
    uint256 Ann = _amp * N_COINS;
    
    for (size_t i = 0; i < 255; ++i) {
        uint256 D_P = D;
        for (const auto& x : _xp) {
            D_P = D_P * D / x;
        }
        // N_COINS^N_COINS = 2^2 = 4 for N_COINS=2
        D_P /= uint256(4);
        
        uint256 Dprev = D;
        
        // Calculate D using integer division
        uint256 numerator = (Ann * S / A_MULTIPLIER + D_P * N_COINS) * D;
        uint256 denominator = ((Ann - A_MULTIPLIER) * D / A_MULTIPLIER + (N_COINS + 1) * D_P);
        D = numerator / denominator;
        
        // Equality with the precision of 1
        if (D > Dprev) {
            if (D - Dprev <= 1) {
                return D;
            }
        } else {
            if (Dprev - D <= 1) {
                return D;
            }
        }
    }
    
    // convergence typically occurs in 4 rounds or less
    throw std::runtime_error("Did not converge");
}

uint256 StableswapMath::get_p(
    const std::array<uint256, N_COINS>& _xp,
    const uint256& _D,
    const std::array<uint256, N_COINS>& _A_gamma
) {
    // dx_0 / dx_1 only
    uint256 ANN = _A_gamma[0] * N_COINS;
    // N_COINS^N_COINS = 4 for N_COINS=2
    uint256 Dr = _D / uint256(4);
    
    for (size_t i = 0; i < N_COINS; ++i) {
        Dr = Dr * _D / _xp[i];
    }
    
    uint256 xp0_A = ANN * _xp[0] / A_MULTIPLIER;
    
    // Return with 10^18 precision
    static const uint256 ten_18 = boost::multiprecision::pow(uint256(10), 18);
    return ten_18 * (xp0_A + Dr * _xp[0] / _xp[1]) / (xp0_A + Dr);
}

uint256 StableswapMath::wad_exp(const int256& x) {
    // Exact implementation of snekmate's _wad_exp function
    // Calculates e^x with 1e18 precision
    
    // If the result is < 1, we return zero. This happens when:
    // x <= (log(1e-18) * 1e18) ~ -4.15e19
    static const int256 MIN_EXP_INPUT("-41446531673892822313");
    if (x <= MIN_EXP_INPUT) {
        return 0;
    }
    
    // When the result is > (2^255 - 1) / 1e18 we cannot represent it
    // This happens when x >= floor(log((2^255 - 1) / 1e18) * 1e18) ~ 135
    static const int256 MAX_EXP_INPUT("135305999368893231589");
    if (x >= MAX_EXP_INPUT) {
        throw std::overflow_error("math: wad_exp overflow");
    }
    
    // x is now in the range (-42, 136) * 1e18. Convert to (-42, 136) * 2^96
    // for higher intermediate precision and a binary base.
    // This base conversion is a multiplication with 1e18 / 2^96 = 5^18 / 2^78
    static const int256 five_pow_18 = boost::multiprecision::pow(int256(5), 18);
    int256 x_scaled = (x << 78) / five_pow_18;
    
    // Reduce the range of x to (-½ ln 2, ½ ln 2) * 2^96 by factoring out powers of two
    // so that exp(x) = exp(x') * 2^k, where k is a signed integer.
    // k = round(x / log(2)) and x' = x - k * log(2)
    // Thus, k is in the range [-61, 195]
    
    // log(2) * 2^96
    static const int256 LOG2_2_96("54916777467707473351141471128");
    
    // Calculate k = round(x_scaled / log(2))
    // Add 2^95 for rounding, then shift right by 96
    int256 k = ((x_scaled << 96) / LOG2_2_96 + (int256(1) << 95)) >> 96;
    
    // Calculate x' = x - k * log(2)
    x_scaled = x_scaled - k * LOG2_2_96;
    
    // Evaluate using a (6, 7)-term rational approximation
    // Since p is monic, we will multiply by a scaling factor later
    
    // First calculate y
    int256 y = (x_scaled + int256("1346386616545796478920950773328")) * x_scaled;
    y = (y >> 96) + int256("57155421227552351082224309758442");
    
    // Calculate p
    int256 p = y + x_scaled - int256("94201549194550492254356042504812");
    p = p * y;
    p = (p >> 96) + int256("28719021644029726153956944680412240");
    p = p * x_scaled;
    p = p + (int256("4385272521454847904659076985693276") << 96);
    
    // Calculate q - we leave p in the 2^192 base
    int256 q = x_scaled - int256("2855989394907223263936484059900");
    q = q * x_scaled;
    q = (q >> 96) + int256("50020603652535783019961831881945");
    
    q = q * x_scaled;
    q = (q >> 96) - int256("533845033583426703283633433725380");
    
    q = q * x_scaled;
    q = (q >> 96) + int256("3604857256930695427073651918091429");
    
    q = q * x_scaled;
    q = (q >> 96) - int256("14423608567350463180887372962807573");
    
    q = q * x_scaled;
    q = (q >> 96) + int256("26449188498355588339934803723976023");
    
    // The polynomial q has no zeros in the range
    // No scaling required as p is already 2^96 too large
    // r is in the range (0.09, 0.25) * 2^96 after division
    int256 r = p / q;
    
    // To finalize, multiply r by:
    // - the scale factor s = ~6.031367120
    // - the factor 2^k from range reduction
    // - the factor 1e18 / 2^96 for base conversion
    // We do this all at once with intermediate result in 2^213 base
    
    // Scale factor * 2^167
    static const uint256 SCALE_FACTOR("3822833074963236453042738258902158003155416615667");
    
    // Convert r to uint256 (handles negative values via two's complement)
    uint256 r_unsigned;
    if (r >= 0) {
        r_unsigned = uint256(r);
    } else {
        // For negative r, we need proper two's complement conversion
        // This shouldn't happen in normal operation but we handle it for completeness
        static const int256 two_pow_256 = boost::multiprecision::pow(int256(2), 256);
        r_unsigned = uint256(two_pow_256 + r);
    }
    
    // Calculate the final result
    // Shift amount: 195 - k
    int shift_amount = 195 - static_cast<int>(k);
    
    uint256 result;
    if (shift_amount > 0) {
        result = (r_unsigned * SCALE_FACTOR) >> shift_amount;
    } else if (shift_amount < 0) {
        // Left shift if shift_amount is negative
        result = (r_unsigned * SCALE_FACTOR) << (-shift_amount);
    } else {
        result = r_unsigned * SCALE_FACTOR;
    }
    
    return result;
}

} // namespace stableswap

// Extern "C" wrapper functions for Python ctypes
#ifdef BUILD_SHARED_LIB
extern "C" {

using namespace stableswap;

char* newton_D(const char* A, const char* gamma, const char* x0, const char* x1) {
    try {
        uint256 amp(A);
        uint256 gam(gamma);
        std::array<uint256, 2> xp = {uint256(x0), uint256(x1)};
        
        uint256 result = StableswapMath::newton_D(amp, gam, xp);
        
        std::string result_str = result.str();
        char* c_str = new char[result_str.length() + 1];
        std::strcpy(c_str, result_str.c_str());
        return c_str;
    } catch (...) {
        return nullptr;
    }
}

char** get_y(const char* A, const char* gamma, const char* x0, const char* x1, const char* D, int i) {
    try {
        uint256 amp(A);
        uint256 gam(gamma);
        std::array<uint256, 2> xp = {uint256(x0), uint256(x1)};
        uint256 d(D);
        
        MathResult result = StableswapMath::get_y(amp, gam, xp, d, i);
        
        char** output = new char*[2];
        
        std::string y_str = result.value.str();
        output[0] = new char[y_str.length() + 1];
        std::strcpy(output[0], y_str.c_str());
        
        std::string k_str = result.unused.str();
        output[1] = new char[k_str.length() + 1];
        std::strcpy(output[1], k_str.c_str());
        
        return output;
    } catch (...) {
        return nullptr;
    }
}

char* get_p(const char* x0, const char* x1, const char* D, const char* A) {
    try {
        std::array<uint256, 2> xp = {uint256(x0), uint256(x1)};
        uint256 d(D);
        std::array<uint256, 2> a_gamma = {uint256(A), uint256("145000000000000")};
        
        uint256 result = StableswapMath::get_p(xp, d, a_gamma);
        
        std::string result_str = result.str();
        char* c_str = new char[result_str.length() + 1];
        std::strcpy(c_str, result_str.c_str());
        return c_str;
    } catch (...) {
        return nullptr;
    }
}

char* wad_exp(const char* x) {
    try {
        int256 x_val(x);
        uint256 result = StableswapMath::wad_exp(x_val);
        
        std::string result_str = result.str();
        char* c_str = new char[result_str.length() + 1];
        std::strcpy(c_str, result_str.c_str());
        return c_str;
    } catch (...) {
        return nullptr;
    }
}

void free_string(char* str) {
    delete[] str;
}

void free_string_array(char** arr, int size) {
    if (arr) {
        for (int i = 0; i < size; i++) {
            if (arr[i]) delete[] arr[i];
        }
        delete[] arr;
    }
}

} // extern "C"
#endif // BUILD_SHARED_LIB
