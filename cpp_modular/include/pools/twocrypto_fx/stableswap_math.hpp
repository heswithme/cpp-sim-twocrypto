// Stableswap math (templated, parity with Vyper StableswapMath)
// Duplicated from cpp/include/stableswap_math.hpp with namespace arb::pools::twocrypto_fx
#pragma once

#include <array>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>

namespace arb {
namespace pools {
namespace twocrypto_fx {

using uint256 = boost::multiprecision::uint256_t;
using int256 = boost::multiprecision::int256_t;

template <typename T>
struct MathResultT {
    T value;
    T unused;
};

template <typename T>
struct MathTraits;

template <>
struct MathTraits<uint256> {
    static constexpr size_t N = 2;

    static uint256 A_MULTIPLIER() {
        return uint256(10000);
    }

    static uint256 PRECISION() {
        static uint256 v("1000000000000000000"); // 1e18
        return v;
    }
};

template <>
struct MathTraits<double> {
    static constexpr size_t N = 2;

    static double A_MULTIPLIER() {
        return 10000.0;
    }

    static double PRECISION() {
        return 1.0;
    }
};

template <>
struct MathTraits<float> {
    static constexpr size_t N = 2;
    static float A_MULTIPLIER() { return 10000.0f; }
    static float PRECISION() { return 1.0f; }
};

template <>
struct MathTraits<long double> {
    static constexpr size_t N = 2;
    static long double A_MULTIPLIER() { return 10000.0L; }
    static long double PRECISION() { return 1.0L; }
};

// Convergence and iteration traits per numeric type
template <typename T>
struct Convergence;

template <>
struct Convergence<uint256> {
    static bool close(const uint256& a, const uint256& b) {
        return (a > b ? a - b : b - a) <= uint256(1);
    }

    static constexpr size_t MAX_IT = 255;
};

template <>
struct Convergence<double> {
    static bool close(double a, double b) {
        return std::fabs(a - b) <= 1e-12 * std::max(1.0, a);
    }

    static constexpr size_t MAX_IT = 256;
};

template <>
struct Convergence<float> {
    static bool close(float a, float b) {
        return std::fabs(a - b) <= 1e-6f * std::max(1.0f, a);
    }
    static constexpr size_t MAX_IT = 256;
};

template <>
struct Convergence<long double> {
    static bool close(long double a, long double b) {
        return fabsl(a - b) <= 1e-12L * std::max<long double>(1.0L, a);
    }
    static constexpr size_t MAX_IT = 256;
};

// Common templated implementation
template <typename T>
struct MathOpsCommon {
    using Traits = MathTraits<T>;

    static MathResultT<T> get_y(
        const T& _amp,
        const T& _gamma,
        const std::array<T, Traits::N>& xp,
        const T& D,
        size_t i
    ) {
        (void)_gamma;

        if (i >= Traits::N) {
            throw std::invalid_argument("i above N");
        }

        T S_  = T(0);
        T c   = D;
        T Ann = _amp * Traits::N;

        for (size_t idx = 0; idx < Traits::N; ++idx) {
            if (idx == i) continue;
            T _x = xp[idx];
            S_ += _x;
            c   = c * D / (_x * Traits::N);
        }

        c = c * D * Traits::A_MULTIPLIER() / (Ann * Traits::N);

        T b = S_ + D * Traits::A_MULTIPLIER() / Ann;
        T y = D;

        for (size_t it = 0; it < Convergence<T>::MAX_IT; ++it) {
            T y_prev = y;
            y = (y * y + c) / (T(2) * y + b - D);
            if (Convergence<T>::close(y, y_prev)) {
                return { y, T(0) };
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            return { y, T(0) };
        }
        throw std::runtime_error("Did not converge");
    }

    static T newton_D(
        const T& _amp,
        const T& _gamma,
        const std::array<T, Traits::N>& _xp,
        const T& K0_prev
    ) {
        (void)_gamma; (void)K0_prev;

        T S = T(0);
        for (const auto& x : _xp) {
            S += x;
        }
        if (S == T(0)) {
            return T(0);
        }

        T D   = S;
        T Ann = _amp * Traits::N;

        for (size_t it = 0; it < Convergence<T>::MAX_IT; ++it) {
            T D_P = D;
            for (const auto& x : _xp) {
                D_P = D_P * D / x;
            }
            D_P /= (T(Traits::N) * T(Traits::N));

            T Dprev = D;

            T num = (Ann * S / Traits::A_MULTIPLIER() + D_P * Traits::N) * D;
            T den = (
                (Ann - Traits::A_MULTIPLIER()) * D / Traits::A_MULTIPLIER() +
                (T(Traits::N) + T(1)) * D_P
            );
            D = num / den;

            if (Convergence<T>::close(D, Dprev)) {
                break;
            }
        }

        return D;
    }

    static T get_p(
        const std::array<T, Traits::N>& _xp,
        const T& _D,
        const std::array<T, Traits::N>& _A_gamma
    ) {
        T ANN = _A_gamma[0] * Traits::N;
        T Dr  = _D / T(Traits::N * Traits::N);

        for (size_t idx = 0; idx < Traits::N; ++idx) {
            Dr = Dr * _D / _xp[idx];
        }

        T xp0_A = ANN * _xp[0] / Traits::A_MULTIPLIER();

        return (
            Traits::PRECISION() * (xp0_A + Dr * _xp[0] / _xp[1])
        ) / (xp0_A + Dr);
    }
};

// Primary MathOps template forwards to common implementation
template <typename T>
struct MathOps : MathOpsCommon<T> {};

// uint256 specialization adds wad_exp used by EMA in pool logic
template <>
struct MathOps<uint256> : MathOpsCommon<uint256> {
    using T = uint256;

    static T wad_exp(const int256& x) {
        static const int256 MIN_EXP_INPUT("-41446531673892822313");
        if (x <= MIN_EXP_INPUT) return 0;

        static const int256 MAX_EXP_INPUT("135305999368893231589");
        if (x >= MAX_EXP_INPUT) throw std::overflow_error("math: wad_exp overflow");

        static const int256 five_pow_18 = boost::multiprecision::pow(int256(5), 18);
        int256 x_scaled = (x << 78) / five_pow_18;

        static const int256 LOG2_2_96("54916777467707473351141471128");
        int256 k = ((x_scaled << 96) / LOG2_2_96 + (int256(1) << 95)) >> 96;
        x_scaled = x_scaled - k * LOG2_2_96;

        int256 y = (x_scaled + int256("1346386616545796478920950773328")) * x_scaled;
        y = (y >> 96) + int256("57155421227552351082224309758442");

        int256 p = y + x_scaled - int256("94201549194550492254356042504812");
        p = p * y;
        p = (p >> 96) + int256("28719021644029726153956944680412240");
        p = p * x_scaled;
        p = p + (int256("4385272521454847904659076985693276") << 96);

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

        int256 r = p / q;

        static const uint256 SCALE_FACTOR("3822833074963236453042738258902158003155416615667");
        uint256 r_unsigned = (r >= 0) ? uint256(r) : uint256(boost::multiprecision::pow(int256(2), 256) + r);

        int shift_amount = 195 - static_cast<int>(k);
        uint256 result;
        if (shift_amount > 0) {
            result = (r_unsigned * SCALE_FACTOR) >> shift_amount;
        } else if (shift_amount < 0) {
            result = (r_unsigned * SCALE_FACTOR) << (-shift_amount);
        } else {
            result = r_unsigned * SCALE_FACTOR;
        }
        return result;
    }
};

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
