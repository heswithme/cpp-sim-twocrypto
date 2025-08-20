#ifndef STABLESWAP_MATH_D_HPP
#define STABLESWAP_MATH_D_HPP

#include <array>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace stableswap_d {

constexpr size_t N_COINS = 2;

inline double A_MULTIPLIER_R() { return 10000.0; }

struct MathResultD {
    double value;
    double unused;  // compatibility with uint version signature
};

class StableswapMathD {
public:
    static MathResultD get_y(
        double _amp,
        double _gamma,  // unused, for compatibility
        const std::array<double, N_COINS>& xp,
        double D,
        size_t i
    ) {
        if (i >= N_COINS) throw std::invalid_argument("i above N_COINS");

        double S_ = 0.0;
        double c = D;
        double Ann = _amp * static_cast<double>(N_COINS);

        for (size_t _i = 0; _i < N_COINS; ++_i) {
            if (_i != i) {
                double _x = xp[_i];
                S_ += _x;
                c = c * D / (_x * static_cast<double>(N_COINS));
            }
        }

        c = c * D * A_MULTIPLIER_R() / (Ann * static_cast<double>(N_COINS));
        double b = S_ + D * A_MULTIPLIER_R() / Ann;
        double y = D;

        for (size_t it = 0; it < 128; ++it) {
            double y_prev = y;
            y = (y * y + c) / (2.0 * y + b - D);
            if (std::fabs(y - y_prev) <= 1e-12 * std::max(1.0, y)) {
                return {y, 0.0};
            }
        }
        // Non-convergence is highly unlikely with doubles at these scales
        return {y, 0.0};
    }

    static double newton_D(
        double _amp,
        double _gamma,  // unused, for compatibility
        const std::array<double, N_COINS>& _xp,
        double K0_prev = 0.0  // unused, for compatibility
    ) {
        (void)_gamma; (void)K0_prev;
        double S = 0.0;
        for (auto x : _xp) S += x;
        if (S <= 0.0) return 0.0;

        double D = S;
        double Ann = _amp * static_cast<double>(N_COINS);
        const double Nf = static_cast<double>(N_COINS);
        const double Npow = 4.0; // N_COINS^N_COINS for N=2

        for (size_t it = 0; it < 128; ++it) {
            double D_P = D;
            for (auto x : _xp) D_P = D_P * D / x;
            D_P /= Npow;

            double Dprev = D;
            double numerator = (Ann * S / A_MULTIPLIER_R() + D_P * Nf) * D;
            double denominator = ((Ann - A_MULTIPLIER_R()) * D / A_MULTIPLIER_R() + (Nf + 1.0) * D_P);
            D = numerator / denominator;
            if (std::fabs(D - Dprev) <= 1e-12 * std::max(1.0, D)) break;
        }
        return D;
    }

    static double get_p(
        const std::array<double, N_COINS>& _xp,
        double _D,
        const std::array<double, N_COINS>& _A_gamma
    ) {
        // dx_0 / dx_1 only
        double ANN = _A_gamma[0] * static_cast<double>(N_COINS);
        double Dr = _D / 4.0;
        for (size_t i = 0; i < N_COINS; ++i) Dr = Dr * _D / _xp[i];
        double xp0_A = ANN * _xp[0] / A_MULTIPLIER_R();
        // Return price with 1.0 base (no 1e18 scaling); caller manages units
        return (xp0_A + Dr * _xp[0] / _xp[1]) / (xp0_A + Dr);
    }
};

} // namespace stableswap_d

#endif // STABLESWAP_MATH_D_HPP
