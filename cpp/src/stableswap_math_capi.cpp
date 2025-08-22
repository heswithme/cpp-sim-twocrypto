// C API wrapper exposing stableswap unified math (uint256) for Python benchmark
#include "stableswap_math.hpp"
#include <string>
#include <array>
#include <cstdlib>
#include <cstring>

using stableswap::uint256;

extern "C" {

static char* alloc_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static uint256 to_u256(const char* s) {
    try {
        return uint256(std::string(s ? s : "0"));
    } catch (...) {
        return uint256(0);
    }
}

// newton_D(A, gamma, x0, x1) -> char*
const char* newton_D(const char* A, const char* gamma, const char* x0, const char* x1) {
    uint256 a = to_u256(A);
    uint256 g = to_u256(gamma);
    std::array<uint256, 2> xp{ to_u256(x0), to_u256(x1) };
    uint256 D = stableswap::MathOps<uint256>::newton_D(a, g, xp, uint256(0));
    return alloc_cstr(D.str());
}

// get_y(A, gamma, x0, x1, D, i) -> char** (length 2)
char** get_y(const char* A, const char* gamma, const char* x0, const char* x1, const char* D, int i) {
    uint256 a = to_u256(A);
    uint256 g = to_u256(gamma);
    std::array<uint256, 2> xp{ to_u256(x0), to_u256(x1) };
    uint256 d = to_u256(D);
    size_t idx = static_cast<size_t>(i);
    auto res = stableswap::MathOps<uint256>::get_y(a, g, xp, d, idx);
    char** arr = static_cast<char**>(std::malloc(sizeof(char*) * 2));
    if (!arr) return nullptr;
    arr[0] = alloc_cstr(res.value.str());
    arr[1] = alloc_cstr(std::string("0"));
    return arr;
}

// get_p(x0, x1, D, A) -> char*
const char* get_p(const char* x0, const char* x1, const char* D, const char* A) {
    std::array<uint256, 2> xp{ to_u256(x0), to_u256(x1) };
    uint256 d = to_u256(D);
    std::array<uint256, 2> A_gamma{ to_u256(A), uint256(0) };
    uint256 p = stableswap::MathOps<uint256>::get_p(xp, d, A_gamma);
    return alloc_cstr(p.str());
}

void free_string(void* p) {
    if (p) std::free(p);
}

void free_string_array(char** arr, int n) {
    if (!arr) return;
    for (int i = 0; i < n; ++i) {
        if (arr[i]) std::free(arr[i]);
    }
    std::free(arr);
}

} // extern "C"

