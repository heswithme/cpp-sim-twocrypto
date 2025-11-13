#pragma once

#include <boost/multiprecision/cpp_int.hpp>

#if defined(SIM_T_INT)
#  define SIM_T_TYPE boost::multiprecision::uint256_t
#elif defined(SIM_T_FLOAT)
#  define SIM_T_TYPE float
#elif defined(SIM_T_LONG_DOUBLE)
#  define SIM_T_TYPE long double
#elif defined(SIM_T_DOUBLE)
#  define SIM_T_TYPE double
#else
#  define SIM_T_TYPE double
#endif

namespace sim {
using RealT = SIM_T_TYPE;
}
