cmake_minimum_required(VERSION 3.14)
project(stableswap_math CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost with required components
find_package(Boost 1.75 REQUIRED COMPONENTS json)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

# Shared library for Python ctypes (integer/uint256)
add_library(stableswap_math_i SHARED
    src/stableswap_math_i.cpp
)

target_link_libraries(stableswap_math_i
    ${Boost_LIBRARIES}
)

target_compile_definitions(stableswap_math_i PRIVATE BUILD_SHARED_LIB)

set_target_properties(stableswap_math_i PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# TwoCrypto pool implementation (integer/uint256)
add_library(twocrypto_i STATIC
    src/twocrypto_i.cpp
    src/stableswap_math_i.cpp
)

target_link_libraries(twocrypto_i
    ${Boost_LIBRARIES}
)

# Benchmark harness for structured testing (integer/uint256)
add_executable(benchmark_harness_i
    src/benchmark_harness_i.cpp
)

target_link_libraries(benchmark_harness_i
    twocrypto_i
    Boost::json
)

# Double-precision TwoCrypto implementation
add_library(twocrypto_d STATIC
    src/twocrypto_d.cpp
)

target_link_libraries(twocrypto_d
    ${Boost_LIBRARIES}
)

add_executable(benchmark_harness_d
    src/benchmark_harness_d.cpp
)

target_link_libraries(benchmark_harness_d
    twocrypto_d
    Boost::json
)


# Enable optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(stableswap_math_i PRIVATE -O3 -march=native)
    target_compile_options(twocrypto_i PRIVATE -O3 -march=native)
    target_compile_options(benchmark_harness_i PRIVATE -O3 -march=native)
    target_compile_options(twocrypto_d PRIVATE -O3 -march=native)
    target_compile_options(benchmark_harness_d PRIVATE -O3 -march=native)
endif()
