cmake_minimum_required(VERSION 3.14)
project(twocrypto CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.75 REQUIRED COMPONENTS json)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

# Shared library exposing templated math (uint256) via C API for Python benchmark
add_library(stableswap_math_i SHARED
    src/stableswap_math_capi.cpp
)
set_target_properties(stableswap_math_i PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Unified templated harness (select type at runtime via arg: i|d)
add_executable(benchmark_harness
    src/benchmark_harness.cpp
)

target_link_libraries(benchmark_harness
    Boost::json
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness PRIVATE -O3 -march=native)
endif()

# Separate typed harnesses to avoid switching/recompiling between modes
add_executable(benchmark_harness_i
    src/benchmark_harness.cpp
)
target_compile_definitions(benchmark_harness_i PRIVATE HARNESS_MODE_I)
target_link_libraries(benchmark_harness_i Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_i PRIVATE -O3 -march=native)
endif()

add_executable(benchmark_harness_d
    src/benchmark_harness.cpp
)
target_compile_definitions(benchmark_harness_d PRIVATE HARNESS_MODE_D)
target_link_libraries(benchmark_harness_d Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_d PRIVATE -O3 -march=native)
endif()


# Float and long double variants
add_executable(benchmark_harness_f
    src/benchmark_harness.cpp
)
target_compile_definitions(benchmark_harness_f PRIVATE HARNESS_MODE_F)
target_link_libraries(benchmark_harness_f Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_f PRIVATE -O3 -march=native)
endif()

add_executable(benchmark_harness_ld
    src/benchmark_harness.cpp
)
target_compile_definitions(benchmark_harness_ld PRIVATE HARNESS_MODE_LD)
target_link_libraries(benchmark_harness_ld Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_ld PRIVATE -O3 -march=native)
endif()
# Standalone arbitrage harness (multi-pool, threaded)
add_executable(arb_harness
    src/arb_harness.cpp
)
target_link_libraries(arb_harness
    Boost::json
)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness PRIVATE -O3 -march=native)
endif()


# Typed arbitrage harness variants
add_executable(arb_harness_f src/arb_harness.cpp)
target_compile_definitions(arb_harness_f PRIVATE ARB_MODE_F)
target_link_libraries(arb_harness_f Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_f PRIVATE -O3 -march=native)
endif()

add_executable(arb_harness_ld src/arb_harness.cpp)
target_compile_definitions(arb_harness_ld PRIVATE ARB_MODE_LD)
target_link_libraries(arb_harness_ld Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_ld PRIVATE -O3 -march=native)
endif()

# IDR-specific arbitrage harness (supports IDR data format)
add_executable(arb_harness_idr src/arb_harness_idr.cpp)
target_link_libraries(arb_harness_idr Boost::json)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_idr PRIVATE -O3 -march=native)
endif()
