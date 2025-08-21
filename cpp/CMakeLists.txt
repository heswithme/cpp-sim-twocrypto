cmake_minimum_required(VERSION 3.14)
project(twocrypto_unified CXX)

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
