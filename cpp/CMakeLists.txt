cmake_minimum_required(VERSION 3.14)
project(stableswap_math CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost with required components
find_package(Boost 1.75 REQUIRED COMPONENTS json)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

# Shared library for Python ctypes
add_library(stableswap_math SHARED
    src/stableswap_math.cpp
)

target_link_libraries(stableswap_math
    ${Boost_LIBRARIES}
)

target_compile_definitions(stableswap_math PRIVATE BUILD_SHARED_LIB)

set_target_properties(stableswap_math PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# TwoCrypto pool implementation
add_library(twocrypto STATIC
    src/twocrypto.cpp
    src/stableswap_math.cpp
)

target_link_libraries(twocrypto
    ${Boost_LIBRARIES}
)

# Benchmark harness for structured testing
add_executable(benchmark_harness
    src/benchmark_harness.cpp
)

target_link_libraries(benchmark_harness
    twocrypto
    Boost::json
)


# Enable optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(stableswap_math PRIVATE -O3 -march=native)
    target_compile_options(twocrypto PRIVATE -O3 -march=native)
    target_compile_options(benchmark_harness PRIVATE -O3 -march=native)
endif()