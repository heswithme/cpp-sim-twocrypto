cmake_minimum_required(VERSION 3.14)
project(arb_harness_modular CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.75 REQUIRED COMPONENTS json)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)
# Also include the original cpp/include for stableswap_math.hpp and twocrypto.hpp
include_directories(${CMAKE_SOURCE_DIR}/../cpp/include)

# Common sources (will grow as we add modules)
set(COMMON_SOURCES
    src/main.cpp
    src/events.cpp
)

# Compiler flags
set(COMMON_FLAGS -Wall -Wextra)
set(RELEASE_FLAGS -O3 -march=native)

# --- arb_harness (default: double) ---
add_executable(arb_harness ${COMMON_SOURCES})
target_link_libraries(arb_harness Boost::json)
target_compile_options(arb_harness PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness PRIVATE ${RELEASE_FLAGS})
endif()

# --- arb_harness_f (float) ---
add_executable(arb_harness_f ${COMMON_SOURCES})
target_compile_definitions(arb_harness_f PRIVATE ARB_MODE_F)
target_link_libraries(arb_harness_f Boost::json)
target_compile_options(arb_harness_f PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_f PRIVATE ${RELEASE_FLAGS})
endif()

# --- arb_harness_ld (long double) ---
add_executable(arb_harness_ld ${COMMON_SOURCES})
target_compile_definitions(arb_harness_ld PRIVATE ARB_MODE_LD)
target_link_libraries(arb_harness_ld Boost::json)
target_compile_options(arb_harness_ld PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_ld PRIVATE ${RELEASE_FLAGS})
endif()

