cmake_minimum_required(VERSION 3.14)
project(stableswap_math CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost
find_package(Boost 1.70 REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

# Shared library for Python ctypes only
add_library(stableswap_math SHARED
    src/stableswap_math.cpp
)

target_link_libraries(stableswap_math
    ${Boost_LIBRARIES}
)

# Add extern "C" wrapper for Python
target_compile_definitions(stableswap_math PRIVATE BUILD_SHARED_LIB)

set_target_properties(stableswap_math PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Enable optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(stableswap_math PRIVATE -O3 -march=native)
endif()

# Installation (optional)
install(TARGETS stableswap_math
    LIBRARY DESTINATION lib
)

install(FILES include/stableswap_math.hpp
    DESTINATION include
)