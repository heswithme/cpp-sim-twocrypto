cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

project(twocrypto_runner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED)

add_library(twocrypto_fx INTERFACE)
target_include_directories(twocrypto_fx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/twocrypto-fx/include
)

set(SIM_REAL_TYPES "double" CACHE STRING "Semicolon-separated list of real types to build (choices: int;float;double;long_double)")

function(add_arb_target target_name real_type)
    add_executable(${target_name} src/arb_harness.cpp)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(${target_name} PRIVATE twocrypto_fx)
    if(TARGET Boost::headers)
        target_link_libraries(${target_name} PRIVATE Boost::headers)
    endif()

    string(TOLOWER "${real_type}" real_lc)
    if(real_lc STREQUAL "int")
        target_compile_definitions(${target_name} PRIVATE SIM_T_INT)
    elseif(real_lc STREQUAL "float")
        target_compile_definitions(${target_name} PRIVATE SIM_T_FLOAT)
    elseif(real_lc STREQUAL "double")
        target_compile_definitions(${target_name} PRIVATE SIM_T_DOUBLE)
    elseif(real_lc STREQUAL "long_double")
        target_compile_definitions(${target_name} PRIVATE SIM_T_LONG_DOUBLE)
    else()
        message(FATAL_ERROR "Unsupported real type '${real_type}'. Use one of: int,float,double,long_double")
    endif()
endfunction()

if(NOT SIM_REAL_TYPES)
    message(FATAL_ERROR "SIM_REAL_TYPES must list at least one real type")
endif()

list(LENGTH SIM_REAL_TYPES _real_type_count)
list(GET SIM_REAL_TYPES 0 _primary_real)
add_arb_target(arb_harness "${_primary_real}")

if(_real_type_count GREATER 1)
    math(EXPR _last_index "${_real_type_count} - 1")
    foreach(idx RANGE 1 ${_last_index})
        list(GET SIM_REAL_TYPES ${idx} _extra_real)
        string(REPLACE " " "_" _suffix "${_extra_real}")
        string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _suffix "${_suffix}")
        add_arb_target("arb_harness_${_suffix}" "${_extra_real}")
    endforeach()
endif()
