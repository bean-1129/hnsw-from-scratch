cmake_minimum_required(VERSION 3.14)
project(hnsw-from-scratch
        VERSION 1.0.0
        DESCRIPTION "HNSW (Malkov & Yashunin) implemented from scratch in C++17"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard, build type and options
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# A single-config generator without an explicit build type produces an
# unoptimized binary, which makes every benchmark number meaningless.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

option(HNSW_BUILD_TESTS "Build the unit tests" ON)
option(HNSW_ENABLE_NATIVE_ARCH "Compile for the host CPU (-march=native)" OFF)
option(HNSW_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(hnsw_core
    src/vector_loader.cpp
    src/distance.cpp
    src/brute_force.cpp
    src/hnsw.cpp
    src/graph.cpp
    src/benchmark.cpp
    src/recall.cpp
    src/utils.cpp
)

target_include_directories(hnsw_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(hnsw_core PRIVATE /W4 /permissive-)
    target_compile_definitions(hnsw_core PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(HNSW_WARNINGS_AS_ERRORS)
        target_compile_options(hnsw_core PRIVATE /WX)
    endif()
    target_compile_options(hnsw_core PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(hnsw_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(HNSW_WARNINGS_AS_ERRORS)
        target_compile_options(hnsw_core PRIVATE -Werror)
    endif()
    target_compile_options(hnsw_core PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

# The distance kernels are plain scalar loops; -march=native lets the compiler
# widen them to the host's SIMD width. Off by default because the resulting
# binary is not portable across machines.
if(HNSW_ENABLE_NATIVE_ARCH AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HNSW_HAS_MARCH_NATIVE)
    if(HNSW_HAS_MARCH_NATIVE)
        target_compile_options(hnsw_core PUBLIC -march=native)
    else()
        check_cxx_compiler_flag("-mcpu=native" HNSW_HAS_MCPU_NATIVE)
        if(HNSW_HAS_MCPU_NATIVE)
            target_compile_options(hnsw_core PUBLIC -mcpu=native)
        endif()
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(hnsw_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Command line tool
# ---------------------------------------------------------------------------
add_executable(hnsw src/main.cpp)
target_link_libraries(hnsw PRIVATE hnsw_core)
if(MSVC)
    target_compile_options(hnsw PRIVATE /W4)
else()
    target_compile_options(hnsw PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(HNSW_BUILD_TESTS)
    enable_testing()
    # Each test is a self-contained executable with its own main(); no external
    # test framework is required to build or run the suite.
    set(HNSW_TESTS
        test_distance
        test_loader
        test_bruteforce
        test_hnsw
        test_recall
    )
    foreach(test_name IN LISTS HNSW_TESTS)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE hnsw_core)
        if(MSVC)
            target_compile_options(${test_name} PRIVATE /W4)
        else()
            target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 300)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS hnsw hnsw_core
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

message(STATUS "hnsw-from-scratch ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  tests      : ${HNSW_BUILD_TESTS}")
message(STATUS "  native arch: ${HNSW_ENABLE_NATIVE_ARCH}")
