cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_STANDARD_REQUIRED "c++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ParallelNumericMethods)

add_compile_options(-mavx2 -mfma -O3)

set(BUILD_PROJECTS "all" CACHE STRING "Specify which subproject to build (default: all)")

if (MSVC)
    add_compile_definitions(__MSVC)
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(src)
