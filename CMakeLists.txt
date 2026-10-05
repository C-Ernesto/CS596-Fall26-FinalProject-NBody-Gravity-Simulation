cmake_minimum_required(VERSION 3.20)

project(nbody_performance_study LANGUAGES CXX)

option(NBODY_ENABLE_CUDA "Build CUDA backends" OFF)
option(NBODY_BUILD_TESTS "Build project tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nbody_core
    src/config.cpp
    src/initialization.cpp
    src/serial.cpp
    src/openmp.cpp
)

target_include_directories(nbody_core PUBLIC include)

if(MSVC)
    target_compile_options(nbody_core PRIVATE /W4)
else()
    target_compile_options(nbody_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody_core PUBLIC NBODY_HAS_OPENMP=1)
endif()

if(NBODY_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(nbody_cuda
        src/cuda/cuda_naive.cu
        src/cuda/cuda_tiled.cu
    )
    target_include_directories(nbody_cuda PUBLIC include)
    set_target_properties(nbody_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

add_executable(nbody src/main.cpp)
target_link_libraries(nbody PRIVATE nbody_core)

if(NBODY_ENABLE_CUDA)
    target_link_libraries(nbody PRIVATE nbody_cuda)
    target_compile_definitions(nbody PRIVATE NBODY_HAS_CUDA=1)
endif()

if(NBODY_BUILD_TESTS)
    include(CTest)
    add_executable(test_initialization tests/test_initialization.cpp)
    target_link_libraries(test_initialization PRIVATE nbody_core)
    add_test(NAME deterministic_initialization COMMAND test_initialization)
endif()

