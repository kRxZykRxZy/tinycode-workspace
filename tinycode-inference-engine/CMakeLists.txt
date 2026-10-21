cmake_minimum_required(VERSION 3.16)
project(tinycode_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TINYCODE_ENABLE_AVX2 "Enable the AVX2 kernels" ON)
option(TINYCODE_ENABLE_NEON "Enable the ARM NEON kernels" ON)
option(TINYCODE_BUILD_TESTS "Build the native tests" OFF)

add_subdirectory(src)

if(TINYCODE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
