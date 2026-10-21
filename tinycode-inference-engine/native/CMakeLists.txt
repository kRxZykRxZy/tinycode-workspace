cmake_minimum_required(VERSION 3.16)
project(tinycode_native_src LANGUAGES CXX)

set(TINYCODE_SOURCES
    core/tensor.cpp
    core/memory.cpp
    cpu/operators_scalar.cpp
)

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
set(TINYCODE_ARCH "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(TINYCODE_ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(TINYCODE_ARCH "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|arm)$")
  set(TINYCODE_ARCH "arm32")
endif()

set(TINYCODE_DEFINES "")

if(TINYCODE_ARCH STREQUAL "x86" AND TINYCODE_ENABLE_AVX2)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2 -mfma" TINYCODE_HAS_AVX2)
  if(TINYCODE_HAS_AVX2)
    list(APPEND TINYCODE_SOURCES cpu/operators_avx2.cpp)
    list(APPEND TINYCODE_DEFINES TINYCODE_USE_AVX2=1)
  endif()
endif()

if(TINYCODE_ARCH MATCHES "^arm" AND TINYCODE_ENABLE_NEON)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mfpu=neon" TINYCODE_HAS_NEON)
  if(TINYCODE_HAS_NEON)
    list(APPEND TINYCODE_SOURCES arm_neon/operators_neon.cpp)
    list(APPEND TINYCODE_DEFINES TINYCODE_USE_NEON=1)
  endif()
endif()

add_library(tinycode_native SHARED ${TINYCODE_SOURCES})

target_include_directories(tinycode_native
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(TINYCODE_DEFINES)
  target_compile_definitions(tinycode_native PRIVATE ${TINYCODE_DEFINES})
endif()

if(TINYCODE_ARCH STREQUAL "x86" AND TINYCODE_HAS_AVX2)
  target_compile_options(tinycode_native PRIVATE -mavx2 -mfma)
endif()

if(TINYCODE_ARCH MATCHES "^arm" AND TINYCODE_HAS_NEON)
  target_compile_options(tinycode_native PRIVATE -mfpu=neon)
endif()

set_target_properties(tinycode_native PROPERTIES
    C_VISIBILITY_PRESET default
    OUTPUT_NAME "tinycode_native")

install(TARGETS tinycode_native LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
