cmake_minimum_required(VERSION 3.16)
project(Adamant VERSION 0.1.0 LANGUAGES C CXX
        DESCRIPTION "Adamant SIMD-VIC by Euclase: SIMD-vectorized inference core for quantized LLMs")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
  message(FATAL_ERROR "Adamant supports Linux and Windows")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "Adamant needs GCC or Clang (Corundum uses the __atomic builtins; on Windows: MinGW-w64 or clang)")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ADAMANT_BUILD_TESTS "Build tests" ON)
option(ADAMANT_BUILD_TOOLS "Build adamant and adamant-view" ON)
option(ADAMANT_AVX2 "Build the AVX2 kernels (selected at run time if the CPU has AVX2+FMA+F16C)" ON)
option(ADAMANT_NATIVE "Compile everything with -march=native" OFF)
option(ADAMANT_WERROR "Treat warnings as errors" OFF)
set(ADAMANT_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. address,undefined")

set(ADAMANT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wno-sign-conversion)
if(ADAMANT_WERROR)
  list(APPEND ADAMANT_WARNINGS -Werror)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND ADAMANT_AVX2)
  set(ADAMANT_HAVE_AVX2 ON)
else()
  set(ADAMANT_HAVE_AVX2 OFF)
endif()

if(ADAMANT_SANITIZE)
  add_compile_options(-fsanitize=${ADAMANT_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${ADAMANT_SANITIZE})
endif()
if(ADAMANT_NATIVE)
  add_compile_options(-march=native)
endif()

# --- Corundum HFT (vendored, C) --------------------------------------------
add_library(corundum STATIC third_party/corundum/corundum.c)
target_include_directories(corundum PUBLIC third_party/corundum)
set_target_properties(corundum PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(CheckLibraryExists)
  check_library_exists(rt shm_open "" ADAMANT_HAVE_LIBRT)
  if(ADAMANT_HAVE_LIBRT)
    target_link_libraries(corundum PUBLIC rt)
  endif()
endif()

# --- the engine ----------------------------------------------------------------
find_package(Threads REQUIRED)
add_library(adamant STATIC
  src/common.cpp
  src/mmap_file.cpp
  src/gguf.cpp
  src/gguf_writer.cpp
  src/quant_scalar.cpp
  src/thread_pool.cpp
  src/tokenizer.cpp
  src/model.cpp
  src/sampler.cpp
  src/stream.cpp)
add_library(Adamant::adamant ALIAS adamant)
target_include_directories(adamant PUBLIC include third_party/euclase)
target_link_libraries(adamant PUBLIC corundum Threads::Threads)
target_compile_options(adamant PRIVATE ${ADAMANT_WARNINGS})

# Kernels: no FMA contraction, so scalar and AVX2 apply float scales identically.
# The scalar reference is kept truly scalar (no auto-vectorization), so "scalar vs AVX2"
# compares one value per instruction against hand-written 256-bit SIMD.
set_source_files_properties(src/quant_scalar.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-tree-vectorize;-fno-tree-slp-vectorize")
if(ADAMANT_HAVE_AVX2)
  target_sources(adamant PRIVATE src/quant_avx2.cpp)
  set(_avx2_flags -mavx2 -mfma -mf16c -ffp-contract=off)
  if(MINGW)
    # GCC on Windows x64 cannot align the stack to 32 bytes for spilled __m256
    # values (GCC bug 54412): make the assembler emit unaligned moves instead.
    list(APPEND _avx2_flags -Wa,-muse-unaligned-vector-move)
  endif()
  set_source_files_properties(src/quant_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_avx2_flags}")
  target_compile_definitions(adamant PUBLIC ADAMANT_HAVE_AVX2=1)
endif()
if(WIN32)
  target_link_libraries(adamant PUBLIC shell32)
endif()

# --- tools ---------------------------------------------------------------------
if(ADAMANT_BUILD_TOOLS)
  add_executable(adamant-cli tools/adamant.cpp)
  set_target_properties(adamant-cli PROPERTIES OUTPUT_NAME adamant)
  target_link_libraries(adamant-cli PRIVATE adamant)
  target_compile_options(adamant-cli PRIVATE ${ADAMANT_WARNINGS})

  add_executable(adamant-view tools/adamant_view.cpp)
  target_link_libraries(adamant-view PRIVATE adamant)
  target_compile_options(adamant-view PRIVATE ${ADAMANT_WARNINGS})
  if(MINGW)
    target_link_options(adamant-cli PRIVATE -static)
    target_link_options(adamant-view PRIVATE -static)
  endif()
endif()

# --- tests ---------------------------------------------------------------------
if(ADAMANT_BUILD_TESTS)
  enable_testing()
  foreach(t test_fp16 test_quant test_kernels test_gguf test_tokenizer test_thread_pool test_model test_stream)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE adamant)
    target_compile_options(${t} PRIVATE ${ADAMANT_WARNINGS})
    if(MINGW)
      target_link_options(${t} PRIVATE -static)
    endif()
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
  endforeach()
  set_tests_properties(test_model test_gguf PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
