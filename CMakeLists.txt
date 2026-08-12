cmake_minimum_required(VERSION 3.20)
project(daba_mm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_executable(daba_mm daba_mm.cu)
# Explicit target property (not just the CMAKE_CUDA_ARCHITECTURES default, which
# CMake didn't reliably apply here) -- 89 = RTX 2000 Ada (Ada Lovelace); override as
# needed. atomicAdd(double*, double) requires >=60 (Pascal+) so this must not be left
# to an older implicit default.
set_target_properties(daba_mm PROPERTIES
    CUDA_ARCHITECTURES "89"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(daba_mm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
