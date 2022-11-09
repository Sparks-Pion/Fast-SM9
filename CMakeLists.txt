cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_ARCHITECTURES 75)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CUDA_NVCC_FLAGS -O3;-G;-g;-w)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Ofast  -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast  -Wall")

project(HUST-SM9)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(miracl)

file (GLOB SM9_LIB_SOURCES  "sm9/*.c" "sm9/*.cu")
file (GLOB SM9_LIB_HEADERS  "sm9/*.h" "sm9/*.cuh")
cuda_add_library(sm9lib STATIC ${SM9_LIB_SOURCES})
target_link_libraries(sm9lib miracl)

file (GLOB CUDA_SOURCES  "cuda/*.cu")
file (GLOB CUDA_HEADERS  "cuda/*.cuh")
cuda_add_library(cudalib STATIC ${CUDA_SOURCES})
# target_link_libraries(cudalib ${CUDA_LIBRARIES})
target_link_libraries(cudalib miracl)

cuda_add_executable(HUST-SM9 SM9Test.c)
target_link_libraries(HUST-SM9 sm9lib cudalib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})#/bin) 