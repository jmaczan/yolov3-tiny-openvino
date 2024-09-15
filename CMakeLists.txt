cmake_minimum_required(VERSION 3.10)
project(yolov3-tiny-openvino)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For Debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
add_compile_definitions($<$<CONFIG:Debug>:DEBUG_BUILD>)

# For Benchmark
set(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -DBENCHMARK_BUILD")

# For Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# OpenVINO

## Debian
# set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")

## macOS (brew)
set(OpenVINO_DIR "/opt/homebrew/Cellar/openvino/2024.3.0/lib/cmake")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# OpenCV

## Debian
# include_directories("/usr/local/include/opencv4")

## macOS (brew)
include_directories(${OpenVINO_INCLUDE_DIRS})

add_executable(yolov3-tiny-openvino src/inference.cpp src/person_detector.cpp)

target_link_libraries(yolov3-tiny-openvino PRIVATE openvino::runtime ${OpenCV_LIBS})

file(COPY ${CMAKE_SOURCE_DIR}/src/person_detector.hpp DESTINATION ${CMAKE_BINARY_DIR})