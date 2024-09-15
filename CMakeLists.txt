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


# Set optimization flags
if(APPLE)
    execute_process(COMMAND sysctl -n machdep.cpu.brand_string
                    OUTPUT_VARIABLE CPU_BRAND
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(CPU_BRAND MATCHES "Apple M([0-9]+)")
        string(REGEX REPLACE ".*Apple M([0-9]+).*" "\\1" APPLE_SILICON_GEN "${CPU_BRAND}")
        set(ARCH_FLAGS "-mcpu=apple-m${APPLE_SILICON_GEN}")
        message(STATUS "Detected Apple Silicon M${APPLE_SILICON_GEN}")
    else()
        set(ARCH_FLAGS "-march=native -mtune=native")
    endif()
elseif(UNIX AND NOT APPLE)
    set(ARCH_FLAGS "-march=native -mtune=native")
elseif(WIN32)
    message(STATUS "Detected Windows")
    set(ARCH_FLAGS "") # TODO
else()
    message(STATUS "Unknown operating system")
    set(ARCH_FLAGS "-march=native -mtune=native")
endif()


# For Debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
add_compile_definitions($<$<CONFIG:Debug>:DEBUG_BUILD>)

# For Benchmark
set(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -DBENCHMARK_BUILD ${ARCH_FLAGS}")

# For Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${ARCH_FLAGS}")


# OpenVINO
if(APPLE)
    set(OpenVINO_DIR "/opt/homebrew/Cellar/openvino/2024.3.0/lib/cmake")
elseif(UNIX AND NOT APPLE)
    set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
endif()

find_package(OpenVINO REQUIRED)


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(APPLE)
    include_directories(${OpenVINO_INCLUDE_DIRS})
elseif(UNIX AND NOT APPLE)
    include_directories("/usr/local/include/opencv4")
endif()


add_executable(yolov3-tiny-openvino src/inference.cpp src/person_detector.cpp)

target_link_libraries(yolov3-tiny-openvino PRIVATE openvino::runtime ${OpenCV_LIBS})

file(COPY ${CMAKE_SOURCE_DIR}/src/person_detector.hpp DESTINATION ${CMAKE_BINARY_DIR})