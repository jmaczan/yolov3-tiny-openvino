cmake_minimum_required(VERSION 3.10)
project(yolov3-tiny-openvino)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debian
# set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")

# macOS (brew)
set(OpenVINO_DIR "/opt/homebrew/Cellar/openvino/2024.3.0/lib/cmake")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenVINO_INCLUDE_DIRS}) # "/usr/local/include/opencv4"

add_executable(yolov3-tiny-openvino src/inference.cpp src/utils.cpp)

target_link_libraries(yolov3-tiny-openvino PRIVATE openvino::runtime ${OpenCV_LIBS})


file(COPY ${CMAKE_SOURCE_DIR}/src/constants.hpp DESTINATION ${CMAKE_BINARY_DIR})