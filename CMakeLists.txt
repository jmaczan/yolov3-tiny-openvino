cmake_minimum_required(VERSION 3.10)
project(yolov3-tiny-openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
include_directories("/usr/local/include/opencv4")
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(yolov3-tiny-openvino "src/inference.cpp" "src/utils.cpp")
target_link_libraries(yolov3-tiny-openvino PRIVATE openvino::runtime ${OpenCV_LIBS})