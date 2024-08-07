g++ -std=c++17 inference.cpp \
    -I/opt/intel/openvino_2023.3.0/runtime/include \
    -L/opt/intel/openvino_2023.3.0/runtime/lib/intel64 \
    -lopenvino \
    -Wl, -rpath=/opt/intel/openvino_2023.3.0/runtime/lib/intel64 \
    -o yolov3-tiny-openvino