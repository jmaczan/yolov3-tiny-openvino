# yolov3-tiny-openvino

High performance human detector using YOLOv3-tiny with OpenVINO and OpenCV in C++

## Build

```sh
chmod +x build_release.sh
./build_release.sh
```

Output path is `build/yolov3-tiny-openvino`

## Run

Download onnx model from here https://docs.openvino.ai/2022.3/omz_models_model_yolo_v3_tiny_onnx.html

```sh
cd build && ./yolov3-tiny-openvino <path_to_model_in_onnx_format> <path_to_input_image> [compile_target]
```

## Performance

Lenovo ThinkPad x230 8GB, Debian 12, 6.1.0-23-amd64:

- Detection: 0.32s

Macbook Air M2 16GB, macOS 14.6.1:

- Detection: 0.15s

### Useful resources

- https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-nv12-input-classification.html
- https://docs.openvino.ai/2022.3/omz_models_model_yolo_v3_tiny_onnx.html
- https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
- https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__runtime__cpp__api.html#_CPPv4N2ov6TensorE
- https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application.html
- https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v3-tiny-onnx/README.md


## Cite
If you use this software in your research, please use the following citation:

```bibtex
@software{Maczan_yolov3tinyopenvino_2024,
author = {Maczan, Jędrzej Paweł},
title = {{yolov3-tiny-openvino - High performance human detector}},
url = {https://github.com/jmaczan/yolov3-tiny-openvino},
year = {2024},
publisher = {GitHub}
}
```


### License

GPL-3.0 license

Jędrzej Maczan, 2024
