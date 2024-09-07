# yolov3-tiny-openvino

High performance human detector using YOLOv3-tiny with OpenVINO and OpenCV in C++

> Designed for a person detection on a very low-end hardware like my Lenovo ThinkPad x230 <3

Still in development

## Benchmark

Benchmarked on macOS 14.6.1, Air M2 16GB with `hyperfine --warmup 3 --runs 50`

```
Time (mean ± σ):     565.9 ms ±  16.8 ms    [User: 931.6 ms, System: 91.6 ms]
Range (min … max):   551.1 ms … 638.6 ms    50 runs
```

More benchmarks on low-end hw will come later

### Useful resources

- https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-nv12-input-classification.html
- https://docs.openvino.ai/2022.3/omz_models_model_yolo_v3_tiny_onnx.html
- https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
- https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__runtime__cpp__api.html#_CPPv4N2ov6TensorE
- https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application.html
- https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v3-tiny-onnx/README.md

### License

GPL-3.0 license

Jędrzej Maczan, 2024
