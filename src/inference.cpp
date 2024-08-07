#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    ov::Core core;
    
    const std::string model_path{argv[2]};
    //char* compile_target = argv[3]; // "CPU" or "GPU"
    //char* image_path
    std::shared_ptr<ov::Model> model = core.read_model("/home/user/models/yolov3-tiny/public/yolo-v3-tiny-onnx/FP32/yolo-v3-tiny-onnx.xml"); // model_path
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU"); // compile_target
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    ov::Tensor input_tensor = infer_request.get_input_tensor();
    float* input_data = input_tensor.data<float>();
    ov::Tensor tensor = imageToTensor(imagePath, inputShape);
}