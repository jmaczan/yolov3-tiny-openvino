#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    ov::Core core;

    const std::string model_path{ argv[1] };
    const std::string compile_target{ argv[2] };
    const std::string image_path{ argv[3] };

    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::CompiledModel compiled_model = core.compile_model(model, compile_target);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    std::string input_tensor_name = model->input(0).get_any_name();
    std::string output_tensor_name = model->output(0).get_any_name();

    ov::Tensor input_tensor = imageToTensor(image_path, model->get_output_element_type(0));
    infer_request.set_tensor(input_tensor_name, input_tensor);
    infer_request.infer();

    ov::Tensor output_tensor = infer_request.get_tensor(output_tensor_name);
    std::cout << output_tensor.get_size();
}