#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
    ov::Core core;

    if (argc < 3)
    {
        std::cerr << "Too few arguments: " << argc << " provided" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const std::string model_path{ argv[1] };
    const std::string image_path{ argv[2] };
    const std::string compile_target{ argc == 4 ? argv[3] : "AUTO" };

    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::CompiledModel compiled_model = core.compile_model(model, compile_target);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor image = imageToTensor(image_path, model->get_output_element_type(0));
    infer_request.set_input_tensor(0, image);

    // either sync
    infer_request.infer();

    // or async
    // infer_request.start_async();
    // infer_request.wait();

    ov::Tensor boxes_scores = infer_request.get_output_tensor(1);

    const float* boxes_scores_buffer = boxes_scores.data<const float>();

    std::cout << boxes_scores.get_size();
}