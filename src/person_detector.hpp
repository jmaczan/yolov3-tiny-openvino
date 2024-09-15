#ifndef PERSON_DETECTOR_HPP
#define PERSON_DETECTOR_HPP

#include <string>
#include <openvino/openvino.hpp>

namespace person_detector {

    class PersonDetector {
    public:
        PersonDetector(const std::string& model_path, const std::string& compile_target = "AUTO");
        void detect(const std::string& image_path);

    private:
        ov::Core core_;
        std::shared_ptr<ov::Model> model_;
        ov::CompiledModel compiled_model_;
        ov::InferRequest infer_request_;
        std::string compile_target_;

        ov::Tensor input_tensor_;
        ov::Tensor boxes_tensor_;
        ov::Tensor scores_tensor_;
        ov::Tensor indices_tensor_;

        ov::Tensor preprocess_input(const std::string& image_path) const;
        void process_outputs();
    };

}

#endif