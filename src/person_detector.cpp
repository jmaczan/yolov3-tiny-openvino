#include "person_detector.hpp"
#include <iostream>
#include <stdexcept>
#include "utils.hpp"

namespace person_detector {
    PersonDetector::PersonDetector(const std::string& model_path, const std::string& compile_target) : core_(), compile_target_(compile_target) {
        try {
            model_ = core_.read_model(model_path);
            compiled_model_ = core_.compile_model(model_, compile_target);
            infer_request_ = compiled_model_.create_infer_request();
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error initializing PersonDetector: " + std::string(e.what()));
        }
    }

    void PersonDetector::detect(const std::string& image_path) {
        try {
            const ov::Tensor input_tensor = preprocess_input(image_path);
            infer_request_.set_input_tensor(input_tensor);

            infer_request_.start_async();
            infer_request_.wait();

            process_outputs();
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error with detection: " + std::string(e.what()));
        }
    }

    ov::Tensor PersonDetector::preprocess_input(const std::string& image_path) const {
        cv::Mat image = cv::imread(image_path);

        if (image.empty()) {
            throw std::runtime_error("Can't read image from " + image_path);
        }

        ov::Shape expected_shape = model_->input().get_shape();
        int expected_height = expected_shape[2];
        int expected_width = expected_shape[3];

        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(expected_width, expected_height));

        return imageToTensor(image_path, model_->get_output_element_type(0));
    }

    void process_outputs() {

    }
}