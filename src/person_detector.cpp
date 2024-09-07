#include "person_detector.hpp"
#include <iostream>
// #include <omp.h>
#include <stdexcept>
#include "constants.hpp"
#include "utils.hpp"

const ov::Tensor image_shape_tensor = ov::Tensor(ov::element::f32, { 1, 2 }, std::vector<float>{YOLO_INPUT_DIMENSIONS_SIZE_T, YOLO_INPUT_CHANNELS_SIZE_T}.data());

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
            infer_request_.set_tensor("input_1", input_tensor);
            infer_request_.set_tensor("image_shape", image_shape_tensor);

            infer_request_.start_async();
            infer_request_.wait();

            process_outputs();
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error with detection: " + std::string(e.what()));
        }
    }

    ov::Tensor PersonDetector::preprocess_input(const std::string& image_path) const {
        static cv::Mat resized_image(YOLO_INPUT_DIMENSIONS, YOLO_INPUT_DIMENSIONS, CV_8UC3);
        static std::vector<float> raw_input_image(YOLO_INPUT_DIMENSIONS_SQUARE * YOLO_INPUT_CHANNELS); // RGB

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR); // BGR

        if (image.empty()) {
            throw std::runtime_error("Can't read image from " + image_path);
        }

        cv::resize(image, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat channels[3];
        cv::split(resized_image, channels);

#pragma omp parallel for
        for (int i = 0; i < YOLO_INPUT_DIMENSIONS_SQUARE; ++i) {
            raw_input_image[i] = channels[2].data[i] * SCALE_FACTOR;
            raw_input_image[i + YOLO_INPUT_DIMENSIONS_SQUARE] = channels[1].data[i] * SCALE_FACTOR;
            raw_input_image[i + 2 * YOLO_INPUT_DIMENSIONS_SQUARE] = channels[0].data[i] * SCALE_FACTOR;
        }

        return ov::Tensor(ov::element::f32, { 1, YOLO_INPUT_CHANNELS_SIZE_T, YOLO_INPUT_DIMENSIONS_SIZE_T, YOLO_INPUT_DIMENSIONS_SIZE_T }, raw_input_image.data());
    }

    void PersonDetector::process_outputs() {

    }
}