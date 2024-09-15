#include "person_detector.hpp"
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

constexpr float CONFIDENCE_THRESHOLD = 0.5;
constexpr int PERSON_CLASS_ID = 0;
constexpr int YOLO_INPUT_DIMENSIONS = 416;
constexpr int YOLO_INPUT_DIMENSIONS_SQUARE = YOLO_INPUT_DIMENSIONS * YOLO_INPUT_DIMENSIONS;
constexpr int YOLO_INPUT_CHANNELS = 3;
constexpr size_t YOLO_INPUT_CHANNELS_SIZE_T = 3;
constexpr size_t YOLO_INPUT_DIMENSIONS_SIZE_T = 416;
constexpr float SCALE_FACTOR = 1.0f / 255.0f;
const ov::Tensor image_shape_tensor = ov::Tensor(ov::element::f32, { 1, 2 }, std::vector<float>{YOLO_INPUT_DIMENSIONS_SIZE_T, YOLO_INPUT_CHANNELS_SIZE_T}.data());

namespace person_detector {
    PersonDetector::PersonDetector(const std::string& model_path, const std::string& compile_target) : core_(), compile_target_(compile_target) {
        try {
            model_ = core_.read_model(model_path);
            compiled_model_ = core_.compile_model(model_, compile_target, {
               ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                });
            infer_request_ = compiled_model_.create_infer_request();
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error initializing PersonDetector: " + std::string(e.what()));
        }
    }

    void PersonDetector::detect(const std::string& image_path) {
        try {
            input_tensor_ = preprocess_input(image_path);
            infer_request_.set_tensor("input_1", input_tensor_);
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

        cv::parallel_for_(cv::Range(0, YOLO_INPUT_DIMENSIONS), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; ++r) {
                for (int c = 0; c < YOLO_INPUT_DIMENSIONS; ++c) {
                    const cv::Vec3b& pixel = resized_image.at<cv::Vec3b>(r, c);
                    int idx = r * YOLO_INPUT_DIMENSIONS + c;
                    raw_input_image[idx] = pixel[2] * SCALE_FACTOR;
                    raw_input_image[idx + YOLO_INPUT_DIMENSIONS_SQUARE] = pixel[1] * SCALE_FACTOR;
                    raw_input_image[idx + 2 * YOLO_INPUT_DIMENSIONS_SQUARE] = pixel[0] * SCALE_FACTOR;
                }
            }
            });

        return ov::Tensor(ov::element::f32, { 1, YOLO_INPUT_CHANNELS_SIZE_T, YOLO_INPUT_DIMENSIONS_SIZE_T, YOLO_INPUT_DIMENSIONS_SIZE_T }, raw_input_image.data());
    }

    void PersonDetector::process_outputs() {
        boxes_tensor_ = infer_request_.get_tensor("yolonms_layer_1");
        scores_tensor_ = infer_request_.get_tensor("yolonms_layer_1:1");
        indices_tensor_ = infer_request_.get_tensor("yolonms_layer_1:2");

        const float* scores_data = scores_tensor_.data<const float>();
        const int32_t* indices_data = indices_tensor_.data<const int32_t>();

        size_t num_candidates = boxes_tensor_.get_shape()[1];
        size_t num_detections = indices_tensor_.get_shape()[1];

        bool person_detected = false;

        for (size_t i = 0; i < num_detections; ++i) {
            int class_idx = indices_data[i * 3 + 1];
            int box_idx = indices_data[i * 3 + 2];

            if (class_idx == PERSON_CLASS_ID) {
                float confidence = scores_data[class_idx * num_candidates + box_idx];
                std::cout << "Confidence: " << confidence << std::endl;
                if (confidence > CONFIDENCE_THRESHOLD) {
                    std::cout << "Person detected" << std::endl;
                    person_detected = true;
                    return;
                }
            }
        }

        std::cout << "No person detected" << std::endl;
    }
}