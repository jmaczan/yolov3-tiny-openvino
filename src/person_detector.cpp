#include "person_detector.hpp"
#include <iostream>
#include <stdexcept>
#include "constants.hpp"
#include "utils.hpp"

const ov::Tensor image_shape_tensor = ov::Tensor(ov::element::f32, { 1, 2 }, std::vector<float>{YOLO_INPUT_DIMENSIONS_SIZE_T, YOLO_INPUT_CHANNELS_SIZE_T}.data());
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int PERSON_CLASS_ID = 0;

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
        ov::Tensor boxes_tensor = infer_request_.get_tensor("yolonms_layer_1");
        ov::Tensor scores_tensor = infer_request_.get_tensor("yolonms_layer_1:1");
        ov::Tensor indices_tensor = infer_request_.get_tensor("yolonms_layer_1:2");

        const float* boxes_data = boxes_tensor.data<const float>();
        const float* scores_data = scores_tensor.data<const float>();
        const int32_t* indices_data = indices_tensor.data<const int32_t>();

        size_t num_candidates = boxes_tensor.get_shape()[1];
        size_t num_classes = scores_tensor.get_shape()[1];
        size_t num_detections = indices_tensor.get_shape()[1];

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        bool person_detected = false;

        for (size_t i = 0; i < num_detections; ++i) {
            int batch_idx = indices_data[i * 3];
            int class_idx = indices_data[i * 3 + 1];
            int box_idx = indices_data[i * 3 + 2];

            if (class_idx != PERSON_CLASS_ID) {
                continue;
            }

            float confidence = scores_data[class_idx * num_candidates + box_idx];

            if (confidence > CONFIDENCE_THRESHOLD) {
                class_ids.push_back(class_idx);
                person_detected = true;
                break;
            }

        }

        std::cout << "Person detected: " << (person_detected ? "yes" : "no") << std::endl;
        std::cout << *class_ids.data() << std::endl;
    }
}