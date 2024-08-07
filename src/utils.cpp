#include "utils.hpp"
#include "constants.hpp"

ov::Tensor imageToTensor(const std::string& image_path, const ov::element::Type image_element_type)
{
    cv::Mat image = cv::imread(image_path);
    cv::Size size = image.size();

    if (size.width != YOLO_INPUT_DIMENSIONS || size.height != YOLO_INPUT_DIMENSIONS)
    {
        cv::resize(image, image, cv::Size(YOLO_INPUT_DIMENSIONS, YOLO_INPUT_DIMENSIONS), 0, 0, cv::INTER_LINEAR);
    }

    return ov::Tensor(image_element_type, ov::Shape{ 1, 3, static_cast<size_t>(YOLO_INPUT_DIMENSIONS), static_cast<size_t>(YOLO_INPUT_DIMENSIONS) });
}