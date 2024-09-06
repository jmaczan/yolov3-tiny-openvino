#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

ov::Tensor imageToTensor(const std::string& image_path, const ov::element::Type image_element_type);

#endif