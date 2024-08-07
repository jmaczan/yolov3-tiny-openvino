#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

ov::Tensor imageToTensor(const std::string& imagePath, const ov::Shape& inputShape);

#endif UTILS_HPP