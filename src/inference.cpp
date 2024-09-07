#include <iostream>
#include "constants.hpp"
#include "person_detector.hpp"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_model> <path_to_image> [compile_target]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const std::string model_path{ argv[1] };
    const std::string image_path{ argv[2] };
    const std::string compile_target{ argc == 4 ? argv[3] : "AUTO" };

    person_detector::PersonDetector detector = person_detector::PersonDetector(model_path, compile_target);

    detector.detect(image_path);
}