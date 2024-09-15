#include <iostream>
#include "person_detector.hpp"
#if defined(DEBUG_BUILD) || defined(BENCHMARK_BUILD)
#include <chrono>
#endif
#ifdef BENCHMARK_BUILD
#include <vector>
#endif

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

#ifdef DEBUG_BUILD
    auto start_timer = std::chrono::high_resolution_clock::now();
#endif

#ifdef BENCHMARK_BUILD
    std::vector<double> execution_times;
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        auto start_timer = std::chrono::high_resolution_clock::now();
#endif
        detector.detect(image_path);
#ifdef BENCHMARK_BUILD
        auto end_timer = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_timer - start_timer;
        execution_times.push_back(duration.count());
    }
    double max_time = *std::max_element(execution_times.begin(), execution_times.end());
    double min_time = *std::min_element(execution_times.begin(), execution_times.end());
    double median_time = execution_times[iterations / 2];  // ofc needs a sorted vector
    double average_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / iterations;
    std::cout << "Execution time statistics (ms)" << " over " << iterations << " iterations:" << std::endl;
    std::cout << "  Min: " << min_time << std::endl;
    std::cout << "  Max: " << max_time << std::endl;
    std::cout << "  Median: " << median_time << std::endl;
    std::cout << "  Average: " << average_time << std::endl;
#endif
#ifdef DEBUG_BUILD
    auto end_timer = std::chrono::high_resolution_clock::now();
    std::cout << "It took " << std::chrono::duration_cast<std::chrono::duration<float>>(end_timer - start_timer) << " seconds to run detection on an image" << std::endl;
#endif
}