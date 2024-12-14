#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "layer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork();
    void train(const std::string& dataPath, int epochs);
    std::string predict(const cv::Mat& image);

private:
    std::vector<Layer> layers;  // This was missing in the private section
    std::vector<double> preprocessImage(const cv::Mat& image);  // This was missing in the private section
    static const int IMAGE_SIZE = 64;  // Added constant for image size
};