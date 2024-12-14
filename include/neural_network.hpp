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
    std::vector<Layer> layers;
    static const int IMAGE_SIZE = 64;
    double learningRate;
    std::vector<double> preprocessImage(const cv::Mat& image);
};