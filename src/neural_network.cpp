#include "neural_network.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

NeuralNetwork::NeuralNetwork() {
    // Initialize layers with proper sizes
    layers.push_back(Layer(IMAGE_SIZE * IMAGE_SIZE * 3, 128));
    layers.push_back(Layer(128, 64));
    layers.push_back(Layer(64, 1));
}

std::vector<double> NeuralNetwork::preprocessImage(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
    
    std::vector<double> input;
    input.reserve(IMAGE_SIZE * IMAGE_SIZE * 3);
    
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; ++c) {
                input.push_back(pixel[c] / 255.0);
            }
        }
    }
    
    return input;
}

void NeuralNetwork::train(const std::string& dataPath, int epochs) {
    std::string trainPath = dataPath + "/training_set";
    
    if (!fs::exists(trainPath)) {
        throw std::runtime_error("Training directory not found at " + trainPath);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        int correct = 0, total = 0;
        
        for (const auto& entry : fs::directory_iterator(trainPath)) {
            std::string label = entry.path().filename().string();
            double target = (label == "dogs") ? 1.0 : 0.0;
            
            for (const auto& image : fs::directory_iterator(entry.path())) {
                cv::Mat img = cv::imread(image.path().string());
                if (img.empty()) continue;
                
                auto input = preprocessImage(img);
                auto output = input;
                
                // Forward pass
                for (auto& layer : layers) {
                    output = layer.forward(output);
                }
                
                // Simple accuracy tracking
                bool prediction = output[0] > 0.5;
                if (prediction == (target > 0.5)) correct++;
                total++;
                
                // Backward pass (simplified)
                double error = output[0] - target;
                std::vector<double> gradients = {error};
                for (auto& layer : layers) {
                    layer.updateWeights(0.001, gradients);
                }
            }
        }
        
        if (total > 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
        }
    }
}

std::string NeuralNetwork::predict(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided for prediction");
    }

    auto input = preprocessImage(image);
    auto output = input;
    
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    
    return output[0] > 0.5 ? "dog" : "cat";
}