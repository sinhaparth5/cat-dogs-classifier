#include "neural_network.hpp"
#include <filesystem>
#include <iostream>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

NeuralNetwork::NeuralNetwork() : learningRate(0.0001) {
    // Improved architecture with more layers and better sizes
    layers.push_back(Layer(IMAGE_SIZE * IMAGE_SIZE * 3, 256));  // Increased first layer size
    layers.push_back(Layer(256, 128));
    layers.push_back(Layer(128, 64));
    layers.push_back(Layer(64, 32));
    layers.push_back(Layer(32, 1));
}

std::vector<double> NeuralNetwork::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Convert to float and normalize to [0, 1]
    image.convertTo(processed, CV_32F, 1.0/255.0);
    
    // Resize
    cv::resize(processed, processed, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
    
    // Normalize further by subtracting mean and dividing by standard deviation
    cv::Scalar mean, stddev;
    cv::meanStdDev(processed, mean, stddev);
    processed = (processed - mean[0]) / stddev[0];

    std::vector<double> input;
    input.reserve(IMAGE_SIZE * IMAGE_SIZE * 3);
    
    // Flatten the image
    for (int i = 0; i < processed.rows; ++i) {
        for (int j = 0; j < processed.cols; ++j) {
            cv::Vec3f pixel = processed.at<cv::Vec3f>(i, j);
            input.push_back(pixel[0]);
            input.push_back(pixel[1]);
            input.push_back(pixel[2]);
        }
    }
    
    return input;
}

void NeuralNetwork::train(const std::string& dataPath, int epochs) {
    std::string trainPath = dataPath + "/training_set";
    
    if (!fs::exists(trainPath)) {
        throw std::runtime_error("Training directory not found at " + trainPath);
    }

    // Collect all image paths first
    std::vector<std::pair<std::string, bool>> trainingData;  // path, isDog
    
    for (const auto& entry : fs::directory_iterator(trainPath)) {
        std::string label = entry.path().filename().string();
        bool isDog = (label == "dogs");
        
        for (const auto& image : fs::directory_iterator(entry.path())) {
            trainingData.emplace_back(image.path().string(), isDog);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        int correct = 0, total = 0;
        
        // Shuffle training data
        std::shuffle(trainingData.begin(), trainingData.end(), gen);
        
        // Mini-batch size
        const int batchSize = 32;
        std::vector<double> batchGradients(layers.back().getOutputSize(), 0.0);
        
        for (size_t i = 0; i < trainingData.size(); ++i) {
            const auto& [imagePath, isDog] = trainingData[i];
            cv::Mat img = cv::imread(imagePath);
            
            if (img.empty()) {
                continue;
            }
            
            auto input = preprocessImage(img);
            auto output = input;
            
            // Forward pass
            for (auto& layer : layers) {
                output = layer.forward(output);
            }
            
            double target = isDog ? 1.0 : 0.0;
            double prediction = output[0];
            
            // Compute accuracy
            bool predictedClass = prediction > 0.5;
            if (predictedClass == isDog) correct++;
            total++;
            
            // Compute gradient for binary cross-entropy loss
            double error = prediction - target;
            
            // Backward pass with improved gradient computation
            std::vector<double> gradients = {error};
            for (int j = layers.size() - 1; j >= 0; --j) {
                gradients = layers[j].backward(gradients, learningRate);
            }
            
            // Update weights every batch_size iterations or at the end
            if ((i + 1) % batchSize == 0 || i == trainingData.size() - 1) {
                for (auto& layer : layers) {
                    layer.updateWeights();
                }
            }
        }
        
        if (total > 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Accuracy: " << (100.0 * correct / total) << "% "
                      << "(" << correct << "/" << total << " images)" << std::endl;
        }
        
        // Reduce learning rate over time
        learningRate *= 0.95;
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