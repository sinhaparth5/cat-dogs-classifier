#include "neural_network.hpp"
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main() {
    try {
        // Get the executable path and set up data directory path
        fs::path executablePath = fs::current_path();
        fs::path dataPath = executablePath / ".." / "data";
        
        // Check if data directory exists
        if (!fs::exists(dataPath)) {
            std::cerr << "Error: Data directory not found at " << dataPath << std::endl;
            std::cerr << "Please ensure the following directory structure exists:" << std::endl;
            std::cerr << "data/" << std::endl;
            std::cerr << "├── training_set/" << std::endl;
            std::cerr << "│   ├── cats/" << std::endl;
            std::cerr << "│   └── dogs/" << std::endl;
            std::cerr << "└── test_set/" << std::endl;
            std::cerr << "    ├── cats/" << std::endl;
            std::cerr << "    └── dogs/" << std::endl;
            return 1;
        }

        NeuralNetwork nn;
        
        // Train the network
        std::cout << "Training the network..." << std::endl;
        std::cout << "Using data directory: " << dataPath << std::endl;
        
        // Check training directory structure
        fs::path trainPath = dataPath / "training_set";
        fs::path testPath = dataPath / "test_set";
        
        if (!fs::exists(trainPath)) {
            std::cerr << "Error: Training directory not found at " << trainPath << std::endl;
            return 1;
        }
        
        if (!fs::exists(trainPath / "cats") || !fs::exists(trainPath / "dogs")) {
            std::cerr << "Error: Missing cats or dogs directory in training path" << std::endl;
            return 1;
        }
        
        // Start training
        nn.train(dataPath.string(), 10);
        
        // Test the network
        std::cout << "\nTesting the network..." << std::endl;
        
        if (!fs::exists(testPath)) {
            std::cerr << "Error: Test directory not found at " << testPath << std::endl;
            return 1;
        }
        
        for (const auto& entry : fs::directory_iterator(testPath)) {
            for (const auto& image : fs::directory_iterator(entry.path())) {
                cv::Mat img = cv::imread(image.path().string());
                if (img.empty()) {
                    std::cerr << "Warning: Could not read image " << image.path() << std::endl;
                    continue;
                }
                
                std::string prediction = nn.predict(img);
                std::cout << "Image: " << image.path().filename() 
                         << " - Predicted: " << prediction << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}