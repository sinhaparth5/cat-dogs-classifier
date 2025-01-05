#pragma once
#include <vector>

class Layer {
public:
    Layer(int inputSize, int outputSize);
    void initializeWeights();
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradients, double learningRate);
    void updateWeights();
    int getOutputSize() const { return outputSize; }

    // Utility functions for activation
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    
private:
    int inputSize, outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;
    std::vector<double> lastOutput;
    std::vector<std::vector<double>> weightGradients;
    std::vector<double> biasGradients;
};