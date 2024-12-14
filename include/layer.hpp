#pragma once
#include <vector>

class Layer {
public:
    Layer(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradients, double learningRate);
    void updateWeights();
    int getOutputSize() const { return outputSize; }
    
private:
    int inputSize, outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;
    std::vector<double> lastOutput;
    std::vector<std::vector<double>> weightGradients;
    std::vector<double> biasGradients;
    
    static double sigmoid(double x);
    static double sigmoid_derivative(double x);
    void initializeWeights();
};