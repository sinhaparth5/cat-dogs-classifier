#pragma once
#include <vector>

class Layer {
public:
    Layer(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input);
    void updateWeights(double learningRate, const std::vector<double>& gradients);
    
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;
    std::vector<double> lastOutput;
    
    static double relu(double x);
};