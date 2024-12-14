#include "layer.hpp"
#include <random>
#include <cmath>

Layer::Layer(int inputSize, int outputSize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, sqrt(2.0 / inputSize));
    
    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);
    
    for (auto& row : weights) {
        for (double& w : row) {
            w = d(gen);
        }
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    lastInput = input;
    lastOutput.resize(weights.size());
    
    for (size_t i = 0; i < weights.size(); ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < weights[i].size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        lastOutput[i] = relu(sum);
    }
    
    return lastOutput;
}

double Layer::relu(double x) {
    return std::max(0.0, x);
}

void Layer::updateWeights(double learningRate, const std::vector<double>& gradients) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learningRate * gradients[i] * lastInput[j];
        }
        biases[i] -= learningRate * gradients[i];
    }
}