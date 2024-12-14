#include "layer.hpp"
#include <random>
#include <cmath>

Layer::Layer(int inputSize, int outputSize) 
    : inputSize(inputSize), outputSize(outputSize) {
    initializeWeights();
}

void Layer::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, sqrt(2.0 / inputSize));  // He initialization
    
    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);
    weightGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biasGradients.resize(outputSize, 0.0);
    
    for (auto& row : weights) {
        for (double& w : row) {
            w = d(gen);
        }
    }
}

double Layer::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Layer::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    lastInput = input;
    lastOutput.resize(outputSize);
    
    for (int i = 0; i < outputSize; ++i) {
        double sum = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            sum += weights[i][j] * input[j];
        }
        lastOutput[i] = sigmoid(sum);  // Using sigmoid instead of ReLU
    }
    
    return lastOutput;
}

std::vector<double> Layer::backward(const std::vector<double>& gradients, double learningRate) {
    std::vector<double> inputGradients(inputSize, 0.0);
    
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            double grad = gradients[i] * lastInput[j];
            weightGradients[i][j] += grad;
            inputGradients[j] += gradients[i] * weights[i][j];
        }
        biasGradients[i] += gradients[i];
    }
    
    return inputGradients;
}

void Layer::updateWeights() {
    const double learningRate = 0.0001;
    const double momentum = 0.9;
    
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] -= learningRate * weightGradients[i][j];
            weightGradients[i][j] *= momentum;
        }
        biases[i] -= learningRate * biasGradients[i];
        biasGradients[i] *= momentum;
    }
}