#include "neuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "utils.hpp"
#include <numeric>

namespace vicetriceNN
{
    void neuralNetwork::addLayer(int input_size, int output_size)
    {
        layers.emplace_back(input_size, output_size);
    }

    std::vector<float> neuralNetwork::forward(const std::vector<float> &input, std::vector<std::vector<float>> &layer_outputs) const
    {
        layer_outputs[0] = input;
        std::vector<float> current(input.size());
        std::copy(input.begin(), input.end(), current.begin());

        for (size_t l = 0; l < layers.size(); ++l)
        {
            current = layers[l].forward(current);
            if (l == layers.size() - 1)
                softmax(current);
            layer_outputs[l + 1] = current;
        }

        return current;
    }

    void neuralNetwork::backward(const std::vector<float> &target, const std::vector<std::vector<float>> &layer_outputs)
    {
        int L = static_cast<int>(layers.size());
        std::vector<float> delta(target.size());

        computeDelta(layer_outputs[L], target, delta);

        for (int l = L - 1; l >= 0; --l)
        {
            const std::vector<float> &prev_output = layer_outputs[l];
            if (l > 0)
            {
                std::vector<float> new_delta(prev_output.size(), 0.f);
                computeNewDelta(prev_output, delta, layers[l], new_delta);
                computeWeights(layers[l], delta, prev_output);
                delta = std::move(new_delta);
            }
            else
            {
                computeWeights(layers[l], delta, prev_output);
            }
        }
    }

    float neuralNetwork::cross_entropy_loss(const std::vector<float> &output, int label)
    {
        const float epsilon = 1e-9f;
        return -std::log(output[label] + epsilon);
    }

    std::vector<float> neuralNetwork::predict(const std::vector<float> &input) const
    {
        std::vector<std::vector<float>> dummy(layers.size() + 1);
        for (size_t l = 0; l < dummy.size(); ++l)
            dummy[l].resize((l == 0 ? layers[0].getInputSize() : layers[l - 1].getOutputSize()));
        return forward(input, dummy);
    }

    void neuralNetwork::train(const mnist_images &dataset, int epochs, int batch_size)
    {
        size_t num_samples = dataset.size();
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(0);
        std::vector<std::vector<float>> layer_outputs(layers.size() + 1);
        std::vector<float> target(layers.back().getOutputSize(), 0.f);
        std::vector<float> output(layers.back().getOutputSize(), 0.f);

        layer_outputs[0].resize(layers[0].getInputSize());
        for (size_t l = 0; l < layers.size(); ++l)
            layer_outputs[l + 1].resize(layers[l].getOutputSize());

        for (int e = 0; e < epochs; ++e)
        {
            std::shuffle(indices.begin(), indices.end(), rng);
            float epoch_loss = 0.f;

            for (size_t start = 0; start < num_samples; start += batch_size)
            {
                size_t end = std::min(start + batch_size, num_samples);
                for (size_t idx = start; idx < end; ++idx)
                {
                    size_t i = indices[idx];
                    int label = dataset.getLabel(i);

                    std::copy(dataset.getImage(i).begin(), dataset.getImage(i).end(), layer_outputs[0].begin());

                    output = forward(layer_outputs[0], layer_outputs);
                    epoch_loss += cross_entropy_loss(output, label);

                    std::fill(target.begin(), target.end(), 0.f);
                    target[label] = 1.f;

                    backward(target, layer_outputs);
                }
            }

            loss = epoch_loss / num_samples;
            std::cout << "Epoch " << e + 1 << "/" << epochs << ", Loss: " << loss << "\n";
        }
    }

    float neuralNetwork::evaluate(const mnist_images &dataset) const
    {
        size_t correct = 0;
        size_t num_samples = dataset.size();

        std::vector<std::vector<float>> layer_outputs(layers.size() + 1);
        layer_outputs[0].resize(layers[0].getInputSize(), 0.f);
        for (size_t l = 0; l < layers.size(); ++l)
            layer_outputs[l + 1].resize(layers[l].getOutputSize(), 0.f);

        std::vector<float> output(layers.back().getOutputSize(), 0.f);

        for (size_t i = 0; i < num_samples; ++i)
        {

            const std::vector<float> &img = dataset.getImage(i);
            std::copy(img.begin(), img.end(), layer_outputs[0].begin());

            output = forward(layer_outputs[0], layer_outputs);

            size_t predicted = 0;
            float max_val = output[0];
            for (size_t j = 1; j < output.size(); ++j)
            {
                if (output[j] > max_val)
                {
                    max_val = output[j];
                    predicted = j;
                }
            }

            if ((int)predicted == dataset.getLabel(i))
                correct++;
        }

        return float(correct) / num_samples;
    }

    void neuralNetwork::computeDelta(const std::vector<float> &output, const std::vector<float> &target, std::vector<float> &delta)
    {
        for (size_t i = 0; i < output.size(); ++i)
            delta[i] = output[i] - target[i];
    }

    void neuralNetwork::computeWeights(neuronLayer &layer, const std::vector<float> &delta, const std::vector<float> &prev_output)
    {
        auto &W = layer.getWeights();
        auto &B = layer.getBiases();

        int in = layer.getInputSize();
        int out = layer.getOutputSize();

        for (int o = 0; o < out; ++o)
        {
            int offset = o * in;
            for (int i = 0; i < in; ++i)
            {
                W[offset + i] -= learning_rate * (delta[o] * prev_output[i] + lambda * W[offset + i]);
            }
            B[o] -= learning_rate * delta[o];
        }
    }

    void neuralNetwork::computeNewDelta(const std::vector<float> &prev_output, const std::vector<float> &old_delta, const neuronLayer &layer, std::vector<float> &new_delta)
    {
        std::fill(new_delta.begin(), new_delta.end(), 0.f);
        int in = layer.getInputSize(), out = layer.getOutputSize();
        const auto &W = layer.getWeights();
        for (int o = 0; o < out; ++o)
        {
            int offset = o * in;
            for (int i = 0; i < in; ++i)
                new_delta[i] += old_delta[o] * W[offset + i];
        }
        for (size_t i = 0; i < new_delta.size(); ++i)
            new_delta[i] *= relu_derivative(prev_output[i]);
    }
}