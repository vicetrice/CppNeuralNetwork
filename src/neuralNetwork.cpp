#include "neuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "utils.hpp"

namespace vicetriceNN
{
    void neuralNetwork::addLayer(int input_size, int output_size)
    {
        layers.emplace_back(input_size, output_size);
    }

    std::vector<float> neuralNetwork::forward(
        const std::vector<float> &input,
        std::vector<std::vector<float>> &layer_outputs) const
    {
        layer_outputs.clear();
        layer_outputs.push_back(input);

        std::vector<float> current = input;

        for (size_t l = 0; l < layers.size(); l++)
        {
            current = layers[l].forward(current);

            if (l == layers.size() - 1)
                current = softmax(current);

            layer_outputs.push_back(current);
        }

        return current;
    }

    void neuralNetwork::backward(
        const std::vector<float> &target,
        const std::vector<std::vector<float>> &layer_outputs)
    {
        int L = static_cast<int>(layers.size());
        std::vector<float> delta;

        {
            const std::vector<float> &output = layer_outputs[L];
            delta.resize(output.size());

            for (size_t i = 0; i < output.size(); i++)
                delta[i] = output[i] - target[i];
        }

        for (int l = L - 1; l >= 0; l--)
        {
            neuronLayer &layer = layers[l];
            auto &W = layer.getWeights();
            auto &B = layer.getBiases();

            const std::vector<float> &prev_output = layer_outputs[l];

            std::vector<float> new_delta;

            if (l > 0)
            {
                new_delta.assign(prev_output.size(), 0.0f);

                for (size_t i = 0; i < W.size(); i++)
                    for (size_t j = 0; j < W[i].size(); j++)
                        new_delta[j] += delta[i] * W[i][j];

                for (size_t j = 0; j < new_delta.size(); j++)
                    new_delta[j] *= relu_derivative(prev_output[j]);
            }

            for (size_t i = 0; i < W.size(); i++)
            {
                for (size_t j = 0; j < W[i].size(); j++)
                    W[i][j] -= learning_rate * (delta[i] * prev_output[j] + lambda * W[i][j]);

                B[i] -= learning_rate * delta[i];
            }

            if (l > 0)
                delta = new_delta;
        }
    }

    float neuralNetwork::cross_entropy_loss(const std::vector<float> &output, int label)
    {
        const float epsilon = 1e-9f;
        return -std::log(output[label] + epsilon);
    }

    std::vector<float> neuralNetwork::predict(const std::vector<float> &input) const
    {
        std::vector<std::vector<float>> dummy;
        return forward(input, dummy);
    }

    void neuralNetwork::train(const mnist_images &dataset, int epochs, int batch_size)
    {
        size_t num_samples = dataset.size();
        std::vector<size_t> indices(num_samples);
        for (size_t i = 0; i < num_samples; i++)
            indices[i] = i;

        std::mt19937 rng(0);

        for (int e = 0; e < epochs; e++)
        {
            std::shuffle(indices.begin(), indices.end(), rng);
            float epoch_loss = 0.0f;

            for (size_t start = 0; start < num_samples; start += batch_size)
            {
                size_t end = std::min(start + batch_size, num_samples);
                for (size_t idx = start; idx < end; idx++)
                {
                    size_t i = indices[idx];
                    const auto &img = dataset.getImage(i);

                    std::vector<float> input_vec(img.size());
                    for (size_t p = 0; p < img.size(); p++)
                        input_vec[p] = img[p].r / 255.0f;

                    int label = dataset.getLabel(i);

                    std::vector<std::vector<float>> layer_outputs;
                    auto output = forward(input_vec, layer_outputs);

                    epoch_loss += cross_entropy_loss(output, label);

                    std::vector<float> target(output.size(), 0.0f);
                    target[label] = 1.0f;

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

        for (size_t i = 0; i < num_samples; i++)
        {
            const auto &img = dataset.getImage(i);
            std::vector<float> input_vec(img.size());
            for (size_t p = 0; p < img.size(); p++)
                input_vec[p] = img[p].r / 255.0f;

            auto output = predict(input_vec);

            size_t predicted = 0;
            float max_val = output[0];
            for (size_t j = 1; j < output.size(); j++)
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

    void neuralNetwork::saveFullModel(const std::string &filename) const
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            return;

        const uint32_t magic = 0x4E4E4655;
        file.write(reinterpret_cast<const char *>(&magic), sizeof(uint32_t));

        file.write(reinterpret_cast<const char *>(&learning_rate), sizeof(float));
        file.write(reinterpret_cast<const char *>(&lambda), sizeof(float));
        file.write(reinterpret_cast<const char *>(&loss), sizeof(float));

        size_t num_layers = layers.size();
        file.write(reinterpret_cast<const char *>(&num_layers), sizeof(size_t));

        for (const auto &layer : layers)
        {
            const auto &W = layer.getWeights();
            const auto &B = layer.getBiases();

            size_t rows = W.size();
            size_t cols = W[0].size();

            file.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));

            for (size_t i = 0; i < rows; i++)
                file.write(reinterpret_cast<const char *>(W[i].data()), cols * sizeof(float));

            file.write(reinterpret_cast<const char *>(B.data()), B.size() * sizeof(float));
        }

        file.close();
    }

    void neuralNetwork::saveWeights(const std::string &filename) const
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            return;

        for (const auto &layer : layers)
        {
            const auto &W = layer.getWeights();
            const auto &B = layer.getBiases();

            size_t rows = W.size();
            size_t cols = W[0].size();

            file.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));

            for (size_t i = 0; i < rows; i++)
                file.write(reinterpret_cast<const char *>(W[i].data()), cols * sizeof(float));

            file.write(reinterpret_cast<const char *>(B.data()), B.size() * sizeof(float));
        }
        file.close();
    }

    bool neuralNetwork::loadWeights(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        for (auto &layer : layers)
        {
            auto &W = layer.getWeights();
            auto &B = layer.getBiases();

            size_t rows, cols;
            file.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
            file.read(reinterpret_cast<char *>(&cols), sizeof(size_t));

            W.resize(rows, std::vector<float>(cols));
            B.resize(rows);

            for (size_t i = 0; i < rows; i++)
                file.read(reinterpret_cast<char *>(W[i].data()), cols * sizeof(float));

            file.read(reinterpret_cast<char *>(B.data()), B.size() * sizeof(float));
        }
        file.close();
        return true;
    }

    bool neuralNetwork::loadFullModel(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        uint32_t magic = 0;
        file.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));

        
        if (magic != 0x4E4E4655)
        {
            file.close();
            return loadWeights(filename); 
        }

        
        file.read(reinterpret_cast<char *>(&learning_rate), sizeof(float));
        file.read(reinterpret_cast<char *>(&lambda), sizeof(float));
        file.read(reinterpret_cast<char *>(&loss), sizeof(float));

        size_t num_layers = 0;
        file.read(reinterpret_cast<char *>(&num_layers), sizeof(size_t));

        if (num_layers != layers.size())
        {
            file.close();
            return false;
        }

        
        for (auto &layer : layers)
        {
            auto &W = layer.getWeights();
            auto &B = layer.getBiases();

            size_t rows, cols;
            file.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
            file.read(reinterpret_cast<char *>(&cols), sizeof(size_t));

            W.resize(rows, std::vector<float>(cols));
            B.resize(rows);

            for (size_t i = 0; i < rows; i++)
                file.read(reinterpret_cast<char *>(W[i].data()), cols * sizeof(float));

            file.read(reinterpret_cast<char *>(B.data()), B.size() * sizeof(float));
        }

        file.close();
        return true;
    }

}
